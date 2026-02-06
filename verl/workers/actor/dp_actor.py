# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
from typing import Dict, Iterable, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis
from verl.utils.seed import seed_everything

__all__ = ['DataParallelPPOActor']


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)
        seed=self.config.get('seed', 42)
        seed_everything(seed)
        self.byol_align_cfg = self.config.get('byol_align', {})
        self.byol_align_enabled = bool(self.byol_align_cfg.get('enabled', False))
        self.byol_align_weight = float(self.byol_align_cfg.get('weight', 0.0))
        self.byol_align_freeze_llm = bool(self.byol_align_cfg.get('freeze_llm', True))
        self.byol_align_debug_print_interval = int(self.byol_align_cfg.get('debug_print_interval', 10))
        self._byol_fsdp_routing_notice_printed = False
        self._actor_update_step = 0

        base_model = self._get_base_actor_module()
        self._base_param_requires_grad = {name: p.requires_grad for name, p in base_model.named_parameters()}
        self._tracked_visual_param_name = None
        self._tracked_visual_param_init = None
        if hasattr(base_model, "visual"):
            for name, param in base_model.visual.named_parameters():
                self._tracked_visual_param_name = name
                self._tracked_visual_param_init = param.detach().float().cpu().clone()
                break
        if self.byol_align_enabled:
            has_heads = hasattr(base_model, "byol_projector") and hasattr(base_model, "byol_predictor")
            print(f"[BYOL ALIGN] enabled={self.byol_align_enabled}, weight={self.byol_align_weight}, "
                  f"freeze_llm={self.byol_align_freeze_llm}, has_heads={has_heads}")

    def _get_base_actor_module(self) -> nn.Module:
        if isinstance(self.actor_module, FSDP):
            return self.actor_module._fsdp_wrapped_module
        return self.actor_module

    def _collect_multi_modal_inputs(self, micro_batch: Dict) -> Dict[str, torch.Tensor]:
        multi_modal_inputs = {}
        if 'multi_modal_inputs' in micro_batch:
            mm_inputs = micro_batch['multi_modal_inputs']
            for key in mm_inputs[0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in mm_inputs], dim=0)
        return multi_modal_inputs

    def _has_byol_alignment_batch_keys(self, data: Dict) -> bool:
        required = ['byol_target_hiddens', 'byol_pred_positions', 'byol_valid_mask']
        return all(k in data for k in required)

    def _set_byol_alignment_routing(self, enabled: bool):
        base_model = self._get_base_actor_module()
        if not self.byol_align_freeze_llm:
            return

        # Runtime requires_grad toggling is fragile with FSDP flattened params.
        # Under FSDP we keep requires_grad untouched and freeze via gradient masking.
        if isinstance(self.actor_module, FSDP):
            if enabled and not self._byol_fsdp_routing_notice_printed:
                print("[BYOL ALIGN] FSDP detected: using grad masking for freeze_llm "
                      "instead of requires_grad toggling.")
                self._byol_fsdp_routing_notice_printed = True
            return

        if enabled:
            for _, param in base_model.named_parameters():
                param.requires_grad = False
            if hasattr(base_model, 'visual'):
                for param in base_model.visual.parameters():
                    param.requires_grad = True
            if hasattr(base_model, 'byol_projector'):
                for param in base_model.byol_projector.parameters():
                    param.requires_grad = True
            if hasattr(base_model, 'byol_predictor'):
                for param in base_model.byol_predictor.parameters():
                    param.requires_grad = True
        else:
            for name, param in base_model.named_parameters():
                if name in self._base_param_requires_grad:
                    param.requires_grad = self._base_param_requires_grad[name]

    def _mask_non_byol_grads(self):
        if not self.byol_align_freeze_llm:
            return
        base_model = self._get_base_actor_module()
        trainable = []
        if hasattr(base_model, 'visual'):
            trainable.extend(list(base_model.visual.parameters()))
        if hasattr(base_model, 'byol_projector'):
            trainable.extend(list(base_model.byol_projector.parameters()))
        if hasattr(base_model, 'byol_predictor'):
            trainable.extend(list(base_model.byol_predictor.parameters()))
        trainable_ids = {id(p) for p in trainable}
        for param in base_model.parameters():
            if id(param) not in trainable_ids:
                param.grad = None

    @staticmethod
    def _grad_norm(params) -> float:
        total = 0.0
        for p in params:
            if p.grad is None:
                continue
            g = p.grad.detach().float()
            total += g.pow(2).sum().item()
        return total**0.5

    def _compute_byol_grad_debug_metrics(self) -> Dict[str, float]:
        base_model = self._get_base_actor_module()
        visual_params = list(base_model.visual.parameters()) if hasattr(base_model, 'visual') else []
        head_params = []
        if hasattr(base_model, 'byol_projector'):
            head_params.extend(list(base_model.byol_projector.parameters()))
        if hasattr(base_model, 'byol_predictor'):
            head_params.extend(list(base_model.byol_predictor.parameters()))

        tracked_ids = {id(p) for p in visual_params + head_params}
        llm_params = [p for p in base_model.parameters() if id(p) not in tracked_ids]
        return {
            'visual_grad_norm': self._grad_norm(visual_params),
            'head_grad_norm': self._grad_norm(head_params),
            'llm_grad_norm': self._grad_norm(llm_params),
        }

    def _compute_tracked_visual_drift(self) -> float:
        if self._tracked_visual_param_name is None or self._tracked_visual_param_init is None:
            return 0.0
        base_model = self._get_base_actor_module()
        for name, param in base_model.visual.named_parameters():
            if name == self._tracked_visual_param_name:
                current = param.detach().float().cpu()
                return (current - self._tracked_visual_param_init).abs().mean().item()
        return 0.0

    def _compute_byol_alignment_loss(self, micro_batch: Dict) -> Tuple[torch.Tensor, int]:
        if not self._has_byol_alignment_batch_keys(micro_batch):
            return None, 0

        base_model = self._get_base_actor_module()
        if not (hasattr(base_model, 'byol_projector') and hasattr(base_model, 'byol_predictor')):
            return None, 0
        if not hasattr(base_model, 'visual'):
            return None, 0

        input_ids = micro_batch['input_ids']
        attention_mask = micro_batch['attention_mask']
        position_ids = micro_batch['position_ids']
        if position_ids.dim() == 3:
            position_ids = position_ids.transpose(0, 1)

        multi_modal_inputs = self._collect_multi_modal_inputs(micro_batch)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                output_hidden_states=True,
                use_cache=False,
            )
        last_hidden = outputs.hidden_states[-1]  # (bsz, seq_len, hidden_dim)

        byol_positions = micro_batch['byol_pred_positions'].long()
        byol_targets = micro_batch['byol_target_hiddens']
        byol_valid = micro_batch['byol_valid_mask'].float()

        if byol_targets.dim() != 3 or byol_positions.dim() != 2 or byol_valid.dim() != 2:
            return None, 0

        seq_len = last_hidden.shape[1]
        in_range = (byol_positions >= 0) & (byol_positions < seq_len)
        valid_mask = (byol_valid > 0.5) & in_range
        valid_pairs = int(valid_mask.sum().item())
        if valid_pairs == 0:
            return None, 0

        clamped_pos = byol_positions.clamp(min=0, max=seq_len - 1)
        gather_idx = clamped_pos.unsqueeze(-1).expand(-1, -1, last_hidden.shape[-1])
        online_hidden = torch.gather(last_hidden, dim=1, index=gather_idx)

        z_online = base_model.byol_projector(online_hidden)
        y_pred = base_model.byol_predictor(z_online)
        y_target = byol_targets.to(device=y_pred.device, dtype=y_pred.dtype)

        y_pred_norm = torch.nn.functional.normalize(y_pred.float(), dim=-1, p=2)
        y_target_norm = torch.nn.functional.normalize(y_target.float(), dim=-1, p=2)
        loss_per_pair = ((y_pred_norm - y_target_norm.detach()) ** 2).sum(dim=-1)
        weight = valid_mask.float()
        byol_loss = (loss_per_pair * weight).sum() / weight.sum().clamp(min=1e-8)
        return byol_loss, valid_pairs

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        multi_modal_inputs = self._collect_multi_modal_inputs(micro_batch)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']
            
            
            #DEBUG
            # print(f"[DEBUG] input_ids.shape: {input_ids.shape}")
            # print(f"[DEBUG] attention_mask.shape: {attention_mask.shape}")
            # print(f"[DEBUG] position_ids.shape: {position_ids.shape}")
            # print(f"[DEBUG] batch_size, seqlen: {batch_size} , {seqlen}")
            
            
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."),
                                                          indices).transpose(0, 1).unsqueeze(
                                                              1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                          indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                # print(f"[DEBUG] input_ids_rmpad.shape: {input_ids_rmpad.shape}")
                # print(f"[DEBUG] position_ids_rmpad.shape: {position_ids_rmpad.shape}")
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           **multi_modal_inputs,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                # print(f"[DEBUG] input_ids.shape: {input_ids.shape}")
                # print(f"[DEBUG] attention_mask.shape: {attention_mask.shape}")
                # print(f"[DEBUG] position_ids.shape: {position_ids.shape}")
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           **multi_modal_inputs,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        
        # Calculate gradient norm
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        
        
        grad_norm_threshold = self.config.grad_norm_threshold  
        
        # Only update if grad_norm is below threshold
        if not torch.isfinite(grad_norm) or (grad_norm_threshold is not None and grad_norm >= grad_norm_threshold):
            # Skip the update
            print(f"[DEBUG] Skipping optimizer step due to high gradient norm: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            print(f"[DEBUG] Performing optimizer step with gradient norm: {grad_norm}")
            self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        if 'loss_mask' in data.batch.keys():
            select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages','loss_mask']
        else:
            select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if 'byol_loss_mean' in data.batch.keys():
            select_keys.append('byol_loss_mean')
        if 'byol_target_hiddens' in data.batch.keys():
            select_keys.append('byol_target_hiddens')
        if 'byol_pred_positions' in data.batch.keys():
            select_keys.append('byol_pred_positions')
        if 'byol_valid_mask' in data.batch.keys():
            select_keys.append('byol_valid_mask')
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                micro_batches = list(micro_batches)

                # Stage A: BYOL alignment update (freeze LLM, update visual + BYOL heads).
                run_byol_align = self.byol_align_enabled and self.byol_align_weight > 0
                if run_byol_align:
                    self._set_byol_alignment_routing(True)
                    self.actor_optimizer.zero_grad()

                    byol_valid_pairs = 0
                    byol_losses = []
                    for micro_data in micro_batches:
                        if isinstance(micro_data, DataProto):
                            micro_data = {**micro_data.batch.to(torch.cuda.current_device()), **micro_data.non_tensor_batch}
                        else:
                            micro_data = micro_data.to(torch.cuda.current_device())

                        byol_loss, valid_pairs = self._compute_byol_alignment_loss(micro_data)
                        if byol_loss is None or valid_pairs <= 0:
                            continue

                        byol_valid_pairs += valid_pairs
                        byol_losses.append(byol_loss.detach().item())
                        scaled_byol_loss = byol_loss * self.byol_align_weight
                        if self.config.use_dynamic_bsz:
                            micro_bsz = micro_data['input_ids'].shape[0]
                            loss = scaled_byol_loss * (micro_bsz / self.config.ppo_mini_batch_size)
                        else:
                            loss = scaled_byol_loss / self.gradient_accumulation
                        loss.backward()

                    if byol_valid_pairs > 0:
                        self._mask_non_byol_grads()
                        grad_debug = self._compute_byol_grad_debug_metrics()
                        byol_grad_norm = self._optimizer_step()
                        visual_drift = self._compute_tracked_visual_drift()
                        byol_loss_mean = sum(byol_losses) / len(byol_losses)
                        byol_metrics = {
                            'actor/byol_align_loss': byol_loss_mean,
                            'actor/byol_align_valid_pairs': float(byol_valid_pairs),
                            'actor/byol_align_grad_norm': byol_grad_norm.detach().item(),
                            'actor/byol_visual_grad_norm': grad_debug['visual_grad_norm'],
                            'actor/byol_head_grad_norm': grad_debug['head_grad_norm'],
                            'actor/byol_llm_grad_norm': grad_debug['llm_grad_norm'],
                            'actor/byol_visual_drift_from_init': visual_drift,
                        }
                        append_to_dict(metrics, byol_metrics)

                        if self._actor_update_step % max(self.byol_align_debug_print_interval, 1) == 0:
                            print(
                                f"[BYOL ALIGN DEBUG] step={self._actor_update_step} "
                                f"loss={byol_loss_mean:.6f} pairs={byol_valid_pairs} "
                                f"visual_grad={grad_debug['visual_grad_norm']:.6f} "
                                f"llm_grad={grad_debug['llm_grad_norm']:.6f} "
                                f"visual_drift={visual_drift:.6f}"
                            )
                    else:
                        self.actor_optimizer.zero_grad()
                        if self._actor_update_step % max(self.byol_align_debug_print_interval, 1) == 0:
                            print(f"[BYOL ALIGN DEBUG] step={self._actor_update_step} no valid BYOL pairs in mini-batch")

                    self._set_byol_alignment_routing(False)

                self.actor_optimizer.zero_grad()

                for micro_data in micro_batches:
                    # Support all hardwares
                    if isinstance(micro_data, DataProto):
                        micro_data = {**micro_data.batch.to(torch.cuda.current_device()), **micro_data.non_tensor_batch}
                    else:
                        micro_data = micro_data.to(torch.cuda.current_device())  # actor device is cpu when using offload
                    responses = micro_data['responses']
                    response_length = responses.size(1)
                    #attention_mask = data['attention_mask']
                    
                    # Note:
                    # In agent setting, the prompt is the initial obs, and reponse is the whole trajectory except initial obs
                    # The loss mask has the same shape of attention mask, which has both prompt and response, it masks:
                    # the prompt, the padding of response (right padded), and the obs given by the environment in the reponse
                    
                    if "loss_mask" in micro_data:
                        loss_mask = micro_data['loss_mask']
                    else:
                        print("DEBUG: warning, loss_mask not found in actor update")
                        loss_mask=micro_data["attention_mask"]
                    response_mask = loss_mask[:, -response_length:]
                    old_log_prob = micro_data['old_log_probs']
                    advantages = micro_data['advantages']

                    clip_ratio = self.config.clip_ratio
                    entropy_coeff = self.config.entropy_coeff

                    # all return: (bsz, response_length)
                    entropy, log_prob = self._forward_micro_batch(micro_batch=micro_data, temperature=temperature)

                    pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                                  log_prob=log_prob,
                                                                                  advantages=advantages,
                                                                                  eos_mask=response_mask,
                                                                                  cliprange=clip_ratio)
                    # compute entropy loss from entropy
                    entropy_loss = verl_F.masked_mean(entropy, response_mask)

                    # compute policy loss
                    policy_loss = pg_loss - entropy_loss * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = micro_data['ref_log_prob']
                        # compute kl loss
                        kld = core_algos.kl_penalty(logprob=log_prob,
                                                    ref_logprob=ref_log_prob,
                                                    kl_penalty=self.config.kl_loss_type)
                        kl_loss = masked_mean(kld, response_mask)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics['actor/kl_loss'] = kl_loss.detach().item()
                        metrics['actor/kl_coef'] = self.config.kl_loss_coef

                    # Optional BYOL aux loss
                    byol_aux_weight = getattr(self.config, "byol_aux_weight", 0.0)
                    if byol_aux_weight > 0 and "byol_loss_mean" in micro_data:
                        aux = micro_data["byol_loss_mean"]
                        policy_loss = policy_loss + byol_aux_weight * aux
                        metrics['actor/byol_aux_loss'] = aux.detach().item() * byol_aux_weight

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        micro_bsz = micro_data['input_ids'].shape[0]
                        loss = policy_loss * (micro_bsz / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        'actor/entropy_loss': entropy_loss.detach().item(),
                        'actor/pg_loss': pg_loss.detach().item(),
                        'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                        'actor/ppo_kl': ppo_kl.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {'actor/grad_norm': grad_norm.detach().item()}
                self._actor_update_step += 1
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
