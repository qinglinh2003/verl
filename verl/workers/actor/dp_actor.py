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
from typing import Iterable, Tuple

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
        glance_config=None,
        glance_f_phi: nn.Module = None,
    ):
        """When optimizer is None, it is Reference Policy.

        Args:
            glance_config: OmegaConf node with GLANCE hyper-parameters (or None).
            glance_f_phi: Pre-FSDP deep-copy of the visual encoder for the
                          momentum target network (or None when GLANCE is disabled).
        """
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

        # ── GLANCE initialisation ──────────────────────────────────────
        self.glance_enabled = (
            glance_config is not None
            and glance_config.get('enabled', False)
            and glance_f_phi is not None
        )
        if self.glance_enabled:
            from vagen.world_model.glance_module import GLANCEReward

            self.glance_config = glance_config
            device = torch.cuda.current_device()

            # Momentum encoder f_phi: frozen, eval-only, lives outside FSDP.
            # Kept on CPU during init so it does not compete with vLLM for
            # GPU memory; moved to GPU on-demand when computing targets.
            self.glance_f_phi = glance_f_phi  # stays on CPU
            self.glance_f_phi.eval()
            for p in self.glance_f_phi.parameters():
                p.requires_grad = False

            # GLANCEReward contains projector g_psi.
            # Also kept on CPU during init; moved to GPU before first use.
            self.glance_reward = GLANCEReward(
                vlm_hidden_dim=int(glance_config.get('vlm_hidden_dim', 2048)),
                visual_dim=int(glance_config.get('visual_dim', 2048)),
                beta=float(glance_config.get('beta', 0.1)),
                drain_eps=float(glance_config.get('drain_eps', 0.1)),
                drain_K=int(glance_config.get('drain_K', 20)),
            )  # stays on CPU
            self.glance_device = device  # target device for later .to() calls

            # Collect FSDP flat-params belonging to the vision encoder so
            # that L_explore gradients can update v alongside projector g_psi.
            # With use_orig_params=False, each FSDP sub-module has a single
            # FlatParameter; we identify visual ones by their module path.
            visual_fsdp_params = []
            for name, mod in self.actor_module.named_modules():
                if 'visual' in name and isinstance(mod, FSDP) and hasattr(mod, '_flat_param'):
                    visual_fsdp_params.append(mod._flat_param)

            glance_lr = float(glance_config.get('projector_lr', 1e-6))
            self.glance_optimizer = torch.optim.AdamW(
                [{'params': list(self.glance_reward.projector_parameters())},
                 {'params': visual_fsdp_params, 'lr': glance_lr}],
                lr=glance_lr,
            )
            self._glance_visual_fsdp_params = visual_fsdp_params

            n_phi = sum(p.numel() for p in self.glance_f_phi.parameters())
            n_psi = sum(p.numel() for p in self.glance_reward.projector.parameters())
            n_ve = sum(p.numel() for p in visual_fsdp_params)
            print(f'[GLANCE] Initialised: '
                  f'f_phi params={n_phi:,} (frozen, on CPU until needed)  '
                  f'projector params={n_psi:,}  '
                  f'visual_encoder FSDP params={n_ve:,} ({len(visual_fsdp_params)} flat tensors)  '
                  f'beta={glance_config.beta}  '
                  f'ema_decay={glance_config.ema_decay}')

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        multi_modal_inputs = {}
        if 'multi_modal_inputs' in micro_batch:
            for key in micro_batch['multi_modal_inputs'][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch['multi_modal_inputs']],
                                                    dim=0)

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

    def compute_glance_hidden_states(self, data: DataProto) -> DataProto:
        """Teacher-forcing forward to extract hidden states at </prediction> positions.

        Token id 68931 ('prediction' without a leading space) uniquely identifies
        the closing </prediction> tag in Qwen2.5-VL tokenization and does not
        appear in normal prose or in the opening <prediction> tag.

        Args:
            data: DataProto with input_ids, attention_mask, position_ids and
                  optionally multi_modal_inputs.
                  meta_info must contain 'micro_batch_size'.

        Returns:
            DataProto with:
                glance_h_pred : float32 (B, max_turns, hidden_size)
                glance_h_mask : float32 (B, max_turns), 1 for valid turns
        """
        # Token id 68931 is the 'prediction' subtoken (no leading space).
        # In context, it can appear in both ><prediction> and </prediction>.
        # We discriminate by checking whether the preceding token's decoded
        # string ends with "</", using the precomputed set passed via meta_info.
        PREDICTION_TOKEN_ID = 68931
        closing_slash_ids: frozenset = data.meta_info.get('glance_closing_slash_ids', frozenset())
        # Fixed max_turns ensures all GPUs produce tensors with identical shape
        # so DataProto.concat across workers does not fail.
        fixed_max_turns: int = data.meta_info.get('glance_max_turns', 0)
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        select_keys = ['input_ids', 'attention_mask', 'position_ids']
        has_mmi = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        if has_mmi:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            micro_batches = data.select(select_keys, ['multi_modal_inputs']).chunk(num_micro_batches)
        else:
            micro_batches = data.select(batch_keys=select_keys).batch.split(micro_batch_size)

        all_h = []
        all_mask = []

        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                mb = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            else:
                mb = micro_batch

            input_ids = mb['input_ids']
            attention_mask = mb['attention_mask']
            position_ids = mb['position_ids']
            multi_modal_inputs = {}
            if 'multi_modal_inputs' in mb:
                for key in mb['multi_modal_inputs'][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inp[key] for inp in mb['multi_modal_inputs']], dim=0)

            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (B,3,S) -> (3,B,S)

            B, seqlen = input_ids.shape

            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    if self.use_remove_padding:
                        input_ids_rmpad, indices, *_ = unpad_input(
                            input_ids.unsqueeze(-1), attention_mask)
                        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                        if position_ids.dim() == 3:
                            position_ids_rmpad = index_first_axis(
                                rearrange(position_ids, "c b s ... -> (b s) c ..."),
                                indices).transpose(0, 1).unsqueeze(1)
                        else:
                            position_ids_rmpad = index_first_axis(
                                rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                indices).transpose(0, 1)

                        output = self.actor_module(
                            input_ids=input_ids_rmpad,
                            attention_mask=None,
                            position_ids=position_ids_rmpad,
                            **multi_modal_inputs,
                            use_cache=False,
                            output_hidden_states=True,
                        )
                        # (1, total_nnz, hidden_size) -> (total_nnz, hidden_size)
                        hs_rmpad = output.hidden_states[-1].squeeze(0).float()
                        last_hidden = pad_input(
                            hidden_states=hs_rmpad,
                            indices=indices,
                            batch=B,
                            seqlen=seqlen,
                        )  # (B, seqlen, hidden_size)
                    else:
                        output = self.actor_module(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            **multi_modal_inputs,
                            use_cache=False,
                            output_hidden_states=True,
                        )
                        last_hidden = output.hidden_states[-1].float()  # (B, seqlen, hidden_size)

            hidden_size = last_hidden.shape[-1]

            pred_positions = []
            for b in range(B):
                raw = (input_ids[b] == PREDICTION_TOKEN_ID).nonzero(as_tuple=True)[0]
                if closing_slash_ids:
                    # Keep only positions where the preceding token ends with "</",
                    # which uniquely identifies the closing </prediction> tag.
                    raw = raw[
                        (raw > 0) &
                        torch.tensor(
                            [input_ids[b, p - 1].item() in closing_slash_ids for p in raw],
                            dtype=torch.bool, device=raw.device,
                        )
                    ]
                pred_positions.append(raw)
            # Use fixed_max_turns as the output dimension when provided so that
            # all data-parallel workers produce identically-shaped tensors.
            # If a sequence has more occurrences than fixed_max_turns (e.g. the
            # model hallucinated extra </prediction> tags), keep only the last
            # fixed_max_turns positions (most recent turns).
            if fixed_max_turns > 0:
                pred_positions = [p[-fixed_max_turns:] if p.numel() > fixed_max_turns else p
                                  for p in pred_positions]
                max_pred = fixed_max_turns
            else:
                local_max = max((p.numel() for p in pred_positions), default=1)
                max_pred = max(local_max, 1)

            h_pad = torch.zeros(B, max_pred, hidden_size, dtype=torch.float32,
                                device=last_hidden.device)
            m_pad = torch.zeros(B, max_pred, dtype=torch.float32,
                                device=last_hidden.device)
            for b in range(B):
                pos = pred_positions[b]
                n = pos.numel()
                if n > 0:
                    h_pad[b, :n] = last_hidden[b, pos]
                    m_pad[b, :n] = 1.0

            all_h.append(h_pad)
            all_mask.append(m_pad)

        global_max = max(h.shape[1] for h in all_h)
        hidden_size = all_h[0].shape[-1]
        B_total = sum(h.shape[0] for h in all_h)
        device = all_h[0].device
        glance_h = torch.zeros(B_total, global_max, hidden_size,
                               dtype=torch.float32, device=device)
        glance_mask = torch.zeros(B_total, global_max,
                                  dtype=torch.float32, device=device)

        offset = 0
        for h, m in zip(all_h, all_mask):
            b, n = h.shape[0], h.shape[1]
            glance_h[offset:offset + b, :n] = h
            glance_mask[offset:offset + b, :n] = m
            offset += b

        valid_counts = glance_mask.sum(dim=-1).tolist()
        print(f'[GLANCE] compute_glance_hidden_states: '
              f'batch={B_total} h_shape={tuple(glance_h.shape)} '
              f'valid_turns_per_seq={[int(v) for v in valid_counts]} '
              f'h_norm_sample={glance_h[0, 0].norm().item():.4f}')

        return DataProto.from_dict(tensors={
            'glance_h_pred': glance_h,
            'glance_h_mask': glance_mask,
        })

    def compute_glance_targets(self, data: DataProto) -> DataProto:
        """Encode next-observation images through the momentum encoder f_phi.

        For each sample and each valid turn, feeds next-obs pixel_values and
        image_grid_thw into glance_f_phi, then mean-pools patch tokens to
        produce a single (d_vis,) target vector per turn.

        Args:
            data: DataProto with:
                - glance_h_mask: (B, max_turns), 1.0 for valid turns
                - non_tensor_batch['glance_next_obs_inputs']: list of B lists,
                  each inner list has max_turns entries (dict or None per turn).
                  Each dict has 'pixel_values' and 'image_grid_thw' from the
                  Qwen2VLImageProcessor.

        Returns:
            DataProto with:
                glance_y_next: (B, max_turns, d_vis), detached target representations
        """
        assert self.glance_enabled, "GLANCE is not enabled"

        device = torch.device(f'cuda:{torch.cuda.current_device()}')
        glance_h_mask = data.batch['glance_h_mask']  # (B, max_turns)
        next_obs_inputs = data.non_tensor_batch['glance_next_obs_inputs']  # list of B items
        B, max_turns = glance_h_mask.shape

        use_momentum = self.glance_config.get('use_momentum', True)
        if use_momentum:
            encoder = self.glance_f_phi
            encoder_dtype = self.glance_f_phi.patch_embed.proj.weight.dtype
            self.glance_f_phi.to(device)
        else:
            # Ablation: use online visual encoder directly (no EMA lag).
            # FSDP model must already be loaded to GPU by the worker.
            unwrapped = self.actor_module
            if hasattr(unwrapped, '_fsdp_wrapped_module'):
                unwrapped = unwrapped._fsdp_wrapped_module
            encoder = unwrapped.visual
            # Cannot access original param dtype through FSDP (use_orig_params=False),
            # but we know the model runs in bf16 (same as autocast dtype).
            encoder_dtype = torch.bfloat16

        # Derive d_vis from the actual merger output dim, not from config.
        d_vis = encoder.merger.mlp[-1].out_features

        y_next = torch.zeros(B, max_turns, d_vis, dtype=torch.float32, device=device)

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                for b in range(B):
                    sample_inputs = next_obs_inputs[b]
                    if sample_inputs is None:
                        continue
                    for t in range(min(len(sample_inputs), max_turns)):
                        if glance_h_mask[b, t] < 0.5:
                            continue
                        turn_input = sample_inputs[t]
                        if turn_input is None:
                            continue

                        pixel_values = turn_input['pixel_values'].to(
                            device=device, dtype=encoder_dtype)
                        grid_thw = turn_input['image_grid_thw'].to(device=device)

                        # visual encoder output: (total_merged_patches, d_vis)
                        # The merger reduces patches by spatial_merge_unit, so
                        # the output length != grid_thw product. Mean-pool all
                        # output patches directly (each obs is typically 1 image).
                        patch_embeds = encoder(pixel_values, grid_thw=grid_thw)
                        y_next[b, t] = patch_embeds.float().mean(dim=0)

        if use_momentum:
            self.glance_f_phi.to('cpu')

        valid_count = int(glance_h_mask.sum().item())
        y_norms = y_next[glance_h_mask > 0.5].norm(dim=-1)
        print(f'[GLANCE] compute_glance_targets: '
              f'y_next={tuple(y_next.shape)} valid={valid_count} '
              f'norm_mean={y_norms.mean().item():.4f} norm_std={y_norms.std().item():.4f}')

        return DataProto.from_dict(tensors={
            'glance_y_next': y_next.detach().cpu(),
        })

    def compute_glance_rewards(self, data: DataProto) -> DataProto:
        """Compute per-turn intrinsic reward from GLANCE prediction error.

        For each valid turn, projects h_{t+1} through g_psi and compares
        against the momentum target y_{t+1} to produce L_explore and
        the normalised intrinsic reward r_i = beta * Normalize(L_explore).

        Args:
            data: DataProto with:
                - glance_h_pred: (B, max_turns, hidden_size)
                - glance_h_mask: (B, max_turns)
                - glance_y_next: (B, max_turns, d_vis)

        Returns:
            DataProto with:
                glance_intrinsic_rewards: (B, max_turns) -- r_i per turn
                glance_l_explore: (B, max_turns) -- raw L_explore per turn
        """
        assert self.glance_enabled, "GLANCE is not enabled"

        h_pred = data.batch['glance_h_pred']   # (B, max_turns, hidden_size)
        h_mask = data.batch['glance_h_mask']    # (B, max_turns)
        y_next = data.batch['glance_y_next']    # (B, max_turns, d_vis)
        B, max_turns = h_mask.shape

        device = torch.device(f'cuda:{torch.cuda.current_device()}')
        self.glance_reward.to(device)

        intrinsic_rewards = torch.zeros(B, max_turns, dtype=torch.float32)
        l_explore_out = torch.zeros(B, max_turns, dtype=torch.float32)

        with torch.no_grad():
            for b in range(B):
                for t in range(max_turns):
                    if h_mask[b, t] < 0.5:
                        continue
                    h = h_pred[b, t].unsqueeze(0).to(device)   # (1, hidden_size)
                    y = y_next[b, t].unsqueeze(0).to(device)   # (1, d_vis)
                    l_exp, r_i = self.glance_reward.compute_intrinsic_reward(h, y)
                    l_explore_out[b, t] = l_exp.item()
                    intrinsic_rewards[b, t] = r_i.item()

        self.glance_reward.to('cpu')

        valid_count = int(h_mask.sum().item())
        valid_mask_cpu = h_mask.cpu() > 0.5
        ri_valid = intrinsic_rewards[valid_mask_cpu]
        le_valid = l_explore_out[valid_mask_cpu]
        print(f'[GLANCE] compute_glance_rewards: '
              f'valid={valid_count} '
              f'l_explore_mean={le_valid.mean().item():.4f} '
              f'r_i_mean={ri_valid.mean().item():.4f} '
              f'r_i_std={ri_valid.std().item():.4f} '
              f'running_std={self.glance_reward.running_std.std.item():.4f}')

        return DataProto.from_dict(tensors={
            'glance_intrinsic_rewards': intrinsic_rewards,
            'glance_l_explore': l_explore_out,
        })

    def update_glance_representation(self, data: DataProto) -> dict:
        """Joint representation learning step (Algorithm 1, Step A).

        Freezes the LLM backbone, performs a forward pass with gradients
        through the vision encoder, computes L_explore through the projector
        g_psi, and updates both projector and vision encoder parameters.

        Args:
            data: DataProto with input_ids, attention_mask, position_ids,
                  multi_modal_inputs, glance_h_mask, glance_y_next.
                  meta_info must contain 'micro_batch_size'.
        Returns:
            Dict of metrics.
        """
        assert self.glance_enabled, "GLANCE is not enabled"

        PREDICTION_TOKEN_ID = 68931
        closing_slash_ids: frozenset = data.meta_info.get('glance_closing_slash_ids', frozenset())
        fixed_max_turns: int = data.meta_info.get('glance_max_turns', 0)
        micro_batch_size = data.meta_info['micro_batch_size']

        glance_h_mask = data.batch['glance_h_mask']   # (B_total, max_turns)
        glance_y_next = data.batch['glance_y_next']   # (B_total, max_turns, d_vis)

        # ── Step 1: Freeze all non-visual parameters ──
        frozen_params = []
        for name, param in self.actor_module.named_parameters():
            if 'visual' not in name:
                if param.requires_grad:
                    param.requires_grad = False
                    frozen_params.append(param)

        self.actor_module.train()

        # Move projector to GPU
        device = torch.device(f'cuda:{torch.cuda.current_device()}')
        self.glance_reward.to(device)

        select_keys = ['input_ids', 'attention_mask', 'position_ids']
        has_mmi = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        if has_mmi:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            micro_batches = data.select(select_keys, ['multi_modal_inputs']).chunk(num_micro_batches)
        else:
            micro_batches = data.select(batch_keys=select_keys).batch.split(micro_batch_size)

        # ── Step 2-4: Forward, compute L_explore, backward, step ──
        self.glance_optimizer.zero_grad()

        total_l_explore = torch.tensor(0.0, device=device)
        total_valid = 0
        sample_offset = 0

        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                mb = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            else:
                mb = micro_batch

            input_ids = mb['input_ids']
            attention_mask = mb['attention_mask']
            position_ids = mb['position_ids']
            multi_modal_inputs = {}
            if 'multi_modal_inputs' in mb:
                for key in mb['multi_modal_inputs'][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inp[key] for inp in mb['multi_modal_inputs']], dim=0)

            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)

            B, seqlen = input_ids.shape

            # Forward WITH gradients (vision encoder unfrozen, LLM frozen)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                if self.use_remove_padding:
                    input_ids_rmpad, indices, *_ = unpad_input(
                        input_ids.unsqueeze(-1), attention_mask)
                    input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                    if position_ids.dim() == 3:
                        position_ids_rmpad = index_first_axis(
                            rearrange(position_ids, "c b s ... -> (b s) c ..."),
                            indices).transpose(0, 1).unsqueeze(1)
                    else:
                        position_ids_rmpad = index_first_axis(
                            rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                            indices).transpose(0, 1)

                    output = self.actor_module(
                        input_ids=input_ids_rmpad,
                        attention_mask=None,
                        position_ids=position_ids_rmpad,
                        **multi_modal_inputs,
                        use_cache=False,
                        output_hidden_states=True,
                    )
                    hs_rmpad = output.hidden_states[-1].squeeze(0).float()
                    last_hidden = pad_input(
                        hidden_states=hs_rmpad,
                        indices=indices,
                        batch=B,
                        seqlen=seqlen,
                    )
                else:
                    output = self.actor_module(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        **multi_modal_inputs,
                        use_cache=False,
                        output_hidden_states=True,
                    )
                    last_hidden = output.hidden_states[-1].float()

            # Extract h at </prediction> positions (same logic as compute_glance_hidden_states)
            pred_positions = []
            for b in range(B):
                raw = (input_ids[b] == PREDICTION_TOKEN_ID).nonzero(as_tuple=True)[0]
                if closing_slash_ids:
                    raw = raw[
                        (raw > 0) &
                        torch.tensor(
                            [input_ids[b, p - 1].item() in closing_slash_ids for p in raw],
                            dtype=torch.bool, device=raw.device,
                        )
                    ]
                if fixed_max_turns > 0 and raw.numel() > fixed_max_turns:
                    raw = raw[-fixed_max_turns:]
                pred_positions.append(raw)

            # Compute L_explore per valid turn and accumulate
            for b in range(B):
                global_b = sample_offset + b
                pos = pred_positions[b]
                for t in range(pos.numel()):
                    if t >= glance_h_mask.shape[1]:
                        break
                    if glance_h_mask[global_b, t] < 0.5:
                        continue
                    h = last_hidden[b, pos[t]].unsqueeze(0)  # (1, d_vlm)
                    y = glance_y_next[global_b, t].unsqueeze(0).to(device).detach()  # (1, d_vis)
                    l_exp = self.glance_reward.compute_l_explore(h, y)  # (1,)
                    total_l_explore = total_l_explore + l_exp.sum()
                    total_valid += 1

            sample_offset += B

        # Average L_explore and backward
        metrics = {}
        if total_valid > 0:
            mean_l_explore = total_l_explore / total_valid
            mean_l_explore.backward()

            # Clip gradients for projector
            proj_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.glance_reward.projector.parameters(), max_norm=1.0)

            self.glance_optimizer.step()

            metrics['glance/repr_l_explore'] = mean_l_explore.detach().item()
            metrics['glance/proj_grad_norm'] = proj_grad_norm.detach().item()
            print(f'[GLANCE] update_glance_representation: '
                  f'valid={total_valid} '
                  f'l_explore={mean_l_explore.item():.4f} '
                  f'proj_grad_norm={proj_grad_norm.item():.4f}')
        else:
            print('[GLANCE] update_glance_representation: no valid turns, skipped')

        self.glance_optimizer.zero_grad()

        # Move projector back to CPU
        self.glance_reward.to('cpu')

        # ── Step 5: Unfreeze LLM parameters ──
        for param in frozen_params:
            param.requires_grad = True

        return metrics

    def update_glance_momentum(self) -> dict:
        """EMA update: phi <- alpha * phi + (1 - alpha) * v  (Algorithm 1, Line 20).

        Uses FSDP.summon_full_params to access the online vision encoder's
        unsharded parameters on each rank, then updates the momentum encoder
        f_phi in-place.

        Returns:
            Dict of metrics (ve_param_norm, phi_param_norm).
        """
        assert self.glance_enabled, "GLANCE is not enabled"
        alpha = float(self.glance_config.get('ema_decay', 0.99))

        device = torch.device(f'cuda:{torch.cuda.current_device()}')
        self.glance_f_phi.to(device)

        # Find the online visual encoder inside the FSDP-wrapped model.
        # actor_module is the FSDP root; the unwrapped module has a .visual attr.
        unwrapped = self.actor_module
        if hasattr(unwrapped, '_fsdp_wrapped_module'):
            unwrapped = unwrapped._fsdp_wrapped_module
        online_visual = unwrapped.visual

        with FSDP.summon_full_params(self.actor_module, writeback=False):
            phi_params = dict(self.glance_f_phi.named_parameters())
            ve_norm_sq = 0.0
            matched = 0
            for name, v_param in online_visual.named_parameters():
                if name in phi_params:
                    phi_p = phi_params[name]
                    v_data = v_param.data.to(device=device, dtype=phi_p.dtype)
                    phi_p.data.mul_(alpha).add_(v_data, alpha=1.0 - alpha)
                    ve_norm_sq += v_data.float().norm().item() ** 2
                    matched += 1

        phi_norm_sq = sum(p.data.float().norm().item() ** 2 for p in self.glance_f_phi.parameters())

        self.glance_f_phi.to('cpu')

        metrics = {
            'glance/ve_param_norm': ve_norm_sq ** 0.5,
            'glance/phi_param_norm': phi_norm_sq ** 0.5,
        }
        print(f'[GLANCE] update_glance_momentum: '
              f'alpha={alpha} matched={matched} '
              f've_norm={ve_norm_sq**0.5:.2f} phi_norm={phi_norm_sq**0.5:.2f}')
        return metrics

    def check_glance_rejuvenation(self, mean_l_explore: float) -> dict:
        """Check for curiosity drain and rejuvenate projector if needed.

        Args:
            mean_l_explore: Mean L_explore from the current iteration
                            (from update_glance_representation).
        Returns:
            Dict of metrics (rejuvenation_triggered, drain_count).
        """
        assert self.glance_enabled, "GLANCE is not enabled"
        triggered = self.glance_reward.check_and_rejuvenate(mean_l_explore)

        if triggered:
            # Reset optimizer state for projector param group (group 0)
            for param in self.glance_reward.projector.parameters():
                if param in self.glance_optimizer.state:
                    del self.glance_optimizer.state[param]
            print(f'[GLANCE] Rejuvenation triggered! Projector re-initialised, '
                  f'optimizer state reset. l_explore={mean_l_explore:.4f}')

        metrics = {
            'glance/rejuvenation_triggered': 1.0 if triggered else 0.0,
            'glance/drain_count': float(self.glance_reward._drain_count),
        }
        return metrics

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        if 'loss_mask' in data.batch.keys():
            select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages','loss_mask']
        else:
            select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
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

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())  # actor device is cpu when using offload
                    responses = data['responses']
                    response_length = responses.size(1)
                    #attention_mask = data['attention_mask']
                    
                    # Note:
                    # In agent setting, the prompt is the initial obs, and reponse is the whole trajectory except initial obs
                    # The loss mask has the same shape of attention mask, which has both prompt and response, it masks:
                    # the prompt, the padding of response (right padded), and the obs given by the environment in the reponse
                    
                    if "loss_mask" in data:
                        loss_mask = data['loss_mask']
                    else:
                        print("DEBUG: warning, loss_mask not found in actor update")
                        loss_mask=data["attention_mask"]
                    response_mask = loss_mask[:, -response_length:]
                    old_log_prob = data['old_log_probs']
                    advantages = data['advantages']

                    clip_ratio = self.config.clip_ratio
                    entropy_coeff = self.config.entropy_coeff

                    # all return: (bsz, response_length)
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

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
                        ref_log_prob = data['ref_log_prob']
                        # compute kl loss
                        kld = core_algos.kl_penalty(logprob=log_prob,
                                                    ref_logprob=ref_log_prob,
                                                    kl_penalty=self.config.kl_loss_type)
                        kl_loss = masked_mean(kld, response_mask)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics['actor/kl_loss'] = kl_loss.detach().item()
                        metrics['actor/kl_coef'] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
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
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
