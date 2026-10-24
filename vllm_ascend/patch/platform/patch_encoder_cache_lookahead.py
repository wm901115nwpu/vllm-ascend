# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

"""Backport the encoder-cache lookahead deadlock fix from vLLM #60553.

Two scheduler offsets of ``num_prefill_lookahead`` (L) point in opposite
directions. When a second multimodal item cannot be allocated, the chunk
stops at ``start - L``. The consumed item is kept until the cursor has
passed its end by that same L. If the gap between the items is smaller
than ``2L``, the cursor never reaches the free point, the next step
schedules 0 tokens, and the request hangs.

Either change on its own only narrows the hang to ``gap < L``. Both are
required, including when the items are adjacent (gap 0).

``disable_chunked_mm_input`` still rolls the chunk back by L. That branch
returns before ``can_allocate``, so this patch does not touch it. Runners
already fall back to token embeddings when the drafter reads into an item
that was not encoded.

Remove this patch once every supported vLLM pin includes #60553.
"""

from __future__ import annotations

import functools
import inspect

from vllm.logger import logger
from vllm.v1.core.sched.scheduler import Scheduler

_PATCH_MARKER = "_vllm_ascend_encoder_cache_lookahead"
_LOOKAHEAD_FREE_MARKERS = ("num_prefill_lookahead", "spec_lookahead")


def _num_new_tokens_at_unallocated_item(
    num_computed_tokens: int,
    num_new_tokens: int,
    start_pos: int,
) -> int:
    """Stop at the next multimodal item without the drafter shift.

    The returned count is capped by the chunk the caller already chose, so
    lookahead cannot grow the step. A cursor already inside the item still
    schedules nothing: the encoder output is not available to decode.
    """
    if num_computed_tokens < start_pos:
        return min(num_new_tokens, start_pos - num_computed_tokens)
    return 0


def _upstream_free_retains_lookahead(method: object) -> bool:
    """Whether ``_free_encoder_inputs`` still waits an extra L tokens.

    The merged #60553 body no longer mentions the lookahead. Skipping the
    wrap on that body avoids stacking a second pass. If the source cannot
    be read, wrap anyway: the extra pass frees nothing already released.
    """
    try:
        source = inspect.getsource(method)  # type: ignore[arg-type]
    except (OSError, TypeError):
        return True
    return any(marker in source for marker in _LOOKAHEAD_FREE_MARKERS)


def _install_schedule_patch(scheduler_cls: type) -> None:
    original = scheduler_cls._try_schedule_encoder_inputs
    if getattr(original, _PATCH_MARKER, False):
        return

    @functools.wraps(original)
    def _try_schedule_encoder_inputs(
        self,
        request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
        shift_computed_tokens: int = 0,
    ):
        if shift_computed_tokens <= 0:
            # No drafter window, so the lookahead stop cannot be reached.
            return original(
                self,
                request,
                num_computed_tokens,
                num_new_tokens,
                encoder_compute_budget,
                shift_computed_tokens,
            )

        manager = self.encoder_cache_manager
        original_can_allocate = manager.can_allocate
        had_instance_can_allocate = "can_allocate" in getattr(manager, "__dict__", {})
        failed_input: list[int] = []

        def _spy_can_allocate(request, input_id, budget, num_embeds_to_schedule):
            allowed = original_can_allocate(request, input_id, budget, num_embeds_to_schedule)
            # The failure branch breaks immediately, so the first rejection
            # is the item the chunk must stop in front of.
            if not allowed and not failed_input:
                failed_input.append(input_id)
            return allowed

        manager.can_allocate = _spy_can_allocate
        try:
            result = original(
                self,
                request,
                num_computed_tokens,
                num_new_tokens,
                encoder_compute_budget,
                shift_computed_tokens,
            )
        finally:
            if had_instance_can_allocate:
                manager.can_allocate = original_can_allocate
            else:
                # Drop the spy so later class-level replacements still apply.
                del manager.can_allocate

        if not failed_input:
            # Includes the disable_chunked_mm_input rollback, which still
            # applies the lookahead shift and never calls can_allocate.
            return result

        start_pos = request.mm_features[failed_input[0]].mm_position.offset
        corrected = _num_new_tokens_at_unallocated_item(num_computed_tokens, num_new_tokens, start_pos)
        if corrected == result[1]:
            # Installed vLLM already stops at the item. Keep its tuple.
            return result
        return (result[0], corrected, *result[2:])

    setattr(_try_schedule_encoder_inputs, _PATCH_MARKER, True)
    scheduler_cls._try_schedule_encoder_inputs = _try_schedule_encoder_inputs


def _install_free_patch(scheduler_cls: type) -> None:
    original = scheduler_cls._free_encoder_inputs
    if getattr(original, _PATCH_MARKER, False):
        return
    if not _upstream_free_retains_lookahead(original):
        return

    @functools.wraps(original)
    def _free_encoder_inputs(self, request) -> None:
        original(self, request)
        cached_encoder_input_ids = self.encoder_cache_manager.get_cached_input_ids(request)
        if not cached_encoder_input_ids:
            return

        # Rejection rollback still holds the cursor back by the number of
        # output placeholders. Lookahead does not.
        num_output_placeholders = getattr(request, "num_output_placeholders", 0)
        cursor = request.num_computed_tokens - num_output_placeholders
        for input_id in list(cached_encoder_input_ids):
            mm_feature = request.mm_features[input_id]
            end_pos = mm_feature.mm_position.offset + mm_feature.mm_position.length
            if getattr(self, "is_encoder_decoder", False) and request.num_computed_tokens > 0:
                # Upstream already drops encoder-decoder inputs on the first
                # generated token. Nothing in that branch used the lookahead.
                continue
            if end_pos <= cursor:
                self._free_encoder_input(request, input_id)

    setattr(_free_encoder_inputs, _PATCH_MARKER, True)
    scheduler_cls._free_encoder_inputs = _free_encoder_inputs


def apply_encoder_cache_lookahead_patch(scheduler_cls: type = Scheduler) -> None:
    """Patch ``scheduler_cls`` so subclasses inherit the #60553 behavior."""
    _install_schedule_patch(scheduler_cls)
    _install_free_patch(scheduler_cls)


apply_encoder_cache_lookahead_patch(Scheduler)
logger.debug_once(
    "Encoder cache lookahead: Scheduler no longer deadlocks when two "
    "multimodal items must occupy the cache one at a time."
)
