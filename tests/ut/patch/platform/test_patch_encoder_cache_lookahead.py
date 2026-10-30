# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Regression tests for the encoder-cache lookahead deadlock backport.

The production patch is loaded by path and applied to a miniature scheduler
that implements the pre-#60553 failure stop and free lag. No NPU, torch, or
installed vLLM package is required.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import OrderedDict
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

PATCH_PATH = (
    Path(__file__).resolve().parents[4] / "vllm_ascend" / "patch" / "platform" / "patch_encoder_cache_lookahead.py"
)
PLATFORM_INIT = Path(__file__).resolve().parents[4] / "vllm_ascend" / "patch" / "platform" / "__init__.py"
PATCH_CATALOG = Path(__file__).resolve().parents[4] / "vllm_ascend" / "patch" / "__init__.py"

# Issue 19585: Kimi-K3 image, chunk, and the dummy encoder-cache budget.
REPORTED_IMAGE_TOKENS = 8740
REPORTED_CHUNK_TOKENS = 8192
REPORTED_ENCODER_CACHE = 16817
REPORTED_PREFIX_TOKENS = 191  # image 0 ends at 8931
REPORTED_SUFFIX_TOKENS = 64  # prompt length 17735
REPORTED_IMAGE_END = REPORTED_PREFIX_TOKENS + REPORTED_IMAGE_TOKENS

# Upstream #60553 scheduler sweep, scaled down from the reported images.
SWEEP_IMAGE_TOKENS = 600
SWEEP_CHUNK_TOKENS = 900
SWEEP_ENCODER_CACHE = 900
SWEEP_PREFIX_TOKENS = 10
SWEEP_SUFFIX_TOKENS = 20


class _Position:
    def __init__(self, offset: int, length: int):
        self.offset = offset
        self.length = length

    def get_num_embeds(self) -> int:
        return self.length


class _Feature:
    def __init__(self, offset: int, length: int, identifier: str):
        self.mm_position = _Position(offset, length)
        self.identifier = identifier


class _EncoderCache:
    """Slot accounting for one-at-a-time encoder residency.

    Mirrors the parts of ``EncoderCacheManager`` the deadlock depends on:
    a referenced entry is not freeable, and ``can_allocate`` may evict
    freeable entries without decrementing free slots until ``allocate``.
    """

    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.num_free_slots = cache_size
        self.num_freeable_slots = cache_size
        self.cached: dict[str, set[str]] = {}
        self.num_embeds_by_hash: dict[str, int] = {}
        self.request_cached_ids: dict[str, set[int]] = {}
        self.freeable: OrderedDict[str, int] = OrderedDict()

    def check_and_update_cache(self, request, input_id: int) -> bool:
        mm_hash = request.mm_features[input_id].identifier
        if mm_hash not in self.cached:
            return False
        if not self.cached[mm_hash]:
            num_embeds = self.freeable.pop(mm_hash)
            self.num_freeable_slots -= num_embeds
        self.cached[mm_hash].add(request.request_id)
        self.request_cached_ids.setdefault(request.request_id, set()).add(input_id)
        return True

    def can_allocate(self, request, input_id: int, encoder_compute_budget: int, num_embeds_to_schedule: int) -> bool:
        num_embeds = request.mm_features[input_id].mm_position.get_num_embeds()
        if num_embeds > encoder_compute_budget:
            return False
        needed = num_embeds + num_embeds_to_schedule
        if needed <= self.num_free_slots:
            return True
        if needed > self.num_freeable_slots:
            return False
        while needed > self.num_free_slots:
            mm_hash, freed = self.freeable.popitem(last=False)
            del self.cached[mm_hash]
            del self.num_embeds_by_hash[mm_hash]
            self.num_free_slots += freed
        return True

    def allocate(self, request, input_id: int) -> None:
        mm_hash = request.mm_features[input_id].identifier
        num_embeds = request.mm_features[input_id].mm_position.get_num_embeds()
        self.cached.setdefault(mm_hash, set()).add(request.request_id)
        self.num_embeds_by_hash.setdefault(mm_hash, num_embeds)
        self.request_cached_ids.setdefault(request.request_id, set()).add(input_id)
        self.num_free_slots -= num_embeds
        self.num_freeable_slots -= num_embeds

    def get_cached_input_ids(self, request) -> set[int]:
        return set(self.request_cached_ids.get(request.request_id, ()))

    def free_encoder_input(self, request, input_id: int) -> None:
        req_id = request.request_id
        mm_hash = request.mm_features[input_id].identifier
        if req_id in self.request_cached_ids:
            self.request_cached_ids[req_id].discard(input_id)
            if not self.request_cached_ids[req_id]:
                del self.request_cached_ids[req_id]
        holders = self.cached.get(mm_hash)
        if not holders or req_id not in holders:
            return
        holders.discard(req_id)
        if not holders:
            num_embeds = self.num_embeds_by_hash[mm_hash]
            self.freeable[mm_hash] = num_embeds
            self.num_freeable_slots += num_embeds


def _features_in_window(mm_features: list[_Feature], start: int, end: int) -> tuple[int, int]:
    """Same half-open overlap rule as ``get_mm_features_in_window``."""
    lo = 0
    while lo < len(mm_features) and mm_features[lo].mm_position.offset + mm_features[lo].mm_position.length <= start:
        lo += 1
    hi = lo
    while hi < len(mm_features) and mm_features[hi].mm_position.offset < end:
        hi += 1
    return lo, hi


def _release_consumed(self, request, retain_tokens: int) -> None:
    cached = self.encoder_cache_manager.get_cached_input_ids(request)
    if not cached:
        return
    placeholders = getattr(request, "num_output_placeholders", 0)
    cursor = request.num_computed_tokens - placeholders
    for input_id in list(cached):
        feature = request.mm_features[input_id]
        end_pos = feature.mm_position.offset + feature.mm_position.length
        if end_pos + retain_tokens <= cursor:
            self._free_encoder_input(request, input_id)


def _try_schedule_encoder_inputs(
    self,
    request,
    num_computed_tokens: int,
    num_new_tokens: int,
    encoder_compute_budget: int,
    shift_computed_tokens: int = 0,
):
    """Pre-#60553 schedule loop, plus the fixed failure branch when selected."""
    if num_new_tokens == 0 or not request.has_encoder_inputs:
        empty = ([], num_new_tokens, encoder_compute_budget, [], [])
        self.last_schedule_result = empty
        return empty

    window_end = num_computed_tokens + num_new_tokens + shift_computed_tokens
    lo, hi = _features_in_window(request.mm_features, num_computed_tokens, window_end)
    scheduled: list[int] = []
    seen: dict[str, int] = {}
    num_embeds_to_schedule = 0
    for index in range(lo, hi):
        feature = request.mm_features[index]
        start_pos = feature.mm_position.offset
        length = feature.mm_position.length
        embeds = feature.mm_position.get_num_embeds()
        if (
            self.scheduler_config.disable_chunked_mm_input
            and num_computed_tokens <= start_pos
            and (num_computed_tokens + num_new_tokens) < (start_pos + length)
        ):
            num_new_tokens = max(0, start_pos - (num_computed_tokens + shift_computed_tokens))
            break
        if feature.identifier in seen:
            continue
        if self.encoder_cache_manager.check_and_update_cache(request, index):
            continue
        if not self.encoder_cache_manager.can_allocate(request, index, encoder_compute_budget, num_embeds_to_schedule):
            if self.failure_applies_lookahead_shift:
                if num_computed_tokens + shift_computed_tokens < start_pos:
                    num_new_tokens = start_pos - (num_computed_tokens + shift_computed_tokens)
                else:
                    num_new_tokens = 0
            elif num_computed_tokens < start_pos:
                num_new_tokens = min(num_new_tokens, start_pos - num_computed_tokens)
            else:
                num_new_tokens = 0
            break
        if min(length, window_end - start_pos) - max(0, num_computed_tokens - start_pos) <= 0:
            continue
        num_embeds_to_schedule += embeds
        encoder_compute_budget -= embeds
        seen[feature.identifier] = embeds
        scheduled.append(index)
    external: list[int] = []
    if self.return_duplicate_slot:
        result = (scheduled, num_new_tokens, encoder_compute_budget, external, [])
    else:
        result = (scheduled, num_new_tokens, encoder_compute_budget, external)
    self.last_schedule_result = result
    return result


def _free_without_lookahead(self, request) -> None:
    _release_consumed(self, request, retain_tokens=0)


class _SchedulerBase:
    failure_applies_lookahead_shift = True
    free_retains_lookahead = True
    return_duplicate_slot = True

    def __init__(self, cache: _EncoderCache, lookahead: int, *, disable_chunked_mm_input: bool = False):
        self.encoder_cache_manager = cache
        self.num_prefill_lookahead = lookahead
        self.is_encoder_decoder = False
        self.scheduler_config = SimpleNamespace(disable_chunked_mm_input=disable_chunked_mm_input)
        self.freed_inputs: list[int] = []

    def _free_encoder_input(self, request, input_id: int) -> None:
        self.freed_inputs.append(input_id)
        self.encoder_cache_manager.free_encoder_input(request, input_id)

    _try_schedule_encoder_inputs = _try_schedule_encoder_inputs


class _BuggyScheduler(_SchedulerBase):
    """Pinned vLLM: shift on can_allocate failure, and free lags by L."""

    def _free_encoder_inputs(self, request) -> None:
        # Pinned body: spec_lookahead = self.num_prefill_lookahead, then
        # free only once end + spec_lookahead <= cursor.
        _release_consumed(self, request, retain_tokens=self.num_prefill_lookahead)


class _FixedScheduler(_SchedulerBase):
    """vLLM after #60553: stop at the item, free when the cursor passes the end."""

    failure_applies_lookahead_shift = False
    free_retains_lookahead = False

    def _free_encoder_inputs(self, request) -> None:
        _free_without_lookahead(self, request)


class _ChildOfBuggy(_BuggyScheduler):
    """Ascend schedulers inherit these methods instead of copying them."""


def _ensure_package(monkeypatch: pytest.MonkeyPatch, name: str) -> ModuleType:
    module = ModuleType(name)
    module.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    module.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, module)
    parent_name, _, child = name.rpartition(".")
    if parent_name:
        parent = sys.modules[parent_name]
        setattr(parent, child, module)
    return module


def _install_vllm_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the production module import when this process has no vLLM."""
    if importlib.util.find_spec("vllm") is not None:
        return

    class _Logger:
        def debug_once(self, *_args, **_kwargs) -> None:
            return None

    class _StubScheduler:
        def _try_schedule_encoder_inputs(
            self,
            request,
            num_computed_tokens,
            num_new_tokens,
            encoder_compute_budget,
            shift_computed_tokens=0,
        ):
            return [], num_new_tokens, encoder_compute_budget, [], []

        def _free_encoder_inputs(self, request) -> None:
            return None

    _ensure_package(monkeypatch, "vllm")
    logger_mod = _ensure_package(monkeypatch, "vllm.logger")
    logger_mod.logger = _Logger()
    _ensure_package(monkeypatch, "vllm.v1")
    _ensure_package(monkeypatch, "vllm.v1.core")
    _ensure_package(monkeypatch, "vllm.v1.core.sched")
    scheduler_mod = _ensure_package(monkeypatch, "vllm.v1.core.sched.scheduler")
    scheduler_mod.Scheduler = _StubScheduler


def _load_patch(monkeypatch: pytest.MonkeyPatch):
    _install_vllm_stub(monkeypatch)
    spec = importlib.util.spec_from_file_location("_test_encoder_cache_lookahead_patch", PATCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def patch_module(monkeypatch: pytest.MonkeyPatch):
    return _load_patch(monkeypatch)


def _make_request(spans: list[tuple[int, int]], *, computed: int = 0, placeholders: int = 0):
    features = [_Feature(offset, length, f"mm-{index}") for index, (offset, length) in enumerate(spans)]
    prompt_len = spans[-1][0] + spans[-1][1] if spans else 0
    return SimpleNamespace(
        request_id="req",
        has_encoder_inputs=bool(spans),
        mm_features=features,
        num_computed_tokens=computed,
        num_output_placeholders=placeholders,
        num_tokens=prompt_len,
    )


def _as_function(method):
    """Accept a class function or a bound method and return the function."""
    return getattr(method, "__func__", method)


def _run_prefill(
    scheduler,
    request,
    *,
    chunk_tokens: int,
    encoder_budget: int,
    lookahead: int,
    schedule_fn=None,
    free_fn=None,
    max_steps: int = 64,
) -> dict:
    schedule = _as_function(schedule_fn or scheduler._try_schedule_encoder_inputs)
    free = _as_function(free_fn or scheduler._free_encoder_inputs)
    steps: list[tuple[int, int, list[int]]] = []
    scheduled_inputs: list[int] = []
    hung = False
    for _ in range(max_steps):
        if request.num_computed_tokens >= request.num_tokens:
            break
        remaining = request.num_tokens - request.num_computed_tokens
        requested = min(chunk_tokens, remaining)
        before = request.num_computed_tokens
        result = schedule(
            scheduler,
            request,
            before,
            requested,
            encoder_budget,
            lookahead,
        )
        num_new = result[1]
        chosen = list(result[0])
        steps.append((before, num_new, chosen))
        if num_new <= 0:
            hung = True
            break
        for input_id in chosen:
            scheduler.encoder_cache_manager.allocate(request, input_id)
            if input_id not in scheduled_inputs:
                scheduled_inputs.append(input_id)
        request.num_computed_tokens = before + num_new
        free(scheduler, request)
    else:
        hung = request.num_computed_tokens < request.num_tokens
    return {
        "hung": hung,
        "steps": steps,
        "scheduled_inputs": scheduled_inputs,
        "computed": request.num_computed_tokens,
    }


def _two_image_request(prefix: int, image_tokens: int, gap: int, suffix: int):
    first = (prefix, image_tokens)
    second = (prefix + image_tokens + gap, image_tokens)
    request = _make_request([first, second])
    request.num_tokens = second[0] + second[1] + suffix
    return request


def test_platform_patch_is_wired():
    platform_init = PLATFORM_INIT.read_text(encoding="utf-8")
    catalog = PATCH_CATALOG.read_text(encoding="utf-8")
    assert "patch_encoder_cache_lookahead" in platform_init
    assert "patch_encoder_cache_lookahead.py" in catalog
    assert "https://github.com/vllm-project/vllm/pull/60553" in catalog


def test_reported_geometry_makes_progress_and_frees_first_image(patch_module):
    """Kimi-K3 shape from issue 19585: 8740 + 8740, chunk 8192, cache 16817, L=1."""
    patch_module.apply_encoder_cache_lookahead_patch(_BuggyScheduler)
    scheduler = _ChildOfBuggy(_EncoderCache(REPORTED_ENCODER_CACHE), lookahead=1)
    request = _two_image_request(
        REPORTED_PREFIX_TOKENS,
        REPORTED_IMAGE_TOKENS,
        gap=0,
        suffix=REPORTED_SUFFIX_TOKENS,
    )
    assert request.num_tokens == 17735
    assert request.mm_features[0].mm_position.offset + REPORTED_IMAGE_TOKENS == REPORTED_IMAGE_END

    result = _run_prefill(
        scheduler,
        request,
        chunk_tokens=REPORTED_CHUNK_TOKENS,
        encoder_budget=REPORTED_ENCODER_CACHE,
        lookahead=1,
    )

    assert result["hung"] is False
    assert result["computed"] == request.num_tokens
    assert result["scheduled_inputs"] == [0, 1]
    crossing = next(step for step in result["steps"] if step[0] < REPORTED_IMAGE_END <= step[0] + step[1])
    # The pinned stop left the cursor on 8930 (end - L). The fix schedules
    # through the last token of the cached image.
    assert crossing[0] == REPORTED_CHUNK_TOKENS
    assert crossing[1] == REPORTED_IMAGE_END - REPORTED_CHUNK_TOKENS
    assert 0 in scheduler.freed_inputs
    assert scheduler.freed_inputs.index(0) < scheduler.freed_inputs.index(1)


def test_either_change_alone_still_hangs_when_gap_is_zero(patch_module):
    patch_module.apply_encoder_cache_lookahead_patch(_BuggyScheduler)
    schedule_fixed = _BuggyScheduler._try_schedule_encoder_inputs
    free_fixed = _BuggyScheduler._free_encoder_inputs
    schedule_buggy = schedule_fixed.__wrapped__
    free_buggy = free_fixed.__wrapped__

    def _outcome(schedule_fn, free_fn) -> bool:
        scheduler = _BuggyScheduler(_EncoderCache(REPORTED_ENCODER_CACHE), lookahead=1)
        request = _two_image_request(
            REPORTED_PREFIX_TOKENS,
            REPORTED_IMAGE_TOKENS,
            gap=0,
            suffix=REPORTED_SUFFIX_TOKENS,
        )
        return _run_prefill(
            scheduler,
            request,
            chunk_tokens=REPORTED_CHUNK_TOKENS,
            encoder_budget=REPORTED_ENCODER_CACHE,
            lookahead=1,
            schedule_fn=schedule_fn,
            free_fn=free_fn,
        )["hung"]

    assert _outcome(schedule_buggy, free_buggy) is True
    assert _outcome(schedule_fixed, free_buggy) is True
    assert _outcome(schedule_buggy, free_fixed) is True
    assert _outcome(schedule_fixed, free_fixed) is False


@pytest.mark.parametrize("lookahead", [0, 1, 3])
@pytest.mark.parametrize("gap", [0, 1, 2, 4, 8])
def test_lookahead_gap_sweep_matches_hang_rule(patch_module, lookahead: int, gap: int):
    """Unpatched hang rule is gap < 2L. The backport finishes every cell."""
    patch_module.apply_encoder_cache_lookahead_patch(_BuggyScheduler)
    schedule_fixed = _BuggyScheduler._try_schedule_encoder_inputs
    free_fixed = _BuggyScheduler._free_encoder_inputs

    def _run(schedule_fn, free_fn) -> dict:
        scheduler = _BuggyScheduler(_EncoderCache(SWEEP_ENCODER_CACHE), lookahead=lookahead)
        request = _two_image_request(SWEEP_PREFIX_TOKENS, SWEEP_IMAGE_TOKENS, gap, SWEEP_SUFFIX_TOKENS)
        return _run_prefill(
            scheduler,
            request,
            chunk_tokens=SWEEP_CHUNK_TOKENS,
            encoder_budget=SWEEP_ENCODER_CACHE,
            lookahead=lookahead,
            schedule_fn=schedule_fn,
            free_fn=free_fn,
        )

    unpatched = _run(schedule_fixed.__wrapped__, free_fixed.__wrapped__)
    patched = _run(schedule_fixed, free_fixed)
    expect_hang = lookahead > 0 and gap < 2 * lookahead
    assert unpatched["hung"] is expect_hang
    assert patched["hung"] is False
    assert patched["scheduled_inputs"] == [0, 1]
    assert patched["computed"] == (SWEEP_PREFIX_TOKENS + 2 * SWEEP_IMAGE_TOKENS + gap + SWEEP_SUFFIX_TOKENS)


@pytest.mark.parametrize("lookahead", [1, 3])
@pytest.mark.parametrize("past_chunk_end", [0, 1, 2])
def test_unallocatable_lookahead_item_does_not_shrink_or_grow_chunk(patch_module, lookahead: int, past_chunk_end: int):
    """An item reached only by lookahead must leave the token budget unchanged."""
    patch_module.apply_encoder_cache_lookahead_patch(_BuggyScheduler)
    budget = 100
    scheduler = _BuggyScheduler(_EncoderCache(300), lookahead=lookahead)
    request = _make_request([(10, 80), (budget + past_chunk_end, 300)])
    request.num_tokens = budget + past_chunk_end + 320

    result = scheduler._try_schedule_encoder_inputs(
        request,
        0,
        budget,
        encoder_compute_budget=1000,
        shift_computed_tokens=lookahead,
    )

    assert result[1] == budget
    assert result[0] == [0]
    assert len(result) == 5
    assert "can_allocate" not in scheduler.encoder_cache_manager.__dict__


def test_disable_chunked_mm_input_keeps_lookahead_shift(patch_module):
    patch_module.apply_encoder_cache_lookahead_patch(_BuggyScheduler)
    scheduler = _BuggyScheduler(_EncoderCache(10), lookahead=3, disable_chunked_mm_input=True)
    request = _make_request([(50, 80)])

    result = scheduler._try_schedule_encoder_inputs(
        request,
        0,
        100,
        encoder_compute_budget=1000,
        shift_computed_tokens=3,
    )

    # start 50, shift 3: rollback stays at 47 and can_allocate is not consulted.
    assert result is scheduler.last_schedule_result
    assert result[1] == 47
    assert result[0] == []


def test_already_correct_schedule_tuple_is_unchanged(patch_module):
    original_free = _FixedScheduler._free_encoder_inputs
    patch_module.apply_encoder_cache_lookahead_patch(_FixedScheduler)
    wrapped_schedule = _FixedScheduler._try_schedule_encoder_inputs

    # #60553's free body has no lookahead lag, so the wrap is skipped.
    assert _FixedScheduler._free_encoder_inputs is original_free
    scheduler = _FixedScheduler(_EncoderCache(300), lookahead=1)
    request = _make_request([(10, 80), (100, 300)])
    request.num_tokens = 420

    result = scheduler._try_schedule_encoder_inputs(
        request,
        0,
        100,
        encoder_compute_budget=1000,
        shift_computed_tokens=1,
    )

    assert result is scheduler.last_schedule_result
    assert result[1] == 100
    # Applying again does not stack wrappers.
    patch_module.apply_encoder_cache_lookahead_patch(_FixedScheduler)
    assert _FixedScheduler._try_schedule_encoder_inputs is wrapped_schedule


def test_free_passes_end_without_lookahead_and_honors_placeholders(patch_module):
    patch_module.apply_encoder_cache_lookahead_patch(_BuggyScheduler)
    cache = _EncoderCache(200)
    scheduler = _BuggyScheduler(cache, lookahead=3)
    request = _make_request([(50, 100)])
    cache.allocate(request, 0)

    request.num_computed_tokens = 149
    request.num_output_placeholders = 0
    scheduler._free_encoder_inputs(request)
    assert cache.get_cached_input_ids(request) == {0}

    # Cursor has passed the end. The old lag (end + L = 153) would keep it.
    request.num_computed_tokens = 150
    scheduler._free_encoder_inputs(request)
    assert cache.get_cached_input_ids(request) == set()

    cache.allocate(request, 0)
    request.num_computed_tokens = 152
    request.num_output_placeholders = 3
    scheduler._free_encoder_inputs(request)
    assert cache.get_cached_input_ids(request) == {0}

    request.num_computed_tokens = 153
    scheduler._free_encoder_inputs(request)
    assert cache.get_cached_input_ids(request) == set()
    assert scheduler.freed_inputs == [0, 0]


def test_fixed_free_is_not_wrapped_and_still_releases_at_end(patch_module):
    original_free = _FixedScheduler._free_encoder_inputs
    patch_module.apply_encoder_cache_lookahead_patch(_FixedScheduler)
    assert _FixedScheduler._free_encoder_inputs is original_free

    cache = _EncoderCache(200)
    scheduler = _FixedScheduler(cache, lookahead=3)
    request = _make_request([(0, 8)])
    cache.allocate(request, 0)
    request.num_computed_tokens = 8
    scheduler._free_encoder_inputs(request)
    assert scheduler.freed_inputs == [0]
    assert cache.get_cached_input_ids(request) == set()


def test_v031_four_tuple_keeps_its_tail_when_the_count_changes(patch_module):
    """v0.31 returns four values. Only the token count is replaced."""

    class _FourTupleScheduler(_BuggyScheduler):
        return_duplicate_slot = False

    patch_module.apply_encoder_cache_lookahead_patch(_FourTupleScheduler)
    scheduler = _FourTupleScheduler(_EncoderCache(10), lookahead=1)
    request = _make_request([(4, 20)])
    result = scheduler._try_schedule_encoder_inputs(
        request,
        0,
        10,
        encoder_compute_budget=100,
        shift_computed_tokens=1,
    )
    assert len(result) == 4
    assert result[0] == []
    assert result[1] == 4
    assert result[2] == 100
    assert result[3] == []


def test_zero_shift_returns_the_upstream_tuple(patch_module):
    patch_module.apply_encoder_cache_lookahead_patch(_BuggyScheduler)
    scheduler = _BuggyScheduler(_EncoderCache(10), lookahead=0)
    request = _make_request([(4, 20)])
    result = scheduler._try_schedule_encoder_inputs(
        request,
        0,
        10,
        encoder_compute_budget=100,
        shift_computed_tokens=0,
    )
    assert result is scheduler.last_schedule_result
    # Shift 0 keeps the pinned stop, which already matches #60553.
    assert result[1] == 4
