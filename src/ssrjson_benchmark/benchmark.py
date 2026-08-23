import enum
import gc
import json
import math
import os
import pathlib
import platform
import re
import subprocess
import sys
import time
from importlib.util import find_spec
from typing import Callable

from . import _ssrjson_benchmark as internal
from .result_types import (
    BenchmarkFinalResult,
    BenchmarkResultPerFile,
    BenchmarkResultPerFileTargetLib,
    SystemInfo,
    summarize_times,
)

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_NS_IN_ONE_S = 1000000000

_NAME_DUMPSTOBYTES = "dumps_to_bytes"
_NAME_DUMPSTOSTR = "dumps_to_str"
_NAME_LOADS = "loads"
# Report section order.
BASE_INDEX_GROUPS = [_NAME_DUMPSTOBYTES, _NAME_DUMPSTOSTR, _NAME_LOADS]
# "load&dump" is the pre-split index group; kept only so result files written
# before the split still render.
_NAME_LOADDUMP = "load&dump"
_BASE_PRINT_NAMES = {
    _NAME_DUMPSTOBYTES: "Dumps to bytes",
    _NAME_DUMPSTOSTR: "Dumps to str",
    _NAME_LOADS: "Loads",
    _NAME_LOADDUMP: "Loads & Dumps to str",
}

# Separator between the base index group and the locality in an index key,
# e.g. "dumps_to_bytes|cold".
_INDEX_SEP = "|"

# The stdlib default separators are (', ', ': '), which makes the baseline
# emit ~7% more bytes than every compact library and inflates every ratio.
_COMPACT = (",", ":")

DEFAULT_MIN_ITERATIONS = 200
DEFAULT_COLD_MULTIPLE = 2.0
DEFAULT_ROUNDS = 5
DEFAULT_STATISTIC = "median"
_FALLBACK_LLC_BYTES = 8 * 1024 * 1024
_WARMUP_MIN = 20
_WARMUP_MAX = 200


def make_index_name(base: str, locality: str) -> str:
    return f"{base}{_INDEX_SEP}{locality}"


def split_index_name(index_name: str) -> tuple[str, str]:
    """Split an index key into (base, locality). Old result files carry no
    locality suffix, in which case locality is an empty string."""
    base, _, locality = index_name.partition(_INDEX_SEP)
    return base, locality


def print_index_name(index_name: str) -> str:
    base, locality = split_index_name(index_name)
    pretty = _BASE_PRINT_NAMES.get(base, base)
    return f"{pretty} ({locality})" if locality else pretty


# ---------------------------------------------------------------------------
# Library availability detection
# ---------------------------------------------------------------------------


def _lib_available(name: str) -> bool:
    return find_spec(name) is not None


def _get_available_third_party_libs() -> list[str]:
    """Return list of available third-party JSON libraries (excluding stdlib json)."""
    candidates = ["ujson", "pydantic_core", "msgspec", "orjson", "ssrjson"]
    return [lib for lib in candidates if _lib_available(lib)]


def _import_lib(name: str):
    """Dynamically import a library by name."""
    import importlib

    return importlib.import_module(name)


# ---------------------------------------------------------------------------
# Benchmark function / group definitions
# ---------------------------------------------------------------------------


class BenchmarkCategory(enum.Enum):
    LOADS = "loads"
    DUMPS = "dumps"
    DUMPS_TO_BYTES = "dumps_to_bytes"


class Locality(enum.Enum):
    """Whether the measured object sits in cache when the call starts.

    HOT keeps a single live copy, so the allocator hands back the same memory
    every iteration. COLD keeps a ring whose total size exceeds the last level
    cache, so each measured object has been evicted by the intervening copies.
    This is the *only* control over object temperature; it is deliberately
    independent of whether copies are needed at all (see InputKind).
    """

    HOT = "hot"
    COLD = "cold"


class InputKind(enum.Enum):
    """What the benchmarked callable is fed, which decides whether a fresh
    object is required per iteration.

    A fresh object is needed only to keep a UTF-8 cache from being reused
    across iterations, so it is needed only when the relevant data is
    non-ASCII -- ASCII PyUnicode objects carry no separate UTF-8 buffer.
    """

    BYTES = "bytes"  # raw bytes: no UTF-8 cache exists, never copy
    STR = "str"  # decoded document: copy when the document is non-ASCII
    OBJECT = "object"  # parsed object: copy when any string in it is non-ASCII
    OBJECT_CACHED = "object_cached"  # parsed object with its cache deliberately warm


class RunOptions:
    """Knobs that apply to every measurement in a run."""

    __slots__ = (
        "statistic",
        "rounds",
        "verify_output",
        "allow_output_mismatch",
    )

    def __init__(
        self,
        statistic: str = "median",
        rounds: int = DEFAULT_ROUNDS,
        verify_output: bool = True,
        allow_output_mismatch: bool = False,
    ):
        self.statistic = statistic
        self.rounds = rounds
        self.verify_output = verify_output
        self.allow_output_mismatch = allow_output_mismatch


class BenchmarkFunction:
    def __init__(self, func: Callable, library_name: str) -> None:
        self.func = func
        self.library_name = library_name


class BenchmarkGroup:
    def __init__(
        self,
        functions: list[BenchmarkFunction],
        base_index_name: str,
        group_name: str,
        category: BenchmarkCategory,
        input_kind: InputKind,
        locality: Locality,
        skip_when_ascii: bool = False,
        unfair_when_non_ascii: bool = False,
    ) -> None:
        self.functions = functions
        self.base_index_name = base_index_name
        self.group_name = group_name
        self.category = category
        self.input_kind = input_kind
        self.locality = locality
        self.skip_when_ascii = skip_when_ascii
        self.unfair_when_non_ascii = unfair_when_non_ascii

    @property
    def index_name(self) -> str:
        return make_index_name(self.base_index_name, self.locality.value)

    @property
    def is_dumps(self) -> bool:
        return self.category != BenchmarkCategory.LOADS


# ---------------------------------------------------------------------------
# Object preparation helpers
# ---------------------------------------------------------------------------


def _gc_prepare():
    gc.collect()
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    return gc_was_enabled


def _check_str_cache(s: str, want_cache: bool):
    _, _, is_ascii, _ = internal.inspect_pyunicode(s)
    return is_ascii or want_cache == internal.pyunicode_has_utf8_cache(s)


def _recursive_check_cache(obj, want_cache: bool):
    if isinstance(obj, str):
        return _check_str_cache(obj, want_cache)
    if isinstance(obj, list):
        for item in obj:
            if not _recursive_check_cache(item, want_cache):
                return False
        return True
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not _recursive_check_cache(key, want_cache):
                return False
            if not _recursive_check_cache(value, want_cache):
                return False
        return True
    return True


def _recursive_copy_obj(obj):
    """Recursively copy a JSON-serializable object, using copy_unicode for str/dict keys."""
    if isinstance(obj, dict):
        return {
            internal.copy_unicode(k): _recursive_copy_obj(v) for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_recursive_copy_obj(item) for item in obj]
    if isinstance(obj, str):
        return internal.copy_unicode(obj)
    return obj


def _copy_bytes(data: bytes) -> bytes:
    """A genuinely distinct bytes object.

    bytes(b), b[:] and b + b"" all return the original object for an exact
    bytes, which would collapse a cold ring into a single shared buffer.
    """
    return bytes(memoryview(data))


def ensure_utf8_cache(encodable):
    import orjson

    orjson.dumps(encodable)


# ---------------------------------------------------------------------------
# CPU pinning and environment capture
# ---------------------------------------------------------------------------


def _read_sys(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _read_sys_int(path: str) -> int | None:
    value = _read_sys(path)
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _linux_cpus() -> list[tuple[int, int, int, str]]:
    """[(cpu_id, capacity, max_freq_khz, siblings_list)] from sysfs."""
    base = "/sys/devices/system/cpu"
    cpus = []
    try:
        entries = os.listdir(base)
    except OSError:
        return cpus
    for entry in entries:
        match = re.fullmatch(r"cpu(\d+)", entry)
        if not match:
            continue
        cpu_id = int(match.group(1))
        capacity = _read_sys_int(f"{base}/{entry}/cpu_capacity") or 0
        freq = _read_sys_int(f"{base}/{entry}/cpufreq/cpuinfo_max_freq") or 0
        siblings = _read_sys(f"{base}/{entry}/topology/thread_siblings_list")
        cpus.append((cpu_id, capacity, freq, siblings or str(cpu_id)))
    return sorted(cpus)


def _speed_ranks(cpus) -> tuple[dict[int, int], str]:
    """Per-CPU speed rank, using whichever sysfs signal actually discriminates.

    cpu_capacity is the natural choice on arm big.LITTLE, but Intel hybrid
    parts report a uniform 1024 for every core while cpuinfo_max_freq correctly
    separates P cores (5.3GHz) from E cores (4.2GHz). Preferring capacity
    unconditionally would fail to see the split on exactly the hardware where
    landing on the wrong core class inverts the comparison.
    """
    by_capacity = {cpu_id: capacity for cpu_id, capacity, _, _ in cpus}
    by_freq = {cpu_id: freq for cpu_id, _, freq, _ in cpus}
    if len(set(by_capacity.values())) > 1:
        return by_capacity, "cpu_capacity"
    if len(set(by_freq.values())) > 1:
        return by_freq, "cpuinfo_max_freq"
    return by_capacity if any(by_capacity.values()) else by_freq, "uniform"


def choose_pin_core() -> tuple[int | None, str]:
    """Pick a core to pin to. Returns (cpu_id, description).

    Prefers the fastest class, and within it the first thread of a physical
    core so the SMT sibling stays free. cpu0 is avoided when there is a choice
    because it typically absorbs more interrupt work.
    """
    if not hasattr(os, "sched_getaffinity"):
        return None, "unsupported-platform"
    allowed = os.sched_getaffinity(0)
    cpus = [c for c in _linux_cpus() if c[0] in allowed]
    if not cpus:
        return None, "no-topology"
    ranks, signal = _speed_ranks(cpus)
    best_rank = max(ranks[c[0]] for c in cpus)
    fastest = [c for c in cpus if ranks[c[0]] == best_rank]
    klass = "performance" if signal != "uniform" else "uniform"

    def is_first_thread(entry) -> bool:
        cpu_id, _, _, siblings = entry
        first = siblings.replace(",", "-").split("-")[0]
        try:
            return int(first) == cpu_id
        except ValueError:
            return True

    primary = [c for c in fastest if is_first_thread(c)] or fastest
    non_zero = [c for c in primary if c[0] != 0] or primary
    chosen = non_zero[0]
    return chosen[0], f"{klass} core via {signal}"


def apply_pin(core: int) -> bool:
    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {core})
            return True
        except OSError:
            return False
    if platform.system() == "Windows":
        try:
            import ctypes

            handle = ctypes.windll.kernel32.GetCurrentProcess()
            return bool(
                ctypes.windll.kernel32.SetProcessAffinityMask(handle, 1 << core)
            )
        except Exception:
            return False
    return False


def _collect_cpu_env(pinned: int | None) -> dict[str, str]:
    """Environment facts that change results but are invisible in the numbers.

    Without these recorded, results contributed from different machines are not
    comparable and nobody can tell why.
    """
    env: dict[str, str] = {}
    governors = set()
    for cpu_id, _, _, _ in _linux_cpus():
        gov = _read_sys(f"/sys/devices/system/cpu/cpu{cpu_id}/cpufreq/scaling_governor")
        if gov:
            governors.add(gov)
    env["governor"] = ",".join(sorted(governors)) if governors else "unknown"

    no_turbo = _read_sys("/sys/devices/system/cpu/intel_pstate/no_turbo")
    boost = _read_sys("/sys/devices/system/cpu/cpufreq/boost")
    if no_turbo is not None:
        env["turbo"] = "off" if no_turbo == "1" else "on"
    elif boost is not None:
        env["turbo"] = "on" if boost == "1" else "off"
    else:
        env["turbo"] = "unknown"

    smt = _read_sys("/sys/devices/system/cpu/smt/active")
    env["smt"] = {"1": "on", "0": "off"}.get(smt or "", "unknown")

    cpus = _linux_cpus()
    tiers = sorted({freq for _, _, freq, _ in cpus if freq}, reverse=True)
    if tiers:
        env["cpu_tiers"] = ", ".join(
            "{}MHz x{}".format(freq // 1000, sum(1 for _, _, f, _ in cpus if f == freq))
            for freq in tiers
        )

    if pinned is None:
        env["pinned_core"] = "none"
        # Unpinned on a hybrid part is the single biggest reproducibility hole:
        # the same comparison inverts between core classes.
        env["pinned_core_class"] = "unpinned-hybrid" if len(tiers) > 1 else "unpinned"
    else:
        freq = _read_sys_int(
            f"/sys/devices/system/cpu/cpu{pinned}/cpufreq/cpuinfo_max_freq"
        )
        siblings = _read_sys(
            f"/sys/devices/system/cpu/cpu{pinned}/topology/thread_siblings_list"
        )
        env["pinned_core"] = str(pinned)
        env["pinned_core_max_mhz"] = str(freq // 1000) if freq else "unknown"
        env["pinned_core_siblings"] = siblings or "unknown"
        if len(tiers) <= 1 or not freq:
            env["pinned_core_class"] = "uniform"
        elif freq == tiers[0]:
            env["pinned_core_class"] = "fastest"
        elif freq == tiers[-1]:
            env["pinned_core_class"] = "efficiency"
        else:
            env["pinned_core_class"] = "performance"

    try:
        env["loadavg"] = "{:.2f} {:.2f} {:.2f}".format(*os.getloadavg())
    except (OSError, AttributeError):
        env["loadavg"] = "unknown"
    env["hash_seed"] = os.environ.get("PYTHONHASHSEED") or (
        "randomized" if sys.flags.hash_randomization else "disabled"
    )
    return env


def calibrate_timer_overhead(samples: int = 20000) -> int:
    """Median nanoseconds the harness itself adds to one measured call.

    Recorded rather than subtracted: it is an additive constant on every
    library, so it compresses ratios slightly, and the reader deserves to see
    how big it is relative to the fastest case (~1.4us).
    """

    def noop(_obj):
        return None

    ring = [None]
    gc_was_enabled = _gc_prepare()
    try:
        _total, times = internal.benchmark_run(
            func=noop, ring=ring, repeat=samples, warmup=200, factory=None
        )
    finally:
        if gc_was_enabled:
            gc.enable()
    times.sort()
    return int(times[len(times) // 2])


# ---------------------------------------------------------------------------
# Last level cache detection
# ---------------------------------------------------------------------------


def _parse_cache_size(text: str) -> int:
    """Parse a sysfs cache size such as '30720K' or '8M' into bytes."""
    text = text.strip().upper()
    mult = 1
    if text.endswith("K"):
        mult, text = 1024, text[:-1]
    elif text.endswith("M"):
        mult, text = 1024 * 1024, text[:-1]
    elif text.endswith("G"):
        mult, text = 1024 * 1024 * 1024, text[:-1]
    return int(float(text) * mult)


def _llc_from_sysfs() -> int:
    """Largest unified cache reported by Linux sysfs for cpu0."""
    best = 0
    base = "/sys/devices/system/cpu/cpu0/cache"
    if not os.path.isdir(base):
        return 0
    for entry in sorted(os.listdir(base)):
        size_path = os.path.join(base, entry, "size")
        type_path = os.path.join(base, entry, "type")
        if not os.path.exists(size_path):
            continue
        try:
            if os.path.exists(type_path):
                with open(type_path) as f:
                    if f.read().strip().lower() not in ("unified", "data"):
                        continue
            with open(size_path) as f:
                best = max(best, _parse_cache_size(f.read()))
        except (OSError, ValueError):
            continue
    return best


def _llc_from_sysctl() -> int:
    """Largest unified cache reported by macOS sysctl."""
    for key in ("hw.l3cachesize", "hw.perflevel0.l2cachesize", "hw.l2cachesize"):
        try:
            out = subprocess.run(
                ["sysctl", "-n", key], capture_output=True, text=True, timeout=5
            )
        except (OSError, subprocess.SubprocessError):
            return 0
        if out.returncode == 0 and out.stdout.strip().isdigit():
            value = int(out.stdout.strip())
            if value > 0:
                return value
    return 0


def _llc_from_cpuinfo() -> int:
    if not _lib_available("cpuinfo"):
        return 0
    import cpuinfo

    info = cpuinfo.get_cpu_info()
    for key in ("l3_cache_size", "l2_cache_size"):
        value = info.get(key)
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, str):
            try:
                return _parse_cache_size(value.replace("iB", "").replace("B", ""))
            except ValueError:
                continue
    return 0


def get_llc_bytes() -> tuple[int, str]:
    """Detect the last level cache size. Returns (bytes, source description).

    The cold ring is sized against this, so the source is recorded in the
    result file: a wrong or fallback value makes cold numbers incomparable
    across machines and the reader needs to be able to see that.
    """
    system = platform.system()
    if system == "Linux":
        value = _llc_from_sysfs()
        if value:
            return value, "linux-sysfs"
    elif system == "Darwin":
        value = _llc_from_sysctl()
        if value:
            return value, "darwin-sysctl"
    value = _llc_from_cpuinfo()
    if value:
        return value, "py-cpuinfo"
    return _FALLBACK_LLC_BYTES, "fallback"


# ---------------------------------------------------------------------------
# Ring construction and measurement
# ---------------------------------------------------------------------------


def _measure_object_bytes(obj) -> int:
    """Bytes one deep copy of *obj* occupies, via tracemalloc.

    Only used to size the cold ring, and only once per file.
    """
    import tracemalloc

    tracemalloc.start()
    try:
        copy = _recursive_copy_obj(obj)
        current, _ = tracemalloc.get_traced_memory()
        del copy
        return int(current)
    finally:
        tracemalloc.stop()


def compute_ring_size(
    locality: Locality,
    copy_bytes: int,
    llc_bytes: int,
    cold_multiple: float,
    repeat: int,
) -> int:
    """Ring length K.

    HOT is always 1 (build, measure, free -- peak is one copy). COLD holds
    enough copies to overflow the last level cache. A single copy already
    larger than the target working set yields K == 1, which is correct: such
    an object cannot sit in cache anyway.
    """
    if locality is Locality.HOT:
        return 1
    if copy_bytes <= 0:
        return 1
    target = int(cold_multiple * llc_bytes)
    return max(1, min(math.ceil(target / copy_bytes), repeat))


def compute_warmup(repeat: int) -> int:
    """Warmup iterations. The old harness used a single call, which is far
    short of where frequency ramp and allocator state settle."""
    return min(max(repeat // 10, _WARMUP_MIN), _WARMUP_MAX)


class _RingSpec:
    """Everything needed to build the ring for one (file, group) pair."""

    __slots__ = ("input_kind", "base", "needs_fresh", "copy_bytes")

    def __init__(self, input_kind: InputKind, base, needs_fresh: bool, copy_bytes: int):
        self.input_kind = input_kind
        self.base = base
        self.needs_fresh = needs_fresh
        self.copy_bytes = copy_bytes

    def make_copy(self):
        if self.input_kind is InputKind.BYTES:
            return _copy_bytes(self.base)
        if self.input_kind is InputKind.STR:
            return internal.copy_unicode(self.base)
        return _recursive_copy_obj(self.base)

    def factory(self):
        """The C-level per-iteration factory, or None when no fresh object is
        needed. Returned as (callable, arg) for PyObject_CallOneArg."""
        if not self.needs_fresh:
            return None, None
        if self.input_kind is InputKind.STR:
            return internal.copy_unicode, self.base
        return _recursive_copy_obj, self.base

    def build_ring(self, ring_size: int) -> list:
        # Always distinct copies: a ring of K references to one object has no
        # working set and would make COLD identical to HOT.
        ring = [self.make_copy() for _ in range(ring_size)]
        if self.input_kind is InputKind.OBJECT_CACHED:
            for item in ring:
                ensure_utf8_cache(item)
            # Every entry comes off the same code path, so verifying one is
            # enough; walking all K would be O(K * strings) per measurement.
            assert _recursive_check_cache(ring[0], True)
        return ring


def _interleaved_measure(
    funcs: list[tuple[str, Callable]],
    spec: _RingSpec,
    ring_size: int,
    repeat: int,
    warmup: int,
    rounds: int,
) -> dict[str, list[int]]:
    """Measure every library, cycling them across *rounds* chunks.

    Running each library to completion in turn makes slow drift (thermal
    throttling, frequency ramp) a systematic bias against whichever library
    runs last -- and ssrjson was last in every group. Interleaving spreads each
    library's samples over the whole wall-clock span, and rotating the order
    each round stops any library from owning a position.

    Every library keeps its own ring for the whole measurement so the rotation
    invariant survives across rounds; peak memory is therefore n_libs rings.
    With rounds == 1 there is nothing to interleave, so rings are built and
    released one library at a time -- that makes --rounds 1 the escape hatch
    for datasets whose single copy is large enough that n_libs of them do not
    fit in RAM.
    """
    factory, factory_arg = spec.factory()

    if rounds <= 1:
        times_seq: dict[str, list[int]] = {}
        for name, func in funcs:
            ring = spec.build_ring(ring_size)
            gc_was_enabled = _gc_prepare()
            try:
                _total, chunk_times = internal.benchmark_run(
                    func=func,
                    ring=ring,
                    repeat=repeat,
                    warmup=warmup,
                    factory=factory,
                    factory_arg=factory_arg,
                )
            finally:
                if gc_was_enabled:
                    gc.enable()
                del ring
            times_seq[name] = chunk_times
        return times_seq

    rings = {name: spec.build_ring(ring_size) for name, _ in funcs}

    base, extra = divmod(repeat, max(1, rounds))
    chunks = [base + (1 if i < extra else 0) for i in range(max(1, rounds))]
    chunks = [c for c in chunks if c > 0]

    times: dict[str, list[int]] = {name: [] for name, _ in funcs}
    calls: dict[str, int] = {name: 0 for name, _ in funcs}

    gc_was_enabled = _gc_prepare()
    try:
        for round_index, chunk in enumerate(chunks):
            shift = round_index % len(funcs)
            order = funcs[shift:] + funcs[:shift]
            # Full warmup once; later rounds only need a short re-settle since
            # the core has been busy continuously.
            round_warmup = warmup if round_index == 0 else min(warmup, _WARMUP_MIN)
            for name, func in order:
                _total, chunk_times = internal.benchmark_run(
                    func=func,
                    ring=rings[name],
                    repeat=chunk,
                    warmup=round_warmup,
                    factory=factory,
                    factory_arg=factory_arg,
                    start_slot=calls[name],
                )
                calls[name] += round_warmup + chunk
                times[name].extend(chunk_times)
    finally:
        if gc_was_enabled:
            gc.enable()
        rings.clear()
    return times


def _verify_output(func: Callable, spec: _RingSpec, is_dumps: bool, expected) -> bool:
    """Check that a library actually produces the expected value.

    Without this a library that emits truncated or wrong output would simply be
    reported as the fastest one.
    """
    sample = spec.make_copy()
    try:
        output = func(sample)
        return (json.loads(output) if is_dumps else output) == expected
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Benchmark group definitions (with optional libraries)
# ---------------------------------------------------------------------------


def _get_benchmark_defs(localities: list[Locality]) -> tuple[BenchmarkGroup, ...]:
    available = _get_available_third_party_libs()
    if not available:
        raise RuntimeError(
            "No third-party JSON libraries are installed. "
            "Install at least one of: ujson, msgspec, orjson, ssrjson"
        )

    libs = {}
    for name in available:
        libs[name] = _import_lib(name)

    def _build_functions(
        func_defs: list[tuple[Callable, str]],
    ) -> list[BenchmarkFunction]:
        """Build list of BenchmarkFunctions, skipping unavailable libraries."""
        result = []
        for func, lib_name in func_defs:
            if lib_name == "json" or lib_name in available:
                result.append(BenchmarkFunction(func, lib_name))
        return result

    # Helper references
    ujson = libs.get("ujson")
    pydantic_core = libs.get("pydantic_core")
    msgspec = libs.get("msgspec")
    orjson = libs.get("orjson")
    ssrjson = libs.get("ssrjson")

    specs: list[tuple] = []

    def add(
        funcs,
        base_index: str,
        group_name: str,
        category: BenchmarkCategory,
        input_kind: InputKind,
        skip_when_ascii: bool = False,
        unfair_when_non_ascii: bool = False,
    ):
        specs.append(
            (
                funcs,
                base_index,
                group_name,
                category,
                input_kind,
                skip_when_ascii,
                unfair_when_non_ascii,
            )
        )

    # loads str
    loads_str_funcs = [(json.loads, "json")]
    if ujson:
        loads_str_funcs.append((ujson.loads, "ujson"))
    if msgspec:
        loads_str_funcs.append((msgspec.json.decode, "msgspec"))
    if orjson:
        loads_str_funcs.append((orjson.loads, "orjson"))
    if ssrjson:
        loads_str_funcs.append((ssrjson.loads, "ssrjson"))
    add(
        loads_str_funcs,
        _NAME_LOADS,
        "loads str",
        BenchmarkCategory.LOADS,
        InputKind.STR,
    )

    # loads bytes
    loads_bytes_funcs = [(json.loads, "json")]
    if ujson:
        loads_bytes_funcs.append((ujson.loads, "ujson"))
    if msgspec:
        loads_bytes_funcs.append((msgspec.json.decode, "msgspec"))
    if orjson:
        loads_bytes_funcs.append((orjson.loads, "orjson"))
    if ssrjson:
        loads_bytes_funcs.append((ssrjson.loads, "ssrjson"))
    add(
        loads_bytes_funcs,
        _NAME_LOADS,
        "loads bytes",
        BenchmarkCategory.LOADS,
        InputKind.BYTES,
    )

    # dumps to str
    dumps_str_funcs = [
        (lambda x: json.dumps(x, ensure_ascii=False, separators=_COMPACT), "json")
    ]
    if ujson:
        dumps_str_funcs.append((lambda x: ujson.dumps(x, ensure_ascii=False), "ujson"))
    if pydantic_core:
        dumps_str_funcs.append(
            (lambda x: pydantic_core.to_json(x).decode("utf-8"), "pydantic_core")
        )
    if msgspec:
        dumps_str_funcs.append(
            (lambda x: msgspec.json.encode(x).decode("utf-8"), "msgspec")
        )
    if orjson:
        dumps_str_funcs.append((lambda x: orjson.dumps(x).decode("utf-8"), "orjson"))
    if ssrjson:
        dumps_str_funcs.append((ssrjson.dumps, "ssrjson"))
    add(
        dumps_str_funcs,
        _NAME_DUMPSTOSTR,
        "dumps to str",
        BenchmarkCategory.DUMPS,
        InputKind.OBJECT,
    )

    # dumps to str (indented2)
    dumps_str_indent_funcs = [
        (lambda x: json.dumps(x, indent=2, ensure_ascii=False), "json")
    ]
    if ujson:
        dumps_str_indent_funcs.append(
            (lambda x: ujson.dumps(x, indent=2, ensure_ascii=False), "ujson")
        )
    if pydantic_core:
        dumps_str_indent_funcs.append(
            (
                lambda x: pydantic_core.to_json(x, indent=2).decode("utf-8"),
                "pydantic_core",
            )
        )
    if orjson:
        dumps_str_indent_funcs.append(
            (
                lambda x: orjson.dumps(x, option=orjson.OPT_INDENT_2).decode("utf-8"),
                "orjson",
            )
        )
    if ssrjson:
        dumps_str_indent_funcs.append((lambda x: ssrjson.dumps(x, indent=2), "ssrjson"))
    add(
        dumps_str_indent_funcs,
        _NAME_DUMPSTOSTR,
        "dumps to str (indented2)",
        BenchmarkCategory.DUMPS,
        InputKind.OBJECT,
    )

    # dumps to bytes
    dumps_bytes_funcs = [
        (
            lambda x: json.dumps(x, ensure_ascii=False, separators=_COMPACT).encode(
                "utf-8"
            ),
            "json",
        )
    ]
    if ujson:
        dumps_bytes_funcs.append(
            (lambda x: ujson.dumps(x, ensure_ascii=False).encode("utf-8"), "ujson")
        )
    if pydantic_core:
        dumps_bytes_funcs.append((pydantic_core.to_json, "pydantic_core"))
    if msgspec:
        dumps_bytes_funcs.append((msgspec.json.encode, "msgspec"))
    if orjson:
        dumps_bytes_funcs.append((orjson.dumps, "orjson"))
    if ssrjson:
        dumps_bytes_funcs.append((ssrjson.dumps_to_bytes, "ssrjson"))
    add(
        dumps_bytes_funcs,
        _NAME_DUMPSTOBYTES,
        "dumps to bytes",
        BenchmarkCategory.DUMPS_TO_BYTES,
        InputKind.OBJECT,
    )

    # dumps to bytes (indented2)
    dumps_bytes_indent_funcs = [
        (lambda x: json.dumps(x, indent=2, ensure_ascii=False).encode("utf-8"), "json")
    ]
    if ujson:
        dumps_bytes_indent_funcs.append(
            (
                lambda x: ujson.dumps(x, indent=2, ensure_ascii=False).encode("utf-8"),
                "ujson",
            )
        )
    if pydantic_core:
        dumps_bytes_indent_funcs.append(
            (
                lambda x: pydantic_core.to_json(x, indent=2),
                "pydantic_core",
            )
        )
    if orjson:
        dumps_bytes_indent_funcs.append(
            (
                lambda x: orjson.dumps(x, option=orjson.OPT_INDENT_2),
                "orjson",
            )
        )
    if ssrjson:
        dumps_bytes_indent_funcs.append(
            (lambda x: ssrjson.dumps_to_bytes(x, indent=2), "ssrjson")
        )
    add(
        dumps_bytes_indent_funcs,
        _NAME_DUMPSTOBYTES,
        "dumps to bytes (indented2)",
        BenchmarkCategory.DUMPS_TO_BYTES,
        InputKind.OBJECT,
    )

    # dumps to bytes (cached) - only relevant for non-ASCII
    dumps_bytes_cached_funcs = [
        (
            lambda x: json.dumps(x, ensure_ascii=False, separators=_COMPACT).encode(
                "utf-8"
            ),
            "json",
        )
    ]
    if ujson:
        dumps_bytes_cached_funcs.append(
            (
                lambda x: ujson.dumps(x, ensure_ascii=False).encode("utf-8"),
                "ujson",
            )
        )
    if pydantic_core:
        dumps_bytes_cached_funcs.append((pydantic_core.to_json, "pydantic_core"))
    if msgspec:
        dumps_bytes_cached_funcs.append((msgspec.json.encode, "msgspec"))
    if orjson:
        dumps_bytes_cached_funcs.append((orjson.dumps, "orjson"))
    if ssrjson:
        dumps_bytes_cached_funcs.append((ssrjson.dumps_to_bytes, "ssrjson"))
    add(
        dumps_bytes_cached_funcs,
        _NAME_DUMPSTOBYTES,
        "dumps to bytes (cached)",
        BenchmarkCategory.DUMPS_TO_BYTES,
        InputKind.OBJECT_CACHED,
        skip_when_ascii=True,
    )

    # dumps to bytes (no cache write) - diagnostic, only relevant for non-ASCII.
    # ssrJSON is forced out of its shipping default here to isolate what the
    # UTF-8 cache write costs. The other libraries cannot be switched, so this
    # group is NOT a like-for-like product comparison and stays out of the
    # aggregate summary; it answers "how fast is the encoder without the cache
    # tax", not "how fast is the package you installed".
    dumps_bytes_nowrite_funcs = [
        (
            lambda x: json.dumps(x, ensure_ascii=False, separators=_COMPACT).encode(
                "utf-8"
            ),
            "json",
        )
    ]
    if ujson:
        dumps_bytes_nowrite_funcs.append(
            (
                lambda x: ujson.dumps(x, ensure_ascii=False).encode("utf-8"),
                "ujson",
            )
        )
    if pydantic_core:
        dumps_bytes_nowrite_funcs.append((pydantic_core.to_json, "pydantic_core"))
    if msgspec:
        dumps_bytes_nowrite_funcs.append((msgspec.json.encode, "msgspec"))
    if orjson:
        dumps_bytes_nowrite_funcs.append((orjson.dumps, "orjson"))
    if ssrjson:
        dumps_bytes_nowrite_funcs.append(
            (
                lambda x: ssrjson.dumps_to_bytes(x, is_write_cache=False),
                "ssrjson",
            )
        )
    add(
        dumps_bytes_nowrite_funcs,
        _NAME_DUMPSTOBYTES,
        "dumps to bytes (no cache write)",
        BenchmarkCategory.DUMPS_TO_BYTES,
        InputKind.OBJECT,
        skip_when_ascii=True,
        unfair_when_non_ascii=True,
    )

    groups = []
    for locality in localities:
        for funcs, base_index, name, category, input_kind, skip_ascii, unfair in specs:
            groups.append(
                BenchmarkGroup(
                    _build_functions(funcs),
                    base_index,
                    name,
                    category=category,
                    input_kind=input_kind,
                    locality=locality,
                    skip_when_ascii=skip_ascii,
                    unfair_when_non_ascii=unfair,
                )
            )
    return tuple(groups)


def _get_benchmark_libraries(localities: list[Locality]) -> list[BenchmarkGroup]:
    return list(_get_benchmark_defs(localities))


# ---------------------------------------------------------------------------
# Inspect helpers
# ---------------------------------------------------------------------------


def _update_inspect_result(old_kind, old_size, old_is_ascii, kind, str_size, is_ascii):
    return (
        max(old_kind, kind),
        old_size + str_size,
        old_is_ascii and is_ascii,
    )


def _inspect_pyunicode_in_json(obj):
    kind = 1
    str_size = 0
    is_ascii = True
    if isinstance(obj, dict):
        for k, v in obj.items():
            _kind, _str_size, _is_ascii, _ = internal.inspect_pyunicode(k)
            kind, str_size, is_ascii = _update_inspect_result(
                kind, str_size, is_ascii, _kind, _str_size, _is_ascii
            )
            _kind, _str_size, _is_ascii = _inspect_pyunicode_in_json(v)
            kind, str_size, is_ascii = _update_inspect_result(
                kind, str_size, is_ascii, _kind, _str_size, _is_ascii
            )
        return kind, str_size, is_ascii
    if isinstance(obj, list):
        for item in obj:
            _kind, _str_size, _is_ascii = _inspect_pyunicode_in_json(item)
            kind, str_size, is_ascii = _update_inspect_result(
                kind, str_size, is_ascii, _kind, _str_size, _is_ascii
            )
        return kind, str_size, is_ascii
    if isinstance(obj, str):
        return internal.inspect_pyunicode(obj)[:3]
    return kind, str_size, is_ascii


def _get_processed_size(func: Callable, sample, is_dumps):
    """Bytes of work one call represents, used only for the GB/s annotation.

    *sample* must be the same kind of object the measured calls were fed, so
    that `loads str` is charged its UCS byte size rather than the UTF-8 length
    of the source file.
    """
    if is_dumps:
        output = func(sample)
        if isinstance(output, bytes):
            return len(output)
        return internal.inspect_pyunicode(output)[1]
    if isinstance(sample, bytes):
        return len(sample)
    return internal.inspect_pyunicode(sample)[1]


# ---------------------------------------------------------------------------
# System info helpers
# ---------------------------------------------------------------------------


def _get_ssrjson_rev():
    if not _lib_available("ssrjson"):
        return "unknown"
    import ssrjson

    return (
        getattr(ssrjson, "__version__", None) or getattr(ssrjson, "ssrjson").__version__
    )


def _get_real_output_file_name():
    rev = _get_ssrjson_rev()
    if not rev or rev == "unknown":
        return "benchmark_result.json"
    return f"benchmark_result_{rev}.json"


def _get_cpu_name() -> str:
    cpuinfo_spec = find_spec("cpuinfo")
    if cpuinfo_spec is not None:
        import cpuinfo

        cpu_name = cpuinfo.get_cpu_info().get("brand_raw", "UnknownCPU")
    else:
        cpu_name: str = platform.processor()
        if cpu_name.strip() == "":
            if os.path.exists("/proc/cpuinfo"):
                with open(file="/proc/cpuinfo", mode="r") as file:
                    cpu_info_lines = file.readlines()
                    for line in cpu_info_lines:
                        if "model name" in line:
                            cpu_name = re.sub(
                                pattern=r"model name\s+:\s+", repl="", string=line
                            )
                            break
            else:
                cpu_name = "UnknownCPU"
    return re.sub(pattern=r"\s+", repl=" ", string=cpu_name).strip()


def _get_mem_total() -> str:
    if _lib_available("psutil"):
        import psutil

        mem_total = psutil.virtual_memory().total // 1024 / (1024**2)
        return f"{mem_total:.3f}GiB"
    return "Unknown"


def _collect_system_info(
    pinned_core: int | None = None, pin_note: str = ""
) -> SystemInfo:
    """Collect system and environment information for the benchmark result."""
    info = SystemInfo()
    info.cpu_env = _collect_cpu_env(pinned_core)
    info.pin_note = pin_note
    info.timer_overhead_ns = calibrate_timer_overhead()
    info.rev = _get_ssrjson_rev()
    info.python = sys.version
    info.os_info = f"{platform.system()} {platform.machine()} {platform.release()} {platform.version()}"
    info.chipset = _get_cpu_name()
    info.memory = _get_mem_total()
    info.generated_time = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())

    for lib_name in ["orjson", "msgspec", "ujson", "pydantic_core"]:
        if _lib_available(lib_name):
            mod = _import_lib(lib_name)
            ver = getattr(mod, "__version__", "?")
        else:
            ver = "N/A"
        setattr(info, f"{lib_name}_ver", ver)

    if _lib_available("ssrjson"):
        import ssrjson

        feats = ssrjson.get_current_features()
        info.simd_flags = str({k: feats[k] for k in ("multi_lib", "simd")})

    return info


def fetch_header(result: BenchmarkFinalResult) -> str:
    """Format header text from system info stored in the benchmark result."""
    with open(os.path.join(_CUR_DIR, "template.md"), "r") as f:
        template = f.read()

    si = result.system_info
    if result.llc_bytes:
        llc = "{:.1f}MiB ({})".format(
            result.llc_bytes / 1024 / 1024, result.llc_source or "?"
        )
    else:
        llc = "N/A"

    env = si.cpu_env or {}
    core = env.get("pinned_core", "unknown")
    if core == "none":
        cpu_line = "NOT PINNED ({})".format(env.get("pinned_core_class", "unknown"))
    else:
        cpu_line = "core {} [{}] {}MHz siblings={}".format(
            core,
            env.get("pinned_core_class", "?"),
            env.get("pinned_core_max_mhz", "?"),
            env.get("pinned_core_siblings", "?"),
        )
    env_line = "governor={} turbo={} smt={} load={} hashseed={}".format(
        env.get("governor", "?"),
        env.get("turbo", "?"),
        env.get("smt", "?"),
        env.get("loadavg", "?"),
        env.get("hash_seed", "?"),
    )
    stat_line = "{} of {} rounds; timer overhead {}ns/call".format(
        result.statistic or "mean", result.rounds or 1, si.timer_overhead_ns
    )
    return template.format(
        REV=si.rev,
        TIME=si.generated_time,
        OS=si.os_info,
        PYTHON=si.python,
        ORJSON_VER=si.orjson_ver,
        MSGSPEC_VER=si.msgspec_ver,
        UJSON_VER=si.ujson_ver,
        PYDANTIC_CORE_VER=si.pydantic_core_ver,
        SIMD_FLAGS=si.simd_flags,
        CHIPSET=si.chipset,
        MEM=si.memory,
        PROCESS_MEM="{:.3f}GiB".format(result.processbytesgb),
        LOCALITY=(
            "{} (cold ring = {:g}x LLC)".format(
                ", ".join(result.locality_modes), result.cold_multiple
            )
            if result.locality_modes
            else "N/A"
        ),
        LLC=llc,
        MIN_ITERS=result.min_iterations,
        CPU=cpu_line,
        ENV=env_line,
        STATISTIC=stat_line,
    )


# ---------------------------------------------------------------------------
# Core benchmark execution
# ---------------------------------------------------------------------------


class _FileContext:
    """Per-file measurement parameters shared by every group and library."""

    __slots__ = (
        "raw_bytes",
        "repeat",
        "warmup",
        "llc_bytes",
        "cold_multiple",
        "object_is_ascii",
        "input_is_ascii",
        "object_copy_bytes",
        "parsed",
        "_text",
    )

    def __init__(
        self,
        raw_bytes: bytes,
        parsed,
        repeat: int,
        warmup: int,
        llc_bytes: int,
        cold_multiple: float,
        object_is_ascii: bool,
        input_is_ascii: bool,
        object_copy_bytes: int,
    ):
        self.raw_bytes = raw_bytes
        self.repeat = repeat
        self.warmup = warmup
        self.llc_bytes = llc_bytes
        self.cold_multiple = cold_multiple
        self.object_is_ascii = object_is_ascii
        self.input_is_ascii = input_is_ascii
        self.object_copy_bytes = object_copy_bytes
        # Parsed object and decoded text are only ever read from -- every ring
        # entry is a fresh copy -- so they are shared across all groups instead
        # of being rebuilt per group.
        self.parsed = parsed
        self._text = None

    @property
    def text(self) -> str:
        if self._text is None:
            self._text = self.raw_bytes.decode("utf-8")
        return self._text

    def make_spec(self, group: BenchmarkGroup) -> _RingSpec:
        """Decide the base object, whether fresh copies are required, and how
        big one copy is.

        Freshness is required only to defeat the UTF-8 cache, so it hinges on
        whether the *relevant* data is non-ASCII -- and the relevant data
        differs between loads and dumps. github.json is the proof: its source
        text is pure ASCII (it uses \\uXXXX escapes) while the parsed object
        holds non-ASCII strings, so the two predicates disagree.
        """
        kind = group.input_kind
        if kind is InputKind.BYTES:
            return _RingSpec(kind, self.raw_bytes, False, len(self.raw_bytes))
        if kind is InputKind.STR:
            copy_bytes = internal.inspect_pyunicode(self.text)[1]
            return _RingSpec(kind, self.text, not self.input_is_ascii, copy_bytes)
        needs_fresh = kind is InputKind.OBJECT and not self.object_is_ascii
        return _RingSpec(kind, self.parsed, needs_fresh, self.object_copy_bytes)


def _run_benchmark(
    cur_result_file: BenchmarkResultPerFile,
    ctx: _FileContext,
    benchmark_group: BenchmarkGroup,
    opts: "RunOptions",
):
    group_name = benchmark_group.group_name
    cur_target = cur_result_file[group_name]
    cur_target.libraries = [bf.library_name for bf in benchmark_group.functions]

    spec = ctx.make_spec(benchmark_group)
    ring_size = compute_ring_size(
        benchmark_group.locality,
        spec.copy_bytes,
        ctx.llc_bytes,
        ctx.cold_multiple,
        ctx.repeat,
    )
    cur_target.ring_size = ring_size
    cur_target.fresh_per_iter = spec.needs_fresh
    cur_target.copy_bytes = spec.copy_bytes
    # ssrJSON's UTF-8 cache writing is globally disabled but orjson has no such
    # switch, so on non-ASCII data the non-cached dumps groups charge orjson for
    # a cache write ssrjson never pays. Those numbers stay on their own chart
    # but are kept out of the aggregate summary; the (write cache) variant is
    # the apples-to-apples comparison.
    cur_target.in_summary = not (
        benchmark_group.unfair_when_non_ascii and not ctx.object_is_ascii
    )

    mismatched: set[str] = set()
    if opts.verify_output:
        for benchmark_target in benchmark_group.functions:
            if not _verify_output(
                benchmark_target.func, spec, benchmark_group.is_dumps, ctx.parsed
            ):
                mismatched.add(benchmark_target.library_name)
        if mismatched and not opts.allow_output_mismatch:
            raise RuntimeError(
                f"[{group_name}] output does not round-trip for: "
                f"{', '.join(sorted(mismatched))}. Benchmarking libraries that "
                f"produce different results compares different work; pass "
                f"--allow-output-mismatch to record and continue anyway."
            )
        for name in sorted(mismatched):
            print(f"  WARNING: {name} output mismatch in [{group_name}]")

    prefix = f"[{group_name}]"
    print(
        prefix
        + (" " * max(0, 50 - len(prefix)))
        + f"repeat={ctx.repeat} warmup={ctx.warmup} ring={ring_size} "
        + f"fresh={'Y' if spec.needs_fresh else 'N'} rounds={opts.rounds} "
        + f"libs={len(benchmark_group.functions)}"
    )
    times_by_lib = _interleaved_measure(
        [(bf.library_name, bf.func) for bf in benchmark_group.functions],
        spec,
        ring_size,
        ctx.repeat,
        ctx.warmup,
        opts.rounds,
    )

    for benchmark_target in benchmark_group.functions:
        name = benchmark_target.library_name
        times = times_by_lib[name]
        summary = summarize_times(times, opts.statistic)
        cur_lib = BenchmarkResultPerFileTargetLib()
        cur_lib.times = times
        cur_lib.repeat_count = len(times)
        cur_lib.speed = sum(times)
        cur_lib.std_dev = summary["std_dev"]
        cur_lib.stat = summary["stat"]
        cur_lib.stat_lo = summary["stat_lo"]
        cur_lib.stat_hi = summary["stat_hi"]
        cur_lib.minimum = summary["min"]
        cur_lib.median = summary["median"]
        cur_lib.mean = summary["mean"]
        cur_lib.p95 = summary["p95"]
        cur_lib.run_values = [summary["stat"]]
        cur_lib.output_ok = name not in mismatched
        cur_target[name] = cur_lib

    baseline_data = cur_target["json"]
    for benchmark_target in benchmark_group.functions:
        cur_lib = cur_target[benchmark_target.library_name]
        if benchmark_target.library_name == "ssrjson":
            # Measure on a throwaway copy so that writing a UTF-8 cache here
            # (the write-cache group does exactly that) cannot touch the shared
            # base object other groups build their rings from.
            size = _get_processed_size(
                benchmark_target.func, spec.make_copy(), benchmark_group.is_dumps
            )
            if cur_lib.stat > 0:
                cur_target.ssrjson_bytes_per_sec = size / (cur_lib.stat / _NS_IN_ONE_S)
        cur_lib.ratio = (
            math.inf if cur_lib.stat == 0 else baseline_data.stat / cur_lib.stat
        )
        # Conservative interval: pair the baseline's low with this library's
        # high and vice versa.
        cur_lib.ratio_lo = (
            baseline_data.stat_lo / cur_lib.stat_hi if cur_lib.stat_hi else 0.0
        )
        cur_lib.ratio_hi = (
            baseline_data.stat_hi / cur_lib.stat_lo if cur_lib.stat_lo else 0.0
        )


def _run_file_benchmark(
    benchmark_groups: list[BenchmarkGroup],
    file: pathlib.Path,
    process_bytes: int,
    min_iterations: int,
    llc_bytes: int,
    cold_multiple: float,
    index_s: str,
    opts: RunOptions,
):
    print(f"Running benchmark for {file.name}, index group: {index_s}")
    with open(file, "rb") as f:
        raw_bytes = f.read()
    base_file_name = os.path.basename(file)
    cur_result_file = BenchmarkResultPerFile()
    cur_result_file.byte_size = bytes_size = len(raw_bytes)
    if bytes_size == 0:
        raise RuntimeError(f"File {file} is empty.")
    parsed = json.loads(raw_bytes)
    kind, str_size, object_is_ascii = _inspect_pyunicode_in_json(parsed)
    assert isinstance(kind, int)
    assert isinstance(str_size, int)
    assert isinstance(object_is_ascii, bool)
    # The decoded document and the parsed object have independent ASCII-ness.
    input_is_ascii = bool(internal.inspect_pyunicode(raw_bytes.decode("utf-8"))[2])
    cur_result_file.pyunicode_size = str_size
    cur_result_file.pyunicode_kind = kind
    cur_result_file.pyunicode_is_ascii = object_is_ascii
    cur_result_file.input_is_ascii = input_is_ascii

    # Equal-bytes budget, floored so that big files still get a usable sample
    # count: canada.json used to be measured 31 times.
    repeat = max(min_iterations, int((process_bytes + bytes_size - 1) // bytes_size))
    ctx = _FileContext(
        raw_bytes=raw_bytes,
        parsed=parsed,
        repeat=repeat,
        warmup=compute_warmup(repeat),
        llc_bytes=llc_bytes,
        cold_multiple=cold_multiple,
        object_is_ascii=object_is_ascii,
        input_is_ascii=input_is_ascii,
        object_copy_bytes=_measure_object_bytes(parsed),
    )

    for benchmark_group in benchmark_groups:
        if benchmark_group.index_name == index_s and (
            not benchmark_group.skip_when_ascii or not object_is_ascii
        ):
            _run_benchmark(cur_result_file, ctx, benchmark_group, opts)
    return base_file_name, cur_result_file


def is_unix_except_macos():
    system = platform.system()
    return system in ("Linux", "AIX", "FreeBSD")


def _filter_groups(
    benchmark_groups: list[BenchmarkGroup],
    only: str | None,
) -> list[BenchmarkGroup]:
    """Filter benchmark groups by the --only flag.

    Valid values match ``BenchmarkCategory`` members: 'loads', 'dumps',
    'dumps_to_bytes'.  ``None`` means no filtering.
    """
    if only is None:
        return benchmark_groups
    category = BenchmarkCategory(only)
    return [g for g in benchmark_groups if g.category == category]


def run_benchmark(
    files: list[pathlib.Path],
    process_bytes: int,
    only: str | None = None,
    localities: list[Locality] | None = None,
    min_iterations: int = DEFAULT_MIN_ITERATIONS,
    cold_multiple: float = DEFAULT_COLD_MULTIPLE,
    llc_bytes: int | None = None,
    opts: RunOptions | None = None,
    pin_core: int | None = None,
    pin: bool = True,
    output_path: str | None = None,
) -> tuple[BenchmarkFinalResult, str]:
    """Run benchmarks and generate a JSON result file. Returns (result, filename)."""
    if localities is None:
        localities = [Locality.HOT, Locality.COLD]
    if opts is None:
        opts = RunOptions()
    if llc_bytes:
        llc_source = "user-specified"
    else:
        llc_bytes, llc_source = get_llc_bytes()

    # Pin before anything is measured. On a hybrid CPU an unpinned process can
    # be scheduled onto an efficiency core, or migrate mid-run, which changes
    # the answer rather than just adding noise.
    pinned_core: int | None = None
    pin_note = "disabled"
    if pin:
        candidate = pin_core if pin_core is not None else choose_pin_core()[0]
        if candidate is None:
            pin_note = "unavailable on this platform"
        elif apply_pin(candidate):
            pinned_core, pin_note = candidate, "ok"
        else:
            pin_note = "failed"
        if pinned_core is None:
            print(
                f"warning: could not pin to a CPU core ({pin_note}); "
                "results will be less reproducible"
            )

    # Run ssrJSON in its shipping configuration: the release wheel writes the
    # UTF-8 cache by default, and so do orjson, msgspec and pydantic_core --
    # which have no switch at all. Pinning the global to True is what makes the
    # default groups a like-for-like comparison. The one group that needs the
    # cache write off asks for it explicitly.
    ssrjson_available = _lib_available("ssrjson")
    old_write_cache_status = None
    if ssrjson_available:
        import ssrjson

        old_write_cache_status = ssrjson.get_current_features()["write_utf8_cache"]
        ssrjson.write_utf8_cache(True)

    try:
        file = output_path or _get_real_output_file_name()

        result = BenchmarkFinalResult()
        result.results = {}

        benchmark_groups = _filter_groups(_get_benchmark_libraries(localities), only)

        result.filenames = [f.name for f in files]
        result.processbytesgb = process_bytes / 1024 / 1024 / 1024
        result.locality_modes = [loc.value for loc in localities]
        result.min_iterations = min_iterations
        result.cold_multiple = cold_multiple
        result.llc_bytes = llc_bytes
        result.llc_source = llc_source
        result.statistic = opts.statistic
        result.rounds = opts.rounds
        result.system_info = _collect_system_info(pinned_core, pin_note)

        # Ordered category names per index group, so the report can lay out
        # subplots without needing to rebuild the group registry.
        result.categories = {}
        for group in benchmark_groups:
            names = result.categories.setdefault(group.index_name, [])
            if group.group_name not in names:
                names.append(group.group_name)

        for index_s in result.categories:
            result.results[index_s] = {}
            for bench_file in files:
                k, v = _run_file_benchmark(
                    benchmark_groups,
                    bench_file,
                    process_bytes,
                    min_iterations,
                    llc_bytes,
                    cold_multiple,
                    index_s,
                    opts,
                )
                result.results[index_s][k] = v
        output_result = result.dumps()

        if os.path.exists(file):
            os.remove(file)

        with open(file, "w", encoding="utf-8") as f:
            f.write(output_result)
        return result, file
    finally:
        if ssrjson_available and old_write_cache_status is not None:
            import ssrjson

            ssrjson.write_utf8_cache(old_write_cache_status)


def parse_file_result(j: dict) -> BenchmarkFinalResult:
    return BenchmarkFinalResult.parse(j)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    return ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0


def merge_run_results(results: list[BenchmarkFinalResult]) -> BenchmarkFinalResult:
    """Combine independent process runs into one result.

    Each library's number becomes the median across runs and its error bar
    becomes the observed run-to-run range. That range is the honest measure of
    reproducibility: the within-run spread cannot see binary/heap layout, which
    is fixed for a process's lifetime and shifts between processes.
    """
    if not results:
        raise ValueError("no run results to merge")
    merged = results[0]
    merged.runs = len(results)

    for index_name, files_dict in merged.results.items():
        for filename, file_result in files_dict.items():
            for group_name, target in file_result.targets.items():
                for lib_name, lib_result in target.lib_results.items():
                    values = []
                    ok = True
                    for other in results:
                        other_file = other.results.get(index_name, {}).get(filename)
                        if other_file is None:
                            continue
                        other_target = other_file.targets.get(group_name)
                        if other_target is None or lib_name not in other_target:
                            continue
                        entry = other_target[lib_name]
                        values.append(entry.stat)
                        ok = ok and entry.output_ok
                    if not values:
                        continue
                    lib_result.run_values = values
                    lib_result.stat = _median(values)
                    lib_result.stat_lo = min(values)
                    lib_result.stat_hi = max(values)
                    lib_result.output_ok = ok

                if "json" not in target:
                    continue
                baseline = target["json"]
                for lib_result in target.lib_results.values():
                    lib_result.ratio = (
                        math.inf
                        if lib_result.stat == 0
                        else baseline.stat / lib_result.stat
                    )
                    lib_result.ratio_lo = (
                        baseline.stat_lo / lib_result.stat_hi
                        if lib_result.stat_hi
                        else 0.0
                    )
                    lib_result.ratio_hi = (
                        baseline.stat_hi / lib_result.stat_lo
                        if lib_result.stat_lo
                        else 0.0
                    )
    return merged
