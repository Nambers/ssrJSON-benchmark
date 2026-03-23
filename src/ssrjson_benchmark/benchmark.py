import enum
import gc
import json
import math
import os
import pathlib
import platform
import re
import sys
import time
from importlib.util import find_spec
from typing import Any, Callable

from . import _ssrjson_benchmark as internal
from .result_types import (
    BenchmarkFinalResult,
    BenchmarkResultPerFile,
    BenchmarkResultPerFileTargetLib,
    SystemInfo,
    compute_std_dev,
)

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_NS_IN_ONE_S = 1000000000

_NAME_LOADDUMP = "load&dump"
_NAME_DUMPSTOBYTES = "dumps_to_bytes"
INDEXED_GROUPS = [_NAME_LOADDUMP, _NAME_DUMPSTOBYTES]
PRINT_INDEX_GROUPS = ["Loads & Dumps to str", "Dumps to bytes"]


# ---------------------------------------------------------------------------
# Library availability detection
# ---------------------------------------------------------------------------


def _lib_available(name: str) -> bool:
    return find_spec(name) is not None


def _get_available_third_party_libs() -> list[str]:
    """Return list of available third-party JSON libraries (excluding stdlib json)."""
    candidates = ["ssrjson", "ujson", "msgspec", "orjson"]
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


class BenchmarkFunction:
    def __init__(self, func: Callable, library_name: str) -> None:
        self.func = func
        self.library_name = library_name


class BenchmarkGroup:
    def __init__(
        self,
        benchmarker: Callable,
        functions: list[BenchmarkFunction],
        index_name: str,
        group_name: str,
        category: BenchmarkCategory,
        input_preprocessor: Callable[[Any], Any] = lambda x: x,
        skip_when_ascii: bool = False,
    ) -> None:
        self.benchmarker = benchmarker
        self.functions = functions
        self.index_name = index_name
        self.group_name = group_name
        self.category = category
        self.input_preprocessor = input_preprocessor
        self.skip_when_ascii = skip_when_ascii

    @property
    def is_dumps(self) -> bool:
        return self.category != BenchmarkCategory.LOADS


# ---------------------------------------------------------------------------
# Benchmarker functions
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


def ensure_utf8_cache(encodable):
    import orjson

    orjson.dumps(encodable)


def _benchmark(
    repeat_time: int, times_per_bin: int, func, data: bytes
) -> tuple[int, list[int]]:
    """Run repeat benchmark for bytes input. Returns (total_ns, per_iteration_times)."""
    gc_was_enabled = _gc_prepare()
    try:
        return internal.benchmark_bytes_load(func, repeat_time, (data,))
    finally:
        if gc_was_enabled:
            gc.enable()


def _benchmark_unicode_arg(
    repeat_time: int, times_per_bin: int, func, unicode: str
) -> tuple[int, list[int]]:
    """Run repeat benchmark, disabling utf-8 cache. Returns (total_ns, per_iteration_times)."""
    gc_was_enabled = _gc_prepare()
    try:
        return internal.benchmark_unicode_invalidate_cache(
            func, repeat_time, times_per_bin, unicode
        )
    finally:
        if gc_was_enabled:
            gc.enable()


def _benchmark_invalidate_dump_cache(
    repeat_time: int, times_per_bin: int, func, raw_bytes: bytes
) -> tuple[int, list[int]]:
    """Invalidate UTF-8 cache for the same input. Returns (total_ns, per_iteration_times)."""
    gc_was_enabled = _gc_prepare()
    try:
        return internal.benchmark_invalidate_dump_cache(
            func, repeat_time, times_per_bin, raw_bytes, json.loads
        )
    finally:
        if gc_was_enabled:
            gc.enable()


def _benchmark_with_dump_cache(
    repeat_time: int, times_per_bin: int, func, raw_bytes: bytes
) -> tuple[int, list[int]]:
    gc_was_enabled = _gc_prepare()
    try:
        data = json.loads(raw_bytes)
        ensure_utf8_cache(data)
        assert _recursive_check_cache(data, True)
        return internal.benchmark_with_dump_cache(func, repeat_time, (data,))
    finally:
        if gc_was_enabled:
            gc.enable()


# ---------------------------------------------------------------------------
# Benchmark group definitions (with optional libraries)
# ---------------------------------------------------------------------------


def _get_benchmark_defs() -> tuple[BenchmarkGroup, ...]:
    available = _get_available_third_party_libs()
    if not available:
        raise RuntimeError(
            "No third-party JSON libraries are installed. "
            "Install at least one of: ssrjson, ujson, msgspec, orjson"
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
    ssrjson = libs.get("ssrjson")
    ujson = libs.get("ujson")
    msgspec = libs.get("msgspec")
    orjson = libs.get("orjson")

    groups = []

    # loads str
    loads_str_funcs = [(json.loads, "json")]
    if ssrjson:
        loads_str_funcs.append((ssrjson.loads, "ssrjson"))
    if ujson:
        loads_str_funcs.append((ujson.loads, "ujson"))
    if msgspec:
        loads_str_funcs.append((msgspec.json.decode, "msgspec"))
    if orjson:
        loads_str_funcs.append((orjson.loads, "orjson"))

    groups.append(
        BenchmarkGroup(
            _benchmark_unicode_arg,
            _build_functions(loads_str_funcs),
            INDEXED_GROUPS[0],
            "loads str",
            category=BenchmarkCategory.LOADS,
            input_preprocessor=lambda x: x.decode("utf-8"),
        )
    )

    # loads bytes
    loads_bytes_funcs = [(json.loads, "json")]
    if ssrjson:
        loads_bytes_funcs.append((ssrjson.loads, "ssrjson"))
    if ujson:
        loads_bytes_funcs.append((ujson.loads, "ujson"))
    if msgspec:
        loads_bytes_funcs.append((msgspec.json.decode, "msgspec"))
    if orjson:
        loads_bytes_funcs.append((orjson.loads, "orjson"))
    groups.append(
        BenchmarkGroup(
            _benchmark,
            _build_functions(loads_bytes_funcs),
            INDEXED_GROUPS[0],
            "loads bytes",
            category=BenchmarkCategory.LOADS,
        )
    )

    # dumps to str
    dumps_str_funcs = [(lambda x: json.dumps(x, ensure_ascii=False), "json")]
    if ssrjson:
        dumps_str_funcs.append((ssrjson.dumps, "ssrjson"))
    if ujson:
        dumps_str_funcs.append((lambda x: ujson.dumps(x, ensure_ascii=False), "ujson"))
    if msgspec:
        dumps_str_funcs.append(
            (lambda x: msgspec.json.encode(x).decode("utf-8"), "msgspec")
        )
    if orjson:
        dumps_str_funcs.append((lambda x: orjson.dumps(x).decode("utf-8"), "orjson"))

    groups.append(
        BenchmarkGroup(
            _benchmark_invalidate_dump_cache,
            _build_functions(dumps_str_funcs),
            INDEXED_GROUPS[0],
            "dumps to str",
            category=BenchmarkCategory.DUMPS,
        )
    )

    # dumps to str (indented2)
    dumps_str_indent_funcs = [
        (lambda x: json.dumps(x, indent=2, ensure_ascii=False), "json")
    ]
    if ssrjson:
        dumps_str_indent_funcs.append((lambda x: ssrjson.dumps(x, indent=2), "ssrjson"))
    if ujson:
        dumps_str_indent_funcs.append(
            (lambda x: ujson.dumps(x, indent=2, ensure_ascii=False), "ujson")
        )
    if msgspec:
        dumps_str_indent_funcs.append(
            (
                lambda x: msgspec.json.format(msgspec.json.encode(x), indent=2).decode(
                    "utf-8"
                ),
                "msgspec",
            )
        )
    if orjson:
        dumps_str_indent_funcs.append(
            (
                lambda x: orjson.dumps(x, option=orjson.OPT_INDENT_2).decode("utf-8"),
                "orjson",
            )
        )
    groups.append(
        BenchmarkGroup(
            _benchmark_invalidate_dump_cache,
            _build_functions(dumps_str_indent_funcs),
            INDEXED_GROUPS[0],
            "dumps to str (indented2)",
            category=BenchmarkCategory.DUMPS,
        )
    )

    # dumps to bytes
    dumps_bytes_funcs = [
        (lambda x: json.dumps(x, ensure_ascii=False).encode("utf-8"), "json")
    ]
    if ssrjson:
        dumps_bytes_funcs.append((ssrjson.dumps_to_bytes, "ssrjson"))
    if ujson:
        dumps_bytes_funcs.append(
            (lambda x: ujson.dumps(x, ensure_ascii=False).encode("utf-8"), "ujson")
        )
    if msgspec:
        dumps_bytes_funcs.append((msgspec.json.encode, "msgspec"))
    if orjson:
        dumps_bytes_funcs.append((orjson.dumps, "orjson"))

    groups.append(
        BenchmarkGroup(
            _benchmark_invalidate_dump_cache,
            _build_functions(dumps_bytes_funcs),
            INDEXED_GROUPS[1],
            "dumps to bytes",
            category=BenchmarkCategory.DUMPS_TO_BYTES,
        )
    )

    # dumps to bytes (indented2)
    dumps_bytes_indent_funcs = [
        (lambda x: json.dumps(x, indent=2, ensure_ascii=False).encode("utf-8"), "json")
    ]
    if ssrjson:
        dumps_bytes_indent_funcs.append(
            (lambda x: ssrjson.dumps_to_bytes(x, indent=2), "ssrjson")
        )
    if ujson:
        dumps_bytes_indent_funcs.append(
            (
                lambda x: ujson.dumps(x, indent=2, ensure_ascii=False).encode("utf-8"),
                "ujson",
            )
        )
    if msgspec:
        dumps_bytes_indent_funcs.append(
            (
                lambda x: msgspec.json.format(msgspec.json.encode(x), indent=2),
                "msgspec",
            )
        )
    if orjson:
        dumps_bytes_indent_funcs.append(
            (
                lambda x: orjson.dumps(x, option=orjson.OPT_INDENT_2),
                "orjson",
            )
        )
    groups.append(
        BenchmarkGroup(
            _benchmark_invalidate_dump_cache,
            _build_functions(dumps_bytes_indent_funcs),
            INDEXED_GROUPS[1],
            "dumps to bytes (indented2)",
            category=BenchmarkCategory.DUMPS_TO_BYTES,
        )
    )

    # dumps to bytes (cached) - only relevant for non-ASCII
    dumps_bytes_cached_funcs = [
        (lambda x: json.dumps(x, ensure_ascii=False).encode("utf-8"), "json")
    ]
    if ssrjson:
        dumps_bytes_cached_funcs.append((ssrjson.dumps_to_bytes, "ssrjson"))
    if ujson:
        dumps_bytes_cached_funcs.append(
            (
                lambda x: ujson.dumps(x, ensure_ascii=False).encode("utf-8"),
                "ujson",
            )
        )
    if msgspec:
        dumps_bytes_cached_funcs.append((msgspec.json.encode, "msgspec"))
    if orjson:
        dumps_bytes_cached_funcs.append((orjson.dumps, "orjson"))
    groups.append(
        BenchmarkGroup(
            _benchmark_with_dump_cache,
            _build_functions(dumps_bytes_cached_funcs),
            INDEXED_GROUPS[1],
            "dumps to bytes (cached)",
            category=BenchmarkCategory.DUMPS_TO_BYTES,
            skip_when_ascii=True,
        )
    )

    # dumps to bytes (write cache) - only relevant for non-ASCII
    dumps_bytes_wcache_funcs = [
        (lambda x: json.dumps(x, ensure_ascii=False).encode("utf-8"), "json")
    ]
    if ssrjson:
        dumps_bytes_wcache_funcs.append(
            (
                lambda x: ssrjson.dumps_to_bytes(x, is_write_cache=True),
                "ssrjson",
            )
        )
    if ujson:
        dumps_bytes_wcache_funcs.append(
            (
                lambda x: ujson.dumps(x, ensure_ascii=False).encode("utf-8"),
                "ujson",
            )
        )
    if msgspec:
        dumps_bytes_wcache_funcs.append((msgspec.json.encode, "msgspec"))
    if orjson:
        dumps_bytes_wcache_funcs.append((orjson.dumps, "orjson"))
    groups.append(
        BenchmarkGroup(
            _benchmark_invalidate_dump_cache,
            _build_functions(dumps_bytes_wcache_funcs),
            INDEXED_GROUPS[1],
            "dumps to bytes (write cache)",
            category=BenchmarkCategory.DUMPS_TO_BYTES,
            skip_when_ascii=True,
        )
    )

    return tuple(groups)


def _get_benchmark_libraries() -> dict[str, BenchmarkGroup]:
    return {x.group_name: x for x in _get_benchmark_defs()}


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


def _get_processed_size(func: Callable, input_data, is_dumps):
    if is_dumps:
        data_obj = json.loads(input_data)
        output = func(data_obj)
        if isinstance(output, bytes):
            size = len(output)
        else:
            size = internal.inspect_pyunicode(output)[1]
    else:
        size = (
            len(input_data)
            if isinstance(input_data, bytes)
            else internal.inspect_pyunicode(input_data)[1]
        )
    return size


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


def _collect_system_info() -> SystemInfo:
    """Collect system and environment information for the benchmark result."""
    info = SystemInfo()
    info.rev = _get_ssrjson_rev()
    info.python = sys.version
    info.os_info = f"{platform.system()} {platform.machine()} {platform.release()} {platform.version()}"
    info.chipset = _get_cpu_name()
    info.memory = _get_mem_total()
    info.generated_time = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())

    for lib_name in ["orjson", "msgspec", "ujson"]:
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
    return template.format(
        REV=si.rev,
        TIME=si.generated_time,
        OS=si.os_info,
        PYTHON=si.python,
        ORJSON_VER=si.orjson_ver,
        MSGSPEC_VER=si.msgspec_ver,
        UJSON_VER=si.ujson_ver,
        SIMD_FLAGS=si.simd_flags,
        CHIPSET=si.chipset,
        MEM=si.memory,
        PROCESS_MEM="{:.3f}GiB".format(result.processbytesgb),
        PER_BIN_MEM="{}MiB".format(result.perbinbytesmb),
    )


# ---------------------------------------------------------------------------
# Core benchmark execution
# ---------------------------------------------------------------------------


def _run_benchmark(
    cur_result_file: BenchmarkResultPerFile,
    repeat_times: int,
    times_per_bin: int,
    input_data: str | bytes,
    benchmark_group: BenchmarkGroup,
):
    group_name = benchmark_group.group_name
    cur_target = cur_result_file[group_name]
    cur_target.libraries = [bf.library_name for bf in benchmark_group.functions]

    input_data = benchmark_group.input_preprocessor(input_data)

    for benchmark_target in benchmark_group.functions:
        prefix = f"[{benchmark_target.library_name}][{benchmark_group.group_name}]"
        print(
            prefix
            + (" " * max(0, 50 - len(prefix)))
            + f"repeat_times={repeat_times} times_per_bin={times_per_bin}"
        )
        total, times = benchmark_group.benchmarker(
            repeat_times, times_per_bin, benchmark_target.func, input_data
        )
        cur_lib = BenchmarkResultPerFileTargetLib()
        cur_lib.speed = total
        cur_lib.times = times
        cur_lib.repeat_count = len(times)
        cur_lib.std_dev = compute_std_dev(times, total)
        cur_target[benchmark_target.library_name] = cur_lib

    baseline_name = "json"
    baseline_data = cur_target[baseline_name]
    for benchmark_target in benchmark_group.functions:
        cur_lib = cur_target[benchmark_target.library_name]
        if benchmark_target.library_name == "ssrjson":
            size = _get_processed_size(
                benchmark_target.func, input_data, benchmark_group.is_dumps
            )
            cur_target.ssrjson_bytes_per_sec = (
                size * repeat_times / (cur_lib.speed / _NS_IN_ONE_S)
            )
        cur_lib.ratio = (
            math.inf
            if baseline_data.speed == 0
            else (baseline_data.speed / cur_lib.speed)
        )


def _run_file_benchmark(
    benchmark_libraries: dict[str, BenchmarkGroup],
    file: pathlib.Path,
    process_bytes: int,
    bin_process_bytes: int,
    index_s: str,
):
    print(f"Running benchmark for {file.name}, index group: {index_s}")
    with open(file, "rb") as f:
        raw_bytes = f.read()
    base_file_name = os.path.basename(file)
    cur_result_file = BenchmarkResultPerFile()
    cur_result_file.byte_size = bytes_size = len(raw_bytes)
    if bytes_size == 0:
        raise RuntimeError(f"File {file} is empty.")
    kind, str_size, is_ascii = _inspect_pyunicode_in_json(json.loads(raw_bytes))
    assert isinstance(kind, int)
    assert isinstance(str_size, int)
    assert isinstance(is_ascii, bool)
    cur_result_file.pyunicode_size = str_size
    cur_result_file.pyunicode_kind = kind
    cur_result_file.pyunicode_is_ascii = is_ascii
    repeat_times = int((process_bytes + bytes_size - 1) // bytes_size)
    times_per_bin = max(1, bin_process_bytes // bytes_size)

    for benchmark_group in benchmark_libraries.values():
        if benchmark_group.index_name == index_s and (
            not benchmark_group.skip_when_ascii or not is_ascii
        ):
            _run_benchmark(
                cur_result_file, repeat_times, times_per_bin, raw_bytes, benchmark_group
            )
    return base_file_name, cur_result_file


def is_unix_except_macos():
    system = platform.system()
    return system in ("Linux", "AIX", "FreeBSD")


def _filter_groups(
    benchmark_libraries: dict[str, BenchmarkGroup],
    only: str | None,
) -> dict[str, BenchmarkGroup]:
    """Filter benchmark groups by the --only flag.

    Valid values match ``BenchmarkCategory`` members: 'loads', 'dumps',
    'dumps_to_bytes'.  ``None`` means no filtering.
    """
    if only is None:
        return benchmark_libraries
    category = BenchmarkCategory(only)
    return {k: v for k, v in benchmark_libraries.items() if v.category == category}


def run_benchmark(
    files: list[pathlib.Path],
    process_bytes: int,
    bin_process_bytes: int,
    only: str | None = None,
) -> tuple[BenchmarkFinalResult, str]:
    """Run benchmarks and generate a JSON result file. Returns (result, filename)."""
    # Disable ssrJSON cache writing globally if available
    ssrjson_available = _lib_available("ssrjson")
    old_write_cache_status = None
    if ssrjson_available:
        import ssrjson

        old_write_cache_status = ssrjson.get_current_features()["write_utf8_cache"]
        ssrjson.write_utf8_cache(False)

    try:
        file = _get_real_output_file_name()

        result = BenchmarkFinalResult()
        result.results = {}

        benchmark_libraries = _get_benchmark_libraries()
        filtered_libraries = _filter_groups(benchmark_libraries, only)

        result.categories = list(filtered_libraries.keys())
        result.filenames = [f.name for f in files]
        result.processbytesgb = process_bytes / 1024 / 1024 / 1024
        result.perbinbytesmb = int(bin_process_bytes / 1024 / 1024)
        result.system_info = _collect_system_info()

        active_index_groups = sorted(
            {g.index_name for g in filtered_libraries.values()}
        )
        for index_s in active_index_groups:
            result.results[index_s] = {}
            for bench_file in files:
                k, v = _run_file_benchmark(
                    filtered_libraries,
                    bench_file,
                    process_bytes,
                    bin_process_bytes,
                    index_s,
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
