import json
import math
from typing import Any


class BenchmarkResultPerFileTargetLib:
    """Per-library benchmark result within a benchmark group for a file."""

    __slots__ = (
        "speed",
        "ratio",
        "std_dev",
        "repeat_count",
        "times",
        "stat",
        "stat_lo",
        "stat_hi",
        "minimum",
        "median",
        "mean",
        "p95",
        "ratio_lo",
        "ratio_hi",
        "run_values",
        "output_ok",
    )

    def __init__(
        self,
        speed: int = 0,
        ratio: float = 0.0,
        std_dev: float = 0.0,
        repeat_count: int = 0,
        times: list[int] | None = None,
    ):
        self.speed = speed
        self.ratio = ratio
        self.std_dev = std_dev
        self.repeat_count = repeat_count
        # times is kept in memory for the summary statistics but not serialized
        self.times = times if times is not None else []
        # Primary statistic (per --statistic) in ns, with its confidence bounds.
        self.stat: float = 0.0
        self.stat_lo: float = 0.0
        self.stat_hi: float = 0.0
        # All four summaries are recorded so a reader can check whether the
        # conclusion depends on the choice: mean and min genuinely disagree
        # when two libraries have different noise profiles.
        self.minimum: float = 0.0
        self.median: float = 0.0
        self.mean: float = 0.0
        self.p95: float = 0.0
        self.ratio_lo: float = 0.0
        self.ratio_hi: float = 0.0
        # Primary statistic from each independent process run, when --runs > 1.
        self.run_values: list[float] = []
        # False when this library's output did not round-trip to the expected
        # object; the chart flags it rather than silently comparing wrong work.
        self.output_ok: bool = True

    def to_dict(self) -> dict:
        return {
            "speed": self.speed,
            "ratio": self.ratio,
            "std_dev": self.std_dev,
            "repeat_count": self.repeat_count,
            "stat": self.stat,
            "stat_lo": self.stat_lo,
            "stat_hi": self.stat_hi,
            "min": self.minimum,
            "median": self.median,
            "mean": self.mean,
            "p95": self.p95,
            "ratio_lo": self.ratio_lo,
            "ratio_hi": self.ratio_hi,
            "run_values": self.run_values,
            "output_ok": self.output_ok,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BenchmarkResultPerFileTargetLib":
        obj = cls(
            speed=d.get("speed", 0),
            ratio=d.get("ratio", 0.0),
            std_dev=d.get("std_dev", 0.0),
            repeat_count=d.get("repeat_count", 0),
        )
        obj.stat = d.get("stat", 0.0)
        obj.stat_lo = d.get("stat_lo", 0.0)
        obj.stat_hi = d.get("stat_hi", 0.0)
        obj.minimum = d.get("min", 0.0)
        obj.median = d.get("median", 0.0)
        obj.mean = d.get("mean", 0.0)
        obj.p95 = d.get("p95", 0.0)
        obj.ratio_lo = d.get("ratio_lo", 0.0)
        obj.ratio_hi = d.get("ratio_hi", 0.0)
        obj.run_values = d.get("run_values", [])
        obj.output_ok = d.get("output_ok", True)
        return obj


class BenchmarkResultPerFileTarget:
    """Results of a single benchmark group for a single file.

    Contains per-library results and metadata about which libraries were tested.
    """

    __slots__ = (
        "libraries",
        "lib_results",
        "ssrjson_bytes_per_sec",
        "ring_size",
        "fresh_per_iter",
        "copy_bytes",
        "in_summary",
    )

    def __init__(self):
        self.libraries: list[str] = []
        self.lib_results: dict[str, BenchmarkResultPerFileTargetLib] = {}
        self.ssrjson_bytes_per_sec: float = 0.0
        # False when this group's comparison is structurally unfair for this
        # file, so it is charted but excluded from the aggregate summary.
        self.in_summary: bool = True
        # Measurement conditions, recorded so a chart can be audited: how many
        # live copies the ring held, whether a fresh copy was built per
        # iteration, and how big one copy is.
        self.ring_size: int = 1
        self.fresh_per_iter: bool = False
        self.copy_bytes: int = 0

    def __getitem__(self, key: str) -> BenchmarkResultPerFileTargetLib:
        return self.lib_results[key]

    def __setitem__(self, key: str, value: BenchmarkResultPerFileTargetLib):
        self.lib_results[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.lib_results

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"libraries": self.libraries}
        for lib_name, lib_result in self.lib_results.items():
            d[lib_name] = lib_result.to_dict()
        d["ssrjson_bytes_per_sec"] = self.ssrjson_bytes_per_sec
        d["ring_size"] = self.ring_size
        d["fresh_per_iter"] = self.fresh_per_iter
        d["copy_bytes"] = self.copy_bytes
        d["in_summary"] = self.in_summary
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BenchmarkResultPerFileTarget":
        obj = cls()
        obj.libraries = d.get("libraries", [])
        obj.ssrjson_bytes_per_sec = d.get("ssrjson_bytes_per_sec", 0.0)
        obj.ring_size = d.get("ring_size", 1)
        obj.fresh_per_iter = d.get("fresh_per_iter", False)
        obj.copy_bytes = d.get("copy_bytes", 0)
        obj.in_summary = d.get("in_summary", True)
        reserved_keys = {
            "libraries",
            "ssrjson_bytes_per_sec",
            "ring_size",
            "fresh_per_iter",
            "copy_bytes",
            "in_summary",
        }
        for k, v in d.items():
            if k not in reserved_keys and isinstance(v, dict):
                obj.lib_results[k] = BenchmarkResultPerFileTargetLib.from_dict(v)
        # If libraries list is empty, infer from lib_results keys for backward compat
        if not obj.libraries and obj.lib_results:
            obj.libraries = list(obj.lib_results.keys())
        return obj


class BenchmarkResultPerFile:
    """All benchmark results for a single file."""

    __slots__ = (
        "byte_size",
        "pyunicode_size",
        "pyunicode_kind",
        "pyunicode_is_ascii",
        "input_is_ascii",
        "targets",
    )

    def __init__(self):
        self.byte_size: int = 0
        self.pyunicode_size: int = 0
        self.pyunicode_kind: int = 0
        # ASCII-ness of the strings inside the parsed object (governs dumps).
        self.pyunicode_is_ascii: bool = True
        # ASCII-ness of the decoded source document (governs loads from str).
        # These genuinely differ: github.json is ASCII source with \uXXXX
        # escapes producing non-ASCII strings.
        self.input_is_ascii: bool = True
        self.targets: dict[str, BenchmarkResultPerFileTarget] = {}

    def __getitem__(self, key: str) -> BenchmarkResultPerFileTarget:
        if key not in self.targets:
            self.targets[key] = BenchmarkResultPerFileTarget()
        return self.targets[key]

    def __setitem__(self, key: str, value: BenchmarkResultPerFileTarget):
        self.targets[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.targets

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "byte_size": self.byte_size,
            "pyunicode_size": self.pyunicode_size,
            "pyunicode_kind": self.pyunicode_kind,
            "pyunicode_is_ascii": self.pyunicode_is_ascii,
            "input_is_ascii": self.input_is_ascii,
        }
        for target_name, target in self.targets.items():
            d[target_name] = target.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BenchmarkResultPerFile":
        obj = cls()
        obj.byte_size = d.get("byte_size", 0)
        obj.pyunicode_size = d.get("pyunicode_size", 0)
        obj.pyunicode_kind = d.get("pyunicode_kind", 0)
        obj.pyunicode_is_ascii = d.get("pyunicode_is_ascii", True)
        obj.input_is_ascii = d.get("input_is_ascii", obj.pyunicode_is_ascii)
        reserved_keys = {
            "byte_size",
            "pyunicode_size",
            "pyunicode_kind",
            "pyunicode_is_ascii",
            "input_is_ascii",
        }
        for k, v in d.items():
            if k not in reserved_keys and isinstance(v, dict):
                obj.targets[k] = BenchmarkResultPerFileTarget.from_dict(v)
        return obj


class SystemInfo:
    """System and environment information collected during benchmark."""

    __slots__ = (
        "rev",
        "python",
        "os_info",
        "chipset",
        "memory",
        "orjson_ver",
        "msgspec_ver",
        "ujson_ver",
        "pydantic_core_ver",
        "simd_flags",
        "generated_time",
        "cpu_env",
        "pin_note",
        "timer_overhead_ns",
    )

    def __init__(self):
        self.rev: str = ""
        self.python: str = ""
        self.os_info: str = ""
        self.chipset: str = ""
        self.memory: str = ""
        self.orjson_ver: str = "N/A"
        self.msgspec_ver: str = "N/A"
        self.ujson_ver: str = "N/A"
        self.pydantic_core_ver: str = "N/A"
        self.simd_flags: str = "N/A"
        self.generated_time: str = ""
        # Environment facts that change results but are invisible in the
        # numbers: governor, turbo, SMT, pinned core and its class, load, hash
        # seed. Without these, results from different machines are not
        # comparable and nobody can tell why.
        self.cpu_env: dict[str, str] = {}
        self.pin_note: str = ""
        self.timer_overhead_ns: int = 0

    def to_dict(self) -> dict:
        return {
            "rev": self.rev,
            "python": self.python,
            "os_info": self.os_info,
            "chipset": self.chipset,
            "memory": self.memory,
            "orjson_ver": self.orjson_ver,
            "msgspec_ver": self.msgspec_ver,
            "ujson_ver": self.ujson_ver,
            "pydantic_core_ver": self.pydantic_core_ver,
            "simd_flags": self.simd_flags,
            "generated_time": self.generated_time,
            "cpu_env": self.cpu_env,
            "pin_note": self.pin_note,
            "timer_overhead_ns": self.timer_overhead_ns,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SystemInfo":
        obj = cls()
        obj.rev = d.get("rev", "")
        obj.python = d.get("python", "")
        obj.os_info = d.get("os_info", "")
        obj.chipset = d.get("chipset", "")
        obj.memory = d.get("memory", "")
        obj.orjson_ver = d.get("orjson_ver", "N/A")
        obj.msgspec_ver = d.get("msgspec_ver", "N/A")
        obj.ujson_ver = d.get("ujson_ver", "N/A")
        obj.pydantic_core_ver = d.get("pydantic_core_ver", "N/A")
        obj.simd_flags = d.get("simd_flags", "N/A")
        obj.generated_time = d.get("generated_time", "")
        obj.cpu_env = d.get("cpu_env", {})
        obj.pin_note = d.get("pin_note", "")
        obj.timer_overhead_ns = d.get("timer_overhead_ns", 0)
        return obj


def _normalize_categories(
    raw, results: dict[str, dict[str, "BenchmarkResultPerFile"]]
) -> dict[str, list[str]]:
    """Coerce the stored categories into {index_name: [group_name, ...]}.

    Result files written before the locality split store a single flat list
    covering every index group. Those are re-bucketed from the measured data
    itself so old reports still render.
    """
    if isinstance(raw, dict):
        return {k: list(v) for k, v in raw.items()}
    flat = list(raw or [])
    normalized: dict[str, list[str]] = {}
    for index_name, files_dict in results.items():
        present: list[str] = []
        for file_result in files_dict.values():
            for group_name in file_result.targets:
                if group_name not in present:
                    present.append(group_name)
        ordered = [name for name in flat if name in present]
        ordered += [name for name in present if name not in ordered]
        normalized[index_name] = ordered
    return normalized


class BenchmarkFinalResult:
    """Top-level benchmark result containing all data."""

    __slots__ = (
        "categories",
        "results",
        "filenames",
        "processbytesgb",
        "locality_modes",
        "min_iterations",
        "cold_multiple",
        "llc_bytes",
        "llc_source",
        "statistic",
        "rounds",
        "runs",
        "system_info",
    )

    def __init__(self):
        # index group name -> ordered group names, so the report can lay out
        # subplots without rebuilding the benchmark group registry (which would
        # break when re-rendering on a machine with different libs installed).
        self.categories: dict[str, list[str]] = {}
        self.results: dict[str, dict[str, BenchmarkResultPerFile]] = {}
        self.filenames: list[str] = []
        self.processbytesgb: float = 0.0
        self.locality_modes: list[str] = []
        self.min_iterations: int = 0
        self.cold_multiple: float = 0.0
        self.llc_bytes: int = 0
        self.llc_source: str = ""
        self.statistic: str = ""
        self.rounds: int = 1
        self.runs: int = 1
        self.system_info: SystemInfo = SystemInfo()

    @classmethod
    def parse(cls, j: dict) -> "BenchmarkFinalResult":
        ret = cls()
        ret.filenames = j.get("filenames", [])
        ret.processbytesgb = j.get("processbytesgb", 0.0)
        ret.locality_modes = j.get("locality_modes", [])
        ret.min_iterations = j.get("min_iterations", 0)
        ret.cold_multiple = j.get("cold_multiple", 0.0)
        ret.llc_bytes = j.get("llc_bytes", 0)
        ret.llc_source = j.get("llc_source", "")
        ret.statistic = j.get("statistic", "mean")
        ret.rounds = j.get("rounds", 1)
        ret.runs = j.get("runs", 1)
        if "system_info" in j:
            ret.system_info = SystemInfo.from_dict(j["system_info"])
        ret.results = {}
        for index_name, files_dict in j.get("results", {}).items():
            ret.results[index_name] = {}
            for filename, file_data in files_dict.items():
                ret.results[index_name][filename] = BenchmarkResultPerFile.from_dict(
                    file_data
                )
        ret.categories = _normalize_categories(j.get("categories", {}), ret.results)
        return ret

    def to_dict(self) -> dict:
        results_dict = {}
        for index_name, files_dict in self.results.items():
            results_dict[index_name] = {}
            for filename, file_result in files_dict.items():
                results_dict[index_name][filename] = file_result.to_dict()
        return {
            "categories": self.categories,
            "results": results_dict,
            "filenames": self.filenames,
            "processbytesgb": self.processbytesgb,
            "locality_modes": self.locality_modes,
            "min_iterations": self.min_iterations,
            "cold_multiple": self.cold_multiple,
            "llc_bytes": self.llc_bytes,
            "llc_source": self.llc_source,
            "statistic": self.statistic,
            "rounds": self.rounds,
            "runs": self.runs,
            "system_info": self.system_info.to_dict(),
        }

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=4)


def compute_std_dev(times: list[int], total: int) -> float:
    """Compute the standard deviation of per-iteration times."""
    n = len(times)
    if n <= 1:
        return 0.0
    mean = total / n
    variance = sum((t - mean) ** 2 for t in times) / (n - 1)
    return math.sqrt(variance)


def _quantile(sorted_times: list[int], q: float) -> float:
    n = len(sorted_times)
    if n == 0:
        return 0.0
    idx = min(n - 1, max(0, int(round(q * (n - 1)))))
    return float(sorted_times[idx])


def summarize_times(times: list[int], statistic: str = "median") -> dict:
    """Summarize per-iteration times.

    The confidence interval for the median is the distribution-free
    order-statistic interval, which needs no normality assumption and is exact.
    This replaces the old error bar, which plotted the standard deviation of a
    *single iteration* onto the summary -- that describes the spread of the
    latency distribution, not the uncertainty of the number being charted, and
    overstates it by roughly sqrt(n).
    """
    n = len(times)
    if n == 0:
        return {
            "stat": 0.0,
            "stat_lo": 0.0,
            "stat_hi": 0.0,
            "min": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "p95": 0.0,
            "std_dev": 0.0,
        }
    ordered = sorted(times)
    total = sum(ordered)
    mean = total / n
    median = (
        float(ordered[n // 2])
        if n % 2
        else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
    )
    minimum = float(ordered[0])
    p95 = _quantile(ordered, 0.95)
    std_dev = compute_std_dev(ordered, total)

    half_width = 1.96 * math.sqrt(n) / 2.0
    lo_idx = max(0, int(math.floor(n / 2.0 - half_width)))
    hi_idx = min(n - 1, int(math.ceil(n / 2.0 + half_width)))
    median_lo, median_hi = float(ordered[lo_idx]), float(ordered[hi_idx])

    if statistic == "min":
        stat, stat_lo, stat_hi = minimum, minimum, minimum
    elif statistic == "mean":
        sem = std_dev / math.sqrt(n) if n > 1 else 0.0
        stat, stat_lo, stat_hi = mean, mean - 1.96 * sem, mean + 1.96 * sem
    else:
        stat, stat_lo, stat_hi = median, median_lo, median_hi

    return {
        "stat": stat,
        "stat_lo": stat_lo,
        "stat_hi": stat_hi,
        "min": minimum,
        "median": median,
        "mean": mean,
        "p95": p95,
        "std_dev": std_dev,
    }
