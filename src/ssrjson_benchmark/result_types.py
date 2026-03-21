import json
import math
from typing import Any


class BenchmarkResultPerFileTargetLib:
    """Per-library benchmark result within a benchmark group for a file."""

    __slots__ = ("speed", "ratio", "std_dev", "times")

    def __init__(
        self,
        speed: int = 0,
        ratio: float = 0.0,
        std_dev: float = 0.0,
        times: list[int] | None = None,
    ):
        self.speed = speed
        self.ratio = ratio
        self.std_dev = std_dev
        self.times = times if times is not None else []

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "speed": self.speed,
            "ratio": self.ratio,
            "std_dev": self.std_dev,
        }
        if self.times:
            d["times"] = self.times
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BenchmarkResultPerFileTargetLib":
        return cls(
            speed=d.get("speed", 0),
            ratio=d.get("ratio", 0.0),
            std_dev=d.get("std_dev", 0.0),
            times=d.get("times", []),
        )


class BenchmarkResultPerFileTarget:
    """Results of a single benchmark group for a single file.

    Contains per-library results and metadata about which libraries were tested.
    """

    __slots__ = ("libraries", "lib_results", "ssrjson_bytes_per_sec")

    def __init__(self):
        self.libraries: list[str] = []
        self.lib_results: dict[str, BenchmarkResultPerFileTargetLib] = {}
        self.ssrjson_bytes_per_sec: float = 0.0

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
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BenchmarkResultPerFileTarget":
        obj = cls()
        obj.libraries = d.get("libraries", [])
        obj.ssrjson_bytes_per_sec = d.get("ssrjson_bytes_per_sec", 0.0)
        reserved_keys = {"libraries", "ssrjson_bytes_per_sec"}
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
        "targets",
    )

    def __init__(self):
        self.byte_size: int = 0
        self.pyunicode_size: int = 0
        self.pyunicode_kind: int = 0
        self.pyunicode_is_ascii: bool = True
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
        reserved_keys = {
            "byte_size",
            "pyunicode_size",
            "pyunicode_kind",
            "pyunicode_is_ascii",
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
        "simd_flags",
        "generated_time",
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
        self.simd_flags: str = "N/A"
        self.generated_time: str = ""

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
            "simd_flags": self.simd_flags,
            "generated_time": self.generated_time,
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
        obj.simd_flags = d.get("simd_flags", "N/A")
        obj.generated_time = d.get("generated_time", "")
        return obj


class BenchmarkFinalResult:
    """Top-level benchmark result containing all data."""

    __slots__ = (
        "categories",
        "results",
        "filenames",
        "processbytesgb",
        "perbinbytesmb",
        "system_info",
    )

    def __init__(self):
        self.categories: list[str] = []
        self.results: dict[str, dict[str, BenchmarkResultPerFile]] = {}
        self.filenames: list[str] = []
        self.processbytesgb: float = 0.0
        self.perbinbytesmb: int = 0
        self.system_info: SystemInfo = SystemInfo()

    @classmethod
    def parse(cls, j: dict) -> "BenchmarkFinalResult":
        ret = cls()
        ret.categories = j["categories"]
        ret.filenames = j["filenames"]
        ret.processbytesgb = j["processbytesgb"]
        ret.perbinbytesmb = j["perbinbytesmb"]
        if "system_info" in j:
            ret.system_info = SystemInfo.from_dict(j["system_info"])
        ret.results = {}
        for index_name, files_dict in j["results"].items():
            ret.results[index_name] = {}
            for filename, file_data in files_dict.items():
                ret.results[index_name][filename] = BenchmarkResultPerFile.from_dict(
                    file_data
                )
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
            "perbinbytesmb": self.perbinbytesmb,
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
