import json
from typing import TYPE_CHECKING, Any


class BenchmarkResultBase(dict):
    ATTRS = ()
    DEFAULT_SECTION_CLS = dict

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __getitem__(self, key: Any) -> Any:
        try:
            return super().__getitem__(key)
        except KeyError:
            cls = getattr(self.__class__, "DEFAULT_SECTION_CLS", None)
            if cls is None:
                raise
            obj = cls()
            super().__setitem__(key, obj)
            return obj

    def __getattr__(self, attrname):
        try:
            return self.__getitem__(attrname)
        except KeyError:
            raise AttributeError(attrname)

    @classmethod
    def parse(cls, j):
        self = cls()
        for k, v in j.items():
            if k in cls.ATTRS:
                setattr(self, k, v)
            else:
                setattr(self, k, cls.DEFAULT_SECTION_CLS.parse(v))
        return self


class BenchmarkResultPerFileTargetLib(BenchmarkResultBase):
    speed: int
    ratio: float
    ATTRS = ("speed", "ratio")


class BenchmarkResultPerFileTarget(BenchmarkResultBase):
    DEFAULT_SECTION_CLS = BenchmarkResultPerFileTargetLib
    ATTRS = ("ssrjson_bytes_per_sec",)
    ssrjson_bytes_per_sec: float

    if TYPE_CHECKING:

        def __getitem__(self, key: str) -> BenchmarkResultPerFileTargetLib: ...


class BenchmarkResultPerFile(BenchmarkResultBase):
    DEFAULT_SECTION_CLS = BenchmarkResultPerFileTarget
    ATTRS = (
        "byte_size",
        "pyunicode_size",
        "pyunicode_kind",
        "pyunicode_is_ascii",
    )
    byte_size: int
    pyunicode_size: int
    pyunicode_kind: int
    pyunicode_is_ascii: bool

    if TYPE_CHECKING:

        def __getitem__(self, key: str) -> BenchmarkResultPerFileTarget: ...


class BenchmarkFinalResult(BenchmarkResultBase):
    categories: list[str]
    results: dict[str, dict[str, BenchmarkResultPerFile]]
    filenames: set[str]

    @classmethod
    def parse(cls, j: dict):
        ret = cls()
        ret.categories = j["categories"]
        ret.results = dict()
        ret.filenames = j["filenames"]
        for k, v in j["results"].items():
            ret.results[k] = dict()
            for a, b in v.items():
                ret.results[k][a] = BenchmarkResultPerFile.parse(b)
        return ret

    def dumps(self):
        return json.dumps(self, ensure_ascii=False, indent=4)
