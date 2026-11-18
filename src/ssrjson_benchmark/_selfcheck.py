"""Self-checks for the ring-buffer measurement loop.

Run with `python -m ssrjson_benchmark._selfcheck`. Assertion-driven on purpose:
the invariants below are the ones that fail silently -- a broken ring still
produces plausible-looking numbers, it just measures the wrong thing.
"""

import gc
import json
import os

from . import _ssrjson_benchmark as internal
from .benchmark import (
    InputKind,
    Locality,
    _inspect_pyunicode_in_json,
    compute_ring_size,
    compute_warmup,
)


def _recording_pair():
    """A factory that stamps each object with its build tick, and a func that
    records what it was handed."""
    state = {"tick": 0}
    seen = []

    def factory(_arg=None):
        state["tick"] += 1
        return ["obj", state["tick"]]

    def func(obj):
        seen.append(obj)
        return None

    return state, seen, factory, func


def check_ring_rotation():
    """Every measured object must have been evicted by exactly K-1 later builds.

    This is the whole point of the ring: it replaces the old bin, where the
    first item in a bin was 1 build old and the last was N builds old.
    """
    repeat = 20
    for k in (1, 2, 3, 7, 20):
        for warmup in (0, 1, 2, 5, 13):
            state, seen, factory, func = _recording_pair()
            ring = [factory() for _ in range(k)]
            internal.benchmark_run(
                func=func,
                ring=ring,
                repeat=repeat,
                warmup=warmup,
                factory=factory,
                factory_arg=None,
            )
            assert len(seen) == warmup + repeat, (k, warmup, len(seen))
            measured = seen[warmup:]
            # Builds completed before global iteration t starts: k (ring) + t.
            ages = {(k + warmup + i) - obj[1] for i, obj in enumerate(measured)}
            assert ages == {k - 1}, (
                f"K={k} warmup={warmup}: object ages {sorted(ages)}, expected {k - 1}"
            )
            ids = [id(obj) for obj in measured]
            assert len(set(ids)) == repeat, (
                f"K={k} warmup={warmup}: an object was measured more than once"
            )
    print("ok  ring rotation: measured object is always exactly K-1 builds old")


def check_no_factory_means_no_allocation():
    """factory=None must reuse the ring untouched -- this is the ASCII path,
    where no UTF-8 cache exists and copying would be pure overhead."""
    state, seen, factory, func = _recording_pair()
    ring = [factory() for _ in range(3)]
    built_ids = [id(obj) for obj in ring]
    internal.benchmark_run(func=func, ring=ring, repeat=9, warmup=2, factory=None)
    assert state["tick"] == 3, f"factory called {state['tick']} times, expected 3"
    assert [id(obj) for obj in ring] == built_ids, "ring was rebuilt"
    # Ring index is continuous across the warmup/measure boundary.
    assert [obj[1] for obj in seen] == [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2], [
        obj[1] for obj in seen
    ]
    print("ok  factory=None: no rebuilds, ring index continuous across warmup")


def check_hot_holds_one_copy():
    """HOT must keep exactly one live copy: that is what makes its peak memory
    a single object regardless of file size or iteration count."""
    live = set()

    class Tracked(list):
        def __init__(self):
            super().__init__()
            live.add(id(self))

        def __del__(self):
            live.discard(id(self))

    peak = {"n": 0}

    def factory(_arg=None):
        obj = Tracked()
        peak["n"] = max(peak["n"], len(live))
        return obj

    def func(_obj):
        peak["n"] = max(peak["n"], len(live))
        return None

    gc.collect()
    live.clear()
    ring = [factory()]
    internal.benchmark_run(
        func=func, ring=ring, repeat=50, warmup=5, factory=factory, factory_arg=None
    )
    del ring
    # The freshly built object briefly coexists with the one being replaced.
    assert peak["n"] <= 2, f"HOT held {peak['n']} live copies, expected at most 2"
    print(f"ok  hot locality: peak {peak['n']} live copies")


def check_ring_size_rules():
    llc = 32 * 1024 * 1024
    assert compute_ring_size(Locality.HOT, 1024, llc, 2.0, 10**6) == 1
    # Cold overflows the cache: 2 * 32MiB / 1MiB
    assert compute_ring_size(Locality.COLD, 1024 * 1024, llc, 2.0, 10**6) == 64
    # A single copy bigger than the target working set cannot be cached anyway.
    assert compute_ring_size(Locality.COLD, 512 * 1024 * 1024, llc, 2.0, 10**6) == 1
    # Never more slots than measured iterations.
    assert compute_ring_size(Locality.COLD, 1024, llc, 2.0, 12) == 12
    # Warmup is adaptive but bounded; the old harness used a single call.
    assert compute_warmup(31) == 20
    assert compute_warmup(1000) == 100
    assert compute_warmup(10**6) == 200
    print("ok  ring size and warmup rules")


def check_ascii_predicates():
    """The dumps predicate and the loads-str predicate genuinely differ.

    github.json is the case that would silently produce a biased measurement if
    a single predicate were used: pure ASCII source text carrying \\uXXXX
    escapes that decode to non-ASCII strings.
    """
    files_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_files")
    path = os.path.join(files_dir, "github.json")
    if not os.path.exists(path):
        print("skip ASCII predicates: bundled _files/github.json not present")
        return
    raw = open(path, "rb").read()
    input_is_ascii = bool(internal.inspect_pyunicode(raw.decode("utf-8"))[2])
    object_is_ascii = _inspect_pyunicode_in_json(json.loads(raw))[2]
    assert input_is_ascii is True, "github.json source text should be ASCII"
    assert object_is_ascii is False, "github.json parsed strings should be non-ASCII"
    # Which predicate each input kind must consult.
    assert InputKind.STR is not InputKind.OBJECT
    print("ok  ASCII predicates disagree on github.json, as they must")


def check_interleaving():
    """Every library must get samples spread across all rounds, and each keeps
    its own ring so the rotation invariant survives the round boundaries."""
    from .benchmark import BenchmarkFunction, _RingSpec, InputKind, _interleaved_measure

    calls: dict[str, list[int]] = {"a": [], "b": [], "c": []}
    order_log: list[str] = []

    def make(name):
        def fn(obj):
            calls[name].append(obj[1])
            order_log.append(name)
            return None

        return fn

    tick = [0]

    def factory(_arg=None):
        tick[0] += 1
        return ["obj", tick[0]]

    class StubSpec(_RingSpec):
        def make_copy(self):
            return factory()

        def factory(self):  # type: ignore[override]
            return factory, None

    spec = StubSpec(InputKind.OBJECT, ["obj", 0], True, 1)

    funcs = [BenchmarkFunction(n, make(n)) for n in ("a", "b", "c")]
    times = _interleaved_measure(
        funcs, spec, ring_size=2, repeat=30, warmup=0, rounds=5
    )

    for name in calls:
        assert len(times[name]) == 30, (name, len(times[name]))
    # No library may own a position: with 5 rounds and 3 libraries the rotation
    # must put each library first at least once.
    firsts = {order_log[i] for i in range(len(order_log)) if i % 30 == 0}
    assert len(firsts) > 1, f"library order never rotated: {firsts}"
    # Objects are never shared between libraries' rings.
    assert not (set(calls["a"]) & set(calls["b"])), "rings leaked between libraries"
    print("ok  interleaving: all libraries sampled every round, order rotates")


def check_statistics():
    """The summary statistic and its interval must behave, and the CI must be
    far narrower than the raw per-iteration spread."""
    from .result_types import summarize_times

    times = [100] * 99 + [10000]  # one outlier
    med = summarize_times(times, "median")
    assert med["median"] == 100, med
    assert med["min"] == 100
    assert med["mean"] > 190, med  # the outlier moves the mean a lot
    assert med["stat"] == med["median"]
    assert med["stat_lo"] <= med["stat"] <= med["stat_hi"]
    # The interval of the median must be much tighter than the raw spread.
    assert (med["stat_hi"] - med["stat_lo"]) < med["std_dev"], med

    mn = summarize_times(times, "min")
    assert mn["stat"] == 100 and mn["stat_lo"] == mn["stat_hi"] == 100
    mean = summarize_times(times, "mean")
    assert mean["stat"] == mean["mean"]
    assert mean["stat_lo"] < mean["stat"] < mean["stat_hi"]
    print("ok  statistics: median resists the outlier that skews the mean")


def check_summary_groups_are_shipping_config():
    """Every group in the aggregate must run all libraries in the
    configuration their wheel ships with.

    ssrJSON writes the UTF-8 cache by default and orjson/msgspec/pydantic_core
    cannot turn it off, so the default groups are like-for-like. The only group
    that forces ssrJSON out of its default is the diagnostic one, and it must
    be the only one excluded.
    """
    from .benchmark import Locality, _get_benchmark_defs

    by_name = {g.group_name: g for g in _get_benchmark_defs([Locality.COLD])}
    # The two ends of the reuse spectrum are both headline results.
    assert not by_name["dumps to bytes"].unfair_when_non_ascii
    assert not by_name["dumps to bytes (indented2)"].unfair_when_non_ascii
    assert not by_name["dumps to bytes (cached)"].unfair_when_non_ascii
    assert not by_name["loads str"].unfair_when_non_ascii
    # ...and the diagnostic is the only exclusion.
    diag = by_name["dumps to bytes (no cache write)"]
    assert diag.unfair_when_non_ascii
    assert diag.skip_when_ascii, "no-cache-write is meaningless on ASCII data"
    excluded = [n for n, g in by_name.items() if g.unfair_when_non_ascii]
    assert excluded == ["dumps to bytes (no cache write)"], excluded
    print("ok  summary keeps shipping-config groups, excludes only the diagnostic")


def check_cache_write_is_on_by_default():
    """The harness must measure ssrJSON as shipped. Forcing the global off was
    the old behaviour and it silently compared a non-default ssrJSON against
    orjson's only configuration."""
    try:
        import ssrjson
    except ImportError:
        print("skip cache write default: ssrjson not installed")
        return
    from .benchmark import Locality, _get_benchmark_defs

    old = ssrjson.get_current_features()["write_utf8_cache"]
    try:
        ssrjson.write_utf8_cache(True)
        by_name = {g.group_name: g for g in _get_benchmark_defs([Locality.HOT])}
        sample = {"k": "你好世界" * 8}

        def ssr_of(group_name):
            group = by_name[group_name]
            return next(f for f in group.functions if f.library_name == "ssrjson")

        # Default group: writes the cache, like orjson does.
        obj = internal.copy_json(sample)
        ssr_of("dumps to bytes")(obj)
        assert internal.pyunicode_has_utf8_cache(obj["k"]), (
            "default dumps_to_bytes did not populate the UTF-8 cache"
        )
        # Diagnostic group: explicitly does not.
        obj = internal.copy_json(sample)
        ssr_of("dumps to bytes (no cache write)")(obj)
        assert not internal.pyunicode_has_utf8_cache(obj["k"]), (
            "no-cache-write group still populated the UTF-8 cache"
        )
    finally:
        ssrjson.write_utf8_cache(old)
    print("ok  default groups write the UTF-8 cache, diagnostic group does not")


def check_baseline_is_compact():
    """The stdlib baseline must not emit more bytes than the compact
    libraries; it inflates every ratio measured against it."""
    import json as _json

    from .benchmark import Locality, _get_benchmark_defs

    sample = {"a": [1, 2, 3], "b": {"c": "d"}}
    groups = {g.group_name: g for g in _get_benchmark_defs([Locality.HOT])}
    for group_name in ("dumps to str", "dumps to bytes"):
        group = groups.get(group_name)
        if group is None:
            continue
        baseline = next(f for f in group.functions if f.library_name == "json")
        out = baseline(sample)
        text = out.decode() if isinstance(out, bytes) else out
        assert ", " not in text and ": " not in text, f"{group_name}: {text}"
        assert _json.loads(text) == sample
    print("ok  stdlib baseline emits compact separators")


def check_compact_twins():
    """A compact twin must be its source minus whitespace: same parsed value,
    same bytes inside every string literal, and loads-only."""
    import pathlib

    from .benchmark import load_inputs, minify_json_bytes

    tricky = b'{ "a b" : [ 1 , "x\\" \\\\ ", "\\n" ,\n\t"\xe4\xb8\xad \\u4e2d" ] }'
    assert minify_json_bytes(tricky) == b'{"a b":[1,"x\\" \\\\ ","\\n","\xe4\xb8\xad \\u4e2d"]}'

    files_dir = pathlib.Path(__file__).resolve().parent / "_files"
    files = sorted(files_dir.glob("*.json"))
    if not files:
        print("skip compact twins: bundled _files not present")
        return
    inputs = load_inputs(files)
    by_name = {i.name: i for i in inputs}
    for i in inputs:
        if not i.loads_only:
            continue
        src = by_name[i.compact_of]
        assert json.loads(i.raw_bytes) == json.loads(src.raw_bytes), i.name
        assert len(i.raw_bytes) < len(src.raw_bytes), i.name
    names = [i.name for i in inputs]
    assert "twitter.min.json" in names and by_name["twitter.json"].has_twin
    assert names.index("twitter.min.json") == names.index("twitter.json") + 1
    # An input that is already compact (under 1% whitespace) gets no twin.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        dense = pathlib.Path(tmp) / "dense.json"
        dense.write_bytes(b'{"k":"' + b"v" * 200 + b'", "n":1}')
        (only,) = load_inputs([dense])
        assert not only.has_twin
    print("ok  compact twins only drop whitespace")


def check_call_path():
    """The C loop must make exactly the call user code makes -- keyword
    arguments included, then the decode/encode -- and no library may be
    wrapped in a Python frame that another library skips."""
    from .benchmark import BenchmarkFunction, Locality, _get_benchmark_defs

    seen = []

    class Out:
        def decode(self, encoding):
            seen.append(("decode", encoding))
            return "s"

    def func(obj, **kwargs):
        seen.append(("call", obj, kwargs))
        return Out()

    bf = BenchmarkFunction("stub", func, then="decode", indent=2, option=5)
    internal.benchmark_run(**bf.run_kwargs(), ring=[7], repeat=2, warmup=1)
    assert seen == [("call", 7, {"indent": 2, "option": 5}), ("decode", "utf-8")] * 3, seen
    for group in _get_benchmark_defs([Locality.HOT]):
        for f in group.functions:
            name = getattr(f.func, "__name__", "")
            assert name != "<lambda>", (group.group_name, f.library_name)
    print("ok  every library is called directly, with its own arguments")


def check_summary_counts_documents_once():
    """A summary panel gets one point per document, whatever its number of
    groups, and groups kept out of the summary -- such as the loads of a
    document whose compact twin stands in for it -- do not count."""
    from .report import _collect_ratios
    from .result_types import (
        BenchmarkFinalResult,
        BenchmarkResultPerFile,
        BenchmarkResultPerFileTargetLib,
    )

    def doc(groups):
        f = BenchmarkResultPerFile()
        for group_name, (ratio, in_summary) in groups.items():
            t = f[group_name]
            t.in_summary = in_summary
            t.libraries = ["json", "ssrjson"]
            for lib, r in (("json", 1.0), ("ssrjson", ratio)):
                lib_result = BenchmarkResultPerFileTargetLib()
                lib_result.ratio = r
                t[lib] = lib_result
        return f

    result = BenchmarkFinalResult()
    result.filenames = ["a.json", "a.min.json", "b.json"]
    result.results = {
        "loads|hot": {
            # Marked by the benchmark because a.min.json stands in for it.
            "a.json": doc({"loads str": (100.0, False), "loads bytes": (100.0, False)}),
            "a.min.json": doc({"loads str": (2.0, True), "loads bytes": (8.0, True)}),
            # The excluded group must not move b's point.
            "b.json": doc({"loads str": (3.0, True), "loads bytes": (50.0, False)}),
        }
    }
    cats = {"loads|hot": ["loads str", "loads bytes"]}
    libs, ratios = _collect_ratios(result, cats, "loads|hot")
    assert libs == ["ssrjson"], libs
    # a is represented by its twin, as geomean(2, 8) = 4; b by its one group.
    assert sorted(round(r, 9) for r in ratios[0]) == [3.0, 4.0], ratios
    print("ok  summary: one point per document, compact twin stands in for loads")


def check_copy_json():
    """The fresh-object factory must return an equal value built from new
    containers and new strings, none of which carries a UTF-8 cache."""
    from .benchmark import _lib_available, ensure_utf8_cache

    src = {"中文键": ["值", {"k": "ascii", "n": [1, 2.5, None, True]}], "a": "é" * 40}
    if _lib_available("orjson"):
        ensure_utf8_cache(src)  # gives every non-ASCII str in src a UTF-8 cache
    out = internal.copy_json(src)
    assert out == src and out is not src
    assert out["中文键"] is not src["中文键"] and out["中文键"][1] is not src["中文键"][1]
    for key, value in out.items():
        assert key is not next(k for k in src if k == key)
        if isinstance(value, str):
            assert not internal.pyunicode_has_utf8_cache(value), key
    assert not internal.pyunicode_has_utf8_cache(next(iter(out)))
    assert list(out) == list(src), "key order must survive the copy"
    print("ok  copy_json: equal value, new containers and strings, no UTF-8 cache")


def check_build_info():
    """The compiler comes from the extension's own ELF .comment section."""
    import sys

    from .benchmark import _elf_build_notes, _get_ssrjson_build

    if sys.platform.startswith("linux"):
        assert _elf_build_notes(internal.__file__), internal.__file__
    assert _elf_build_notes(__file__) == []
    print(f"ok  ssrjson build: {_get_ssrjson_build()}")


def main() -> int:
    check_copy_json()
    check_build_info()
    check_compact_twins()
    check_call_path()
    check_summary_counts_documents_once()
    check_ring_rotation()
    check_no_factory_means_no_allocation()
    check_hot_holds_one_copy()
    check_ring_size_rules()
    check_ascii_predicates()
    check_interleaving()
    check_statistics()
    check_summary_groups_are_shipping_config()
    check_cache_write_is_on_by_default()
    check_baseline_is_compact()
    print("\nall self-checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
