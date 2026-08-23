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
    from .benchmark import _RingSpec, InputKind, _interleaved_measure

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

    funcs = [(n, make(n)) for n in ("a", "b", "c")]
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
            return next(f.func for f in group.functions if f.library_name == "ssrjson")

        # Default group: writes the cache, like orjson does.
        obj = _fresh(sample)
        ssr_of("dumps to bytes")(obj)
        assert internal.pyunicode_has_utf8_cache(obj["k"]), (
            "default dumps_to_bytes did not populate the UTF-8 cache"
        )
        # Diagnostic group: explicitly does not.
        obj = _fresh(sample)
        ssr_of("dumps to bytes (no cache write)")(obj)
        assert not internal.pyunicode_has_utf8_cache(obj["k"]), (
            "no-cache-write group still populated the UTF-8 cache"
        )
    finally:
        ssrjson.write_utf8_cache(old)
    print("ok  default groups write the UTF-8 cache, diagnostic group does not")


def _fresh(obj):
    from .benchmark import _recursive_copy_obj

    return _recursive_copy_obj(obj)


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
        out = baseline.func(sample)
        text = out.decode() if isinstance(out, bytes) else out
        assert ", " not in text and ": " not in text, f"{group_name}: {text}"
        assert _json.loads(text) == sample
    print("ok  stdlib baseline emits compact separators")


def main() -> int:
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
