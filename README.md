# ssrJSON-benchmark

<div align="center">

[![PyPI - Version](https://img.shields.io/pypi/v/ssrjson-benchmark)](https://pypi.org/project/ssrjson-benchmark/) [![PyPI - Wheel](https://img.shields.io/pypi/wheel/ssrjson-benchmark)](https://pypi.org/project/ssrjson-benchmark/)

The [ssrJSON](https://github.com/Antares0982/ssrjson) benchmark repository.

</div>

## Benchmark Results

The benchmark results can be found in [website results](https://ikuyo.dev/ssrJSON-benchmark/) or [GitHub results](https://github.com/Nambers/ssrJSON-benchmark/tree/main/results). Contributing your benchmark result is welcomed.

Quick jump for

* [x86-64-v4, AVX512](https://github.com/Nambers/ssrJSON-benchmark/tree/main/results/AVX512)
* [x86-64-v3, AVX2](https://github.com/Nambers/ssrJSON-benchmark/tree/main/results/AVX2)
* [x86-64-v2, SSE4.2](https://github.com/Nambers/ssrJSON-benchmark/tree/main/results/SSE4.2)
* [aarch64, NEON](https://github.com/Nambers/ssrJSON-benchmark/tree/main/results/NEON)

## Usage

```bash
pip install ssrjson-benchmark[all]  # Install all dependencies for benchmarking and printing PDF / Markdown
# pip install ssrjson-benchmark[benchmark]  # Only install third-party JSON libraries for benchmarking
# pip install ssrjson-benchmark[visual]  # Only install dependencies for generating PDF / Markdown report
# pip install ssrjson-benchmark  # Clean install without any dependency
python -m ssrjson_benchmark full -h # Run benchmark + generate PDF report in one command
# python -m ssrjson_benchmark benchmark -h  # Run benchmark and generate JSON benchmark result
# python -m ssrjson_benchmark print -h # Generate report from previously saved JSON benchmark result
```

To contribute a result, run the defaults in five fresh processes (about 20 minutes on a Ryzen 9 9900X):

```bash
python -m ssrjson_benchmark full --runs 5
```

For a quick local check, `python -m ssrjson_benchmark full --locality hot` takes about two minutes.

## Notes

* Libraries benchmarked are json, [ujson](https://github.com/ultrajson/ultrajson), [pydantic](https://github.com/pydantic/pydantic), [msgspec](https://github.com/jcrist/msgspec), [orjson](https://github.com/ijl/orjson) and [ssrJSON](https://github.com/Antares0982/ssrjson); where a library only produces the other output type, a single `decode("utf-8")` / `encode("utf-8")` is appended rather than dropping it from the comparison.
* The UTF-8 cache is a `dumps_to_bytes` concern only, since `dumps_to_str` and `loads` never encode a `str` to UTF-8.
* The cache is primed with `orjson.dumps`, and cache-related groups are skipped for all-ASCII inputs because ASCII `PyUnicode` carries no separate UTF-8 buffer to invalidate.
* A fresh input object is built per measured call only when the relevant data is non-ASCII, and "relevant" differs per test: `loads str` depends on the source document, `dumps` on the parsed object (`github.json` is ASCII text whose `\uXXXX` escapes decode to non-ASCII strings).
* msgspec is absent from the indented groups because it has no indent option, and `msgspec.json.format` re-parses finished output, which is a different operation.
* `hot` keeps one live copy so the object is in cache when the call starts, while `cold` rotates a ring sized at a multiple of the last level cache (`--cold-working-set-multiple`) so every measured object has been evicted by the same number of intervening copies.
* Fresh objects in a cold ring are used once each, and the whole ring is freed and rebuilt when it runs out instead of refilling a slot after every call, because a copy built while its neighbours are alive reuses pymalloc's free lists in reverse order: a cold github.json dump took about 13µs on a copy from a whole-ring build and 30-40µs on a refilled one, so call order alone made one library look twice as fast as another.
* The first few calls after a rebuild run up to 1.6x slow (ctm.json: orjson's first four, ssrJSON's first two), so every rebuild is followed by 8 unmeasured calls, without which ctm.json measured 44% slow on a 2x ring that had to be rebuilt mid-chunk.
* Hot frees its copy before building the next one, for the same reason: building while the old copy was alive made hot ctm.json dumps drift from 300µs to 900µs over a run, penalising whichever library ran later, while freeing first holds them at 230-270µs.
* The cold ring defaults to 8x the LLC because there every bundled file's cold dumps were within 2.5% of a 16x ring, while smaller rings were off by up to 40% (instruments.json at 1x, twitter.json by 14-16% at 1x and 2x), whereas loads read their input sequentially and the prefetcher keeps them within a few percent of hot at any size.
* `ssrJSON build` in the header records whether the extension is PGO-optimized (as reported by ssrJSON 0.0.20 and later), how it was installed (package index, nix, a local source tree, or a git commit), the compiler named in the binary, and `assertions on` for a build without `NDEBUG`, because a `pip install` from source reports the same version as the release wheel yet is built without PGO unless `SSRJSON_ENABLE_PGO=1` is set, and a Debug build is `-O0`.
* An LLC source of `fallback` in the header means detection failed and cold results are not comparable across machines, so set `--llc-bytes` explicitly.
* `--bin-process-megabytes` is deprecated and ignored, because it used to control how cold the measured object was through what looked like a memory-only knob.
* `--min-iterations` exists because an equal-bytes budget alone leaves the largest inputs with only a few dozen samples.
* The defaults are 100 iterations and 0.05 GiB per test case because beyond ~60 samples the median's interval is a fraction of a percent hot (about a percent cold), far below the run-to-run spread that `--runs` exposes, so `--min-iterations 200 --process-gigabytes 0.1` restores the previous, ~3x slower budget.
* Contributed results use `--runs 5` rather than a larger `--process-gigabytes`, because run-to-run spread (median 1.7%, p90 3.3% across three processes) is about eight times the within-process interval, and only more processes reduce it.
* Pinning is on by default because an unpinned process on a hybrid CPU can land on an efficiency core and invert which library wins, not merely add noise.
* `min`, `median`, `mean` and `p95` are all recorded in the JSON, since mean and min genuinely disagree about the winner when two libraries have different noise profiles.
* Error bars are the distribution-free confidence interval of the charted statistic, or the run-to-run range under `--runs > 1`, not the standard deviation of a single iteration, and the `Runs` line of the header says which.
* Under `--runs > 1` each process contributes one ratio per library, its json time over that library's time measured alongside it, and the bar is the median of those with their range, because pairing json's fastest process with another library's slowest counts a whole-process slowdown twice and puts a bar on json's own 1x.
* Libraries are interleaved across `--rounds` chunks in rotating order, so slow drift such as thermal throttling hits every library equally instead of penalising whichever runs last.
* `--runs N` repeats the whole benchmark in N fresh processes, which is the only way to see binary and heap layout effects, and those are the size of the gap between the closest libraries.
* Every library's output is verified to round-trip before it is timed, so a mismatch aborts unless `--allow-output-mismatch` is passed, which hatches and marks the offending bars instead.
* The stdlib `json` baseline uses `separators=(",", ":")`, because its default separators emit ~7% more bytes and inflated every ratio measured against it.
* Only the call itself is timed: building the input and freeing the result both sit outside the measured window.
* The C timing loop makes each library's call itself, with that library's own arguments, because wrapping only the libraries that need an argument or a conversion in a Python lambda charged them ~10ns that the others skipped, a sizeable share of a small call.
* The clock is `CLOCK_MONOTONIC` on Linux, `CLOCK_UPTIME_RAW` on macOS, whose `CLOCK_MONOTONIC` only has microsecond resolution, and `QueryPerformanceCounter` on Windows.
* A call shorter than 100 clock steps (the `resolution` in the header) is timed in batches per sample, one batch size for every library in a group, because the median of samples rounded to a coarse clock snaps to a whole step: on a 1µs clock a 750ns and a 960ns call come out tied.
* A fresh process keeps its core busy for half a second before measuring, because an idle core can take tens of milliseconds to reach full clock (24% slow for the first 75ms on a Ryzen 9 9900X under amd-pstate-epp).
* The garbage collector is off while measuring, which slightly understates the fast libraries: with it on, json loads of twitter.json took 4.7% longer while orjson and ssrJSON were unaffected.
* The summary has one panel per operation, since ratios to json for loads and dumps sit on very different scales and a pooled average mostly reflects how many points each operation contributes.
* Each document is one value in a summary box, the geometric mean over that operation's groups, so a non-ASCII file does not outweigh an ASCII one by having the extra cached-dumps group, and the whiskers reach the slowest and fastest document rather than hiding any as outliers.
* In the loads panel a pretty-printed document counts through its `.min.json` twin, because compact JSON is what loads meets in practice, and both remain charted.
* Every library in the summary runs in the configuration its wheel ships with, including ssrJSON's UTF-8 cache writing, so `dumps to bytes (no cache write)` is charted per file but left out of the summary: it forces ssrJSON out of its default to isolate the cache-write cost.
* In the hot dumps-to-bytes section, a non-ASCII file's chart shows the UTF-8 cache break-even, how many dumps of one object it takes before writing the cache pays for itself (`ssrjson.write_utf8_cache(False)` is faster below it), and files whose cache-write cost is below the measurement's resolution get no break-even, which under `--runs > 1` means the cost has to show in every process, with the break-even the median of each process's own, since a process that is slow overall moves all three groups together.
* Using your own dataset needs no code changes, just point `-d` at a directory of `*.json` files and everything per-file is derived from the data.
* Each input that is at least 1% insignificant whitespace is also measured as a `<name>.min.json` twin in the same run, `loads` only, because the bundled files are pretty-printed while most JSON on the wire is compact (ctm.json is 70% whitespace, twitter.json 38%).
* The twin is made by deleting whitespace outside string literals rather than by re-serializing, which would also rewrite number spellings and `\uXXXX` escapes, and `--no-compact` turns it off.
* Data the libraries disagree about aborts the run, which in practice means integers wider than 64 bits (rejected by orjson and ssrJSON) and `NaN`/`Infinity` (rejected by orjson and msgspec).
* Peak memory is a single cold ring, `--cold-working-set-multiple` times the LLC, because every library walks the same ring with one running call count.
* ssrJSON, msgspec, orjson and pydantic_core keep short strings cached across calls, so parsing one document over and over is their best case: when every call brought dict keys never seen before, loads of a 100KB slice of twitter.json slowed by 22% (ssrJSON), 18% (msgspec), 10-14% (orjson, pydantic_core), 7% (ujson) and 2% (json), so the results fit a service that parses one schema repeatedly and flatter one whose keys keep changing.
* songci.json is the real-world CJK document, kept as one all-UCS2 case because none of its strings has a character outside the BMP: 1,000 Song ci poems, unmodified `宋词/ci.song.0.json` from [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry) (MIT License, Copyright (c) 2016 JackeyGao).
