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

## Notes

* Libraries benchmarked are json, [ujson](https://github.com/ultrajson/ultrajson), [pydantic](https://github.com/pydantic/pydantic), [msgspec](https://github.com/jcrist/msgspec), [orjson](https://github.com/ijl/orjson) and [ssrJSON](https://github.com/Antares0982/ssrjson); where a library only produces the other output type, a single `decode("utf-8")` / `encode("utf-8")` is appended rather than dropping it from the comparison.
* The UTF-8 cache is a `dumps_to_bytes` concern only, since `dumps_to_str` and `loads` never encode a `str` to UTF-8.
* The cache is primed with `orjson.dumps`, and cache-related groups are skipped for all-ASCII inputs because ASCII `PyUnicode` carries no separate UTF-8 buffer to invalidate.
* A fresh input object is built per measured call only when the relevant data is non-ASCII, and "relevant" differs per test: `loads str` depends on the source document, `dumps` on the parsed object (`github.json` is ASCII text whose `\uXXXX` escapes decode to non-ASCII strings).
* `hot` keeps one live copy so the object is in cache when the call starts, while `cold` rotates a ring sized at a multiple of the last level cache (`--cold-working-set-multiple`) so every measured object has been evicted by the same number of intervening copies.
* An LLC source of `fallback` in the header means detection failed and cold results are not comparable across machines, so set `--llc-bytes` explicitly.
* `--bin-process-megabytes` is deprecated and ignored, because it used to control how cold the measured object was through what looked like a memory-only knob.
* `--min-iterations` exists because an equal-bytes budget alone leaves the largest inputs with only a few dozen samples.
* Pinning is on by default because an unpinned process on a hybrid CPU can land on an efficiency core and invert which library wins, not merely add noise.
* `min`, `median`, `mean` and `p95` are all recorded in the JSON, since mean and min genuinely disagree about the winner when two libraries have different noise profiles.
* Error bars are the distribution-free confidence interval of the charted statistic, or the run-to-run range under `--runs > 1`, not the standard deviation of a single iteration.
* Libraries are interleaved across `--rounds` chunks in rotating order, so slow drift such as thermal throttling hits every library equally instead of penalising whichever runs last.
* `--runs N` repeats the whole benchmark in N fresh processes, which is the only way to see binary and heap layout effects, and those are the size of the gap between the closest libraries.
* Every library's output is verified to round-trip before it is timed, so a mismatch aborts unless `--allow-output-mismatch` is passed, which hatches and marks the offending bars instead.
* The stdlib `json` baseline uses `separators=(",", ":")`, because its default separators emit ~7% more bytes and inflated every ratio measured against it.
* Only the call itself is timed: building the input and freeing the result both sit outside the measured window.
* Using your own dataset needs no code changes, just point `-d` at a directory of `*.json` files and everything per-file is derived from the data.
* Data the libraries disagree about aborts the run, which in practice means integers wider than 64 bits (rejected by orjson and ssrJSON) and `NaN`/`Infinity` (rejected by orjson and msgspec).
* Peak memory is `library_count x ring_size x one_copy` because interleaving keeps every library's ring alive at once, and `--rounds 1` trades the interleaving protection for one ring at a time (measured 400MiB -> 67MiB on twitter.json).
* Both ssrJSON and orjson use a global short-key cache while decoding, so `loads` results may not reflect production conditions.
* simple_object.json and simple_object_zh.json are fast-path probes rather than real-world data.
