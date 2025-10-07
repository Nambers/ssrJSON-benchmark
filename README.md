# ssrJSON-benchmark

<div align="center">

[![PyPI - Version](https://img.shields.io/pypi/v/ssrjson-benchmark)](https://pypi.org/project/ssrjson-benchmark/) [![PyPI - Wheel](https://img.shields.io/pypi/wheel/ssrjson-benchmark)](https://pypi.org/project/ssrjson-benchmark/)

The [ssrJSON](https://github.com/Antares0982/ssrjson) benchmark repository.

</div>

## Benchmark Results

The benchmark results can be found in [website results](https://ikuyo.dev/ssrJSON-benchmark/) or [GitHub results](https://github.com/Nambers/ssrJSON-benchmark/tree/main/results). Contributing your benchmark result is welcomed.

Quick jump for

* [x86-64-v2, SSE4.2](https://github.com/Nambers/ssrJSON-benchmark/tree/main/results/SSE4.2)
* [x86-64-v3, AVX2](https://github.com/Nambers/ssrJSON-benchmark/tree/main/results/AVX2)
* [x86-64-v4, AVX512](https://github.com/Nambers/ssrJSON-benchmark/tree/main/results/AVX512)

## Usage

```bash
# you may need to install `svglib`, `reportlab` and `py-cpuinfo` as well
pip install ssrjson-benchmark
python -m ssrjson_benchmark
```

## Benchmark options

* `-m` output in Markdown instead of PDF.
* `-f <json_path>` used exists benchmark json result.
* `--process-bytes <bytes_num>` Total process bytes per test, default 1e8.

## Notes

* This repository conducts benchmarking using json, [orjson](https://github.com/ijl/orjson), [ujson](https://github.com/ultrajson/ultrajson), and [ssrJSON](https://github.com/Antares0982/ssrjson). The `dumps` benchmark produces str objects, comparing three operations: `json.dumps`, `orjson.dumps` followed by decode, and `ssrjson.dumps`. The `dumps_to_bytes` benchmark produces bytes objects, comparing three functions: `json.dumps` followed by encode, `orjson.dumps`, and `ssrjson.dumps_to_bytes`.
* When orjson handles non-ASCII strings, if the cache of the `PyUnicodeObject`’s UTF-8 representation does not exist, it invokes the `PyUnicode_AsUTF8AndSize` function to obtain the UTF-8 encoding. This function then caches the UTF-8 representation within the `PyUnicodeObject`. If the same `PyUnicodeObject` undergoes repeated encode-decode operations, subsequent calls after the initial one will execute more quickly due to this caching. However, in real-world production scenarios, it is uncommon to perform JSON encode-decode repeatedly on the exact same string object; even identical strings are unlikely to be the same object instance. To achieve benchmark results that better reflect practical use cases, we employ `ssrjson.run_unicode_accumulate_benchmark` and `_benchmark_invalidate_dump_cache` functions, which ensure that new `PyUnicodeObject`s are different for each input every time. (ref: [orjson#586](https://github.com/ijl/orjson/issues/586))
* The performance of JSON encoding is primarily constrained by the speed of writing to the buffer, whereas decoding performance is mainly limited by the frequent invocation of CPython interfaces for object creation. During decoding, both ssrJSON and orjson employ short key caching to reduce the number of object creations, and this caching mechanism is global in both cases. As a result, decoding benchmark tests may not accurately reflect the conditions encountered in real-world production environments.
* The files simple_object.json and simple_object_zh.json do not represent real-world data; they are solely used to compare the performance of the fast path. Therefore, the benchmark results should not be interpreted as indicative of actual performance.
