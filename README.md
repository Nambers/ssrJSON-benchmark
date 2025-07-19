# ssrJSON-benchmark

The [ssrJSON](https://github.com/Antares0982/ssrjson) benchmark repository.

## Benchmark Results

The benchmark results can be found in [results](results). Contributing your benchmark result is welcomed.

Quick jump for

* [x86-64-v2, SSE4.2](results/SSE4.2)
* [x86-64-v3, AVX2](results/AVX2)
* [x86-64-v4, AVX512](results/AVX512)

## Usage

To generate a benchmark report, you need to install `ssrJSON` either by fetched [PyPi](https://pypi.org/project/ssrjson/) or built from [source](https://github.com/Antares0982/ssrjson), and toolkit(`ssrjson_benchmark`) from this repo by:

```bash
python -m build
pip install dist/*.whl
```

Then run the benchmark script:

```bash
python benchmark.py
```

## Benchmark options

* `-m` output in Markdown instead of PDF.
* `-f <json_path>` used exists benchmark json result.
* `--process-bytes <bytes_num>` Total process bytes per test, default 1e8.

## Notes

* This repository conducts benchmarking using json, orjson, and ssrJSON. The `dumps` benchmark produces str objects, comparing three operations: `json.dumps`, `orjson.dumps` followed by decode, and `ssrjson.dumps`. The `dumps_to_bytes` benchmark produces bytes objects, comparing three functions: `json.dumps` followed by encode, `orjson.dumps`, and `ssrjson.dumps_to_bytes`.
* The ssrJSON built with the `BUILD_BENCHMARK` option includes several additional C functions specifically designed for executing benchmarks. These functions utilize high-precision timing APIs, and within the loop, only the time spent on the actual `PyObject_Call` invocations is measured.
* When orjson handles non-ASCII strings, if the cache of the `PyUnicodeObject`’s UTF-8 representation does not exist, it invokes the `PyUnicode_AsUTF8AndSize` function to obtain the UTF-8 encoding. This function then caches the UTF-8 representation within the `PyUnicodeObject`. If the same `PyUnicodeObject` undergoes repeated encode-decode operations, subsequent calls after the initial one will execute more quickly due to this caching. However, in real-world production scenarios, it is uncommon to perform JSON encode-decode repeatedly on the exact same string object; even identical strings are unlikely to be the same object instance. To achieve benchmark results that better reflect practical use cases, we employ `ssrjson.run_unicode_accumulate_benchmark` and `benchmark_invalidate_dump_cache` functions, which ensure that new `PyUnicodeObject`s are different for each input every time.

* The performance of JSON encoding is primarily constrained by the speed of writing to the buffer, whereas decoding performance is mainly limited by the frequent invocation of CPython interfaces for object creation. During decoding, both ssrJSON and orjson employ short key caching to reduce the number of object creations, and this caching mechanism is global in both cases. As a result, decoding benchmark tests may not accurately reflect the conditions encountered in real-world production environments.

* The files simple_object.json and simple_object_zh.json do not represent real-world data; they are solely used to compare the performance of the fast path. Therefore, the benchmark results should not be interpreted as indicative of actual performance.
