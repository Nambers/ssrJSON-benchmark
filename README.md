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
python -m ssrjson_benchmark full  # Run benchmark + generate PDF report in one command
# python -m ssrjson_benchmark benchmark  # Run benchmark and generate JSON benchmark result
# python -m ssrjson_benchmark print  # Generate report from previously saved JSON benchmark result
```

## Notes

* This repository conducts benchmarking using json, [ujson](https://github.com/ultrajson/ultrajson),[pydantic](https://github.com/pydantic/pydantic), [msgspec](https://github.com/jcrist/msgspec), [orjson](https://github.com/ijl/orjson), and [ssrJSON](https://github.com/Antares0982/ssrjson). The benchmark for `dumps_to_str` aims to produce a `str` object. If a JSON library's dumps-related interface only outputs a `bytes` object, it will be substituted with dumps followed by a single `decode("utf-8")` operation. Similarly, for the `dumps_to_bytes` test, if the JSON library's dumps-related interface only outputs a `str` object, it will be replaced with dumps followed by a single `encode("utf-8")` operation.
* To ensure the accuracy of benchmark results, this repository differentiates between scenarios with and without UTF-8 caches when testing `dumps_to_bytes`. For `dumps_to_str` and `loads`, since these methods are unrelated to encoding `str` objects to UTF-8, the data sources do not involve any UTF-8 cache, and no distinction is made in their tests.
  * Cache writing of ssrJSON is disabled globally when running benchmark.
  * We use `orjson.dumps` to create UTF-8 cache for all benchmark targets.
  * Test with UTF-8 cache is skipped when the whole JSON object is ASCII.
* The performance of JSON encoding is primarily constrained by the speed of writing to the buffer, whereas decoding performance is mainly limited by the frequent invocation of CPython interfaces for object creation. During decoding, both ssrJSON and orjson employ short key caching to reduce the number of object creations, and this caching mechanism is global in both libraries. As a result, decoding benchmark tests may not accurately reflect the conditions encountered in real-world production environments.
* The files simple_object.json and simple_object_zh.json do not represent real-world data; they are used to compare the performance of the fast path. Therefore, the benchmark results from these test cases should not be interpreted as indicative of actual performance in production environment.
