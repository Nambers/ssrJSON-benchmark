# ssrJSON-benchmark

The [ssrJSON](https://github.com/Antares0982/ssrjson) benchmark repository.

## Usage

ssrJSON is required to be built with `BUILD_BENCHMARK` option on

```bash
mkdir build
cd build
cmake -DBUILD_BENCHMARK=ON ..
cmake --build .
```

Running `benchmark.py` will generate a report

```bash
python benchmark.py
```

## Notes

* The ssrJSON built with the `BUILD_BENCHMARK` option includes several additional C functions specifically designed for executing benchmarks. These functions utilize high-precision timing APIs, and within the loop, only the time spent on the actual `PyObject_Call` invocations is measured.
* When orjson handles non-ASCII strings, if the cache of the `PyUnicodeObject`’s UTF-8 representation does not exist, it invokes the `PyUnicode_AsUTF8AndSize` function to obtain the UTF-8 encoding. This function then caches the UTF-8 representation within the `PyUnicodeObject`. If the same `PyUnicodeObject` undergoes repeated encode-decode operations, subsequent calls after the initial one will execute more quickly due to this caching. However, in real-world production scenarios, it is uncommon to perform JSON encode-decode repeatedly on the exact same string object; even identical strings are unlikely to be the same object instance. To achieve benchmark results that better reflect practical use cases, we employ `ssrjson.run_unicode_accumulate_benchmark` and `benchmark_invalidate_dump_cache` functions, which ensure that new `PyUnicodeObject`s are different for each input every time.

* The performance of JSON encoding is primarily constrained by the speed of writing to the buffer, whereas decoding performance is mainly limited by the frequent invocation of CPython interfaces for object creation. During decoding, both ssrJSON and orjson employ short key caching to reduce the number of object creations, and this caching mechanism is global in both cases. As a result, decoding benchmark tests may not accurately reflect the conditions encountered in real-world production environments.

