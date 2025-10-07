rm -rf build
mkdir build
cmake -B build .
cmake --build build
cp build/_ssrjson_benchmark.so src/ssrjson_benchmark/_ssrjson_benchmark.so
