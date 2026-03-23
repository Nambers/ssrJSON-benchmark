/*==============================================================================
 Copyright (c) 2025 Antares <antares0982@gmail.com>

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 *============================================================================*/

#include <Python.h>
#include <stdbool.h>

/** compiler builtin check (since gcc 10.0, clang 2.6, icc 2021) */
#ifndef has_builtin
#    ifdef __has_builtin
#        define has_builtin(x) __has_builtin(x)
#    else
#        define has_builtin(x) 0
#    endif
#endif

/** unlikely for compiler */
#ifndef unlikely
#    if has_builtin(__builtin_expect)
#        define unlikely(expr) __builtin_expect(!!(expr), 0)
#    else
#        define unlikely(expr) (expr)
#    endif
#endif

typedef unsigned long long usize;
#if defined(_WIN32) || defined(_WIN64)
#    include <windows.h>

inline LARGE_INTEGER _get_frequency(void) {
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    return frequency;
}

usize perf_counter(void) {
    static LARGE_INTEGER *frequency = NULL;
    if (!frequency) {
        frequency = (LARGE_INTEGER *)malloc(sizeof(LARGE_INTEGER));
        if (!frequency) {
            return 0;
        }
        *frequency = _get_frequency();
    }
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (usize)((counter.QuadPart * 1000000000LL) / frequency->QuadPart);
}

#else
#    include <time.h>

usize perf_counter(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (usize)ts.tv_sec * 1000000000LL + (usize)ts.tv_nsec;
}

#endif

typedef struct UsizeArray {
    usize *data;
    usize size;
} UsizeArray;

typedef struct PyObjectArray {
    PyObject **data;
    usize size;
} PyObjectArray;

static UsizeArray _static_times_buf = {NULL, 0};
static PyObjectArray _static_pyobj_buf = {NULL, 0};

/**
 * Ensure the static UsizeArray has at least `needed` capacity.
 * Returns 0 on success, -1 on failure (sets PyErr_NoMemory).
 */
static int _ensure_usize_capacity(UsizeArray *arr, usize needed) {
    if (arr->size >= needed) return 0;
    usize *new_data = (usize *)PyMem_Realloc(arr->data, needed * sizeof(usize));
    if (unlikely(!new_data)) {
        PyErr_NoMemory();
        return -1;
    }
    arr->data = new_data;
    arr->size = needed;
    return 0;
}

/**
 * Ensure the static PyObjectArray has at least `needed` capacity.
 * Returns 0 on success, -1 on failure (sets PyErr_NoMemory).
 */
static int _ensure_pyobj_capacity(PyObjectArray *arr, usize needed) {
    if (arr->size >= needed) return 0;
    PyObject **new_data = (PyObject **)PyMem_Realloc(arr->data, needed * sizeof(PyObject *));
    if (unlikely(!new_data)) {
        PyErr_NoMemory();
        return -1;
    }
    arr->data = new_data;
    arr->size = needed;
    return 0;
}

typedef struct PyUnicodeCopyInfo {
    Py_ssize_t size;
    int kind;
    Py_UCS4 max_char;
    bool valid;
} PyUnicodeCopyInfo;

PyObject *_copy_unicode(PyObject *unicode, PyUnicodeCopyInfo *unicode_copy_info) {
    if (!unicode_copy_info->valid) {
        // create copy of unicode object.
        int kind = PyUnicode_KIND(unicode);
        Py_UCS4 max_char;
        if (kind == 4) {
            max_char = 0x10ffff;
        } else if (kind == 2) {
            max_char = 0xffff;
        } else if (PyUnicode_IS_ASCII(unicode)) {
            max_char = 0x7f;
        } else {
            max_char = 0xff;
        }
        //
        unicode_copy_info->size = PyUnicode_GET_LENGTH(unicode);
        unicode_copy_info->kind = kind;
        unicode_copy_info->max_char = max_char;
        unicode_copy_info->valid = true;
    }

    PyObject *unicode_copy = PyUnicode_New(unicode_copy_info->size, unicode_copy_info->max_char);
    if (!unicode_copy) return NULL;
    memcpy(PyUnicode_DATA(unicode_copy), PyUnicode_DATA(unicode),
           unicode_copy_info->size * unicode_copy_info->kind);
    return unicode_copy;
}

PyObject *_parse_additional_args(PyObject *additional_args) {
    Py_ssize_t new_args_count = 1;
    if (additional_args) {
        new_args_count += PyTuple_GET_SIZE(additional_args);
    }
    PyObject *new_args = PyTuple_New(new_args_count);
    if (!new_args) {
        return NULL;
    }
    if (additional_args) {
        for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(additional_args); i++) {
            PyObject *item = PyTuple_GET_ITEM(additional_args, i);
            Py_INCREF(item);
            PyTuple_SET_ITEM(new_args, i + 1, item);
        }
    }
    return new_args;
}

PyObject *copy_unicode_list_invalidate_cache(PyObject *self, PyObject *args, PyObject *kwargs) {
    static const char *kwlist[] = {"s", "size", NULL};
    PyObject *s;
    usize size;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OK", (char **)kwlist, &s, &size)) {
        PyErr_SetString(PyExc_TypeError, "Invalid argument");
        return NULL;
    }
    if (!PyUnicode_CheckExact(s)) {
        PyErr_SetString(PyExc_TypeError, "First argument must be str, not other types or subclass of str");
        return NULL;
    }
    PyObject *ret = PyList_New(size);
    if (!ret) {
        return NULL;
    }
    PyUnicodeCopyInfo unicode_copy_info;
    unicode_copy_info.valid = false;
    for (usize i = 0; i < size; i++) {
        PyObject *s_copy = _copy_unicode(s, &unicode_copy_info);
        if (!s_copy) {
            Py_DECREF(ret);
            return NULL;
        }
        PyList_SET_ITEM(ret, i, s_copy);
    }
    return ret;
}

/**
 * Build a Python tuple (total, [t0, t1, ...]) from a C array of per-iteration times.
 * Does NOT free times_buf (caller manages the buffer lifetime).
 */
static PyObject *_build_times_result(usize *times_buf, usize count, usize total) {
    PyObject *times_list = NULL;
    PyObject *total_obj = NULL;
    PyObject *ret = NULL;

    times_list = PyList_New(count);
    if (unlikely(!times_list)) goto fail;
    for (usize i = 0; i < count; i++) {
        PyObject *val = PyLong_FromUnsignedLongLong(times_buf[i]);
        if (unlikely(!val)) goto fail;
        PyList_SET_ITEM(times_list, i, val);
    }

    total_obj = PyLong_FromUnsignedLongLong(total);
    if (unlikely(!total_obj)) goto fail;
    ret = PyTuple_New(2);
    if (unlikely(!ret)) goto fail;
    PyTuple_SET_ITEM(ret, 0, total_obj);
    PyTuple_SET_ITEM(ret, 1, times_list);
    return ret;

fail:
    Py_XDECREF(times_list);
    Py_XDECREF(total_obj);
    return NULL;
}

PyObject *run_object_accumulate_benchmark(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
    PyObject *callable;
    usize repeat;
    PyObject *call_args;
    static const char *kwlist[] = {"func", "repeat", "args", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OKO", (char **)kwlist,
                                     &callable, &repeat, &call_args)) {
        PyErr_SetString(PyExc_TypeError, "Invalid argument");
        return NULL;
    }
    if (!PyCallable_Check(callable)) {
        PyErr_SetString(PyExc_TypeError, "First argument must be callable");
        return NULL;
    }
    if (!PyTuple_Check(call_args)) {
        PyErr_SetString(PyExc_TypeError, "Third argument must be tuple");
        return NULL;
    }
    if (_ensure_usize_capacity(&_static_times_buf, repeat)) return NULL;
    usize *times_buf = _static_times_buf.data;
    usize total = 0;
    for (usize i = 0; i < repeat; i++) {
        usize start = perf_counter();
        PyObject *result = PyObject_Call(callable, call_args, NULL);
        usize end = perf_counter();
        if (unlikely(!result)) {
            if (!PyErr_Occurred()) {
                PyErr_SetString(PyExc_RuntimeError, "Failed to call callable");
            }
            return NULL;
        }
        Py_DECREF(result);
        usize elapsed = end - start;
        times_buf[i] = elapsed;
        total += elapsed;
    }
    return _build_times_result(times_buf, repeat, total);
}

/**
 * Warmup 1 call, then time `repeat` calls.
 * Returns (total_ns, [per_iter_ns]).
 * Shared implementation for benchmark_bytes_load and benchmark_with_dump_cache.
 */
static PyObject *_benchmark_warmup_repeat(PyObject *callable, usize repeat,
                                          PyObject *call_args) {
    // Warmup
    PyObject *warmup = PyObject_Call(callable, call_args, NULL);
    if (unlikely(!warmup)) {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_RuntimeError, "Failed to call callable");
        return NULL;
    }
    Py_DECREF(warmup);

    if (_ensure_usize_capacity(&_static_times_buf, repeat)) return NULL;
    usize *times_buf = _static_times_buf.data;
    usize total = 0;
    for (usize i = 0; i < repeat; i++) {
        usize start = perf_counter();
        PyObject *result = PyObject_Call(callable, call_args, NULL);
        usize end = perf_counter();
        if (unlikely(!result)) {
            if (!PyErr_Occurred())
                PyErr_SetString(PyExc_RuntimeError, "Failed to call callable");
            return NULL;
        }
        Py_DECREF(result);
        usize elapsed = end - start;
        times_buf[i] = elapsed;
        total += elapsed;
    }
    return _build_times_result(times_buf, repeat, total);
}

/**
 * benchmark_bytes_load(func, repeat, args)
 */
PyObject *benchmark_bytes_load(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
    PyObject *callable;
    usize repeat;
    PyObject *call_args;
    static const char *kwlist[] = {"func", "repeat", "args", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OKO", (char **)kwlist,
                                     &callable, &repeat, &call_args)) {
        PyErr_SetString(PyExc_TypeError, "Invalid argument");
        return NULL;
    }
    if (!PyCallable_Check(callable)) {
        PyErr_SetString(PyExc_TypeError, "First argument must be callable");
        return NULL;
    }
    if (!PyTuple_Check(call_args)) {
        PyErr_SetString(PyExc_TypeError, "Third argument must be tuple");
        return NULL;
    }
    return _benchmark_warmup_repeat(callable, repeat, call_args);
}

/**
 * Warmup with items[0], then time callable(items[1..bin_size]) one by one.
 * Writes elapsed times into times_buf[*buf_idx ..] and accumulates into *total.
 * Returns 0 on success, -1 on failure (Python exception already set).
 * Does NOT PyMem_FREE items — caller is responsible.
 */
static int _benchmark_bin(PyObject *callable, PyObject **items, usize bin_size,
                          usize *times_buf, usize *buf_idx, usize *total) {
    // Warmup with items[0]
    PyObject *warmup_args = PyTuple_New(1);
    if (unlikely(!warmup_args)) return -1;
    Py_INCREF(items[0]);
    PyTuple_SET_ITEM(warmup_args, 0, items[0]);
    PyObject *warmup = PyObject_Call(callable, warmup_args, NULL);
    Py_DECREF(warmup_args);
    if (unlikely(!warmup)) {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_RuntimeError, "Failed to call callable");
        return -1;
    }
    Py_DECREF(warmup);
    // Measure items[1..bin_size]
    for (usize i = 1; i <= bin_size; i++) {
        PyObject *iter_args = PyTuple_New(1);
        if (unlikely(!iter_args)) return -1;
        Py_INCREF(items[i]);
        PyTuple_SET_ITEM(iter_args, 0, items[i]);
        usize start = perf_counter();
        PyObject *result = PyObject_Call(callable, iter_args, NULL);
        usize end = perf_counter();
        Py_DECREF(iter_args);
        if (unlikely(!result)) {
            if (!PyErr_Occurred())
                PyErr_SetString(PyExc_RuntimeError, "Failed to call callable");
            return -1;
        }
        Py_DECREF(result);
        usize elapsed = end - start;
        times_buf[(*buf_idx)++] = elapsed;
        *total += elapsed;
    }
    return 0;
}

/**
 * Helper to Py_DECREF a partially-filled array of PyObjects.
 * Does NOT free the array itself (caller manages the buffer lifetime).
 */
static void _decref_pyobj_array(PyObject **arr, usize count) {
    for (usize i = 0; i < count; i++) Py_DECREF(arr[i]);
}

/**
 * benchmark_unicode_invalidate_cache(func, repeat, times_per_bin, unicode)
 *
 * Binned benchmark: for each bin, create copies of `unicode` without UTF-8 cache,
 * warmup with first copy, then time the rest. Returns (total_ns, [per_iter_ns]).
 */
PyObject *benchmark_unicode_invalidate_cache(PyObject *self, PyObject *args,
                                             PyObject *kwargs) {
    PyObject *callable;
    usize repeat;
    usize times_per_bin;
    PyObject *unicode;
    static const char *kwlist[] = {"func", "repeat", "times_per_bin", "unicode", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OKKO", (char **)kwlist,
                                     &callable, &repeat, &times_per_bin, &unicode)) {
        PyErr_SetString(PyExc_TypeError, "Invalid argument");
        return NULL;
    }
    if (!PyCallable_Check(callable)) {
        PyErr_SetString(PyExc_TypeError, "First argument must be callable");
        return NULL;
    }
    if (!PyUnicode_CheckExact(unicode)) {
        PyErr_SetString(PyExc_TypeError, "unicode argument must be str");
        return NULL;
    }

    usize copies_count = 0;

    if (_ensure_usize_capacity(&_static_times_buf, repeat)) return NULL;
    usize *times_buf = _static_times_buf.data;
    // The pyobj buffer needs at most (times_per_bin + 1) entries
    if (_ensure_pyobj_capacity(&_static_pyobj_buf, times_per_bin + 1)) return NULL;
    PyObject **copies = _static_pyobj_buf.data;

    PyUnicodeCopyInfo unicode_copy_info;
    unicode_copy_info.valid = false;
    usize times_left = repeat;
    usize total = 0;
    usize buf_idx = 0;
    while (times_left != 0) {
        usize cur_bin_size = times_left < times_per_bin ? times_left : times_per_bin;
        times_left -= cur_bin_size;
        usize list_size = cur_bin_size + 1;
        copies_count = 0;
        for (usize i = 0; i < list_size; i++) {
            copies[i] = _copy_unicode(unicode, &unicode_copy_info);
            if (unlikely(!copies[i])) goto fail;
            copies_count = i + 1;
        }
        if (_benchmark_bin(callable, copies, cur_bin_size, times_buf, &buf_idx, &total))
            goto fail;
        _decref_pyobj_array(copies, copies_count);
        copies_count = 0;
    }
    return _build_times_result(times_buf, repeat, total);

fail:
    _decref_pyobj_array(copies, copies_count);
    return NULL;
}

/**
 * benchmark_invalidate_dump_cache(func, repeat, times_per_bin, raw_bytes, loads_func)
 *
 * Binned benchmark: for each bin, call loads_func(raw_bytes) to create fresh objects
 * without UTF-8 cache, warmup with first, then time the rest. Returns (total_ns, [per_iter_ns]).
 */
PyObject *benchmark_invalidate_dump_cache(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
    PyObject *callable;
    usize repeat;
    usize times_per_bin;
    PyObject *raw_bytes;
    PyObject *loads_func;
    static const char *kwlist[] = {"func", "repeat", "times_per_bin", "raw_bytes", "loads_func", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OKKOO", (char **)kwlist,
                                     &callable, &repeat, &times_per_bin, &raw_bytes, &loads_func)) {
        PyErr_SetString(PyExc_TypeError, "Invalid argument");
        return NULL;
    }
    if (!PyCallable_Check(callable)) {
        PyErr_SetString(PyExc_TypeError, "func must be callable");
        return NULL;
    }
    if (!PyCallable_Check(loads_func)) {
        PyErr_SetString(PyExc_TypeError, "loads_func must be callable");
        return NULL;
    }

    PyObject *loads_args = NULL;
    usize objects_count = 0;

    loads_args = PyTuple_New(1);
    if (unlikely(!loads_args)) goto fail;
    Py_INCREF(raw_bytes);
    PyTuple_SET_ITEM(loads_args, 0, raw_bytes);

    if (_ensure_usize_capacity(&_static_times_buf, repeat)) goto fail;
    usize *times_buf = _static_times_buf.data;
    // The pyobj buffer needs at most (times_per_bin + 1) entries
    if (_ensure_pyobj_capacity(&_static_pyobj_buf, times_per_bin + 1)) goto fail;
    PyObject **objects = _static_pyobj_buf.data;

    usize times_left = repeat;
    usize total = 0;
    usize buf_idx = 0;
    while (times_left != 0) {
        usize cur_bin_size = times_left < times_per_bin ? times_left : times_per_bin;
        times_left -= cur_bin_size;
        usize list_size = cur_bin_size + 1;
        objects_count = 0;
        for (usize i = 0; i < list_size; i++) {
            objects[i] = PyObject_Call(loads_func, loads_args, NULL);
            if (unlikely(!objects[i])) goto fail;
            objects_count = i + 1;
        }
        if (_benchmark_bin(callable, objects, cur_bin_size, times_buf, &buf_idx, &total))
            goto fail;
        _decref_pyobj_array(objects, objects_count);
        objects_count = 0;
    }
    Py_DECREF(loads_args);
    return _build_times_result(times_buf, repeat, total);

fail:
    _decref_pyobj_array(_static_pyobj_buf.data, objects_count);
    Py_XDECREF(loads_args);
    return NULL;
}

/**
 * benchmark_with_dump_cache(func, repeat, args)
 */
PyObject *benchmark_with_dump_cache(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
    PyObject *callable;
    usize repeat;
    PyObject *call_args;
    static const char *kwlist[] = {"func", "repeat", "args", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OKO", (char **)kwlist,
                                     &callable, &repeat, &call_args)) {
        PyErr_SetString(PyExc_TypeError, "Invalid argument");
        return NULL;
    }
    if (!PyCallable_Check(callable)) {
        PyErr_SetString(PyExc_TypeError, "First argument must be callable");
        return NULL;
    }
    if (!PyTuple_Check(call_args)) {
        PyErr_SetString(PyExc_TypeError, "Third argument must be tuple");
        return NULL;
    }
    return _benchmark_warmup_repeat(callable, repeat, call_args);
}

PyObject *run_object_benchmark(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
    PyObject *callable;
    PyObject *call_args;
    static const char *kwlist[] = {"func", "args", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", (char **)kwlist,
                                     &callable, &call_args)) {
        PyErr_SetString(PyExc_TypeError, "Invalid argument");
        goto fail;
    }
    if (!PyCallable_Check(callable)) {
        PyErr_SetString(PyExc_TypeError, "First argument must be callable");
        goto fail;
    }
    if (!PyTuple_Check(call_args)) {
        PyErr_SetString(PyExc_TypeError, "Second argument must be tuple");
        goto fail;
    }
    usize start = perf_counter();
    PyObject *result = PyObject_Call(callable, call_args, NULL);
    usize end = perf_counter();
    if (unlikely(!result)) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to call callable");
        }
        goto fail;
    } else {
        Py_DECREF(result);
    }
    usize total;
    total = end - start;
    return PyLong_FromUnsignedLongLong(total);
fail:;
    return NULL;
}

PyObject *inspect_pyunicode(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *unicode;
    PyObject *t1 = NULL, *t2 = NULL, *t3 = NULL, *t4 = NULL;
    static const char *kwlist[] = {"unicode", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char **)kwlist,
                                     &unicode)) {
        goto fail;
    }
    if (!PyUnicode_Check(unicode)) {
        PyErr_SetString(PyExc_TypeError, "First argument must be unicode");
        goto fail;
    }
    PyASCIIObject *u = (PyASCIIObject *)unicode;
    int length = u->length;
    int kind = u->state.kind;
    int ascii = u->state.ascii;
    int interned = u->state.interned;
    t1 = PyLong_FromLong(kind);
    if (!t1)
        goto fail;
    t2 = PyLong_FromLong(kind * length);
    if (!t2)
        goto fail;
    t3 = PyBool_FromLong(ascii);
    if (!t3)
        goto fail;
    t4 = PyBool_FromLong(interned);
    if (!t4)
        goto fail;
    PyObject *ret = PyTuple_New(4);
    if (!ret)
        goto fail;
    PyTuple_SET_ITEM(ret, 0, t1);
    PyTuple_SET_ITEM(ret, 1, t2);
    PyTuple_SET_ITEM(ret, 2, t3);
    PyTuple_SET_ITEM(ret, 3, t4);
    return ret;

fail:;
    Py_XDECREF(t1);
    Py_XDECREF(t2);
    Py_XDECREF(t3);
    Py_XDECREF(t4);
    return NULL;
}

PyObject *pyunicode_has_utf8_cache(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *unicode;
    static const char *kwlist[] = {"unicode", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char **)kwlist,
                                     &unicode)) {
        goto fail;
    }
    if (!PyUnicode_Check(unicode)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be unicode");
        goto fail;
    }
    PyASCIIObject *a = (PyASCIIObject *)unicode;
    if (!a->state.compact) {
        PyErr_SetString(PyExc_TypeError, "Cannot check non-compact unicode");
        goto fail;
    }
    if (a->state.ascii) {
        PyErr_SetString(PyExc_TypeError, "Unicode is ASCII");
        goto fail;
    }
    PyCompactUnicodeObject *u = (PyCompactUnicodeObject *)unicode;
    bool has_cache = (u->utf8 != NULL);
    if (has_cache) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
fail:;
    return NULL;
}

static PyMethodDef ssrjson_benchmark_methods[] = {
        {"copy_unicode_list_invalidate_cache", (PyCFunction)copy_unicode_list_invalidate_cache, METH_VARARGS | METH_KEYWORDS, "Copy unicode list invalidate cache."},
        {"run_object_accumulate_benchmark", (PyCFunction)run_object_accumulate_benchmark, METH_VARARGS | METH_KEYWORDS, "Benchmark."},
        {"run_object_benchmark", (PyCFunction)run_object_benchmark, METH_VARARGS | METH_KEYWORDS, "Benchmark."},
        {"benchmark_bytes_load", (PyCFunction)benchmark_bytes_load, METH_VARARGS | METH_KEYWORDS, "Warmup + repeat benchmark for bytes input. Returns (total_ns, [per_iter_ns])."},
        {"benchmark_unicode_invalidate_cache", (PyCFunction)benchmark_unicode_invalidate_cache, METH_VARARGS | METH_KEYWORDS, "Binned benchmark with unicode cache invalidation. Returns (total_ns, [per_iter_ns])."},
        {"benchmark_invalidate_dump_cache", (PyCFunction)benchmark_invalidate_dump_cache, METH_VARARGS | METH_KEYWORDS, "Binned benchmark invalidating dump cache via loads. Returns (total_ns, [per_iter_ns])."},
        {"benchmark_with_dump_cache", (PyCFunction)benchmark_with_dump_cache, METH_VARARGS | METH_KEYWORDS, "Warmup + repeat benchmark with cache enabled. Returns (total_ns, [per_iter_ns])."},
        {"inspect_pyunicode", (PyCFunction)inspect_pyunicode, METH_VARARGS | METH_KEYWORDS, "Inspect PyUnicode."},
        {"pyunicode_has_utf8_cache", (PyCFunction)pyunicode_has_utf8_cache, METH_VARARGS | METH_KEYWORDS, "Check if str has UTF-8 cache."},
        {NULL, NULL, 0, NULL} /* Sentinel */
};

static PyModuleDef_Slot ssrjson_module_slots[] = {
#if PY_MINOR_VERSION >= 14 && !defined(Py_GIL_DISABLED)
        {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
        {0, NULL} /* Sentinel */
};

static struct PyModuleDef moduledef = {
        .m_base = PyModuleDef_HEAD_INIT,
        .m_name = "_ssrjson_benchmark",
        .m_size = 0,
        .m_methods = ssrjson_benchmark_methods,
        .m_slots = ssrjson_module_slots,
};

PyMODINIT_FUNC PyInit__ssrjson_benchmark(void) {
    return PyModuleDef_Init(&moduledef);
}
