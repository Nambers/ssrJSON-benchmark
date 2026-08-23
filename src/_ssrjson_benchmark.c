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

usize perf_counter(void) {
    static LONGLONG frequency = 0;
    if (frequency == 0) {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        frequency = f.QuadPart;
    }
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    /* Split into whole seconds and remainder before scaling to nanoseconds.
       Scaling the raw counter first overflows int64 after ~15 minutes of
       uptime, and the subsequent division does not survive the wraparound. */
    LONGLONG seconds = counter.QuadPart / frequency;
    LONGLONG rest = counter.QuadPart % frequency;
    return (usize)seconds * 1000000000ULL +
           (usize)((rest * 1000000000LL) / frequency);
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

static UsizeArray _static_times_buf = {NULL, 0};

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

/**
 * Replace ring[slot] with a freshly built object. PyList_SetItem steals the new
 * reference and drops the previous one, so the old copy is released here.
 * Returns 0 on success, -1 on failure (Python exception already set).
 */
static int _ring_refill(PyObject *ring, Py_ssize_t slot, PyObject *factory,
                        PyObject *factory_arg) {
    PyObject *fresh = PyObject_CallOneArg(factory, factory_arg);
    if (unlikely(!fresh)) {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_RuntimeError, "Failed to call factory");
        return -1;
    }
    if (unlikely(PyList_SetItem(ring, slot, fresh) < 0)) return -1;
    return 0;
}

/**
 * Call func(ring[slot]) once. When `elapsed` is non-NULL the call is timed and
 * the duration in nanoseconds is stored there. Returns 0 on success, -1 on
 * failure (Python exception already set).
 */
static int _call_slot(PyObject *func, PyObject *ring, Py_ssize_t slot,
                      usize *elapsed) {
    /* Borrowed -> owned, so the item cannot vanish underneath the call.
       Both the incref and the decref sit outside the timed window. */
    PyObject *item = PyList_GET_ITEM(ring, slot);
    Py_INCREF(item);
    usize start = 0, end = 0;
    if (elapsed) start = perf_counter();
    PyObject *result = PyObject_CallOneArg(func, item);
    if (elapsed) end = perf_counter();
    Py_DECREF(item);
    if (unlikely(!result)) {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_RuntimeError, "Failed to call callable");
        return -1;
    }
    Py_DECREF(result);
    if (elapsed) *elapsed = end - start;
    return 0;
}

/**
 * benchmark_run(func, ring, repeat, warmup, factory=None, factory_arg=None,
 *               start_slot=0)
 *
 * Ring-buffer benchmark. Iteration i measures func(ring[(start_slot + i) % K]);
 * when a factory is supplied that slot is immediately rebuilt (untimed), so
 * every measured object was built exactly K iterations earlier. K == 1
 * degenerates to "build one, measure, free" (hot); K > 1 holds a working set of
 * K live copies (cold). factory=None reuses the ring as-is and allocates
 * nothing.
 *
 * start_slot lets a caller split one measurement into several rounds (to
 * interleave libraries) while keeping the rotation continuous: pass the total
 * number of calls already made against this ring.
 *
 * Returns (total_ns, [per_iter_ns]).
 */
PyObject *benchmark_run(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *func;
    PyObject *ring;
    usize repeat;
    usize warmup;
    PyObject *factory = Py_None;
    PyObject *factory_arg = Py_None;
    usize start_slot = 0;
    static const char *kwlist[] = {"func",        "ring",    "repeat",
                                   "warmup",      "factory", "factory_arg",
                                   "start_slot",  NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOKK|OOK", (char **)kwlist,
                                     &func, &ring, &repeat, &warmup, &factory,
                                     &factory_arg, &start_slot)) {
        return NULL;
    }
    if (!PyCallable_Check(func)) {
        PyErr_SetString(PyExc_TypeError, "func must be callable");
        return NULL;
    }
    if (!PyList_Check(ring)) {
        PyErr_SetString(PyExc_TypeError, "ring must be a list");
        return NULL;
    }
    Py_ssize_t ring_size = PyList_GET_SIZE(ring);
    if (ring_size < 1) {
        PyErr_SetString(PyExc_ValueError, "ring must have at least 1 element");
        return NULL;
    }
    if (repeat < 1) {
        PyErr_SetString(PyExc_ValueError, "repeat must be positive");
        return NULL;
    }
    if (factory == Py_None) {
        factory = NULL;
    } else if (!PyCallable_Check(factory)) {
        PyErr_SetString(PyExc_TypeError, "factory must be callable or None");
        return NULL;
    }

    if (_ensure_usize_capacity(&_static_times_buf, repeat)) return NULL;
    usize *times_buf = _static_times_buf.data;
    usize k = (usize)ring_size;

    /* Warmup walks the ring exactly like the measured loop does, so the steady
       state reached here is the state the measurements start from. */
    for (usize i = 0; i < warmup; i++) {
        Py_ssize_t slot = (Py_ssize_t)((start_slot + i) % k);
        if (unlikely(_call_slot(func, ring, slot, NULL))) return NULL;
        if (factory && unlikely(_ring_refill(ring, slot, factory, factory_arg)))
            return NULL;
    }

    usize total = 0;
    for (usize i = 0; i < repeat; i++) {
        /* Continue the ring index across the warmup/measure boundary (and across
           rounds, via start_slot). Restarting at 0 would make the first K
           measured objects younger than K builds whenever warmup is not a
           multiple of K, breaking the invariant that every measured object has
           been evicted by exactly K-1 later builds. */
        Py_ssize_t slot = (Py_ssize_t)((start_slot + warmup + i) % k);
        usize elapsed;
        if (unlikely(_call_slot(func, ring, slot, &elapsed))) return NULL;
        times_buf[i] = elapsed;
        total += elapsed;
        if (factory && unlikely(_ring_refill(ring, slot, factory, factory_arg)))
            return NULL;
    }
    return _build_times_result(times_buf, repeat, total);
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

/**
 * copy_unicode(unicode) -> str
 *
 * Copy a str without its UTF-8 cache. Doubles as the per-iteration factory for
 * the `loads str` benchmarks, so it is called with a single positional arg.
 */
PyObject *copy_unicode(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *unicode;
    static const char *kwlist[] = {"unicode", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char **)kwlist, &unicode)) {
        return NULL;
    }
    if (!PyUnicode_CheckExact(unicode)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be str, not other types or subclass of str");
        return NULL;
    }
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
    Py_ssize_t size = PyUnicode_GET_LENGTH(unicode);
    PyObject *unicode_copy = PyUnicode_New(size, max_char);
    if (!unicode_copy) return NULL;
    memcpy(PyUnicode_DATA(unicode_copy), PyUnicode_DATA(unicode),
           (size_t)size * (size_t)kind);
    return unicode_copy;
}

static PyMethodDef ssrjson_benchmark_methods[] = {
        {"copy_unicode", (PyCFunction)copy_unicode, METH_VARARGS | METH_KEYWORDS, "Copy a unicode object without UTF-8 cache."},
        {"benchmark_run", (PyCFunction)benchmark_run, METH_VARARGS | METH_KEYWORDS, "Ring-buffer benchmark. Returns (total_ns, [per_iter_ns])."},
        {"inspect_pyunicode", (PyCFunction)inspect_pyunicode, METH_VARARGS | METH_KEYWORDS, "Inspect PyUnicode."},
        {"pyunicode_has_utf8_cache", (PyCFunction)pyunicode_has_utf8_cache, METH_VARARGS | METH_KEYWORDS, "Check if str has UTF-8 cache."},
        {NULL, NULL, 0, NULL} /* Sentinel */
};

static PyModuleDef_Slot ssrjson_module_slots[] = {
/* Py_GIL_DISABLED is only defined on free-threaded builds (3.13t+), which are
   exactly the builds where this slot matters: without it the runtime re-enables
   the GIL on import and any free-threaded benchmark run is meaningless. */
#ifdef Py_GIL_DISABLED
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
