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

#elif defined(__APPLE__)
#    include <time.h>

/* Apple's CLOCK_MONOTONIC is derived from gettimeofday and only has
   microsecond resolution, coarser than the fastest calls measured here.
   CLOCK_UPTIME_RAW reads mach_absolute_time (24MHz ticks on Apple silicon). */
usize perf_counter(void) {
    return (usize)clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}

#else
#    include <time.h>

usize perf_counter(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (usize)ts.tv_sec * 1000000000LL + (usize)ts.tv_nsec;
}

#endif

/**
 * timer_resolution(samples=100000) -> int
 *
 * Smallest positive step between two back-to-back clock reads, in ns: the
 * finest duration a single sample can resolve, whichever of the clock's tick
 * and the cost of reading it is larger.
 */
PyObject *timer_resolution(PyObject *self, PyObject *args, PyObject *kwargs) {
    usize samples = 100000;
    static const char *kwlist[] = {"samples", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|K", (char **)kwlist, &samples)) {
        return NULL;
    }
    usize best = 0;
    usize prev = perf_counter();
    for (usize i = 0; i < samples; i++) {
        usize now = perf_counter();
        usize step = now - prev;
        if (step && (!best || step < best)) best = step;
        prev = now;
    }
    return PyLong_FromUnsignedLongLong(best);
}

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
 * Release ring[slot], leaving None in its place. Called for every slot a sample
 * used before any of them is rebuilt: a copy built while the old one is still
 * alive is carved from pymalloc's free lists in reverse order and its layout
 * degrades call after call (hot ctm.json dumps drifted from 300us to 900us),
 * whereas freeing first empties the pools and the new copy is laid out in
 * order every time. Returns 0 on success, -1 on failure.
 */
static int _ring_release(PyObject *ring, Py_ssize_t slot) {
    return PyList_SetItem(ring, slot, Py_NewRef(Py_None));
}

/**
 * Put a freshly built object into ring[slot] (released beforehand by
 * _ring_release). Returns 0 on success, -1 on failure (Python exception
 * already set).
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

/* At most this many keyword arguments per measured call. */
#define CALL_MAX_KWARGS 8

/**
 * How one library is called: func(item, **kwargs), then optionally
 * result.<then>("utf-8"). Every library goes through this same C path, so no
 * library pays for a Python-level wrapper frame that another one skips.
 *
 * argv[0] is scratch space for PY_VECTORCALL_ARGUMENTS_OFFSET, argv[1] is the
 * measured object and argv[2..] are the keyword values named by kwnames.
 */
typedef struct CallSpec {
    PyObject *func;
    PyObject *kwnames; /* tuple of keyword names, or NULL */
    PyObject *argv[2 + CALL_MAX_KWARGS];
    PyObject *then;     /* method name, or NULL */
    PyObject *then_arg; /* "utf-8", owned */
} CallSpec;

static void _call_spec_clear(CallSpec *spec) {
    Py_CLEAR(spec->kwnames);
    Py_CLEAR(spec->then_arg);
    for (int i = 0; i < 2 + CALL_MAX_KWARGS; i++) Py_CLEAR(spec->argv[i]);
}

/**
 * Fill *spec* from Python-level arguments. kwargs may be None or a dict with
 * str keys; then may be None or a method name, called with "utf-8".
 * Returns 0 on success, -1 on failure (Python exception set). Either way the
 * caller releases *spec* with _call_spec_clear().
 */
static int _call_spec_init(CallSpec *spec, PyObject *func, PyObject *kwargs,
                           PyObject *then) {
    memset(spec, 0, sizeof(*spec));
    spec->func = func;
    if (kwargs != Py_None) {
        if (!PyDict_Check(kwargs)) {
            PyErr_SetString(PyExc_TypeError, "kwargs must be a dict or None");
            return -1;
        }
        Py_ssize_t n = PyDict_GET_SIZE(kwargs);
        if (n > CALL_MAX_KWARGS) {
            PyErr_SetString(PyExc_ValueError, "too many keyword arguments");
            return -1;
        }
        if (n > 0) {
            spec->kwnames = PyTuple_New(n);
            if (!spec->kwnames) return -1;
            Py_ssize_t pos = 0, i = 0;
            PyObject *key, *value;
            while (PyDict_Next(kwargs, &pos, &key, &value)) {
                if (!PyUnicode_Check(key)) {
                    PyErr_SetString(PyExc_TypeError, "kwargs keys must be str");
                    return -1;
                }
                PyTuple_SET_ITEM(spec->kwnames, i, Py_NewRef(key));
                spec->argv[2 + i] = Py_NewRef(value);
                i++;
            }
        }
    }
    if (then != Py_None) {
        if (!PyUnicode_Check(then)) {
            PyErr_SetString(PyExc_TypeError, "then must be a str or None");
            return -1;
        }
        spec->then = then;
        spec->then_arg = PyUnicode_InternFromString("utf-8");
        if (!spec->then_arg) return -1;
    }
    return 0;
}

/**
 * One sample: `batch` back-to-back calls on ring slots first, first+1, ...
 * (mod k). When `elapsed` is non-NULL the batch is timed as a whole and its
 * duration in nanoseconds is stored there. Taking the items and freeing the
 * results both sit outside the timed window; rebuilding the slots is left to
 * the caller. `items` and `results` are scratch arrays of `batch` entries.
 * Returns 0 on success, -1 on failure (Python exception already set).
 */
static int _call_batch(CallSpec *spec, PyObject *ring, usize k, usize first,
                       usize batch, PyObject **items, PyObject **results,
                       usize *elapsed) {
    /* Borrowed -> owned, so no item can vanish underneath its call. */
    for (usize b = 0; b < batch; b++) {
        items[b] = PyList_GET_ITEM(ring, (Py_ssize_t)((first + b) % k));
        Py_INCREF(items[b]);
        results[b] = NULL;
    }
    PyObject **argv = spec->argv + 1;
    size_t nargsf = 1 | PY_VECTORCALL_ARGUMENTS_OFFSET;
    int ok = 1;
    usize start = 0, end = 0;
    if (elapsed) start = perf_counter();
    for (usize b = 0; b < batch; b++) {
        argv[0] = items[b];
        PyObject *result = PyObject_Vectorcall(spec->func, argv, nargsf, spec->kwnames);
        if (spec->then && result) {
            /* Same work as `func(x).decode("utf-8")` in user code, including
               freeing the intermediate object. */
            PyObject *converted = PyObject_CallMethodOneArg(result, spec->then, spec->then_arg);
            Py_DECREF(result);
            result = converted;
        }
        results[b] = result;
        if (unlikely(!result)) {
            ok = 0;
            break;
        }
    }
    if (elapsed) end = perf_counter();
    argv[0] = NULL;
    for (usize b = 0; b < batch; b++) {
        Py_XDECREF(results[b]);
        Py_DECREF(items[b]);
    }
    if (unlikely(!ok)) {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_RuntimeError, "Failed to call callable");
        return -1;
    }
    if (elapsed) *elapsed = end - start;
    return 0;
}

/**
 * benchmark_run(func, ring, repeat, warmup, factory=None, factory_arg=None,
 *               start_slot=0, kwargs=None, then=None, batch=1)
 *
 * Ring-buffer benchmark. Call j measures
 * func(ring[(start_slot + j) % K], **kwargs), followed by
 * result.<then>("utf-8") when then is given. A sample is `batch` consecutive
 * calls timed together, so calls shorter than the clock can resolve are still
 * measured to a fraction of a tick; `repeat` and `warmup` count samples.
 * When a factory is supplied every slot a sample used is released and then
 * rebuilt right after it (untimed), so each measured object was built between
 * K - batch and K - 1 calls earlier, and with batch == 1 exactly K - 1.
 * K == batch degenerates to "free, build, measure" (hot); a larger K holds a
 * working set of K live copies (cold). factory=None reuses the ring as-is and
 * allocates nothing.
 *
 * start_slot lets a caller split one measurement into several rounds (to
 * interleave libraries) while keeping the rotation continuous: pass the total
 * number of calls already made against this ring.
 *
 * Returns (total_ns, [per_sample_ns]); divide by batch for per-call times.
 */
PyObject *benchmark_run(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *func;
    PyObject *ring;
    usize repeat;
    usize warmup;
    PyObject *factory = Py_None;
    PyObject *factory_arg = Py_None;
    usize start_slot = 0;
    PyObject *call_kwargs = Py_None;
    PyObject *then = Py_None;
    usize batch = 1;
    static const char *kwlist[] = {"func",       "ring",    "repeat",
                                   "warmup",     "factory", "factory_arg",
                                   "start_slot", "kwargs",  "then",
                                   "batch",      NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOKK|OOKOOK", (char **)kwlist,
                                     &func, &ring, &repeat, &warmup, &factory,
                                     &factory_arg, &start_slot, &call_kwargs,
                                     &then, &batch)) {
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
    usize k = (usize)ring_size;
    if (batch < 1) {
        PyErr_SetString(PyExc_ValueError, "batch must be positive");
        return NULL;
    }
    /* With fresh objects, a batch wider than the ring would measure a slot
       twice before it is rebuilt, the second time with a warm UTF-8 cache. */
    if (factory && batch > k) {
        PyErr_SetString(PyExc_ValueError, "batch must not exceed the ring size when a factory is given");
        return NULL;
    }

    if (_ensure_usize_capacity(&_static_times_buf, repeat)) return NULL;
    usize *times_buf = _static_times_buf.data;

    PyObject **scratch = PyMem_Calloc(2 * batch, sizeof(PyObject *));
    if (!scratch) return PyErr_NoMemory();
    PyObject **items = scratch, **results = scratch + batch;

    CallSpec spec;
    PyObject *ret = NULL;
    if (_call_spec_init(&spec, func, call_kwargs, then)) goto done;

    /* Warmup walks the ring exactly like the measured loop does, and the ring
       index continues across the warmup/measure boundary (and across rounds,
       via start_slot): restarting at 0 would make the first measured objects
       younger than the ring whenever warmup is not a multiple of K. */
    usize total = 0;
    for (usize i = 0; i < warmup + repeat; i++) {
        usize first = start_slot + i * batch;
        int measured = i >= warmup;
        usize elapsed = 0;
        if (unlikely(_call_batch(&spec, ring, k, first, batch, items, results,
                                 measured ? &elapsed : NULL)))
            goto done;
        if (measured) {
            times_buf[i - warmup] = elapsed;
            total += elapsed;
        }
        if (factory) {
            for (usize b = 0; b < batch; b++) {
                Py_ssize_t slot = (Py_ssize_t)((first + b) % k);
                if (unlikely(_ring_release(ring, slot))) goto done;
            }
            for (usize b = 0; b < batch; b++) {
                Py_ssize_t slot = (Py_ssize_t)((first + b) % k);
                if (unlikely(_ring_refill(ring, slot, factory, factory_arg))) goto done;
            }
        }
    }
    ret = _build_times_result(times_buf, repeat, total);
done:
    _call_spec_clear(&spec);
    PyMem_Free(scratch);
    return ret;
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

static PyObject *_copy_unicode_impl(PyObject *unicode) {
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
    return _copy_unicode_impl(unicode);
}

static PyObject *_copy_json_impl(PyObject *obj);

static PyObject *_copy_json_list(PyObject *obj) {
    Py_ssize_t n = PyList_GET_SIZE(obj);
    PyObject *out = PyList_New(n);
    if (!out) return NULL;
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = _copy_json_impl(PyList_GET_ITEM(obj, i));
        if (!item) {
            Py_DECREF(out);
            return NULL;
        }
        PyList_SET_ITEM(out, i, item);
    }
    return out;
}

static PyObject *_copy_json_dict(PyObject *obj) {
    PyObject *out = PyDict_New();
    if (!out) return NULL;
    Py_ssize_t pos = 0;
    PyObject *key, *value;
    while (PyDict_Next(obj, &pos, &key, &value)) {
        PyObject *k = _copy_json_impl(key);
        PyObject *v = k ? _copy_json_impl(value) : NULL;
        int rc = v ? PyDict_SetItem(out, k, v) : -1;
        Py_XDECREF(k);
        Py_XDECREF(v);
        if (rc < 0) {
            Py_DECREF(out);
            return NULL;
        }
    }
    return out;
}

/* Deep copy of a parsed JSON value: new dicts and lists, every str (keys
   included) copied without its UTF-8 cache, other scalars shared. */
static PyObject *_copy_json_impl(PyObject *obj) {
    if (PyUnicode_CheckExact(obj)) return _copy_unicode_impl(obj);
    int is_list = PyList_CheckExact(obj);
    if (!is_list && !PyDict_CheckExact(obj)) return Py_NewRef(obj);
    if (Py_EnterRecursiveCall(" while copying a JSON value")) return NULL;
    PyObject *out = is_list ? _copy_json_list(obj) : _copy_json_dict(obj);
    Py_LeaveRecursiveCall();
    return out;
}

/**
 * copy_json(obj) -> object
 *
 * The per-iteration factory for dumps of non-ASCII data, in C because a
 * Python recursive copy costs far more wall-clock than the call it feeds.
 */
PyObject *copy_json(PyObject *self, PyObject *obj) {
    return _copy_json_impl(obj);
}

static PyMethodDef ssrjson_benchmark_methods[] = {
        {"copy_unicode", (PyCFunction)copy_unicode, METH_VARARGS | METH_KEYWORDS, "Copy a unicode object without UTF-8 cache."},
        {"copy_json", (PyCFunction)copy_json, METH_O, "Deep-copy a parsed JSON value, every str without UTF-8 cache."},
        {"benchmark_run", (PyCFunction)benchmark_run, METH_VARARGS | METH_KEYWORDS, "Ring-buffer benchmark. Returns (total_ns, [per_sample_ns])."},
        {"timer_resolution", (PyCFunction)timer_resolution, METH_VARARGS | METH_KEYWORDS, "Smallest positive step between back-to-back clock reads, in ns."},
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
