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

typedef struct PyUnicodeNewCallArg {
    Py_ssize_t size;
    int kind;
    Py_UCS4 max_char;
    bool valid;
} PyUnicodeNewCallArg;

PyObject *_copy_unicode(PyObject *unicode, PyUnicodeNewCallArg *call_arg) {
    if (!call_arg->valid) {
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
        call_arg->size = PyUnicode_GET_LENGTH(unicode);
        call_arg->kind = kind;
        call_arg->max_char = max_char;
        call_arg->valid = true;
    }

    PyObject *unicode_copy = PyUnicode_New(call_arg->size, call_arg->max_char);
    memcpy(PyUnicode_DATA(unicode_copy), PyUnicode_DATA(unicode),
           call_arg->size * call_arg->kind);
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

PyObject *run_unicode_accumulate_benchmark(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
    PyObject *callable;
    usize repeat;
    PyObject *unicode;
    PyObject *additional_args = NULL;
    static const char *kwlist[] = {"func", "repeat", "unicode", "args", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OKO|O", (char **)kwlist,
                                     &callable, &repeat, &unicode,
                                     &additional_args)) {
        PyErr_SetString(PyExc_TypeError, "Invalid argument");
        goto fail;
    }
    if (!PyCallable_Check(callable)) {
        PyErr_SetString(PyExc_TypeError, "First argument must be callable");
        goto fail;
    }
    if (!PyUnicode_Check(unicode)) {
        PyErr_SetString(PyExc_TypeError, "Third argument must be unicode");
        goto fail;
    }
    PyUnicodeNewCallArg call_arg;
    call_arg.valid = false;
    usize total = 0;
    for (usize i = 0; i < repeat; i++) {
        // create copy of unicode object.
        PyObject *new_args = _parse_additional_args(additional_args);
        if (!new_args)
            goto fail;
        PyObject *unicode_copy = _copy_unicode(unicode, &call_arg);
        if (!unicode_copy) {
            Py_DECREF(new_args);
            goto fail;
        }
        PyTuple_SET_ITEM(new_args, 0, unicode_copy);
        usize start = perf_counter();
        PyObject *result = PyObject_Call(callable, new_args, NULL);
        usize end = perf_counter();
        assert(unicode_copy->ob_refcnt == 1);
        Py_DECREF(new_args);
        unicode_copy = NULL;
        new_args = NULL;
        if (unlikely(!result)) {
            if (!PyErr_Occurred()) {
                PyErr_SetString(PyExc_RuntimeError, "Failed to call callable");
            }
            goto fail;
        } else {
            Py_DECREF(result);
        }
        total += end - start;
    }
    return PyLong_FromUnsignedLongLong(total);
fail:;
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
        goto fail;
    }
    if (!PyCallable_Check(callable)) {
        PyErr_SetString(PyExc_TypeError, "First argument must be callable");
        goto fail;
    }
    if (!PyTuple_Check(call_args)) {
        PyErr_SetString(PyExc_TypeError, "Third argument must be tuple");
        goto fail;
    }
    usize total = 0;
    for (usize i = 0; i < repeat; i++) {
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
        total += end - start;
    }
    return PyLong_FromUnsignedLongLong(total);
fail:;
    return NULL;
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

static PyMethodDef ssrjson_benchmark_methods[] = {
        {"run_unicode_accumulate_benchmark", (PyCFunction)run_unicode_accumulate_benchmark, METH_VARARGS | METH_KEYWORDS, "Benchmark."},
        {"run_object_accumulate_benchmark", (PyCFunction)run_object_accumulate_benchmark, METH_VARARGS | METH_KEYWORDS, "Benchmark."},
        {"run_object_benchmark", (PyCFunction)run_object_benchmark, METH_VARARGS | METH_KEYWORDS, "Benchmark."},
        {"inspect_pyunicode", (PyCFunction)inspect_pyunicode, METH_VARARGS | METH_KEYWORDS, "Inspect PyUnicode."},
        {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_ssrjson_benchmark",       /* m_name */
        0,                         /* m_doc */
        0,                         /* m_size */
        ssrjson_benchmark_methods, /* m_methods */
        NULL,                      /* m_slots */
        NULL,                      /* m_traverse */
        NULL,                      /* m_clear */
        NULL                       /* m_free */
};

PyMODINIT_FUNC PyInit__ssrjson_benchmark(void) {
    PyObject *module;
    // check if module already exists
    if ((module = PyState_FindModule(&moduledef)) != NULL) {
        Py_INCREF(module);
        return module;
    }
    // create module
    module = PyModule_Create(&moduledef);
    if (module == NULL) {
        return NULL;
    }
    return module;
}
