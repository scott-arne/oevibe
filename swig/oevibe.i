// swig/oevibe.i
// SWIG interface file for oevibe Python bindings
%module(directors="1", threads="1") _oevibe

%{
#include "oevibe/oevibe.h"
#include <oechem.h>
#include <oegrid.h>

using namespace OEVIBE;
%}

// ============================================================================
// Forward declarations for cross-module SWIG type resolution
// ============================================================================
// These enable typemaps for OpenEye types whose definitions live in the
// OpenEye SWIG runtime (v4). Only types you actually use in your wrapped API
// need full #include — forward declarations suffice for the typemaps.

namespace OEChem {
    class OEMolBase;
    class OEMCMolBase;
    class OEMol;
    class OEGraphMol;
    class OEAtomBase;
    class OEBondBase;
    class OEConfBase;
    class OEMatchBase;
    class OEMolDatabase;
    class oemolistream;
    class oemolostream;
    class OEQMol;
    class OEResidue;
    class OEUniMolecularRxn;
}

namespace OEBio {
    class OEDesignUnit;
    class OEHierView;
    class OEHierResidue;
    class OEHierFragment;
    class OEHierChain;
    class OEInteractionHint;
    class OEInteractionHintContainer;
}

namespace OEDocking {
    class OEReceptor;
}

namespace OEPlatform {
    class oeifstream;
    class oeofstream;
    class oeisstream;
    class oeosstream;
}

namespace OESystem {
    class OEScalarGrid;
    class OERecord;
    class OEMolRecord;
}

namespace OEMolPotential {
    class OEMolFunc1;
    class OEMolFunc2;
}

// ============================================================================
// Cross-runtime SWIG compatibility layer
// ============================================================================
// OpenEye's Python bindings use SWIG runtime v4; our module uses v5.
// Since the runtimes are separate, SWIG_TypeQuery cannot access OpenEye types.
// We use Python isinstance for type safety and directly extract the void*
// pointer from the SwigPyObject struct layout (stable across SWIG versions).
//
// This approach enables passing OpenEye objects between Python and C++ without
// serialization. The macros below generate the boilerplate for each type.

%{
// Minimal SwigPyObject layout compatible across SWIG runtime versions.
// The actual struct may have more fields, but ptr is always first after
// PyObject_HEAD.
struct _SwigPyObjectCompat {
    PyObject_HEAD
    void *ptr;
};

static void* _oevibe_extract_swig_ptr(PyObject* obj) {
    PyObject* thisAttr = PyObject_GetAttrString(obj, "this");
    if (!thisAttr) {
        PyErr_Clear();
        return NULL;
    }
    void* ptr = ((_SwigPyObjectCompat*)thisAttr)->ptr;
    Py_DECREF(thisAttr);
    return ptr;
}

// ---- Type checker generator macro ----
// Generates a cached isinstance checker for an OpenEye Python type.
// TAG:    identifier suffix (e.g., oemolbase)
// MODULE: Python module string (e.g., "openeye.oechem")
// CLASS:  Python class name string (e.g., "OEMolBase")
#define DEFINE_OE_TYPE_CHECKER(TAG, MODULE, CLASS) \
    static PyObject* _oevibe_oe_##TAG##_type = NULL; \
    static bool _oevibe_is_##TAG(PyObject* obj) { \
        if (!_oevibe_oe_##TAG##_type) { \
            PyObject* mod = PyImport_ImportModule(MODULE); \
            if (mod) { \
                _oevibe_oe_##TAG##_type = PyObject_GetAttrString(mod, CLASS); \
                Py_DECREF(mod); \
            } \
            if (!_oevibe_oe_##TAG##_type) return false; \
        } \
        return PyObject_IsInstance(obj, _oevibe_oe_##TAG##_type) == 1; \
    }

// ---- Molecule types (openeye.oechem) ----
DEFINE_OE_TYPE_CHECKER(oemolbase,    "openeye.oechem", "OEMolBase")
DEFINE_OE_TYPE_CHECKER(oemcmolbase,  "openeye.oechem", "OEMCMolBase")
DEFINE_OE_TYPE_CHECKER(oemol,        "openeye.oechem", "OEMol")
DEFINE_OE_TYPE_CHECKER(oegraphmol,   "openeye.oechem", "OEGraphMol")
DEFINE_OE_TYPE_CHECKER(oeqmol,       "openeye.oechem", "OEQMol")

// ---- Atom / bond / conformer / residue (openeye.oechem) ----
DEFINE_OE_TYPE_CHECKER(oeatombase,   "openeye.oechem", "OEAtomBase")
DEFINE_OE_TYPE_CHECKER(oebondbase,   "openeye.oechem", "OEBondBase")
DEFINE_OE_TYPE_CHECKER(oeconfbase,   "openeye.oechem", "OEConfBase")
DEFINE_OE_TYPE_CHECKER(oeresidue,    "openeye.oechem", "OEResidue")
DEFINE_OE_TYPE_CHECKER(oematchbase,  "openeye.oechem", "OEMatchBase")

// ---- Molecule I/O (openeye.oechem) ----
DEFINE_OE_TYPE_CHECKER(oemolistream, "openeye.oechem", "oemolistream")
DEFINE_OE_TYPE_CHECKER(oemolostream, "openeye.oechem", "oemolostream")
DEFINE_OE_TYPE_CHECKER(oemoldatabase,"openeye.oechem", "OEMolDatabase")

// ---- Reactions (openeye.oechem) ----
DEFINE_OE_TYPE_CHECKER(oeunimolecularrxn, "openeye.oechem", "OEUniMolecularRxn")

// ---- Platform streams (openeye.oechem) ----
DEFINE_OE_TYPE_CHECKER(oeifstream,   "openeye.oechem", "oeifstream")
DEFINE_OE_TYPE_CHECKER(oeofstream,   "openeye.oechem", "oeofstream")
DEFINE_OE_TYPE_CHECKER(oeisstream,   "openeye.oechem", "oeisstream")
DEFINE_OE_TYPE_CHECKER(oeosstream,   "openeye.oechem", "oeosstream")

// ---- Records (openeye.oechem) ----
DEFINE_OE_TYPE_CHECKER(oerecord,     "openeye.oechem", "OERecord")
DEFINE_OE_TYPE_CHECKER(oemolrecord,  "openeye.oechem", "OEMolRecord")

// ---- Bio / hierarchy (openeye.oechem) ----
DEFINE_OE_TYPE_CHECKER(oedesignunit, "openeye.oechem", "OEDesignUnit")
DEFINE_OE_TYPE_CHECKER(oehierview,   "openeye.oechem", "OEHierView")
DEFINE_OE_TYPE_CHECKER(oehierresidue,"openeye.oechem", "OEHierResidue")
DEFINE_OE_TYPE_CHECKER(oehierfragment,"openeye.oechem","OEHierFragment")
DEFINE_OE_TYPE_CHECKER(oehierchain,  "openeye.oechem", "OEHierChain")
DEFINE_OE_TYPE_CHECKER(oeinteractionhint,          "openeye.oechem", "OEInteractionHint")
DEFINE_OE_TYPE_CHECKER(oeinteractionhintcontainer, "openeye.oechem", "OEInteractionHintContainer")

// ---- Grid (openeye.oegrid) ----
DEFINE_OE_TYPE_CHECKER(oescalargrid, "openeye.oegrid", "OEScalarGrid")

// ---- Docking (openeye.oedocking) ----
DEFINE_OE_TYPE_CHECKER(oereceptor,   "openeye.oedocking", "OEReceptor")

#undef DEFINE_OE_TYPE_CHECKER

// OEMolFunc1 is a virtual base of every OpenEye potential, so it does not sit at offset 0 of the
// object a proxy points at (measured: +48 in OEMMFF) and there is no shared type table to adjust
// the pointer with. oevibe.MolFuncEngine does the upcast with OpenEye's own runtime and passes the
// resulting proxy here, whose exact type is OEMolFunc1 and whose pointer is already adjusted.
// Accepting a subclass proxy instead would silently read 48 bytes short of the subobject.
//
// Every OpenEye binding module defines its OWN proxy class for OEMolPotential::OEMolFunc1, and the
// last one imported wins the shared SWIG type registry: in a process that has imported
// openeye.oequacpac, oeff.OEGenericFF.GetComponents() hands back oequacpac.OEMolFunc1 proxies.
// Any module's OEMolFunc1 carries the adjusted pointer, so the class is recognised by provenance
// -- it must BE what openeye.<its __module__> exports under OEMolFunc1 -- rather than compared
// against one class cached in advance, which rejected legitimate callers by import order.
//
// That __module__ already carries the openeye. prefix when the caller wrote `import openeye.oeff`
// rather than `from openeye import oeff`, so it is normalised to the leaf before being qualified;
// prepending unconditionally sent this looking for openeye.openeye.oeff and refused every
// potential. oevibe.oemolfunc._openeye_leaf_name is the Python side of the same normalisation.
static bool _oevibe_is_exact_molfunc1(PyObject* obj) {
    PyObject* type = reinterpret_cast<PyObject*>(Py_TYPE(obj));
    // Cheap reject first: overload resolution runs this over every argument, and only a type
    // named OEMolFunc1 is worth asking the import system about.
    PyObject* name = PyObject_GetAttrString(type, "__name__");
    if (!name) {
        PyErr_Clear();
        return false;
    }
    const bool named_molfunc1 = PyUnicode_Check(name) &&
                                PyUnicode_CompareWithASCIIString(name, "OEMolFunc1") == 0;
    Py_DECREF(name);
    if (!named_molfunc1) {
        return false;
    }
    PyObject* module_name = PyObject_GetAttrString(type, "__module__");
    if (!module_name || !PyUnicode_Check(module_name)) {
        Py_XDECREF(module_name);
        PyErr_Clear();
        return false;
    }
    PyObject* leaf = PyObject_CallMethod(module_name, "removeprefix", "s", "openeye.");
    Py_DECREF(module_name);
    if (!leaf) {
        PyErr_Clear();
        return false;
    }
    PyObject* qualified = PyUnicode_FromFormat("openeye.%U", leaf);
    Py_DECREF(leaf);
    if (!qualified) {
        PyErr_Clear();
        return false;
    }
    PyObject* mod = PyImport_Import(qualified);
    Py_DECREF(qualified);
    if (!mod) {
        PyErr_Clear();
        return false;
    }
    PyObject* declared = PyObject_GetAttrString(mod, "OEMolFunc1");
    Py_DECREF(mod);
    if (!declared) {
        PyErr_Clear();
        return false;
    }
    bool exact = (declared == type);
    Py_DECREF(declared);
    return exact;
}

// ---- OEScalarGrid return-type helper (zero-copy pointer swap) ----
static PyObject* _oevibe_wrap_as_oe_grid(OESystem::OEScalarGrid* grid) {
    if (!grid) {
        Py_RETURN_NONE;
    }
    PyObject* oegrid_mod = PyImport_ImportModule("openeye.oegrid");
    if (!oegrid_mod) {
        delete grid;
        return NULL;
    }
    PyObject* grid_cls = PyObject_GetAttrString(oegrid_mod, "OEScalarGrid");
    Py_DECREF(oegrid_mod);
    if (!grid_cls) {
        delete grid;
        return NULL;
    }
    PyObject* oe_grid = PyObject_CallNoArgs(grid_cls);
    Py_DECREF(grid_cls);
    if (!oe_grid) {
        delete grid;
        return NULL;
    }
    PyObject* thisAttr = PyObject_GetAttrString(oe_grid, "this");
    if (!thisAttr) {
        PyErr_Clear();
        Py_DECREF(oe_grid);
        delete grid;
        return NULL;
    }
    _SwigPyObjectCompat* swig_this = (_SwigPyObjectCompat*)thisAttr;
    delete reinterpret_cast<OESystem::OEScalarGrid*>(swig_this->ptr);
    swig_this->ptr = grid;
    Py_DECREF(thisAttr);
    return oe_grid;
}
%}

// ============================================================================
// Typemap generator macros
// ============================================================================

// Generate const-ref and non-const-ref typemaps for a cross-runtime OpenEye type.
// CPP_TYPE: fully qualified C++ type (e.g., OEChem::OEMolBase)
// CHECKER:  isinstance checker function name
// ERR_MSG:  error message on type mismatch
%define OE_CROSS_RUNTIME_REF_TYPEMAPS(CPP_TYPE, CHECKER, ERR_MSG)

%typemap(in) const CPP_TYPE& (void *argp = 0, int res = 0) {
    res = SWIG_ConvertPtr($input, &argp, $descriptor, 0);
    if (!SWIG_IsOK(res)) {
        if (CHECKER($input)) {
            argp = _oevibe_extract_swig_ptr($input);
            if (argp) res = SWIG_OK;
        }
    }
    if (!SWIG_IsOK(res)) {
        SWIG_exception_fail(SWIG_ArgError(res), ERR_MSG);
    }
    if (!argp) {
        SWIG_exception_fail(SWIG_NullReferenceError, "Null reference.");
    }
    $1 = reinterpret_cast< $1_ltype >(argp);
}

%typemap(typecheck, precedence=10) const CPP_TYPE& {
    void *vptr = 0;
    int res = SWIG_ConvertPtr($input, &vptr, $descriptor, SWIG_POINTER_NO_NULL);
    $1 = SWIG_IsOK(res) ? 1 : CHECKER($input) ? 1 : 0;
}

%typemap(in) CPP_TYPE& (void *argp = 0, int res = 0) {
    res = SWIG_ConvertPtr($input, &argp, $descriptor, 0);
    if (!SWIG_IsOK(res)) {
        if (CHECKER($input)) {
            argp = _oevibe_extract_swig_ptr($input);
            if (argp) res = SWIG_OK;
        }
    }
    if (!SWIG_IsOK(res)) {
        SWIG_exception_fail(SWIG_ArgError(res), ERR_MSG);
    }
    if (!argp) {
        SWIG_exception_fail(SWIG_NullReferenceError, "Null reference.");
    }
    $1 = reinterpret_cast< $1_ltype >(argp);
}

%typemap(typecheck, precedence=10) CPP_TYPE& {
    void *vptr = 0;
    int res = SWIG_ConvertPtr($input, &vptr, $descriptor, SWIG_POINTER_NO_NULL);
    $1 = SWIG_IsOK(res) ? 1 : CHECKER($input) ? 1 : 0;
}

%enddef

// Generate nullable-pointer typemaps (accepts None) for a cross-runtime type.
%define OE_CROSS_RUNTIME_NULLABLE_PTR_TYPEMAPS(CPP_TYPE, CHECKER, ERR_MSG)

%typemap(in) const CPP_TYPE* (void *argp = 0, int res = 0) {
    if ($input == Py_None) {
        $1 = NULL;
    } else {
        res = SWIG_ConvertPtr($input, &argp, $descriptor, 0);
        if (!SWIG_IsOK(res)) {
            if (CHECKER($input)) {
                argp = _oevibe_extract_swig_ptr($input);
                if (argp) res = SWIG_OK;
            }
        }
        if (!SWIG_IsOK(res)) {
            SWIG_exception_fail(SWIG_ArgError(res), ERR_MSG);
        }
        $1 = reinterpret_cast< $1_ltype >(argp);
    }
}

%typemap(typecheck, precedence=10) const CPP_TYPE* {
    if ($input == Py_None) {
        $1 = 1;
    } else {
        void *vptr = 0;
        int res = SWIG_ConvertPtr($input, &vptr, $descriptor, 0);
        $1 = SWIG_IsOK(res) ? 1 : CHECKER($input) ? 1 : 0;
    }
}

%enddef

// ============================================================================
// Typemap declarations for all OpenEye types
// ============================================================================
// Each type gets const-ref and non-const-ref typemaps. Types that commonly
// appear as optional parameters also get nullable-pointer typemaps.
// These are inert until a wrapped function signature uses the type.

// ---- Molecule hierarchy (OEChem) ----
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEChem::OEMolBase,    _oevibe_is_oemolbase,    "Expected OEMolBase-derived object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEChem::OEMCMolBase,  _oevibe_is_oemcmolbase,  "Expected OEMCMolBase-derived object (OEMCMolBase or OEMol).")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEChem::OEMol,        _oevibe_is_oemol,        "Expected OEMol object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEChem::OEGraphMol,   _oevibe_is_oegraphmol,   "Expected OEGraphMol object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEChem::OEQMol,       _oevibe_is_oeqmol,       "Expected OEQMol object.")

// ---- Atom / bond / conformer / residue / match (OEChem) ----
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEChem::OEAtomBase,   _oevibe_is_oeatombase,   "Expected OEAtomBase-derived object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEChem::OEBondBase,   _oevibe_is_oebondbase,   "Expected OEBondBase-derived object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEChem::OEConfBase,   _oevibe_is_oeconfbase,   "Expected OEConfBase-derived object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEChem::OEResidue,    _oevibe_is_oeresidue,    "Expected OEResidue object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEChem::OEMatchBase,  _oevibe_is_oematchbase,  "Expected OEMatchBase-derived object.")

// ---- Molecule I/O (OEChem) ----
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEChem::oemolistream,  _oevibe_is_oemolistream, "Expected oemolistream object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEChem::oemolostream,  _oevibe_is_oemolostream, "Expected oemolostream object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEChem::OEMolDatabase, _oevibe_is_oemoldatabase,"Expected OEMolDatabase object.")

// ---- Reactions (OEChem) ----
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEChem::OEUniMolecularRxn, _oevibe_is_oeunimolecularrxn, "Expected OEUniMolecularRxn object.")

// ---- Platform streams (OEPlatform) ----
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEPlatform::oeifstream, _oevibe_is_oeifstream, "Expected oeifstream object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEPlatform::oeofstream, _oevibe_is_oeofstream, "Expected oeofstream object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEPlatform::oeisstream, _oevibe_is_oeisstream, "Expected oeisstream object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEPlatform::oeosstream, _oevibe_is_oeosstream, "Expected oeosstream object.")

// ---- Records (OESystem) ----
OE_CROSS_RUNTIME_REF_TYPEMAPS(OESystem::OERecord,    _oevibe_is_oerecord,    "Expected OERecord object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OESystem::OEMolRecord, _oevibe_is_oemolrecord, "Expected OEMolRecord object.")

// ---- Bio / hierarchy (OEBio) ----
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEBio::OEDesignUnit,   _oevibe_is_oedesignunit, "Expected OEDesignUnit object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEBio::OEHierView,     _oevibe_is_oehierview,   "Expected OEHierView object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEBio::OEHierResidue,  _oevibe_is_oehierresidue,"Expected OEHierResidue object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEBio::OEHierFragment,  _oevibe_is_oehierfragment,"Expected OEHierFragment object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEBio::OEHierChain,    _oevibe_is_oehierchain,  "Expected OEHierChain object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEBio::OEInteractionHint,          _oevibe_is_oeinteractionhint,          "Expected OEInteractionHint object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEBio::OEInteractionHintContainer, _oevibe_is_oeinteractionhintcontainer, "Expected OEInteractionHintContainer object.")

// ---- Grid (OESystem) ----
OE_CROSS_RUNTIME_REF_TYPEMAPS(OESystem::OEScalarGrid, _oevibe_is_oescalargrid, "Expected OEScalarGrid-derived object.")
OE_CROSS_RUNTIME_NULLABLE_PTR_TYPEMAPS(OESystem::OEScalarGrid, _oevibe_is_oescalargrid, "Expected OEScalarGrid or None.")

// OEScalarGrid return-type typemap (wraps C++ grid as native openeye.oegrid object)
%typemap(out) OESystem::OEScalarGrid* {
    $result = _oevibe_wrap_as_oe_grid($1);
    if (!$result) SWIG_fail;
}

// ---- Docking (OEDocking) ----
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEDocking::OEReceptor, _oevibe_is_oereceptor, "Expected OEReceptor object.")

// ---- Potentials (OEMolPotential) ----
// Not OE_CROSS_RUNTIME_REF_TYPEMAPS: that would accept every subclass proxy, and the pointer it
// holds is the most-derived one. See _oevibe_is_exact_molfunc1 for why only the exact type is safe.
%typemap(in) const OEMolPotential::OEMolFunc1& (void *argp = 0) {
    if (!_oevibe_is_exact_molfunc1($input)) {
        SWIG_exception_fail(SWIG_TypeError,
                            "expected an openeye.oeff.OEMolFunc1 produced by "
                            "oevibe.MolFuncEngine; pass the potential to oevibe.MolFuncEngine "
                            "rather than constructing the wrapped class directly.");
    }
    argp = _oevibe_extract_swig_ptr($input);
    if (!argp) {
        SWIG_exception_fail(SWIG_NullReferenceError, "Null reference.");
    }
    $1 = reinterpret_cast< $1_ltype >(argp);
}

%typemap(typecheck, precedence=10) const OEMolPotential::OEMolFunc1& {
    $1 = _oevibe_is_exact_molfunc1($input) ? 1 : 0;
}

// ============================================================================
// GIL: held by default, released only for Energy, Gradient, Optimize, Frequencies, and Scan
// ============================================================================
// A feature applies only to declarations parsed after it, so this block precedes every
// wrapper-generating directive. Plan A engines are native C++ and never call back into Python,
// so releasing the GIL around a calculation is safe. The constructor keeps the GIL: it copies
// the caller's CalculationOptions, which another Python thread could be editing, and it only
// validates, so there is no calculation to overlap with. Optimize, Frequencies, and Scan read
// their options for the whole run, so their wrappers copy them while the GIL is still held (the
// typemaps before Calculation.h below).
%feature("nothreadallow", "1");
%threadallow OEVIBE::Calculation::Energy;
%threadallow OEVIBE::Calculation::Gradient;
%threadallow OEVIBE::Calculation::Optimize;
%threadallow OEVIBE::Calculation::Frequencies;
// Scan reads its ScanOptions for the whole run, so its wrapper copies them while the GIL is still
// held (the typemap before Calculation.h below).
%threadallow OEVIBE::Calculation::Scan;

// ============================================================================
// STL support
// ============================================================================
// The C++-style iterator methods (begin, end, find, ...) return iterators that do not keep
// their container alive, so they are not exported. This also drops the vectors' iterator() and
// __iter__, so Python iterates a vector by index through __getitem__, which stays valid when the
// vector reallocates.
#define SWIG_NO_EXPORT_ITERATOR_METHODS
%include "std_string.i"
%include "std_vector.i"
%include "std_map.i"
%include "stdint.i"
%include "exception.i"
%include "std_shared_ptr.i"

// std_vector.i wraps front(), back() and pop_back() straight onto the C++ methods, so calling any
// of them on an empty vector is undefined behavior rather than an IndexError, and get_allocator
// only hands back an opaque proxy. Nothing in OEVibe needs them, so every wrapped vector drops
// them; pop() stays because std_vector.i range-checks it.
%define OEVIBE_SAFE_VECTOR(Vector...)
%ignore Vector::front;
%ignore Vector::back;
%ignore Vector::pop_back;
%ignore Vector::get_allocator;
%enddef

// Indexing a vector of a class type returns a pointer into the vector's storage. The element's
// reference to the vector keeps that storage allocated but not in place, so every method that
// can reallocate, shrink, swap, or replace it leaves elements taken earlier dangling. Such a
// vector is read-only here: C++ fills it, and the container typemaps convert a plain Python
// list the other way without any of these.
%define OEVIBE_READONLY_VECTOR(Vector...)
OEVIBE_SAFE_VECTOR(Vector)
%ignore Vector::append;
%ignore Vector::push_back;
%ignore Vector::insert;
%ignore Vector::assign;
%ignore Vector::resize;
%ignore Vector::reserve;
%ignore Vector::pop;
%ignore Vector::erase;
%ignore Vector::clear;
%ignore Vector::swap;
%ignore Vector::__setitem__;
%ignore Vector::__delitem__;
// SWIG releases before 4.5 also generate the Python 2 slice methods.
%ignore Vector::__setslice__;
%ignore Vector::__delslice__;
%enddef

OEVIBE_SAFE_VECTOR(std::vector<double>)
OEVIBE_SAFE_VECTOR(std::vector<unsigned int>)
OEVIBE_SAFE_VECTOR(std::vector<std::string>)
%template(DoubleVector) std::vector<double>;
%template(UnsignedVector) std::vector<unsigned int>;
%template(StringVector) std::vector<std::string>;

// Python iterates a map over a snapshot of its keys, values or items. The library's versions
// hold a C++ iterator that erasing its entry leaves dangling, and its iteritems() calls the
// iterator() method the define above removes.
%define OEVIBE_SNAPSHOT_MAP_ITERATION(Map...)
%ignore Map::key_iterator;
%ignore Map::value_iterator;
%extend Map {
%pythoncode %{
    def __iter__(self):
        return iter(self.keys())

    def iterkeys(self):
        return iter(self.keys())

    def itervalues(self):
        return iter(self.values())

    def iteritems(self):
        return iter(self.items())
%}
}
%enddef
OEVIBE_SNAPSHOT_MAP_ITERATION(std::map<std::string, std::string>)
OEVIBE_SNAPSHOT_MAP_ITERATION(std::map<unsigned int, double>)
%template(StringStringMap) std::map<std::string, std::string>;
%template(UnsignedDoubleMap) std::map<unsigned int, double>;

// ============================================================================
// Exceptions: each C++ class maps to a Python class rooted at RuntimeError
// ============================================================================
%{
static PyObject* g_oevibe_error = NULL;
static PyObject* g_options_error = NULL;
static PyObject* g_engine_unavailable_error = NULL;
static PyObject* g_input_error = NULL;
static PyObject* g_unsupported_error = NULL;

// Idempotent: if module initialization runs twice, the existing class is reused.
static int oevibe_add_exception(PyObject* module, PyObject** slot, const char* qualified,
                                const char* name, PyObject* base) {
    if (*slot == NULL) {
        *slot = PyErr_NewException(qualified, base, NULL);
        if (*slot == NULL) {
            return -1;
        }
    }
    return PyModule_AddObjectRef(module, name, *slot);
}

// A director method returned with a Python exception set. Converting it to the matching C++
// exception is what lets spec 7.5 treat a Python engine exactly like a native one:
// UnsupportedError becomes UNSUPPORTED, everything else ENGINE_ERROR carrying the Python
// message. The GIL is held here, because a director callback always takes it.
static bool oevibe_is_instance(PyObject* value, PyObject* type) {
    return type != NULL && value != NULL && PyObject_IsInstance(value, type) == 1;
}

[[noreturn]] static void oevibe_throw_from_python() {
    PyObject* type = NULL;
    PyObject* value = NULL;
    PyObject* traceback = NULL;
    PyErr_Fetch(&type, &value, &traceback);
    PyErr_NormalizeException(&type, &value, &traceback);
    std::string message = "Python engine raised an exception";
    if (value != NULL) {
        PyObject* text = PyObject_Str(value);
        if (text != NULL) {
            // The module is built against the stable ABI, which has no PyUnicode_AsUTF8, so the
            // message goes through a bytes object instead.
            PyObject* encoded = PyUnicode_AsUTF8String(text);
            if (encoded != NULL) {
                const char* utf8 = PyBytes_AsString(encoded);
                if (utf8 != NULL && *utf8 != '\0') {
                    message = utf8;
                }
                Py_DECREF(encoded);
            }
            Py_DECREF(text);
        }
    }
    const bool unsupported = oevibe_is_instance(value, g_unsupported_error);
    const bool input = oevibe_is_instance(value, g_input_error);
    const bool options = oevibe_is_instance(value, g_options_error);
    const bool unavailable = oevibe_is_instance(value, g_engine_unavailable_error);
    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(traceback);
    PyErr_Clear();
    if (unsupported) throw OEVIBE::UnsupportedError(message);
    if (input) throw OEVIBE::InputError(message);
    if (options) throw OEVIBE::OptionsError(message);
    if (unavailable) throw OEVIBE::EngineUnavailableError(message);
    throw std::runtime_error(message);
}
%}

// The %init block below runs inside a function whose failure value depends on the SWIG
// version. SWIG 4.4 introduced multi-phase initialization, which emits %init code into
// SWIG_mod_exec, a Py_mod_exec slot returning int (-1 on failure); earlier versions emit it into
// the single-phase PyInit function, which returns the module object (NULL on failure). This
// SWIG_VERSION is SWIG's own version, defined by the SWIG preprocessor in every release.
#if SWIG_VERSION >= 0x040400
%{
#define OEVIBE_INIT_FAILURE -1
%}
#else
%{
#define OEVIBE_INIT_FAILURE NULL
%}
#endif

%init %{
    if (oevibe_add_exception(m, &g_oevibe_error, "oevibe.OEVibeError", "OEVibeError",
                             PyExc_RuntimeError) < 0 ||
        oevibe_add_exception(m, &g_options_error, "oevibe.OptionsError", "OptionsError",
                             g_oevibe_error) < 0 ||
        oevibe_add_exception(m, &g_engine_unavailable_error, "oevibe.EngineUnavailableError",
                             "EngineUnavailableError", g_oevibe_error) < 0 ||
        oevibe_add_exception(m, &g_input_error, "oevibe.InputError", "InputError",
                             g_oevibe_error) < 0 ||
        oevibe_add_exception(m, &g_unsupported_error, "oevibe.UnsupportedError",
                             "UnsupportedError", g_oevibe_error) < 0) {
        return OEVIBE_INIT_FAILURE;
    }
%}

%exception {
    try {
        $action
    } catch (const OEVIBE::OptionsError& e) {
        PyErr_SetString(g_options_error, e.what());
        SWIG_fail;
    } catch (const OEVIBE::EngineUnavailableError& e) {
        PyErr_SetString(g_engine_unavailable_error, e.what());
        SWIG_fail;
    } catch (const OEVIBE::InputError& e) {
        PyErr_SetString(g_input_error, e.what());
        SWIG_fail;
    } catch (const OEVIBE::UnsupportedError& e) {
        PyErr_SetString(g_unsupported_error, e.what());
        SWIG_fail;
    } catch (const OEVIBE::OEVibeError& e) {
        PyErr_SetString(g_oevibe_error, e.what());
        SWIG_fail;
    } catch (const std::exception& e) {
        SWIG_exception(SWIG_RuntimeError, e.what());
    } catch (...) {
        SWIG_exception(SWIG_RuntimeError, "Unknown C++ exception");
    }
}

// ============================================================================
// Scope of the Python API in Plan A
// ============================================================================
// Engines, the registry, and MolecularSystem reach Python with the director engines of Plan C.
// The predicate overload of FreezeAtoms is replaced by a Python method below, so no OpenEye
// predicate crosses the runtime boundary (spec 6.6).
%ignore OEVIBE::ConstraintSet::FreezeAtoms;
%ignore OEVIBE::ConstraintSet::GetConstraints;
%ignore OEVIBE::ConstraintSet::GetRestraints;
%ignore OEVIBE::Constraint;
%ignore OEVIBE::Restraint;
%ignore OEVIBE::Provenance::resolved_options;
%ignore OEVIBE::Method::operator<;
%rename(__eq__) OEVIBE::Method::operator==;
%rename(__ne__) OEVIBE::Method::operator!=;
%rename(__eq__) OEVIBE::Solvent::operator==;
%ignore OEVIBE::property_names;
%ignore OEVIBE::ResultSet::Add;
%ignore OEVIBE::ResultSet::Finalize;

// SWIG drops ``const`` from reference returns, so every setter of a result would be callable on
// the views Python receives. A changed energy or provenance would silently contradict the
// finalized relative energies and the SD tags already written, so results are read-only here:
// C++ produces them and Python reads them.
%ignore OEVIBE::ResultSet::AddWarningToAll;
%ignore OEVIBE::ConformerResult::SetConformerIndex;
%ignore OEVIBE::ConformerResult::SetStatus;
%ignore OEVIBE::ConformerResult::AddWarning;
%ignore OEVIBE::ConformerResult::SetAtomIndices;
%ignore OEVIBE::ConformerResult::SetEnergy;
%ignore OEVIBE::ConformerResult::SetRelativeEnergy;
%ignore OEVIBE::ConformerResult::ClearRelativeEnergy;
%ignore OEVIBE::ConformerResult::SetCoordinates;
%ignore OEVIBE::ConformerResult::SetGradient;
%ignore OEVIBE::ConformerResult::SetCharges;
%ignore OEVIBE::ConformerResult::SetDipole;
%ignore OEVIBE::ConformerResult::SetOrbitalEnergies;
%ignore OEVIBE::ConformerResult::SetBondOrders;
%ignore OEVIBE::ConformerResult::SetHessian;
%ignore OEVIBE::ConformerResult::SetFrequencies;
%ignore OEVIBE::ConformerResult::SetThermochemistry;
%ignore OEVIBE::ConformerResult::SetOptimizationStats;
%ignore OEVIBE::ConformerResult::SetProvenance;
%ignore OEVIBE::ConformerResult::SetTimings;

// Scan results are read-only for the same reason. GetMolecule is replaced by the Python method
// in the %extend below. The cross-runtime typemaps convert OpenEye arguments, and the only
// OpenEye return typemap swaps a heap OEScalarGrid pointer into a new openeye.oegrid object, so
// an OEMol returned by value would reach Python as a proxy openeye.oechem cannot use. As for
// MolecularSystem::CopyMolecule (spec 6.2), Python passes in a genuine openeye.oechem.OEMol.
%ignore OEVIBE::ScanPoint::SetAngles;
%ignore OEVIBE::ScanPoint::SetStatus;
%ignore OEVIBE::ScanPoint::SetEnergy;
%ignore OEVIBE::ScanPoint::SetRelativeEnergy;
%ignore OEVIBE::ScanPoint::ClearRelativeEnergy;
%ignore OEVIBE::ScanPoint::SetCoordinates;
%ignore OEVIBE::ScanResult::AddDihedral;
%ignore OEVIBE::ScanResult::AddPoint;
%ignore OEVIBE::ScanResult::SetTruncated;
%ignore OEVIBE::ScanResult::AddWarning;
%ignore OEVIBE::ScanResult::SetProvenance;
%ignore OEVIBE::ScanResult::SetMolecule;
%ignore OEVIBE::ScanResult::Finalize;
%ignore OEVIBE::ScanResult::GetMolecule;

// A Calculation resolves its engine and settings from the options at construction, so Python
// receives a copy of them (see the %extend below); an alias that could be edited would
// desynchronize the calculation from the tags it writes. The %ignore is lifted again before
// the %extend because it would otherwise hide the replacement too.
%ignore OEVIBE::Calculation::GetOptions;

// Python must never hold a pointer into a C++ object it does not keep alive, so getters that
// return a reference set ``_parent`` on the result.
%pythonappend OEVIBE::ResultSet::Get %{ val._parent = self %}
%pythonappend OEVIBE::ResultSet::GetLowest %{ val._parent = self %}
%pythonappend OEVIBE::ConformerResult::GetOptimizationStats %{ val._parent = self %}
%pythonappend OEVIBE::ConformerResult::GetProvenance %{ val._parent = self %}
%pythonappend OEVIBE::ConformerResult::GetTimings %{ val._parent = self %}
%pythonappend OEVIBE::ConformerResult::GetThermochemistry %{ val._parent = self %}
%pythonappend OEVIBE::OptimizerOptions::GetConstraints %{ val._parent = self %}
%pythonappend OEVIBE::ThermoOptions::GetOptimizerOptions %{ val._parent = self %}
%pythonappend OEVIBE::ScanOptions::GetOptimizerOptions %{ val._parent = self %}
%pythonappend OEVIBE::ScanResult::GetPoint %{ val._parent = self %}
%pythonappend OEVIBE::ScanResult::GetProvenance %{ val._parent = self %}
%pythonappend OEVIBE::CalculationOptions::GetMethod %{ val._parent = self %}
%pythonappend OEVIBE::CalculationOptions::GetSolvent %{ val._parent = self %}

// Energy, Gradient, Optimize, Frequencies, and Scan release the GIL (above), so the molecule they
// read must not be shared with another thread that is using it.
%feature("docstring") OEVIBE::Calculation
"Runs tasks over every conformer of a molecule with one resolved engine (spec 5.6).

Not thread-safe; use one per thread. Construction holds the GIL. Energy, Gradient, Optimize,
Frequencies, and Scan release it while they evaluate; Optimize, Frequencies, and Scan copy their
task options before releasing it. The same molecule must not be passed to calculations from more
than one thread at a time."

// ============================================================================
// Version
// ============================================================================
#define OEVIBE_VERSION_MAJOR 0
#define OEVIBE_VERSION_MINOR 1
#define OEVIBE_VERSION_PATCH 0

// ============================================================================
// Wrapped API
// ============================================================================
%include "oevibe/Units.h"
// The Python package passes its interpreter's library directory at import (spec 8.3). No
// %threadallow names these two functions, so they keep the GIL under the nothreadallow default
// above: they only copy a string, so releasing it would buy nothing, and holding it keeps the
// setter ordered with the Calculation constructor, which also holds the GIL and is where an
// engine first reads the directory.
%include "oevibe/engines/RuntimeLibraries.h"
%include "oevibe/Method.h"
%include "oevibe/Preset.h"
%include "oevibe/Solvent.h"
%include "oevibe/Property.h"
// Results hand out bond orders as copies and nothing in Python passes a vector to C++, so this
// one takes the read-only treatment the macro describes.
OEVIBE_READONLY_VECTOR(std::vector<OEVIBE::BondOrder>)
%template(BondOrderVector) std::vector<OEVIBE::BondOrder>;

// ---- Engines (spec 8.1, 8.2, 8.7) --------------------------------------------------------
// A Python engine is a director subclass, so the GIL rules differ from the rest of the module:
// a director callback can happen while Energy, Gradient, Optimize, Frequencies or Scan has
// released the GIL, and SWIG's -threads support takes it back around every callback and around
// the director destructor. Nothing here may be called with the registry mutex held
// (EngineRegistry.h explains why).
//
// SWIG's default is to print the traceback and return a default value, which would turn a
// failed Python engine into a silently wrong answer. Converting the pending exception into the
// matching C++ one is what lets spec 7.5 treat a Python engine exactly like a native one.
%feature("director:except") {
    if ($error != NULL) {
        oevibe_throw_from_python();
    }
}
%feature("director") OEVIBE::Engine;
%feature("director") OEVIBE::EngineSession;

// The snapshot is immutable C++ state; returning its OEMolBase by reference would hand Python a
// proxy openeye.oechem cannot use, so the molecule crosses as a copy the caller owns.
%ignore OEVIBE::MolecularSystem::GetMolecule;
// A MolecularSystem reaches Python only as a borrowed view that lives for one director callback,
// so a reference-returning getter would hand out a window into storage C++ frees first. A
// ``_parent`` keepalive cannot help here: the proxy it would pin does not own the C++ object
// either. These four return copies instead, as Provenance::GetResolvedOptions already does.
%ignore OEVIBE::MolecularSystem::GetAtomicNumbers;
%ignore OEVIBE::MolecularSystem::GetOEAtomIndices;
%ignore OEVIBE::MolecularSystem::GetConformerIndices;
%ignore OEVIBE::MolecularSystem::GetConformerCoordinates;
%include "oevibe/MolecularSystem.h"
// A declaration takes its features when it is parsed, so the ignores above hid the header's
// reference-returning getters; these lift them again for the %extend replacements only, exactly
// as the Calculation::GetOptions block further down does.
%rename("%s") OEVIBE::MolecularSystem::GetAtomicNumbers;
%rename("%s") OEVIBE::MolecularSystem::GetOEAtomIndices;
%rename("%s") OEVIBE::MolecularSystem::GetConformerIndices;
%rename("%s") OEVIBE::MolecularSystem::GetConformerCoordinates;
%extend OEVIBE::MolecularSystem {
    std::vector<unsigned int> GetAtomicNumbers() const { return $self->GetAtomicNumbers(); }
    std::vector<unsigned int> GetOEAtomIndices() const { return $self->GetOEAtomIndices(); }
    std::vector<unsigned int> GetConformerIndices() const { return $self->GetConformerIndices(); }
    std::vector<double> GetConformerCoordinates(unsigned int position) const {
        return $self->GetConformerCoordinates(position);
    }
%pythoncode %{
    @property
    def molecule(self):
        """A fresh ``openeye.oechem.OEMol`` copy of the snapshot (spec 8.7)."""
        from openeye import oechem

        mol = oechem.OEMol()
        self.CopyMolecule(mol)
        return mol
%}
}

// Every vector an engine declaration mentions is instantiated before the header is parsed: a
// declaration takes the typemaps that exist when SWIG reaches it, and a %template that lands
// afterwards leaves it with the opaque-pointer director typemaps, which reject the list a
// Python engine returns from GetMethods or GetKeywordSpecs. Forward declarations are all the
// instantiation needs; Engine.h below supplies the definitions.
namespace OEVIBE {
struct KeywordSpec;
struct MethodCapabilities;
}
OEVIBE_READONLY_VECTOR(std::vector<OEVIBE::Method>)
%template(MethodVector) std::vector<OEVIBE::Method>;
OEVIBE_SAFE_VECTOR(std::vector<OEVIBE::SolventModel>)
%template(SolventModelVector) std::vector<OEVIBE::SolventModel>;
// A vector of a scoped enum is the one container std_vector.i does not convert on its own: its
// element traits say "pointer", so every solvent model would cross as an opaque pointer that
// no engine can supply and that the read path leaks. (The library's own half-measure, a
// traits_asval it drags in behind the container's ``front``, is both unreachable from the
// pointer path and removed here with ``front`` itself.) These two specializations put the
// element on the same footing as every other wrapped container's: an integer in, an integer
// out.
%fragment("OEVibeSolventModelTraits", "header", fragment="StdTraits",
          fragment=SWIG_AsVal_frag(int), fragment=SWIG_From_frag(int)) {
namespace swig {
template <>
struct traits_asptr<OEVIBE::SolventModel> {
    static int asptr(PyObject* obj, OEVIBE::SolventModel** val) {
        int number = 0;
        const int res = SWIG_AsVal(int)(obj, &number);
        if (SWIG_IsOK(res) && val != 0) {
            // The caller owns the value, as the SWIG_NEWOBJ result says.
            *val = new OEVIBE::SolventModel(static_cast<OEVIBE::SolventModel>(number));
            return SWIG_NEWOBJ;
        }
        return res;
    }
};
template <>
struct traits_from<OEVIBE::SolventModel> {
    static PyObject* from(const OEVIBE::SolventModel& val) {
        return SWIG_From(int)(static_cast<int>(val));
    }
};
}
}
%fragment("OEVibeSolventModelTraits");
OEVIBE_READONLY_VECTOR(std::vector<OEVIBE::MethodCapabilities>)
%template(MethodCapabilitiesVector) std::vector<OEVIBE::MethodCapabilities>;
OEVIBE_READONLY_VECTOR(std::vector<OEVIBE::KeywordSpec>)
%template(KeywordSpecVector) std::vector<OEVIBE::KeywordSpec>;

// A director callback's reference arguments point into storage C++ frees when the call ends:
// the MolecularSystem dies with the Calculation's per-call state, the EngineSettings with the
// Calculation, the EvaluationRequest with the single evaluation. SWIG's default hands Python a
// non-owning proxy over each, so an engine that keeps one -- the natural thing to do with the
// system it was asked to build a session for -- reads freed memory afterwards and gets silent
// garbage rather than an error. Each crosses as a copy Python owns instead. The snapshot copy
// is the only costly one, and it happens once per session, which is once per conformer.
%define OEVIBE_DIRECTORIN_COPY(Type...)
%typemap(directorin) const Type& {
    $input = SWIG_NewPointerObj(new Type($1), $descriptor(Type*),
                                SWIG_POINTER_OWN | %newpointer_flags);
}
%enddef
OEVIBE_DIRECTORIN_COPY(OEVIBE::MolecularSystem)
OEVIBE_DIRECTORIN_COPY(OEVIBE::EngineSettings)
OEVIBE_DIRECTORIN_COPY(OEVIBE::EvaluationRequest)

// %naturalvar gives a container member value semantics, but a member of a wrapped class type
// still comes back as a bare pointer into its parent with ownership off, and no ``_parent``
// keepalive can save it: ``find_method(engine, method).method`` outlives its temporary, and the
// parent of a member read inside a director callback may itself be a borrowed view. Within the
// engine structs these two cross as copies; the typemaps are cleared after Engine.h so that
// CalculationOptions' getters, which Python already holds through ``_parent``, keep the
// behavior Plan A tested.
%define OEVIBE_COPY_REF_OUT(Type...)
%typemap(out) Type& {
    $result = SWIG_NewPointerObj(new Type(*$1), $descriptor(Type*),
                                 SWIG_POINTER_OWN | %newpointer_flags);
}
%typemap(out) const Type& = Type&;
%enddef
OEVIBE_COPY_REF_OUT(OEVIBE::Method)
OEVIBE_COPY_REF_OUT(OEVIBE::Solvent)

// The engine structs are plain data that Python both fills and reads, so every member that is
// not a primitive crosses as a copy: a sequence or mapping going in, a tuple, dict or owned
// proxy coming out. Without this a container member would be settable only from a wrapped
// vector, and reading one off an EngineSettings or MolecularSystem borrowed for a single
// director callback would hand Python a view into storage C++ frees first.
%naturalvar OEVIBE::KeywordSpec::methods;
%naturalvar OEVIBE::MethodCapabilities::method;
%naturalvar OEVIBE::MethodCapabilities::solvent_models;
%naturalvar OEVIBE::MethodCapabilities::solvents;
%naturalvar OEVIBE::MethodCapabilities::elements;
%naturalvar OEVIBE::EngineSettings::method;
%naturalvar OEVIBE::EngineSettings::solvent;
%naturalvar OEVIBE::EngineSettings::keywords;
%naturalvar OEVIBE::Evaluation::gradient;
%naturalvar OEVIBE::Evaluation::hessian;
%naturalvar OEVIBE::Evaluation::charges;
%naturalvar OEVIBE::Evaluation::dipole;
%naturalvar OEVIBE::Evaluation::orbital_energies;
%naturalvar OEVIBE::Evaluation::bond_orders;
%naturalvar OEVIBE::Evaluation::messages;

// SWIG has no std::optional support, so find_method's return would reach Python as an opaque
// pointer with no destructor: unreadable, and leaked on every call. An absent method becomes
// None and a present one a MethodCapabilities copy the caller owns.
%typemap(out) std::optional<OEVIBE::MethodCapabilities> {
    if ($1.has_value()) {
        $result = SWIG_NewPointerObj(new OEVIBE::MethodCapabilities($1.value()),
                                     $descriptor(OEVIBE::MethodCapabilities*), SWIG_POINTER_OWN);
    } else {
        $result = SWIG_Py_Void();
    }
}

%shared_ptr(OEVIBE::Engine)
%shared_ptr(OEVIBE::EngineSession)

// A Python CreateSession returns a session whose last Python reference dies when the method
// returns, while C++ evaluates it for the rest of the conformer. SWIG's __disown__ would have
// the C++ director hold the proxy while the proxy (a %shared_ptr one) holds the C++ object: a
// cycle that leaks every session. Instead C++ receives a shared_ptr whose deleter owns the
// Python reference, so releasing the last shared_ptr frees the proxy, which frees the session.
// The deleter takes the GIL: spike S1 found sessions are destroyed inside RunConformers, which
// runs with the GIL released.
%typemap(directorout) std::shared_ptr<OEVIBE::EngineSession> {
    void* argp = 0;
    int newmem = 0;
    const int res = SWIG_ConvertPtrAndOwn($input, &argp,
                                          $descriptor(std::shared_ptr<OEVIBE::EngineSession>*),
                                          0, &newmem);
    if (!SWIG_IsOK(res) || argp == NULL) {
        Swig::DirectorTypeMismatchException::raise(
            SWIG_ErrorType(SWIG_TypeError), "CreateSession must return an EngineSession");
    }
    std::shared_ptr<OEVIBE::EngineSession>* owned =
        reinterpret_cast<std::shared_ptr<OEVIBE::EngineSession>*>(argp);
    OEVIBE::EngineSession* session = owned->get();
    // A Python object of a derived wrapped class converts through a cast that allocates the
    // shared_ptr it hands back; the caller owns that copy, and the proxy still holds the one
    // that keeps the session alive.
    if ((newmem & SWIG_CAST_NEW_MEMORY) != 0) {
        delete owned;
    }
    if (session == NULL) {
        Swig::DirectorTypeMismatchException::raise(SWIG_ErrorType(SWIG_ValueError),
                                                   "CreateSession returned a null session");
    }
    PyObject* keepalive = $input;
    Py_INCREF(keepalive);
    std::shared_ptr<void> guard(static_cast<void*>(0), [keepalive](void*) {
        SWIG_PYTHON_THREAD_BEGIN_BLOCK;
        Py_DECREF(keepalive);
        SWIG_PYTHON_THREAD_END_BLOCK;
    });
    $result = std::shared_ptr<OEVIBE::EngineSession>(guard, session);
}
%include "oevibe/Engine.h"
%clear OEVIBE::Method&;
%clear const OEVIBE::Method&;
%clear OEVIBE::Solvent&;
%clear const OEVIBE::Solvent&;
// Indexing an EngineVector hands back an owned shared_ptr copy, not a pointer into the
// vector's storage, so this one needs only the usual hardening.
OEVIBE_SAFE_VECTOR(std::vector<std::shared_ptr<OEVIBE::Engine> >)
%template(EngineVector) std::vector<std::shared_ptr<OEVIBE::Engine> >;

// %shared_ptr keeps a registered engine's C++ director alive, but a director's ``self`` is
// borrowed, so the last Python reference going away leaves every later callback -- the first
// of which is the GetName inside the next Calculation -- reading freed memory. The module
// therefore keeps every engine it registers for the life of the process (spec 8.7). Nothing is
// ever dropped: registering over a name replaces it in the registry, but a Calculation built
// earlier still holds a shared_ptr to the engine it resolved and goes on calling it.
//
// The keepalive is taken BEFORE the native call, not after. Registration mutates a
// process-wide registry, so an append that failed afterwards -- a MemoryError, an asynchronous
// exception delivered between the two -- would raise out of a registry that already owns the
// engine, and unwinding a temporary would then free the very Python object the director
// borrows. Taking it first inverts the failure: the registry is never touched, and the only
// cost is that an engine the registry goes on to reject stays retained. That is bounded memory
// under user error. Nothing undoes the append on a failed Register, because a compensating pop
// would open the same window it closed.
//
// The one exemption is a _MolFuncEngine, which is native C++ holding a C++ copy of an OpenEye
// potential and borrows no Python state at all; retaining it would pin that copy for the life
// of the process on every re-registration of the same name. Retention stays the default for
// everything else, including anything the predicate cannot classify.
%pythonprepend OEVIBE::EngineRegistry::Register %{
    if not _is_exact_molfunc_engine(engine):
        _REGISTERED_ENGINES.append(engine)
%}
%include "oevibe/EngineRegistry.h"

%include "oevibe/Constraints.h"
%include "oevibe/Options.h"
// EngineMolFunc reaches Python only as the engine behind as_oemolfunc: its OEMolFunc2 base means
// nothing across the runtime boundary, so the raw-pointer operators stay hidden and Python uses
// the vector-based Evaluate (spec 8.8).
//
// 402 rather than 401: the module forward-declares OEMolPotential::OEMolFunc2 above, so SWIG
// reports an incomplete base rather than an unknown one. Leaving that declaration out to get
// 401 instead is not an option -- the cross-runtime typemaps need the name.
%warnfilter(402) OEVIBE::EngineMolFunc;
%rename("_EngineMolFunc") OEVIBE::EngineMolFunc;
%ignore OEVIBE::EngineMolFunc::operator();
%ignore OEVIBE::EngineMolFunc::GetFComponents;
// CreateCopy stays wrapped: it is how as_oemolfunc copies a function without holding on to the
// caller's mutable CalculationOptions. The C++ constructor already copied the options, so the
// copy is insulated from anything the caller does to theirs afterwards. %newobject because the
// returned pointer is a fresh heap object Python must own.
%newobject OEVIBE::EngineMolFunc::CreateCopy;
// SWIG's Python backend does not wrap nested classes (325). Left nested, Evaluate's by-value
// return arrives as an opaque SwigPyObject with no attributes and no destructor -- every call
// prints "no destructor found" and leaks the struct. Flattening gives it a real proxy; the name
// is private because nothing outside as_oemolfunc has any use for it.
%feature("flatnested") OEVIBE::EngineMolFunc::FunctionValue;
%rename("_EngineMolFuncValue") OEVIBE::EngineMolFunc::FunctionValue;
%naturalvar OEVIBE::EngineMolFunc::FunctionValue::gradient;
// Read-only like every other value this module hands to Python: C++ fills these and Python only
// reads them, so a setter would be a write with nowhere to go. Named rather than a %immutable
// block, because the rest of OEAdapters.h is classes, not values.
%immutable OEVIBE::EngineMolFunc::FunctionValue::energy;
%immutable OEVIBE::EngineMolFunc::FunctionValue::gradient;
// The Python factory oevibe.MolFuncEngine upcasts the potential before calling this; see the
// OEMolFunc1 typemap above for why a direct call would be wrong.
%rename("_MolFuncEngine") OEVIBE::MolFuncEngine;
%shared_ptr(OEVIBE::MolFuncEngine)
%include "oevibe/OEAdapters.h"
// The member variables of the result structs are filled in C++, and Python receives them as
// views into a result, so they are exposed without setters. A vector member would otherwise be
// returned as a mutable DoubleVector view into the result; %naturalvar returns a tuple copy.
%naturalvar OEVIBE::Thermochemistry::frequencies;
%naturalvar OEVIBE::Thermochemistry::principal_moments;
%immutable;
%include "oevibe/Result.h"
%mutable;
// Optimize, Frequencies, and Scan run with the GIL released and read their options throughout,
// so the wrappers convert them into private copies first; argument conversion runs before the
// GIL is released, so no other Python thread can edit the caller's object mid-copy or mid-run.
// The typemaps are cleared after Calculation.h so no other function with an ``options``
// parameter picks them up.
%typemap(in) const OEVIBE::OptimizerOptions& options
    (void *argp = 0, int res = 0, OEVIBE::OptimizerOptions snapshot) {
    res = SWIG_ConvertPtr($input, &argp, $descriptor, 0);
    if (!SWIG_IsOK(res)) {
        SWIG_exception_fail(SWIG_ArgError(res), "Expected OptimizerOptions object.");
    }
    if (!argp) {
        SWIG_exception_fail(SWIG_NullReferenceError, "Null reference.");
    }
    snapshot = *reinterpret_cast< $1_ltype >(argp);
    $1 = &snapshot;
}
%typemap(in) const OEVIBE::ThermoOptions& options
    (void *argp = 0, int res = 0, OEVIBE::ThermoOptions snapshot) {
    res = SWIG_ConvertPtr($input, &argp, $descriptor, 0);
    if (!SWIG_IsOK(res)) {
        SWIG_exception_fail(SWIG_ArgError(res), "Expected ThermoOptions object.");
    }
    if (!argp) {
        SWIG_exception_fail(SWIG_NullReferenceError, "Null reference.");
    }
    snapshot = *reinterpret_cast< $1_ltype >(argp);
    $1 = &snapshot;
}
// Scan runs with the GIL released and reads its ScanOptions throughout, so it gets the same
// private copy.
%typemap(in) const OEVIBE::ScanOptions& options
    (void *argp = 0, int res = 0, OEVIBE::ScanOptions snapshot) {
    res = SWIG_ConvertPtr($input, &argp, $descriptor, 0);
    if (!SWIG_IsOK(res)) {
        SWIG_exception_fail(SWIG_ArgError(res), "Expected ScanOptions object.");
    }
    if (!argp) {
        SWIG_exception_fail(SWIG_NullReferenceError, "Null reference.");
    }
    snapshot = *reinterpret_cast< $1_ltype >(argp);
    $1 = &snapshot;
}
// Entry-point engines have to be in the registry before a Calculation resolves its method, and
// spec 8.7 loads them lazily on the first registry query. Constructing a Calculation is such a
// query, and it is the one path into the registry that does not go through oevibe.engines.
%pythonprepend OEVIBE::Calculation::Calculation %{
    from oevibe.engines import ensure_engines_loaded as _ensure_engines_loaded

    _ensure_engines_loaded()
%}
%include "oevibe/Calculation.h"
%clear const OEVIBE::OptimizerOptions& options;
%clear const OEVIBE::ThermoOptions& options;
%clear const OEVIBE::ScanOptions& options;

// A declaration takes its features when it is parsed, so this lifts the %ignore only for the
// %extend below, not for the header's const-reference method parsed above.
%rename("%s") OEVIBE::Calculation::GetOptions;
%extend OEVIBE::Calculation {
    OEVIBE::CalculationOptions GetOptions() const {
        return $self->GetOptions();
    }
}

%extend OEVIBE::Provenance {
    std::map<std::string, std::string> GetResolvedOptions() const {
        return $self->resolved_options;
    }
}

%extend OEVIBE::ConstraintSet {
%pythoncode %{
    def FreezeAtoms(self, mol, predicate):
        """Freeze every atom of ``mol`` that matches an OpenEye atom predicate.

        The predicate is evaluated here in Python, so built-in and Python-defined predicates
        both work and no predicate crosses into C++ (spec 6.6).
        """
        self.FreezeAtomIndices([atom.GetIdx() for atom in mol.GetAtoms(predicate)])
%}
}

%extend OEVIBE::ResultSet {
%pythoncode %{
    def __len__(self):
        return self.Size()

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.Get(index)

    def __iter__(self):
        return (self[i] for i in range(len(self)))
%}
}

%extend OEVIBE::ScanResult {
%pythoncode %{
    def GetMolecule(self):
        """Return an ``openeye.oechem.OEMol`` with one conformer per point that has an energy.

        Conformers follow grid order and carry the SD data ``ScanAngles``, ``Energy``,
        ``RelativeEnergy``, and ``Status`` under the ``OEVibe.<method>.<solvent-tag>.`` prefix.
        """
        from openeye import oechem

        mol = oechem.OEMol()
        self.CopyMolecule(mol)
        return mol
%}
}

// ============================================================================
// Module-level Python code: enums and exception aliases
// ============================================================================
%pythoncode %{
import enum as _enum

__version__ = "0.1.0"

# Every engine ever registered from Python except the one exempt native type, kept alive for
# the life of the process: a registered engine's C++ director borrows its Python object, so
# dropping the last reference would leave the registry calling into freed memory (see the
# Register keepalive in swig/oevibe.i).
_REGISTERED_ENGINES = []


def _is_exact_molfunc_engine(engine):
    """:returns: Whether ``engine`` is exactly the native ``_MolFuncEngine`` proxy.

    Exact type, like the ``_oevibe_is_exact_molfunc1`` guard in swig/oevibe.i, and for the same
    reason: ``isinstance`` would also answer yes for a Python subclass, which carries Python
    state of its own and must keep being retained.
    """
    return type(engine) is _MolFuncEngine

OEVibeError = _oevibe.OEVibeError
OptionsError = _oevibe.OptionsError
EngineUnavailableError = _oevibe.EngineUnavailableError
InputError = _oevibe.InputError
UnsupportedError = _oevibe.UnsupportedError


class Preset(_enum.IntEnum):
    FAST = Preset_FAST
    STANDARD = Preset_STANDARD
    THOROUGH = Preset_THOROUGH


class SolventModel(_enum.IntEnum):
    AUTO = SolventModel_AUTO
    ALPB = SolventModel_ALPB
    GBSA = SolventModel_GBSA
    SHEFFIELD = SolventModel_SHEFFIELD


class Property(_enum.IntEnum):
    CHARGES = Property_CHARGES
    DIPOLE = Property_DIPOLE
    ORBITAL_ENERGIES = Property_ORBITAL_ENERGIES
    BOND_ORDERS = Property_BOND_ORDERS


class PartialChargePolicy(_enum.IntEnum):
    NONE = PartialChargePolicy_NONE
    ASSIGN_AM1BCC_ELF10 = PartialChargePolicy_ASSIGN_AM1BCC_ELF10
    USE_INPUT = PartialChargePolicy_USE_INPUT


class KeywordType(_enum.IntEnum):
    BOOL = KeywordType_BOOL
    INT = KeywordType_INT
    DOUBLE = KeywordType_DOUBLE
    STRING = KeywordType_STRING


class Algorithm(_enum.IntEnum):
    AUTO = Algorithm_AUTO
    ANC_RFO = Algorithm_ANC_RFO
    LBFGS = Algorithm_LBFGS


class ConformerStatus(_enum.IntEnum):
    OK = ConformerStatus_OK
    NOT_CONVERGED = ConformerStatus_NOT_CONVERGED
    ENGINE_ERROR = ConformerStatus_ENGINE_ERROR
    UNSUPPORTED = ConformerStatus_UNSUPPORTED


class ScanPointStatus(_enum.IntEnum):
    OK = ScanPointStatus_OK
    NOT_CONVERGED = ScanPointStatus_NOT_CONVERGED
    ENGINE_ERROR = ScanPointStatus_ENGINE_ERROR
    UNSUPPORTED = ScanPointStatus_UNSUPPORTED
    NOT_ATTEMPTED = ScanPointStatus_NOT_ATTEMPTED
%}
