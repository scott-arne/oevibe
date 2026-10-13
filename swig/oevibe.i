// swig/oevibe.i
// SWIG interface file for oevibe Python bindings
%module(threads="1") _oevibe

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

// ============================================================================
// GIL: held by default, released only for Energy, Gradient, Optimize, and Frequencies
// ============================================================================
// A feature applies only to declarations parsed after it, so this block precedes every
// wrapper-generating directive. Plan A engines are native C++ and never call back into Python,
// so releasing the GIL around a calculation is safe. The constructor keeps the GIL: it copies
// the caller's CalculationOptions, which another Python thread could be editing, and it only
// validates, so there is no calculation to overlap with. Optimize and Frequencies read their
// options for the whole run, so their wrappers copy them while the GIL is still held (the
// typemaps before Calculation.h below).
%feature("nothreadallow", "1");
%threadallow OEVIBE::Calculation::Energy;
%threadallow OEVIBE::Calculation::Gradient;
%threadallow OEVIBE::Calculation::Optimize;
%threadallow OEVIBE::Calculation::Frequencies;

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
%pythonappend OEVIBE::CalculationOptions::GetMethod %{ val._parent = self %}
%pythonappend OEVIBE::CalculationOptions::GetSolvent %{ val._parent = self %}

// Energy, Gradient, Optimize, and Frequencies release the GIL (above), so the molecule they read
// must not be shared with another thread that is using it.
%feature("docstring") OEVIBE::Calculation
"Runs tasks over every conformer of a molecule with one resolved engine (spec 5.6).

Not thread-safe; use one per thread. Construction holds the GIL. Energy, Gradient, Optimize, and
Frequencies release it while they evaluate; Optimize and Frequencies copy their options before
releasing it. The same molecule must not be passed to calculations from more than one thread at
a time."

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
// Indexing returns a pointer into the vector's storage. The element's reference to the vector
// keeps that storage allocated but not in place, so every method that can reallocate, shrink,
// swap, or replace it would leave elements taken earlier dangling. Results hand out bond orders
// as copies and nothing in Python passes a vector to C++, so the vector is read-only here.
%ignore std::vector<OEVIBE::BondOrder>::append;
%ignore std::vector<OEVIBE::BondOrder>::push_back;
%ignore std::vector<OEVIBE::BondOrder>::insert;
%ignore std::vector<OEVIBE::BondOrder>::assign;
%ignore std::vector<OEVIBE::BondOrder>::resize;
%ignore std::vector<OEVIBE::BondOrder>::reserve;
%ignore std::vector<OEVIBE::BondOrder>::pop;
%ignore std::vector<OEVIBE::BondOrder>::pop_back;
%ignore std::vector<OEVIBE::BondOrder>::erase;
%ignore std::vector<OEVIBE::BondOrder>::clear;
%ignore std::vector<OEVIBE::BondOrder>::swap;
%ignore std::vector<OEVIBE::BondOrder>::__setitem__;
%ignore std::vector<OEVIBE::BondOrder>::__delitem__;
// SWIG releases before 4.5 also generate the Python 2 slice methods.
%ignore std::vector<OEVIBE::BondOrder>::__setslice__;
%ignore std::vector<OEVIBE::BondOrder>::__delslice__;
%template(BondOrderVector) std::vector<OEVIBE::BondOrder>;
%include "oevibe/Constraints.h"
%include "oevibe/Options.h"
// The member variables of the result structs are filled in C++, and Python receives them as
// views into a result, so they are exposed without setters. A vector member would otherwise be
// returned as a mutable DoubleVector view into the result; %naturalvar returns a tuple copy.
%naturalvar OEVIBE::Thermochemistry::frequencies;
%naturalvar OEVIBE::Thermochemistry::principal_moments;
%immutable;
%include "oevibe/Result.h"
%mutable;
// Optimize and Frequencies run with the GIL released and read their options throughout, so the
// wrappers convert them into private copies first; argument conversion runs before the GIL is
// released, so no other Python thread can edit the caller's object mid-copy or mid-run. The
// typemaps are cleared after Calculation.h so no other function with an ``options`` parameter
// picks them up.
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
%include "oevibe/Calculation.h"
%clear const OEVIBE::OptimizerOptions& options;
%clear const OEVIBE::ThermoOptions& options;

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

// ============================================================================
// Module-level Python code: enums and exception aliases
// ============================================================================
%pythoncode %{
import enum as _enum

__version__ = "0.1.0"

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


class Algorithm(_enum.IntEnum):
    AUTO = Algorithm_AUTO
    ANC_RFO = Algorithm_ANC_RFO
    LBFGS = Algorithm_LBFGS


class ConformerStatus(_enum.IntEnum):
    OK = ConformerStatus_OK
    NOT_CONVERGED = ConformerStatus_NOT_CONVERGED
    ENGINE_ERROR = ConformerStatus_ENGINE_ERROR
    UNSUPPORTED = ConformerStatus_UNSUPPORTED
%}
