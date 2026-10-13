#ifndef OEVIBE_ENGINES_XTBLIBRARY_H
#define OEVIBE_ENGINES_XTBLIBRARY_H

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "engines/RuntimeLibrary.h"

namespace OEVIBE {
namespace detail {

/// The one xtb C API version OEVibe drives, 1.0.0 (xtb 6.7.x). C API 2.0 changes
/// ``xtb_setSolvent`` and adds model selection (spec 8.5).
constexpr int XTB_SUPPORTED_API_VERSION = 10000;

/// The GFN0-xTB parameter file of an xtb installation.
constexpr const char* XTB_GFN0_PARAMETER_FILE_NAME = "param_gfn0-xtb.txt";

/// Why ``gfn-ff`` is unavailable in sub-project 1 (spec 8.5; probe 2026-10-11).
extern const char* const XTB_GFNFF_UNAVAILABLE_REASON;

/// :returns: Whether an ``xtb_getAPIVersion()`` value is a C API OEVibe supports (10000 only).
bool xtb_api_version_supported(int version);

/// :returns: ``major.minor.patch`` of an ``xtb_getAPIVersion()`` value (10000 is ``1.0.0``).
std::string xtb_api_version_text(int version);

/// :param library_path: The path libxtb was loaded from.
/// :param xtbpath: The ``XTBPATH`` value: directories separated by ``:``, as xtb reads it.
/// :param xtbhome: The ``XTBHOME`` value: one directory.
/// :returns: Candidate GFN0-xTB parameter files in search order: each ``xtbpath`` directory,
///     then ``xtbhome``, then ``<prefix>/share/xtb`` where ``<prefix>`` is the parent of the
///     library's directory. Empty entries, and a library path without a directory, add nothing.
std::vector<std::string> xtb_gfn0_parameter_candidates(const std::string& library_path,
                                                       const std::string& xtbpath,
                                                       const std::string& xtbhome);

/// :returns: The first candidate that is a regular file, or "" when none is.
std::string xtb_gfn0_parameter_file(const std::string& library_path, const std::string& xtbpath,
                                    const std::string& xtbhome);

/// The same with ``XTBPATH`` and ``XTBHOME`` read from the process environment. OEVibe only
/// reads them: the file is passed to ``xtb_loadGFN0xTB``, so nothing sets a variable.
std::string xtb_gfn0_parameter_file(const std::string& library_path);

/// :returns: Why GFN0-xTB cannot run when no candidate exists, naming every path tried.
std::string xtb_gfn0_missing_reason(const std::string& library_path, const std::string& xtbpath,
                                    const std::string& xtbhome);

/// The same with ``XTBPATH`` and ``XTBHOME`` read from the process environment.
std::string xtb_gfn0_missing_reason(const std::string& library_path);

/// The process-wide mutex that serializes the libxtb calls OEVibe's xtb engine makes, the
/// per-engine mutex of spec 8.3. The xtb C API is not thread-safe, so every engine instance and
/// session holds it around each call, whichever libxtb file it loaded. It does not cover
/// libxtb's own dependencies: libxtb links libtblite and libdftd4, so a thread inside
/// TbliteEngine holding ``tblite_mutex()`` can enter those same images while this mutex is held.
std::mutex& xtb_mutex();

/// An opaque xtb C API handle (environment, molecule, calculator, or results).
using XtbHandle = void*;

/// The libxtb C API 1.0 entry points OEVibe calls, declared from the documented C ABI.
/// Positions are Bohr, energies Hartree, gradients Hartree/Bohr, and dipoles e Bohr.
struct XtbFunctions {
    int (*get_api_version)() = nullptr;
    XtbHandle (*new_environment)() = nullptr;
    void (*del_environment)(XtbHandle*) = nullptr;
    int (*check_environment)(XtbHandle) = nullptr;
    void (*get_error)(XtbHandle, char*, const int*) = nullptr;
    void (*set_verbosity)(XtbHandle, int) = nullptr;
    /// (env, natoms, numbers, positions, charge, uhf, lattice, periodic)
    XtbHandle (*new_molecule)(XtbHandle, const int*, const int*, const double*, const double*,
                              const int*, const double*, const bool*) = nullptr;
    void (*del_molecule)(XtbHandle*) = nullptr;
    /// (env, mol, positions, lattice)
    void (*update_molecule)(XtbHandle, XtbHandle, const double*, const double*) = nullptr;
    XtbHandle (*new_calculator)() = nullptr;
    void (*del_calculator)(XtbHandle*) = nullptr;
    /// (env, mol, calc, parameter file)
    void (*load_gfn0)(XtbHandle, XtbHandle, XtbHandle, char*) = nullptr;
    void (*set_accuracy)(XtbHandle, XtbHandle, double) = nullptr;
    /// (env, mol, calc, results)
    void (*singlepoint)(XtbHandle, XtbHandle, XtbHandle, XtbHandle) = nullptr;
    XtbHandle (*new_results)() = nullptr;
    void (*del_results)(XtbHandle*) = nullptr;
    void (*get_energy)(XtbHandle, XtbHandle, double*) = nullptr;
    void (*get_gradient)(XtbHandle, XtbHandle, double*) = nullptr;
    void (*get_dipole)(XtbHandle, XtbHandle, double*) = nullptr;
    void (*get_charges)(XtbHandle, XtbHandle, double*) = nullptr;
    void (*get_bond_orders)(XtbHandle, XtbHandle, double*) = nullptr;
    void (*get_nao)(XtbHandle, XtbHandle, int*) = nullptr;
    void (*get_orbital_eigenvalues)(XtbHandle, XtbHandle, double*) = nullptr;
    void (*get_orbital_occupations)(XtbHandle, XtbHandle, double*) = nullptr;
};

/// A loaded libxtb whose C API version OEVibe supports and whose entry points all resolved.
class XtbLibrary {
public:
    /// Checks ``xtb_getAPIVersion()`` (under ``xtb_mutex()``) and resolves every entry point.
    /// :returns: The library, or null with ``reason`` naming the unsupported version or every
    ///     missing entry point.
    static std::shared_ptr<const XtbLibrary> Open(std::shared_ptr<RuntimeLibrary> runtime,
                                                  std::string& reason);
    const XtbFunctions& Functions() const;
    /// :returns: The path the library was loaded from.
    const std::string& GetPath() const;
    int GetApiVersion() const;

private:
    XtbLibrary() = default;

    std::shared_ptr<RuntimeLibrary> runtime_;
    XtbFunctions functions_;
    int api_version_ = 0;
};

}  // namespace detail
}  // namespace OEVIBE

#endif  // OEVIBE_ENGINES_XTBLIBRARY_H
