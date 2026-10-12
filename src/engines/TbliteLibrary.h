#ifndef OEVIBE_ENGINES_TBLITELIBRARY_H
#define OEVIBE_ENGINES_TBLITELIBRARY_H

#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include "engines/RuntimeLibrary.h"

namespace OEVIBE {
namespace detail {

/// ``tblite_get_version()`` encodes a release as major * 10000 + minor * 100 + patch.
/// :returns: Whether the release is 0.6.x or 0.7.x, the series whose C API OEVibe uses (spec 8.4).
bool tblite_version_supported(int version);

/// :returns: ``major.minor.patch`` of an encoded release, for example ``0.6.0`` for 600.
std::string tblite_version_text(int version);

/// :returns: A reason when <prefix>/conda-meta records tblite 0.6.x with dftd4 4.2.0 build 1,
///     where <prefix> is the parent of the library's directory; nothing otherwise. With that
///     pairing a GFN2-xTB single point never returns (it hangs inside dftd4).
std::optional<std::string> known_bad_tblite_pairing(const std::string& library_path);

/// Serializes every libtblite call in the process (spec 8.3). Every TbliteEngine instance shares
/// it, because instances may load the same library.
std::mutex& tblite_mutex();

/// The libtblite C API functions OEVibe calls. tblite's handles are pointers to opaque structs,
/// so each is ``void*`` here. Written from the C ABI, not from tblite's headers (spec 3).
struct TbliteApi {
    int (*get_version)();
    void* (*new_context)();
    void (*delete_context)(void** context);
    int (*check_context)(void* context);
    void (*get_context_error)(void* context, char* buffer, const int* buffer_size);
    void (*set_context_verbosity)(void* context, int verbosity);
    void* (*new_error)();
    void (*delete_error)(void** error);
    int (*check_error)(void* error);
    void (*clear_error)(void* error);
    void (*get_error)(void* error, char* buffer, const int* buffer_size);
    void* (*new_structure)(void* error, int num_atoms, const int* numbers, const double* positions,
                           const double* charge, const int* unpaired, const double* lattice,
                           const bool* periodic);
    void (*delete_structure)(void** structure);
    void (*update_structure_geometry)(void* error, void* structure, const double* positions,
                                      const double* lattice);
    void* (*new_gfn2_calculator)(void* context, void* structure, void* config);
    void* (*new_gfn1_calculator)(void* context, void* structure, void* config);
    void (*delete_calculator)(void** calculator);
    void (*set_calculator_accuracy)(void* context, void* calculator, double accuracy);
    void (*set_calculator_max_iter)(void* context, void* calculator, int max_iterations);
    void (*set_calculator_temperature)(void* context, void* calculator, double temperature);
    void* (*new_alpb_solvation_solvent)(void* error, void* structure, char* solvent, int version,
                                        int reference_state);
    void (*calculator_push_back)(void* context, void* calculator, void** container);
    void (*delete_container)(void** container);
    void* (*new_result)();
    void (*delete_result)(void** result);
    void (*get_singlepoint)(void* context, void* structure, void* calculator, void* result);
    void (*get_result_energy)(void* error, void* result, double* energy);
    void (*get_result_gradient)(void* error, void* result, double* gradient);
    void (*get_result_number_of_orbitals)(void* error, void* result, int* orbitals);
    void (*get_result_number_of_spins)(void* error, void* result, int* spins);
    void (*get_result_charges)(void* error, void* result, double* charges);
    void (*get_result_dipole)(void* error, void* result, double* dipole);
    void (*get_result_bond_orders)(void* error, void* result, double* bond_orders);
    void (*get_result_orbital_energies)(void* error, void* result, double* energies);
    void (*get_result_orbital_occupations)(void* error, void* result, double* occupations);
};

/// Resolves every function of ``api`` from ``library``.
/// :returns: The names of missing symbols, comma-separated, or an empty string.
std::string bind_tblite_api(const RuntimeLibrary& library, TbliteApi& api);

/// A loaded, version-checked libtblite. Sessions share it, so the library stays open while any
/// session lives.
struct TbliteLibrary {
    std::shared_ptr<RuntimeLibrary> library;
    TbliteApi api{};
    int version = 0;
};

}  // namespace detail
}  // namespace OEVIBE

#endif  // OEVIBE_ENGINES_TBLITELIBRARY_H
