#ifndef OEVIBE_SOLVENT_H
#define OEVIBE_SOLVENT_H

#include <string>

namespace OEVIBE {

// NOLINTBEGIN(performance-enum-size)
/// Implicit-solvent model; AUTO resolves per method (spec 9).
enum class SolventModel { AUTO, ALPB, GBSA, SHEFFIELD };
// NOLINTEND(performance-enum-size)

/// :returns: ``auto``, ``alpb``, ``gbsa``, or ``sheffield``.
/// :raises OptionsError: For a value outside the enumerators.
std::string solvent_model_name(SolventModel model);

/// A named implicit solvent, or gas phase when default-constructed (spec 9).
class Solvent {
public:
    /// Gas phase.
    Solvent();
    /// :param name: A Minnesota Solvent Descriptor Database name such as ``water``; stored
    ///     lowercase. Engines validate whether they can model it.
    /// :raises OptionsError: When the name is empty or contains whitespace, or the model is
    ///     outside the enumerators.
    explicit Solvent(const std::string& name, SolventModel model = SolventModel::AUTO);

    bool IsGasPhase() const;
    /// :returns: The name, or an empty string for gas phase.
    const std::string& GetName() const;
    SolventModel GetModel() const;
    /// :returns: ``gas`` or ``<model>-<name>``, the SD tag component (spec 6.5).
    /// :raises OptionsError: When a named solvent still has the AUTO model.
    std::string GetTag() const;
    /// :returns: A copy with a concrete model; gas phase is returned unchanged.
    Solvent WithModel(SolventModel model) const;

    bool operator==(const Solvent& other) const;

private:
    std::string name_;
    SolventModel model_ = SolventModel::AUTO;
};

}  // namespace OEVIBE

#endif  // OEVIBE_SOLVENT_H
