#ifndef OEVIBE_ERROR_H
#define OEVIBE_ERROR_H

#include <stdexcept>

namespace OEVIBE {

/// Base class of every exception OEVibe throws.
class OEVibeError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/// An option value or combination is invalid (spec 6.3).
class OptionsError : public OEVibeError {
public:
    using OEVibeError::OEVibeError;
};

/// The resolved engine or method cannot run here; the message carries the reason (spec 8.2).
class EngineUnavailableError : public OEVibeError {
public:
    using OEVibeError::OEVibeError;
};

/// The input molecule cannot be used as given (spec 6.1).
class InputError : public OEVibeError {
public:
    using OEVibeError::OEVibeError;
};

/// Thrown by an engine's ``CreateSession`` for a system it cannot handle. A calculation reports
/// it as ``UNSUPPORTED`` and never lets it escape (spec 8.1).
class UnsupportedError : public OEVibeError {
public:
    using OEVibeError::OEVibeError;
};

}  // namespace OEVIBE

#endif  // OEVIBE_ERROR_H
