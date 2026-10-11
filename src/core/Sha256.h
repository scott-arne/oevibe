#ifndef OEVIBE_CORE_SHA256_H
#define OEVIBE_CORE_SHA256_H

#include <map>
#include <string>

namespace OEVIBE {
namespace detail {

/// :returns: The lowercase hexadecimal SHA-256 digest (FIPS 180-4) of ``data``.
std::string sha256_hex(const std::string& data);

/// :returns: SHA-256 of the canonical serialization ``key=value\n`` in key order, the options
///     digest recorded in ``Provenance`` and used later as a cache key (spec 5.7).
std::string options_digest(const std::map<std::string, std::string>& resolved);

}  // namespace detail
}  // namespace OEVIBE

#endif  // OEVIBE_CORE_SHA256_H
