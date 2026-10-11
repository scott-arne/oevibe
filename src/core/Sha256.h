#ifndef OEVIBE_CORE_SHA256_H
#define OEVIBE_CORE_SHA256_H

#include <map>
#include <string>

namespace OEVIBE {
namespace detail {

/// :returns: The lowercase hexadecimal SHA-256 digest (FIPS 180-4) of ``data``.
std::string sha256_hex(const std::string& data);

/// :returns: SHA-256 of the length-prefixed serialization in key order. Each entry is encoded as
///     ``<key_len>:<key><value_len>:<value>`` with lengths as decimal integers (e.g.
///     ``{"a": "xy"}`` becomes ``1:a2:xy``). This encoding is injective for arbitrary keys and
///     values, unlike ``key=value\n`` which collides (e.g. ``{a: "x\nb=y"}`` and ``{a: "x",
///     b: "y"}`` both serialize to ``a=x\nb=y\n``). The digest is recorded in ``Provenance`` and
///     used later as a cache key (spec 5.7).
std::string options_digest(const std::map<std::string, std::string>& resolved);

}  // namespace detail
}  // namespace OEVIBE

#endif  // OEVIBE_CORE_SHA256_H
