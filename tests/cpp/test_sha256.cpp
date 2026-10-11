#include <gtest/gtest.h>

#include <map>
#include <string>

#include "core/Sha256.h"

using namespace OEVIBE;

// FIPS 180-4 test vectors.
TEST(Sha256Test, KnownVectors) {
    EXPECT_EQ(detail::sha256_hex(""),
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EXPECT_EQ(detail::sha256_hex("abc"),
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    EXPECT_EQ(detail::sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

TEST(Sha256Test, OptionsDigestIsCanonical) {
    const std::map<std::string, std::string> a = {{"method", "mmff94"}, {"preset", "standard"}};
    std::map<std::string, std::string> b;
    b["preset"] = "standard";
    b["method"] = "mmff94";
    EXPECT_EQ(detail::options_digest(a), detail::options_digest(b));
    EXPECT_EQ(detail::options_digest(a).size(), 64u);
    b["preset"] = "fast";
    EXPECT_NE(detail::options_digest(a), detail::options_digest(b));
}
