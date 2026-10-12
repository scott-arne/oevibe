#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "engines/RuntimeLibrary.h"
#include "oevibe/Units.h"
#include "oevibe/engines/RuntimeLibraries.h"
#include "test_runtime_libraries.h"

using namespace OEVIBE;

namespace {

// A variable no real engine reads, so a developer's engine settings cannot leak in.
constexpr const char* TEST_VARIABLE = "OEVIBE_TEST_RUNTIME_LIBRARY";

// A library and function every supported platform has: libSystem lives in the dyld shared
// cache on macOS, and libm is a separate shared object on Linux.
#ifdef __APPLE__
constexpr const char* SYSTEM_LIBRARY = "/usr/lib/libSystem.B.dylib";
#else
constexpr const char* SYSTEM_LIBRARY = "libm.so.6";
#endif

detail::LibrarySearch test_search() {
    detail::LibrarySearch search;
    search.engine = "test";
    search.environment_variable = TEST_VARIABLE;
    search.file_names = {"liboevibe-test.1.dylib", "liboevibe-test.dylib"};
    return search;
}

/// :returns: The loader message recorded for ``candidate`` in a load error, or "" without one.
std::string message_for(const std::string& error, const std::string& candidate) {
    const std::string prefix = "\n  " + candidate + ": ";
    const std::size_t at = error.find(prefix);
    if (at == std::string::npos) {
        return "";
    }
    const std::size_t start = at + prefix.size();
    return error.substr(start, error.find('\n', start) - start);
}

}  // namespace

TEST(RuntimeLibraryTest, CandidatesFollowTheSpecOrder) {
    const test::ScopedEnvironment variable(TEST_VARIABLE, "/variable/liboevibe-test.dylib");
    const test::ScopedEnvironment conda("CONDA_PREFIX", "/conda");
    const test::ScopedPythonLibraryDirectory python("/python/lib");
    detail::LibrarySearch search = test_search();
    search.explicit_path = "/explicit/liboevibe-test.dylib";
    EXPECT_EQ(detail::library_candidates(search),
              (std::vector<std::string>{
                  "/explicit/liboevibe-test.dylib", "/variable/liboevibe-test.dylib",
                  "/conda/lib/liboevibe-test.1.dylib", "/conda/lib/liboevibe-test.dylib",
                  "/python/lib/liboevibe-test.1.dylib", "/python/lib/liboevibe-test.dylib",
                  "liboevibe-test.1.dylib", "liboevibe-test.dylib"}));
}

TEST(RuntimeLibraryTest, UnsetSourcesAreSkippedAndRepeatsDropped) {
    {
        const test::ScopedEnvironment variable(TEST_VARIABLE, nullptr);
        const test::ScopedEnvironment conda("CONDA_PREFIX", nullptr);
        const test::ScopedPythonLibraryDirectory python("");
        EXPECT_EQ(detail::library_candidates(test_search()),
                  (std::vector<std::string>{"liboevibe-test.1.dylib", "liboevibe-test.dylib"}));
    }
    // An empty variable counts as unset, and a Python prefix that is the conda environment adds
    // nothing new.
    const test::ScopedEnvironment variable(TEST_VARIABLE, "");
    const test::ScopedEnvironment conda("CONDA_PREFIX", "/env/");
    const test::ScopedPythonLibraryDirectory python("/env/lib");
    EXPECT_EQ(detail::library_candidates(test_search()),
              (std::vector<std::string>{"/env/lib/liboevibe-test.1.dylib",
                                        "/env/lib/liboevibe-test.dylib", "liboevibe-test.1.dylib",
                                        "liboevibe-test.dylib"}));
}

TEST(RuntimeLibraryTest, PlatformFileNames) {
#ifdef __APPLE__
    EXPECT_EQ(detail::library_file_names("tblite", "0"),
              (std::vector<std::string>{"libtblite.0.dylib", "libtblite.dylib"}));
    EXPECT_EQ(detail::library_file_names("xtb", "6"),
              (std::vector<std::string>{"libxtb.6.dylib", "libxtb.dylib"}));
#else
    EXPECT_EQ(detail::library_file_names("tblite", "0"),
              (std::vector<std::string>{"libtblite.so.0", "libtblite.so"}));
    EXPECT_EQ(detail::library_file_names("xtb", "6"),
              (std::vector<std::string>{"libxtb.so.6", "libxtb.so"}));
#endif
}

TEST(RuntimeLibraryTest, LoadsASystemLibraryAndResolvesAFunction) {
    detail::LibrarySearch search;
    search.engine = "system";
    search.explicit_path = SYSTEM_LIBRARY;
    std::string error = "stale";
    const std::shared_ptr<detail::RuntimeLibrary> library =
        detail::RuntimeLibrary::Load(search, error);
    ASSERT_NE(library, nullptr) << error;
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(library->GetPath(), SYSTEM_LIBRARY);
    void* symbol = library->Symbol("cos");
    ASSERT_NE(symbol, nullptr);
    const auto cosine = reinterpret_cast<double (*)(double)>(symbol);
    EXPECT_DOUBLE_EQ(cosine(0.0), 1.0);
    EXPECT_NEAR(cosine(Units::PI), -1.0, 1e-15);
    EXPECT_EQ(library->Symbol("oevibe_no_such_symbol"), nullptr);
    EXPECT_FALSE(detail::image_path_of(symbol).empty());
    EXPECT_EQ(detail::image_path_of(nullptr), "");
}

TEST(RuntimeLibraryTest, AMissingLibraryNamesEveryCandidate) {
    const test::ScopedEnvironment variable(TEST_VARIABLE, "/missing/variable/liboevibe-test.dylib");
    const test::ScopedEnvironment conda("CONDA_PREFIX", "/missing/conda");
    const test::ScopedPythonLibraryDirectory python("/missing/python/lib");
    detail::LibrarySearch search = test_search();
    search.explicit_path = "/missing/explicit/liboevibe-test.dylib";
    std::string error;
    EXPECT_EQ(detail::RuntimeLibrary::Load(search, error), nullptr);
    EXPECT_NE(error.find("cannot load the test library"), std::string::npos) << error;
    const std::vector<std::string> candidates = detail::library_candidates(search);
    ASSERT_EQ(candidates.size(), 8u);
    for (const std::string& candidate : candidates) {
        EXPECT_FALSE(message_for(error, candidate).empty()) << candidate << " in\n" << error;
    }
}

TEST(RuntimeLibraryTest, PythonLibraryDirectoryIsProcessWide) {
    const test::ScopedPythonLibraryDirectory python("/prefix/lib");
    EXPECT_EQ(get_python_library_directory(), "/prefix/lib");
    set_python_library_directory("");
    EXPECT_EQ(get_python_library_directory(), "");
}
