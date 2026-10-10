#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "oevibe/EngineRegistry.h"
#include "oevibe/Error.h"
#include "test_support.h"

using namespace OEVIBE;

TEST(EngineRegistryTest, RegisterFindAndList) {
    auto engine = test::register_fake("fake-registry-basic");
    EngineRegistry& registry = EngineRegistry::Instance();
    EXPECT_EQ(registry.Find("fake-registry-basic"), engine);
    EXPECT_EQ(registry.Find("no-such-engine"), nullptr);
    bool listed = false;
    for (const auto& item : registry.List()) {
        listed = listed || item->GetName() == "fake-registry-basic";
    }
    EXPECT_TRUE(listed);
}

TEST(EngineRegistryTest, DuplicateNamesNeedReplace) {
    test::register_fake("fake-registry-dup");
    auto second = std::make_shared<test::FakeEngine>("fake-registry-dup");
    EXPECT_THROW(EngineRegistry::Instance().Register(second), OptionsError);
    EXPECT_NO_THROW(EngineRegistry::Instance().Register(second, true));
    EXPECT_EQ(EngineRegistry::Instance().Find("fake-registry-dup"), second);
}

TEST(EngineRegistryTest, BuiltinNamesAreReserved) {
    for (const char* name : {"oeff", "tblite", "xtb"}) {
        EXPECT_TRUE(is_builtin_engine_name(name));
        EXPECT_THROW(
            EngineRegistry::Instance().Register(std::make_shared<test::FakeEngine>(name), true),
            OptionsError);
    }
    EXPECT_THROW(EngineRegistry::Instance().Register(nullptr), OptionsError);
}

TEST(EngineRegistryTest, ResolvesACustomMethod) {
    auto engine = std::make_shared<test::FakeEngine>(
        "fake-resolve", std::vector<MethodCapabilities>{test::fake_capabilities("resolve-only")});
    EngineRegistry::Instance().Register(engine, true);
    EXPECT_EQ(EngineRegistry::Instance().ResolveForMethod(Method("resolve-only")), engine);
}

TEST(EngineRegistryTest, AmbiguousOrMissingMethodsAreOptionsErrors) {
    for (const char* name : {"fake-ambiguous-a", "fake-ambiguous-b"}) {
        EngineRegistry::Instance().Register(
            std::make_shared<test::FakeEngine>(
                name, std::vector<MethodCapabilities>{test::fake_capabilities("ambiguous")}),
            true);
    }
    EXPECT_THROW(EngineRegistry::Instance().ResolveForMethod(Method("ambiguous")), OptionsError);
    EXPECT_THROW(EngineRegistry::Instance().ResolveForMethod(Method("nobody-provides-this")),
                 OptionsError);
}

// Plan B registers tblite; until then a built-in GFN method has no engine.
TEST(EngineRegistryTest, BuiltinMethodWithoutItsEngineIsUnavailable) {
    EXPECT_THROW(EngineRegistry::Instance().ResolveForMethod(Method(GFN2_XTB)),
                 EngineUnavailableError);
}

// Registration, lookup, and listing race from several threads; ThreadSanitizer checks the
// registry's locking.
TEST(EngineRegistryTest, ConcurrentRegisterFindAndListAreSafe) {
    constexpr int THREAD_COUNT = 8;
    constexpr int ROUNDS = 50;
    std::vector<std::thread> workers;
    for (int t = 0; t < THREAD_COUNT; ++t) {
        workers.emplace_back([t] {
            const std::string name = "fake-concurrent-" + std::to_string(t);
            for (int round = 0; round < ROUNDS; ++round) {
                EngineRegistry::Instance().Register(std::make_shared<test::FakeEngine>(name), true);
                EXPECT_NE(EngineRegistry::Instance().Find(name), nullptr);
                EXPECT_FALSE(EngineRegistry::Instance().List().empty());
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
    for (int t = 0; t < THREAD_COUNT; ++t) {
        EXPECT_NE(EngineRegistry::Instance().Find("fake-concurrent-" + std::to_string(t)), nullptr);
    }
}
