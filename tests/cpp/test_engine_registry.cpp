#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "engines/Builtins.h"
#include "oevibe/EngineRegistry.h"
#include "oevibe/Error.h"
#include "test_support.h"

using namespace OEVIBE;

namespace {

/// Counts ``GetName`` calls so tests can tell when the registry runs engine code.
class NameCountingEngine : public test::FakeEngine {
public:
    using test::FakeEngine::FakeEngine;

    std::string GetName() const override {
        ++name_calls;
        return FakeEngine::GetName();
    }

    mutable std::atomic<int> name_calls{0};
};

// Access checking is part of substitution, so a private RegisterBuiltin selects the primary
// template.
template <typename T, typename = void>
struct HasPublicRegisterBuiltin : std::false_type {};

template <typename T>
struct HasPublicRegisterBuiltin<
    T, std::void_t<decltype(std::declval<T&>().RegisterBuiltin(std::shared_ptr<Engine>()))>>
    : std::true_type {};

}  // namespace

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

// A Python engine's GetName needs the GIL, so calling it while holding the registry mutex can
// deadlock against a thread that holds the GIL and waits for the mutex (Plan C). Lookups must
// therefore compare names cached at registration.
TEST(EngineRegistryTest, LookupsDoNotCallEngineCodeUnderTheLock) {
    auto counted = std::make_shared<NameCountingEngine>("fake-name-counted");
    EngineRegistry::Instance().Register(counted, true);
    test::register_fake("fake-name-counted-other");
    counted->name_calls = 0;

    EXPECT_EQ(EngineRegistry::Instance().Find("fake-name-counted"), counted);
    EXPECT_NE(EngineRegistry::Instance().Find("fake-name-counted-other"), nullptr);
    EXPECT_EQ(EngineRegistry::Instance().Find("no-such-engine"), nullptr);
    EXPECT_FALSE(EngineRegistry::Instance().List().empty());
    EXPECT_EQ(counted->name_calls.load(), 0);
}

// Built-in registration skips the reserved-name and duplicate checks, so a public entry point
// would let any caller take over a built-in engine's routing.
TEST(EngineRegistryTest, BuiltinRegistrationIsNotPublic) {
    EXPECT_FALSE(HasPublicRegisterBuiltin<EngineRegistry>::value);
}

TEST(EngineRegistryTest, BuiltinRegistrationRejectsANullEngine) {
    EXPECT_THROW(detail::BuiltinRegistration::Register(EngineRegistry::Instance(), nullptr),
                 OptionsError);
}
