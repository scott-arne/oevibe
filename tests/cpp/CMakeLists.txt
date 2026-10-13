include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(googletest)

# One executable; each component adds its test_<component>.cpp here.
add_executable(oevibe_tests
    test_main.cpp
    test_oevibe.cpp
    test_error.cpp
    test_units.cpp
    test_method.cpp
    test_preset.cpp
    test_solvent.cpp
    test_constraints.cpp
    test_options.cpp
    test_molecular_system.cpp
    test_input_prep.cpp
    test_engine.cpp
    test_engine_registry.cpp
    test_molfunc_engine.cpp
    test_oeff_engine.cpp
    test_result.cpp
    test_sha256.cpp
    test_calculation.cpp
    test_calculation_optimize.cpp
    test_geometry.cpp
    test_constraint_math.cpp
    test_model_hessian.cpp
    test_lbfgs.cpp
    test_anc_rfo.cpp
    test_optimizer.cpp
    test_runtime_library.cpp
    test_tblite_engine.cpp
    test_electronic_properties.cpp
    test_xtb_engine.cpp
    test_engine_integration.cpp
    test_hessian.cpp
    test_frequencies.cpp
    test_thermochemistry.cpp
    test_calculation_frequencies.cpp
    test_scan_grid.cpp
    test_wavefront.cpp
)

target_link_libraries(oevibe_tests
    PRIVATE
        oevibe
        oevibe_eigen
        OpenEye::OEOmega2
        GTest::gtest
)

# Tests reach internal headers under src/ to test components directly.
target_include_directories(oevibe_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)

include(GoogleTest)

# The OpenEye static libraries are not instrumented, so AddressSanitizer reports false
# container-overflow errors from libc++ containers inside OpenEye frames. Only builds compiled
# with -fsanitize=address get the option.
set(OEVIBE_TEST_PROPERTIES "")
if(CMAKE_CXX_FLAGS MATCHES "-fsanitize=[^ ]*address")
    set(OEVIBE_TEST_PROPERTIES PROPERTIES ENVIRONMENT "ASAN_OPTIONS=detect_container_overflow=0")
endif()
gtest_discover_tests(oevibe_tests DISCOVERY_TIMEOUT 120 ${OEVIBE_TEST_PROPERTIES})
