include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(googletest)

add_executable(oevibe_tests
    test_oevibe.cpp
)

target_link_libraries(oevibe_tests
    PRIVATE
        oevibe
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(oevibe_tests)
