cmake_minimum_required(VERSION 3.16)

if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()

project(oevibe VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(OEVIBE_BUILD_TESTS "Build tests" ON)
option(OEVIBE_BUILD_PYTHON "Build Python bindings" ON)
option(OEVIBE_UNIVERSAL2 "Build universal2 binary for macOS" OFF)
option(OEVIBE_USE_STABLE_ABI "Use Python stable ABI" ON)

if(APPLE AND OEVIBE_UNIVERSAL2)
    set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64" CACHE STRING "Build architectures" FORCE)
elseif(APPLE)
    unset(CMAKE_OSX_ARCHITECTURES CACHE)
endif()

if(APPLE AND NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS version")
endif()

# Dependencies
include(FetchContent)

FetchContent_Declare(
    cmake_openeye
    GIT_REPOSITORY https://github.com/scott-arne/cmake-openeye.git
    GIT_TAG v1.2.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(cmake_openeye)
list(APPEND CMAKE_MODULE_PATH "${cmake_openeye_SOURCE_DIR}")

if(OEVIBE_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(SWIG 4.0)
endif()

if(OPENEYE_USE_SHARED)
    find_package(OpenEyePython REQUIRED)
endif()

find_package(OpenEye REQUIRED)

# Library
set(OEVIBE_SOURCES
    src/oevibe.cpp
)

add_library(oevibe ${OEVIBE_SOURCES})

set_target_properties(oevibe PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(oevibe
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(oevibe
    PUBLIC
        OpenEye::OEChem
)

# Tests
if(OEVIBE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Python bindings
if(OEVIBE_BUILD_PYTHON AND SWIG_FOUND)
    add_subdirectory(swig)
endif()

# Install
install(TARGETS oevibe
    EXPORT oevibe-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/oevibe DESTINATION include)

install(EXPORT oevibe-targets
    FILE OEVIBETargets.cmake
    NAMESPACE OEVIBE::
    DESTINATION lib/cmake/oevibe
)
