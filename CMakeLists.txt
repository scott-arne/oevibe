cmake_minimum_required(VERSION 3.21)

if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()

project(oevibe VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(OEVIBE_BUILD_TESTS "Build tests" ON)
option(OEVIBE_BUILD_PYTHON "Build Python bindings (requires shared OpenEye libraries)" OFF)
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

# Eigen is header-only. SOURCE_SUBDIR names a directory without a CMakeLists.txt, so
# FetchContent only downloads the sources and Eigen's own test and install targets stay out.
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR oevibe-header-only
)
FetchContent_MakeAvailable(eigen)
add_library(oevibe_eigen INTERFACE)
target_include_directories(oevibe_eigen SYSTEM INTERFACE ${eigen_SOURCE_DIR})
# The static OpenEye libraries embed an older Eigen whose template instantiations (for example
# SelfAdjointEigenSolver<MatrixXd>) have the same mangled names but a different class layout.
# When optimized, ld64 coalesces our weak definitions onto OpenEye's, so a solver constructed
# with the 3.4 layout is driven by 3.3 code and corrupts the heap. Renaming the namespace gives
# every Eigen symbol of ours a distinct name; it must apply to every translation unit that
# includes Eigen, so it lives on this target rather than in the sources.
target_compile_definitions(oevibe_eigen INTERFACE Eigen=OEVibeEigen)

if(OEVIBE_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    # Required so that a Python build without SWIG fails at configure time instead of skipping
    # the module. The minimum matches the swig build requirement in pyproject.toml.
    find_package(SWIG 4.2.1 REQUIRED)
endif()

if(OPENEYE_USE_SHARED)
    find_package(OpenEyePython REQUIRED)
endif()

find_package(OpenEye REQUIRED)

# The Python module must link the openeye-toolkits wheel's shared libraries. With static OpenEye
# it would carry a second OpenEye runtime into the Python process, and objects passed between
# the two runtimes have their generic data (including SD tags) relabeled, because each runtime
# keeps its own tag registry. The check reads what FindOpenEye resolved on this configure rather
# than the cached OpenEye_LIBRARY_TYPE, which a reconfigure would keep stale.
if(OEVIBE_BUILD_PYTHON)
    get_target_property(_OEVIBE_OECHEM_LOCATION OpenEye::OEChem IMPORTED_LOCATION)
    if(NOT _OEVIBE_OECHEM_LOCATION AND CMAKE_BUILD_TYPE)
        string(TOUPPER "${CMAKE_BUILD_TYPE}" _OEVIBE_CONFIG)
        get_target_property(_OEVIBE_OECHEM_LOCATION OpenEye::OEChem
            IMPORTED_LOCATION_${_OEVIBE_CONFIG})
    endif()
    set(_OEVIBE_OECHEM_EXTENSION "")
    if(_OEVIBE_OECHEM_LOCATION)
        cmake_path(GET _OEVIBE_OECHEM_LOCATION EXTENSION LAST_ONLY _OEVIBE_OECHEM_EXTENSION)
    endif()
    if(NOT OPENEYE_USE_SHARED
            OR NOT "${_OEVIBE_OECHEM_EXTENSION}" STREQUAL "${CMAKE_SHARED_LIBRARY_SUFFIX}")
        message(FATAL_ERROR "OEVIBE_BUILD_PYTHON requires the shared OpenEye libraries of the "
            "openeye-toolkits wheel, but OPENEYE_USE_SHARED resolved to ${OPENEYE_USE_SHARED} "
            "and OpenEye::OEChem links ${_OEVIBE_OECHEM_LOCATION}. "
            "Configure with the python preset.")
    endif()
    set(OpenEye_LIBRARY_TYPE SHARED)
endif()

# Library
set(OEVIBE_SOURCES
    src/Method.cpp
    src/Preset.cpp
    src/Solvent.cpp
    src/Constraints.cpp
    src/Options.cpp
    src/MolecularSystem.cpp
    src/core/InputPrep.cpp
    src/Engine.cpp
    src/EngineRegistry.cpp
    src/engines/Builtins.cpp
    src/core/MolFuncSession.cpp
    src/OEAdapters.cpp
    src/engines/OEFFEngine.cpp
    src/Result.cpp
    src/core/Sha256.cpp
    src/Calculation.cpp
    src/core/WriteBack.cpp
    src/core/Geometry.cpp
    src/core/ConstraintMath.cpp
    src/core/ModelHessian.cpp
    src/core/AncCoordinates.cpp
    src/core/Lbfgs.cpp
    src/core/AncRfo.cpp
    src/core/Optimizer.cpp
    src/engines/RuntimeLibrary.cpp
    src/engines/TbliteLibrary.cpp
    src/engines/TbliteEngine.cpp
    src/engines/ElectronicProperties.cpp
    src/engines/XtbLibrary.cpp
    src/engines/XtbEngine.cpp
    src/core/Hessian.cpp
)

add_library(oevibe ${OEVIBE_SOURCES})

set_target_properties(oevibe PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(oevibe
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Eigen is an implementation detail of src/core; BUILD_INTERFACE keeps it out of the export.
target_link_libraries(oevibe
    PUBLIC
        OpenEye::OEChem
        OpenEye::OEFF
        OpenEye::OEMMFF
        OpenEye::OESmirnoff
        OpenEye::OESheffield
        OpenEye::OEQuacpac
    PRIVATE
        $<BUILD_INTERFACE:oevibe_eigen>
        ${CMAKE_DL_LIBS}
)

# Tests
if(OEVIBE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Python bindings
if(OEVIBE_BUILD_PYTHON)
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
