cmake_minimum_required(VERSION 3.20)

project(hvd-core
    VERSION 0.1.0
    DESCRIPTION "Holographic-variational NTSC/PAL Y/C separator — decoder engine (no decode-orc SDK dependency)"
    LANGUAGES CXX
)

# Consumers add this directory (typically as a git submodule) with
# add_subdirectory() and link against the `hvd_core` target. It never
# includes or links anything from decode-orc: see PORTING.md in the plugin
# repo for the split rationale. This CMakeLists.txt is also fully standalone
# — `cmake -S hvd-core -B build && cmake --build build && ctest --test-dir build`
# builds and runs the whole engine test suite with nothing else checked out.

# --- MSVC: keep <windows.h> from defining min/max as MACROS ------------------
# See the long comment in the plugin's own CMakeLists.txt for the full story;
# kept here too since hvd_core is buildable (and testable) fully standalone
# and some engine headers pull in platform headers on Windows.
if(WIN32)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Only default tests ON when hvd-core is the top-level project (a direct
# `cmake -S hvd-core -B build`, or a standalone checkout). When pulled in via
# add_subdirectory() by a consumer (e.g. orc-plugin_hvd, where hvd-core lives
# at ../hvd-core as a git submodule), default tests OFF so the consumer's own
# `BUILD_TESTS` / test targets aren't duplicated — the consumer can still
# force this back ON with -DHVD_CORE_BUILD_TESTS=ON.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(HVD_CORE_IS_TOP_LEVEL ON)
else()
    set(HVD_CORE_IS_TOP_LEVEL OFF)
endif()
option(HVD_CORE_BUILD_TESTS "Build hvd-core's own engine tests" ${HVD_CORE_IS_TOP_LEVEL})

# --- FFTW3 (single precision): the engine's only third-party dependency -------
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(FFTW3F IMPORTED_TARGET fftw3f)
endif()
if(NOT FFTW3F_FOUND)
    find_library(FFTW3F_LIBRARY NAMES fftw3f libfftw3f-3 fftw3f-3 libfftw3f
        HINTS ENV FFTW_DIR PATH_SUFFIXES lib)
    find_path(FFTW3F_INCLUDE_DIR NAMES fftw3.h
        HINTS ENV FFTW_DIR PATH_SUFFIXES include)
    if(NOT FFTW3F_LIBRARY OR NOT FFTW3F_INCLUDE_DIR)
        message(FATAL_ERROR
            "FFTW3 single-precision (fftw3f) not found. Pass "
            "-DFFTW3F_LIBRARY=<...> and -DFFTW3F_INCLUDE_DIR=<dir of fftw3.h>. "
            "Note: the single-precision 'f' variant is required.")
    endif()
    add_library(PkgConfig::FFTW3F INTERFACE IMPORTED)
    set_target_properties(PkgConfig::FFTW3F PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${FFTW3F_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${FFTW3F_LIBRARY}")
endif()

# --- FFTW3 threading: deliberately NOT used ------------------------------
# See variational.cpp / fft2d.h's SetThreadCount doc comment: kept as a
# no-op on purpose (own OpenMP parallelism already covers the hot loops;
# a threaded fftw3f pulls in its own OpenMP runtime and conflicts with it).

# --- hvd_core: the SDK-free numerical library (engine + frame bridge) ---------
# NOTE: variational.cpp uses #pragma omp parallel for on its hot elementwise
# loops (the IRLS/CG solver that dominates decode time) — see the OpenMP
# block below for why -fopenmp is forced explicitly rather than relying
# solely on find_package(OpenMP).
find_package(OpenMP QUIET)
if(NOT OpenMP_CXX_FOUND)
    message(WARNING
        "OpenMP not detected by find_package(OpenMP): if the compiler is "
        "GCC or Clang we still force -fopenmp explicitly below, so this is "
        "informational, not necessarily a problem. On MinGW-w64, OpenMP "
        "just needs libgomp to be installed.")
endif()

add_library(hvd_core STATIC
    src/engine/ntsc_geometry.cpp
    src/engine/lockin.cpp
    src/engine/fft2d.cpp
    src/engine/holographic_init.cpp
    src/engine/variational.cpp
    src/engine/colour.cpp
    src/engine/motion.cpp
    src/engine/temporal.cpp
    src/engine/sequence.cpp
    src/engine/engine.cpp
    src/frame_bridge.cpp
)
target_include_directories(hvd_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(hvd_core PUBLIC PkgConfig::FFTW3F)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # PUBLIC: needs to reach every consumer's compile AND link command (the
    # plugin DLL in orc-plugin_hvd, and every engine test below).
    target_compile_options(hvd_core PUBLIC -fopenmp)
    target_link_options(hvd_core PUBLIC -fopenmp)
elseif(OpenMP_CXX_FOUND)
    target_link_libraries(hvd_core PUBLIC OpenMP::OpenMP_CXX)
endif()

find_package(Threads REQUIRED)
target_link_libraries(hvd_core PUBLIC Threads::Threads)
set_target_properties(hvd_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(HVD_CORE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
