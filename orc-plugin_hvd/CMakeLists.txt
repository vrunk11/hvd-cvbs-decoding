cmake_minimum_required(VERSION 3.20)

project(orc-plugin-hvd-chroma-decoder
    VERSION 0.1.0
    DESCRIPTION "Decode-Orc holographic-variational NTSC chroma decoder plugin"
    LANGUAGES CXX
)

set(ORC_PLUGIN_VERSION "${PROJECT_VERSION}" CACHE STRING
    "Plugin version string embedded in StagePluginDescriptor")

# On tagged CI builds, derive plugin version from tag names like v1.2.3
# unless the caller explicitly overrides ORC_PLUGIN_VERSION.
if(ORC_PLUGIN_VERSION STREQUAL PROJECT_VERSION)
    if(DEFINED ENV{GITHUB_REF_TYPE} AND "$ENV{GITHUB_REF_TYPE}" STREQUAL "tag" AND DEFINED ENV{GITHUB_REF_NAME})
        set(_orc_ref_name "$ENV{GITHUB_REF_NAME}")
        if(_orc_ref_name MATCHES "^v(.+)$")
            set(ORC_PLUGIN_VERSION "${CMAKE_MATCH_1}")
            set(ORC_PLUGIN_VERSION "${ORC_PLUGIN_VERSION}" CACHE STRING
                "Plugin version string embedded in StagePluginDescriptor" FORCE)
        endif()
    endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_TESTS "Build plugin + engine tests" ON)

# --- decode-orc plugin SDK: in-tree source tree OR installed package ---------
if(DEFINED ORC_INTREE_SDK_DIR)
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/DecodeOrcPluginSDKHelpers.cmake")

    add_library(orc-sdk-includes INTERFACE)

    find_package(fmt CONFIG QUIET)
    if(fmt_FOUND)
        target_link_libraries(orc-sdk-includes INTERFACE fmt::fmt-header-only)
    else()
        file(GLOB _orc_fmt_include_candidates "/nix/store/*-fmt-*/include")
        list(LENGTH _orc_fmt_include_candidates _orc_fmt_include_count)
        if(_orc_fmt_include_count GREATER 0)
            list(GET _orc_fmt_include_candidates 0 _orc_fmt_include_dir)
            target_include_directories(orc-sdk-includes INTERFACE ${_orc_fmt_include_dir})
        endif()
    endif()

    add_library(orc-plugin-sdk INTERFACE)
    target_include_directories(orc-plugin-sdk INTERFACE
        ${ORC_INTREE_SDK_DIR}/orc/sdk/include
    )
    target_link_libraries(orc-plugin-sdk INTERFACE orc-sdk-includes)
else()
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/DecodeOrcPluginSDKHelpers.cmake")
    find_package(decode-orc-plugin-sdk REQUIRED)
endif()

# --- FFTW3 (single precision): the engine's only third-party dependency -------
# Declared by the plugin itself (not inherited from the host), as the SDK allows.
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

# --- FFTW3 threading (optional): lets FFTW's own 2-D transforms use
# multiple threads internally — this is on top of, not instead of, the
# OpenMP/std::thread parallelism elsewhere in hvd_core; it's the last major
# unparallelised chunk of a single frame's decode (HolographicInit does 3-4
# full-picture 2-D FFTs per frame).
#
# This does NOT search for a separate library file. Some FFTW builds ship
# threading as a distinct fftw3f_threads/fftw3f_omp library, but vcpkg's
# fftw3[threads] (or [openmp]) feature compiles the threading symbols
# (fftwf_init_threads, fftwf_plan_with_nthreads) directly INTO the base
# fftw3f library instead — verified against an actual vcpkg x64-mingw-
# dynamic build, which produces only libfftw3f.dll/.dll.a, no separate
# threads/omp artifact at all. So instead of looking for a file, we just
# try to actually link a program calling those symbols against whatever
# fftw3f we already have — the only test that's actually correct across
# both packaging styles.
include(CheckCXXSourceCompiles)
get_target_property(_fftw3f_test_incs PkgConfig::FFTW3F INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(_fftw3f_test_libs PkgConfig::FFTW3F INTERFACE_LINK_LIBRARIES)
set(CMAKE_REQUIRED_INCLUDES ${_fftw3f_test_incs})
set(CMAKE_REQUIRED_LIBRARIES ${_fftw3f_test_libs})
check_cxx_source_compiles("
    #include <fftw3.h>
    int main() {
      fftwf_init_threads();
      fftwf_plan_with_nthreads(1);
      return 0;
    }
" HVD_FFTW_THREADS_AVAILABLE)
unset(CMAKE_REQUIRED_INCLUDES)
unset(CMAKE_REQUIRED_LIBRARIES)

if(HVD_FFTW_THREADS_AVAILABLE)
    message(STATUS
        "FFTW3 threading symbols available in fftw3f: FFTW's own "
        "transforms will use multiple threads too, not just the caller's "
        "OpenMP/std::thread parallelism.")
else()
    message(WARNING
        "fftwf_init_threads/fftwf_plan_with_nthreads not available in the "
        "linked fftw3f — FFTW's 2-D transforms will run single-threaded. "
        "This is currently the largest remaining unparallelised chunk of "
        "per-frame decode time. If using vcpkg, make sure vcpkg.json "
        "requests fftw3's 'threads' or 'openmp' feature. The build "
        "proceeds fine either way — this is a perf nice-to-have, not a "
        "hard requirement.")
endif()

# --- hvd_core: the SDK-free numerical library (engine + frame bridge) ---------
# NOTE: variational.cpp already has #pragma omp parallel for on its hot
# elementwise loops (gradient, curvature, dot products — the IRLS/CG solver
# that dominates decode time). Those pragmas were previously silent no-ops:
# nothing here ever requested OpenMP, so every compiler just ignored them
# and ran single-threaded. This is very likely why the plugin was observed
# using ~1 core's worth of CPU regardless of how many are available.
#
# We add -fopenmp directly (compile AND link) for GCC/Clang instead of
# relying solely on find_package(OpenMP)'s OpenMP::OpenMP_CXX imported
# target: on at least one MinGW-w64 setup, that target correctly added
# -fopenmp at compile time (so #pragma omp lowered to real GOMP_* calls)
# but did NOT propagate a link-time flag/library to consumers, so hvd_core
# itself built fine (a static archive never resolves its own external
# symbols) while the plugin DLL and any test exercising variational.cpp
# failed at link with "undefined reference to GOMP_parallel/omp_get_num_
# threads/...". Passing -fopenmp to the LINK command directly is what
# actually fixes that: GCC's driver pulls in libgomp (and libpthread)
# automatically whenever -fopenmp is present at link time, no need to name
# the library by hand. find_package(OpenMP) is kept only for the
# diagnostic message below.
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
    src/engine/engine.cpp
    src/frame_bridge.cpp
)
target_include_directories(hvd_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(hvd_core PUBLIC PkgConfig::FFTW3F)
if(HVD_FFTW_THREADS_AVAILABLE)
    # No extra target_link_libraries: the threading symbols are already
    # part of PkgConfig::FFTW3F, just above — see the detection comment.
    target_compile_definitions(hvd_core PUBLIC HVD_HAVE_FFTW_THREADS)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # PUBLIC: needs to reach every consumer's compile AND link command
    # (the plugin DLL, every tests/engine/*_test, hvd_stage_smoke_test,
    # hvd_plugin_entrypoints_test — anything that pulls in variational.cpp
    # transitively through hvd_core).
    target_compile_options(hvd_core PUBLIC -fopenmp)
    target_link_options(hvd_core PUBLIC -fopenmp)
elseif(OpenMP_CXX_FOUND)
    # MSVC or anything else find_package(OpenMP) actually handles cleanly.
    target_link_libraries(hvd_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# Frame-level parallel export (hvd_chroma_decoder_stage.cpp, part of the
# plugin target below, not hvd_core itself) uses std::thread directly
# rather than OpenMP (each worker owns its own HvdEngine — see that file
# for why this can't just be another #pragma omp loop). Declared PUBLIC
# here so it propagates to that target through LINK_LIBRARIES hvd_core.
find_package(Threads REQUIRED)
target_link_libraries(hvd_core PUBLIC Threads::Threads)
set_target_properties(hvd_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- the plugin ---------------------------------------------------------------
set(HVD_PLUGIN_TARGET "orc-stage-plugin-hvd-chroma-decoder")

orc_add_stage_plugin(
    ${HVD_PLUGIN_TARGET}
    OUTPUT_NAME ${HVD_PLUGIN_TARGET}
    PLUGIN_VERSION ${ORC_PLUGIN_VERSION}
    SOURCES
        src/plugin.cpp
        src/hvd_chroma_decoder_stage.cpp
    LINK_LIBRARIES
        hvd_core
)

target_include_directories(${HVD_PLUGIN_TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
