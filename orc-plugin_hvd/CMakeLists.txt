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

# --- MSVC: keep <windows.h> from defining min/max as MACROS ------------------
# The SDK's <orc/support/stage_instructions.h> includes <windows.h> on _WIN32
# without setting NOMINMAX, and orc/plugin/orc_stage_tooling.h pulls it in. Once
# that header is seen, `max` is a function-like macro for the REST of the
# translation unit, so std::max(0, n) expands to std::((0)>(n)?(0):(n)) and MSVC
# reports C2589 "illegal token on right side of '::'" at every such call — with
# <algorithm> correctly included, which is what makes the error look unrelated
# to its cause.
#
# Set here, before any target is created, so it covers hvd_core, the plugin and
# every test regardless of which SDK branch (in-tree or find_package) is taken
# and regardless of include order. A per-file #define would only protect the
# file that remembers to carry it.
if(WIN32)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_TESTS "Build plugin + engine tests" ON)
# Engine-only mode: build hvd_core + the SDK-free engine tests WITHOUT any
# decode-orc SDK checkout. This is what the fast CI job and quick local
# numerical work use: it exercises the whole numerical core (including the
# PAL lock-in/carrier tests) with only cmake + a C++17 compiler + fftw3f.
option(HVD_ENGINE_ONLY "Build only hvd_core and engine tests (no SDK, no plugin)" OFF)

# --- decode-orc plugin SDK: in-tree source tree OR installed package ---------
if(HVD_ENGINE_ONLY)
    # No SDK targets in engine-only mode.
elseif(DEFINED ORC_INTREE_SDK_DIR)
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

# --- FFTW3 threading: deliberately NOT used ------------------------------
# This project used to detect and, if present, use fftw3f's own internal
# threading (fftwf_init_threads/fftwf_plan_with_nthreads) on top of the
# OpenMP/std::thread parallelism already in hvd_core. That is removed: a
# threaded fftw3f (vcpkg's fftw3[threads] feature, see vcpkg.json) links its
# own OpenMP runtime, which alongside this project's own -fopenmp one
# produced DLL/symbol conflicts the host (decode-orc, built without threaded
# FFTW) doesn't have. The transforms here are small enough per call
# (HolographicInit's 3-4 full-picture 2-D FFTs per frame) that the measured
# speed gain from threading FFTW itself was negligible next to the
# OpenMP-parallel IRLS/CG solver that dominates decode time and remains
# fully threaded. See fft2d.h's SetThreadCount doc comment: it is kept as a
# no-op so callers don't need conditional compilation of their own.

# --- FFmpeg (libav*): real-container video export (.mkv/FFV1, .mp4/MPEG-4, NUT) --
# Only needed by the plugin target's video_writer.cpp, not hvd_core or the
# engine-only tests -- HVD_ENGINE_ONLY builds skip this block entirely.
if(NOT HVD_ENGINE_ONLY)
    if(PkgConfig_FOUND)
        pkg_check_modules(HVD_FFMPEG IMPORTED_TARGET
            libavformat libavcodec libavutil libswscale)
    endif()
    if(NOT HVD_FFMPEG_FOUND)
        find_library(HVD_AVFORMAT_LIBRARY NAMES avformat)
        find_library(HVD_AVCODEC_LIBRARY NAMES avcodec)
        find_library(HVD_AVUTIL_LIBRARY NAMES avutil)
        find_library(HVD_SWSCALE_LIBRARY NAMES swscale)
        find_path(HVD_FFMPEG_INCLUDE_DIR NAMES libavformat/avformat.h)
        if(NOT HVD_AVFORMAT_LIBRARY OR NOT HVD_AVCODEC_LIBRARY OR
           NOT HVD_AVUTIL_LIBRARY OR NOT HVD_SWSCALE_LIBRARY OR
           NOT HVD_FFMPEG_INCLUDE_DIR)
            message(FATAL_ERROR
                "FFmpeg (libavformat/libavcodec/libavutil/libswscale) not "
                "found. Install it (e.g. via vcpkg -- see vcpkg.json) or "
                "pass -DHVD_AVFORMAT_LIBRARY=<...> etc. by hand.")
        endif()
        add_library(PkgConfig::HVD_FFMPEG INTERFACE IMPORTED)
        set_target_properties(PkgConfig::HVD_FFMPEG PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${HVD_FFMPEG_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES
                "${HVD_AVFORMAT_LIBRARY};${HVD_AVCODEC_LIBRARY};${HVD_AVUTIL_LIBRARY};${HVD_SWSCALE_LIBRARY}")
    endif()
    # else: pkg_check_modules(... IMPORTED_TARGET) already created
    # PkgConfig::HVD_FFMPEG for us.
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
    src/engine/sequence.cpp
    src/engine/engine.cpp
    src/frame_bridge.cpp
)
target_include_directories(hvd_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(hvd_core PUBLIC PkgConfig::FFTW3F)

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
if(NOT HVD_ENGINE_ONLY)
set(HVD_PLUGIN_TARGET "orc-stage-plugin-hvd-chroma-decoder")

orc_add_stage_plugin(
    ${HVD_PLUGIN_TARGET}
    OUTPUT_NAME ${HVD_PLUGIN_TARGET}
    PLUGIN_VERSION ${ORC_PLUGIN_VERSION}
    SOURCES
        src/plugin.cpp
        src/hvd_chroma_decoder_stage.cpp
        src/video_writer.cpp
    LINK_LIBRARIES
        hvd_core
)

target_include_directories(${HVD_PLUGIN_TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(TARGET PkgConfig::HVD_FFMPEG)
    target_link_libraries(${HVD_PLUGIN_TARGET} PRIVATE PkgConfig::HVD_FFMPEG)
endif()
endif()  # NOT HVD_ENGINE_ONLY

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
