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
# Set here, before any target is created, so it covers the plugin and every
# test regardless of which SDK branch (in-tree or find_package) is taken and
# regardless of include order. A per-file #define would only protect the file
# that remembers to carry it. (hvd-core sets this too, independently, since
# it's buildable standalone — see hvd-core/CMakeLists.txt.)
if(WIN32)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_TESTS "Build plugin tests" ON)

# --- hvd-core: the decoder engine, consumed as a git submodule ---------------
# Lives at ../hvd-core by default (a sibling directory checked out as a git
# submodule of the umbrella repo — see .gitmodules at the repository root).
# Override with -DHVD_CORE_DIR=<path> to point at a different checkout (e.g.
# a local hvd-core working copy while developing both together).
set(HVD_CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../hvd-core" CACHE PATH
    "Path to the hvd-core decoder engine (git submodule by default)")
if(NOT EXISTS "${HVD_CORE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "hvd-core not found at '${HVD_CORE_DIR}'. This plugin consumes the "
        "decoder engine as a git submodule: run "
        "'git submodule update --init --recursive' from the repository "
        "root, or pass -DHVD_CORE_DIR=<path> to point at a local checkout.")
endif()
if(NOT TARGET hvd_core)
    add_subdirectory(${HVD_CORE_DIR} ${CMAKE_BINARY_DIR}/hvd-core)
endif()

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

# --- FFmpeg (libav*): real-container video export (.mkv/FFV1, .mp4/MPEG-4, NUT) --
# Only needed by the plugin target's video_writer.cpp, not hvd_core.
find_package(PkgConfig QUIET)
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

# --- the plugin ---------------------------------------------------------------
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

# --- release tooling: print the actual compiled descriptor -------------------
# Tiny standalone executable (SDK headers only, no hvd_core/FFTW) that prints
# plugin_id/host_abi/toolchain_tag straight out of kPluginDescriptor for THIS
# build. CI uses its output to fill orc-plugin-manifest.yaml so the ABI number
# and toolchain tag can never drift from what the binary actually embeds —
# see scripts/print_plugin_build_info.cpp and scripts/gen_manifest_fragment.py.
if(TARGET orc::plugin-sdk)
    set(_hvd_sdk_target_for_tools orc::plugin-sdk)
elseif(TARGET orc-plugin-sdk)
    set(_hvd_sdk_target_for_tools orc-plugin-sdk)
endif()
if(_hvd_sdk_target_for_tools)
    add_executable(hvd_print_build_info scripts/print_plugin_build_info.cpp)
    target_link_libraries(hvd_print_build_info PRIVATE ${_hvd_sdk_target_for_tools})
    target_include_directories(hvd_print_build_info PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    if(ORC_PLUGIN_VERSION)
        target_compile_definitions(hvd_print_build_info PRIVATE
            ORC_STAGE_PLUGIN_VERSION="${ORC_PLUGIN_VERSION}")
    endif()
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
