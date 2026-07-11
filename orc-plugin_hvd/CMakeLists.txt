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

# --- hvd_core: the SDK-free numerical library (engine + frame bridge) ---------
add_library(hvd_core STATIC
    src/engine/ntsc_geometry.cpp
    src/engine/lockin.cpp
    src/engine/fft2d.cpp
    src/engine/holographic_init.cpp
    src/engine/variational.cpp
    src/engine/colour.cpp
    src/engine/engine.cpp
    src/frame_bridge.cpp
)
target_include_directories(hvd_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(hvd_core PUBLIC PkgConfig::FFTW3F)
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
