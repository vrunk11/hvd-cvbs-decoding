# ---------------------------------------------------------------------------
# SDK-level tests (need the decode-orc SDK target): stage metadata + plugin
# entrypoints. Mirror the skeleton's tests, adapted to HvdChromaDecoderStage.
#
# The engine test suite (SDK-free: gradients_test, lockin_test, pal_lockin_test,
# ...) now lives in ../../hvd-core/tests — see that project's CMakeLists.txt.
# It's built there whenever hvd-core is the top-level project, or here too if
# you configure with -DHVD_CORE_BUILD_TESTS=ON.
# ---------------------------------------------------------------------------
if(DEFINED ORC_INTREE_SDK_DIR AND TARGET orc-plugin-sdk)
    set(HVD_TEST_SDK_TARGET orc-plugin-sdk)
elseif(TARGET orc::plugin-sdk)
    set(HVD_TEST_SDK_TARGET orc::plugin-sdk)
elseif(TARGET orc-plugin-sdk)
    set(HVD_TEST_SDK_TARGET orc-plugin-sdk)
else()
    message(FATAL_ERROR "Neither orc::plugin-sdk nor orc-plugin-sdk is available for tests")
endif()

add_executable(hvd_stage_smoke_test
    stage_smoke_test.cpp
    ../src/hvd_chroma_decoder_stage.cpp
    ../src/video_writer.cpp
)
target_link_libraries(hvd_stage_smoke_test PRIVATE ${HVD_TEST_SDK_TARGET} hvd_core)
target_include_directories(hvd_stage_smoke_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
if(TARGET PkgConfig::HVD_FFMPEG)
    target_link_libraries(hvd_stage_smoke_test PRIVATE PkgConfig::HVD_FFMPEG)
    # See HVD_FFMPEG_WIN32_SYSLIBS in ../CMakeLists.txt: linked directly here
    # too, not just on the plugin target, for the same reason — a static
    # FFmpeg needs these on every final link that pulls it in, and this test
    # links video_writer.cpp (and therefore FFmpeg) directly, independent of
    # the plugin .dll.
    if(HVD_FFMPEG_WIN32_SYSLIBS)
        target_link_libraries(hvd_stage_smoke_test PRIVATE ${HVD_FFMPEG_WIN32_SYSLIBS})
    endif()
endif()
add_test(NAME hvd_stage_smoke_test COMMAND hvd_stage_smoke_test)

add_executable(hvd_plugin_entrypoints_test
    plugin_entrypoints_test.cpp
    ../src/plugin.cpp
    ../src/hvd_chroma_decoder_stage.cpp
    ../src/video_writer.cpp
)
target_link_libraries(hvd_plugin_entrypoints_test PRIVATE ${HVD_TEST_SDK_TARGET} hvd_core)
target_include_directories(hvd_plugin_entrypoints_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
if(TARGET PkgConfig::HVD_FFMPEG)
    target_link_libraries(hvd_plugin_entrypoints_test PRIVATE PkgConfig::HVD_FFMPEG)
    if(HVD_FFMPEG_WIN32_SYSLIBS)
        target_link_libraries(hvd_plugin_entrypoints_test PRIVATE ${HVD_FFMPEG_WIN32_SYSLIBS})
    endif()
endif()
add_test(NAME hvd_plugin_entrypoints_test COMMAND hvd_plugin_entrypoints_test)
