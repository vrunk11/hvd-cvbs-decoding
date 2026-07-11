# ---------------------------------------------------------------------------
# SDK-level tests (need the decode-orc SDK target): stage metadata + plugin
# entrypoints. Mirror the skeleton's tests, adapted to HvdChromaDecoderStage.
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
)
target_link_libraries(hvd_stage_smoke_test PRIVATE ${HVD_TEST_SDK_TARGET} hvd_core)
target_include_directories(hvd_stage_smoke_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
add_test(NAME hvd_stage_smoke_test COMMAND hvd_stage_smoke_test)

add_executable(hvd_plugin_entrypoints_test
    plugin_entrypoints_test.cpp
    ../src/plugin.cpp
    ../src/hvd_chroma_decoder_stage.cpp
)
target_link_libraries(hvd_plugin_entrypoints_test PRIVATE ${HVD_TEST_SDK_TARGET} hvd_core)
target_include_directories(hvd_plugin_entrypoints_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
add_test(NAME hvd_plugin_entrypoints_test COMMAND hvd_plugin_entrypoints_test)

# ---------------------------------------------------------------------------
# Engine tests (SDK-free): link only hvd_core. These verify the numerical core
# (adjoint identity, lossless split, lock-in, colour, end-to-end bridge).
# ---------------------------------------------------------------------------
set(HVD_ENGINE_TESTS
    gradients_test
    lossless_identity_test
    lockin_test
    colour_test
    frame_bridge_test
    motion_test)

foreach(t ${HVD_ENGINE_TESTS})
    add_executable(${t} engine/${t}.cpp)
    target_include_directories(${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${t} PRIVATE hvd_core)
    add_test(NAME ${t} COMMAND ${t})
endforeach()
