# ---------------------------------------------------------------------------
# Engine tests (SDK-free): link only hvd_core. These verify the numerical core
# (adjoint identity, lossless split, lock-in, colour, end-to-end bridge).
#
# Moved here unchanged from orc-plugin_hvd/tests/CMakeLists.txt when hvd-core
# was split out into its own (git-submodule-able) project — these tests never
# depended on the decode-orc SDK, only on hvd_core.
# ---------------------------------------------------------------------------
set(HVD_ENGINE_TESTS
    gradients_test
    lossless_identity_test
    lockin_test
    colour_test
    frame_bridge_test
    motion_test
    motion_v2_test
    temporal_test
    sequence_test
    field_order_test
    variational_v2_test
    pal_lockin_test
    chroma_reference_test
    gain_chain_test
    pal_aniso_group_test
    preview_aspect_test)

foreach(t ${HVD_ENGINE_TESTS})
    add_executable(${t} engine/${t}.cpp)
    target_include_directories(${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${t} PRIVATE hvd_core)
    add_test(NAME ${t} COMMAND ${t})
endforeach()
