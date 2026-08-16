/*
 * File:        plugin.h
 * Module:      orc-stage-plugin-hvd-chroma-decoder
 * Purpose:     Plugin entrypoint metadata for HvdChromaDecoderStage
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2026 hvd-decode contributors
 */

#pragma once

#include <cstdint>
#include <orc/abi/orc_plugin_sdk.h>

#ifndef ORC_STAGE_PLUGIN_VERSION
#define ORC_STAGE_PLUGIN_VERSION "dev"
#endif

namespace orc::plugins::hvd {

// Stage identifier used during registration and project serialization.
inline constexpr const char* kStageName = "hvd_chroma_decoder";

// Human-readable stage label shown in the UI.
inline constexpr const char* kStageDisplayName = "HVD Chroma Decoder";

// Connectivity archetype.
inline constexpr orc::NodeType kStageNodeType = NodeType::SINK;

inline constexpr uint32_t kStageMinInputs = 1;
inline constexpr uint32_t kStageMaxInputs = 1;
inline constexpr uint32_t kStageMinOutputs = 0;
inline constexpr uint32_t kStageMaxOutputs = 0;

// NTSC-only, like the reference decoder.
// NTSC + 625-line PAL. VERIFIED against orc/stage/node_type.h: the
// both-standards value is ALL ("Works with any format (NTSC, PAL,
// PAL-M, etc.)") — an earlier revision guessed ANY offline; ALL is the
// real name.
inline constexpr orc::VideoFormatCompatibility kStageCompatibleFormats =
    VideoFormatCompatibility::ALL;

// NOTE: SinkCategory and the plugin-declared menu category are gone as of
// ABI 12 (orc-sdk abi_history.yaml, abi: 12). NodeTypeInfo no longer carries
// a sink_category/menu_category field: the Add Stage menu group is derived
// from NodeType via NodeTypeInfo::category()/stage_category_for(), so a SINK
// stage is always filed under "Sink" and this plugin can no longer place
// itself under an invented category ("Chroma decode" is gone). kStageSinkCategory
// and kStageMenuCategory were removed accordingly — see plugin.cpp, which no
// longer cross-checks them against get_node_type_info().

static_assert(kStageName[0] != '\0', "kStageName must not be empty");
static_assert(kStageDisplayName[0] != '\0', "kStageDisplayName must not be empty");
static_assert(kStageMaxInputs >= kStageMinInputs, "max inputs >= min inputs");
static_assert(kStageMaxOutputs >= kStageMinOutputs, "max outputs >= min outputs");

// The macro fills in host ABI version, plugin API version and toolchain tag
// from the SDK the plugin is compiled against (the host rejects the plugin
// unless all three match it exactly at load time).
inline constexpr orc::StagePluginDescriptor kPluginDescriptor =
    ORC_STAGE_PLUGIN_DESCRIPTOR(
        "org.decodeorc.stage.hvd_chroma_decoder",
        ORC_STAGE_PLUGIN_VERSION,
        "GPL-3.0-or-later",
        /*is_core_plugin=*/false);

}  // namespace orc::plugins::hvd
