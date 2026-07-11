/*
 * File:        plugin.h
 * Module:      orc-stage-plugin-hvd-chroma-decoder
 * Purpose:     Plugin entrypoint metadata for HvdChromaDecoderStage
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2026 hvd-decode contributors
 */

#pragma once

#include <orc/plugin/orc_plugin_sdk.h>

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
inline constexpr orc::VideoFormatCompatibility kStageCompatibleFormats =
    VideoFormatCompatibility::NTSC_ONLY;

// External / third-party plugin.
inline constexpr orc::SinkCategory kStageSinkCategory = SinkCategory::THIRD_PARTY;

// UI menu category group shown in the node palette.
inline constexpr const char* kStageMenuCategory = "Chroma decode";

static_assert(kStageName[0] != '\0', "kStageName must not be empty");
static_assert(kStageDisplayName[0] != '\0', "kStageDisplayName must not be empty");
static_assert(kStageMenuCategory[0] != '\0', "kStageMenuCategory must not be empty");
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
