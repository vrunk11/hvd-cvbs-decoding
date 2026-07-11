/*
 * File:        hvd_chroma_decoder_stage.h
 * Module:      orc-stage-plugin-hvd-chroma-decoder
 * Purpose:     Holographic-variational NTSC Y/C separator stage + wrapper repr
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2026 hvd-decode contributors
 */

#pragma once

#include <orc/plugin/orc_plugin_sdk.h>

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "engine/hvd_config.h"
#include "frame_bridge.h"

namespace orc::plugins::hvd {

// ---------------------------------------------------------------------------
// HvdDecodedRepresentation
// ---------------------------------------------------------------------------
// Wraps a composite VideoFrameRepresentation and exposes the holographic-
// variational Y/C split. Decoding is lazy and cached per frame; the split is
// lossless, so get_frame() recombines luma + (chroma - chroma_dc) == composite.
class HvdDecodedRepresentation : public VideoFrameRepresentationWrapper,
                                 public Artifact {
public:
    HvdDecodedRepresentation(
        std::shared_ptr<const VideoFrameRepresentation> source,
        ::hvd::HvdConfig config);

    std::string type_name() const override
    {
        return "hvd_decoded_frame_representation";
    }

    bool has_separate_channels() const override { return true; }

    const sample_type* get_frame(FrameID id) const override;
    const sample_type* get_frame_luma(FrameID id) const override;
    const sample_type* get_frame_chroma(FrameID id) const override;

private:
    const ::hvd::YcFrameS16& decoded(FrameID id) const;
    ::hvd::FrameParams frame_params() const;

    ::hvd::HvdConfig config_;

    mutable std::mutex mutex_;
    mutable std::map<FrameID, ::hvd::YcFrameS16> yc_cache_;
    mutable std::map<FrameID, std::vector<sample_type>> composite_cache_;
};

// ---------------------------------------------------------------------------
// HvdChromaDecoderStage
// ---------------------------------------------------------------------------
class HvdChromaDecoderStage : public DAGStage,
                             public ParameterizedStage,
                             public IStagePreviewCapability {
public:
    HvdChromaDecoderStage();

    std::string version() const override { return "0.1.0"; }
    ORC_STAGE_INSTRUCTIONS_MD

    NodeTypeInfo get_node_type_info() const override;

    std::vector<ArtifactPtr> execute(
        const std::vector<ArtifactPtr>& inputs,
        const std::map<std::string, ParameterValue>& parameters,
        ObservationContext& observation_context) override;

    size_t required_input_count() const override { return 1; }
    size_t output_count() const override { return 1; }

    std::shared_ptr<const VideoFrameRepresentation> process(
        std::shared_ptr<const VideoFrameRepresentation> source) const;

    // ParameterizedStage
    std::vector<ParameterDescriptor> get_parameter_descriptors(
        VideoSystem project_format, SourceType source_type) const override;
    using ParameterizedStage::get_parameter_descriptors;
    std::map<std::string, ParameterValue> get_parameters() const override;
    bool set_parameters(
        const std::map<std::string, ParameterValue>& params) override;

    // IStagePreviewCapability
    StagePreviewCapability get_preview_capability() const override;

private:
    void refresh_status();

    ::hvd::HvdConfig config_;
    mutable std::shared_ptr<const VideoFrameRepresentation> cached_output_;
};

}  // namespace orc::plugins::hvd
