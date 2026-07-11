/*
 * File:        hvd_chroma_decoder_stage.cpp
 * Module:      orc-stage-plugin-hvd-chroma-decoder
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2026 hvd-decode contributors
 */

#include "hvd_chroma_decoder_stage.h"

#include <orc/stage/cvbs_signal_constants.h>

#include <algorithm>
#include <cmath>
#include <utility>

namespace orc::plugins::hvd {

namespace {
constexpr const char* kLambdaC = "lambda_c";
constexpr const char* kCharbonnierEps = "charbonnier_eps";
constexpr const char* kChromaEps = "chroma_eps";
constexpr const char* kStructureCoupling = "structure_coupling";
constexpr const char* kCgIterations = "cg_iterations";
constexpr const char* kNtscJ = "ntsc_j";
constexpr const char* kAcc = "acc";
constexpr const char* kMonochrome = "monochrome";
constexpr const char* kSymmetryVariant = "symmetry_variant";
}  // namespace

// ===========================================================================
// HvdDecodedRepresentation
// ===========================================================================

HvdDecodedRepresentation::HvdDecodedRepresentation(
    std::shared_ptr<const VideoFrameRepresentation> source,
    ::hvd::HvdConfig config)
    : VideoFrameRepresentationWrapper(std::move(source)),
      Artifact(ArtifactID("hvd_decoded_frame"), Provenance{}),
      config_(config)
{
}

::hvd::FrameParams HvdDecodedRepresentation::frame_params() const
{
    ::hvd::FrameParams fp;
    const auto params = source_ ? source_->get_video_parameters() : std::nullopt;
    if (!params.has_value()) return fp;
    const SourceParameters& sp = *params;

    fp.frame_width = sp.frame_width_nominal;
    fp.frame_height = sp.frame_height;
    fp.field1_lines = static_cast<int>(field1_lines(sp.system));
    fp.active_video_start = sp.active_video_start;
    fp.active_video_end = sp.active_video_end;
    const auto burst = colour_burst_range(sp.system);
    fp.colour_burst_start = burst.first;
    fp.colour_burst_end = burst.second;
    fp.first_active_frame_line = sp.first_active_frame_line;
    fp.last_active_frame_line = sp.last_active_frame_line;
    fp.black_level = static_cast<float>(sp.black_level);
    fp.white_level = static_cast<float>(sp.white_level);
    fp.blanking_level = static_cast<float>(sp.blanking_level);
    fp.chroma_dc = sp.chroma_dc_offset >= 0
                       ? static_cast<float>(sp.chroma_dc_offset)
                       : static_cast<float>(sp.blanking_level);
    fp.sample_rate = sample_rate_from_system(sp.system);
    return fp;
}

const ::hvd::YcFrameS16& HvdDecodedRepresentation::decoded(FrameID id) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = yc_cache_.find(id);
    if (it != yc_cache_.end()) return it->second;

    const ::hvd::FrameParams fp = frame_params();
    const sample_type* frame = source_ ? source_->get_frame(id) : nullptr;
    ::hvd::YcFrameS16 yc;
    if (frame && fp.frame_width > 0 && fp.frame_height > 0) {
        yc = ::hvd::DecodeFrameBuffer(frame, fp, config_);
    }
    auto [ins, ok] = yc_cache_.emplace(id, std::move(yc));
    (void)ok;
    return ins->second;
}

const HvdDecodedRepresentation::sample_type*
HvdDecodedRepresentation::get_frame_luma(FrameID id) const
{
    const ::hvd::YcFrameS16& yc = decoded(id);
    return yc.luma.empty() ? nullptr : yc.luma.data();
}

const HvdDecodedRepresentation::sample_type*
HvdDecodedRepresentation::get_frame_chroma(FrameID id) const
{
    const ::hvd::YcFrameS16& yc = decoded(id);
    return yc.chroma.empty() ? nullptr : yc.chroma.data();
}

const HvdDecodedRepresentation::sample_type*
HvdDecodedRepresentation::get_frame(FrameID id) const
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = composite_cache_.find(id);
        if (it != composite_cache_.end()) return it->second.data();
    }
    const ::hvd::YcFrameS16& yc = decoded(id);
    if (yc.luma.empty()) return source_ ? source_->get_frame(id) : nullptr;

    const int dc = static_cast<int>(std::lround(yc.chroma_dc));
    std::vector<sample_type> composite(yc.luma.size());
    for (size_t i = 0; i < composite.size(); ++i) {
        const int v = static_cast<int>(yc.luma[i]) +
                      (static_cast<int>(yc.chroma[i]) - dc);
        composite[i] = static_cast<sample_type>(std::clamp(v, 0, 1023));
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto [ins, ok] = composite_cache_.emplace(id, std::move(composite));
    (void)ok;
    return ins->second.data();
}

// ===========================================================================
// HvdChromaDecoderStage
// ===========================================================================

HvdChromaDecoderStage::HvdChromaDecoderStage() { refresh_status(); }

void HvdChromaDecoderStage::refresh_status()
{
    set_configuration_status(config_.cg_iterations == 0
                                 ? ConfigurationStatus::Yellow
                                 : ConfigurationStatus::Green);
}

NodeTypeInfo HvdChromaDecoderStage::get_node_type_info() const
{
    return NodeTypeInfo{
        NodeType::TRANSFORM, "hvd_chroma_decoder", "HVD Chroma Decoder",
        "Holographic-variational NTSC Y/C separator (experimental). "
        "Outputs a lossless Y/C split.",
        1, 1, 1, 1, VideoFormatCompatibility::NTSC_ONLY,
        SinkCategory::THIRD_PARTY, "Chroma decode"};
}

std::vector<ArtifactPtr> HvdChromaDecoderStage::execute(
    const std::vector<ArtifactPtr>& inputs,
    const std::map<std::string, ParameterValue>& parameters,
    ObservationContext&)
{
    if (inputs.empty() || !inputs[0]) {
        cached_output_.reset();
        return {};
    }
    auto vfr = std::dynamic_pointer_cast<const VideoFrameRepresentation>(inputs[0]);
    if (!vfr) {
        return {inputs[0]};  // not video data: pass through unchanged
    }
    if (!parameters.empty()) set_parameters(parameters);

    auto output = process(vfr);
    cached_output_ = output;

    return {std::dynamic_pointer_cast<Artifact>(
        std::const_pointer_cast<VideoFrameRepresentation>(output))};
}

std::shared_ptr<const VideoFrameRepresentation>
HvdChromaDecoderStage::process(
    std::shared_ptr<const VideoFrameRepresentation> source) const
{
    if (!source) return nullptr;
    return std::make_shared<HvdDecodedRepresentation>(std::move(source), config_);
}

// ------------------------------------------------------------- parameters

std::vector<ParameterDescriptor>
HvdChromaDecoderStage::get_parameter_descriptors(VideoSystem, SourceType) const
{
    auto real = [](double mn, double mx, double def) {
        return ParameterConstraints{ParameterValue{mn}, ParameterValue{mx},
                                    ParameterValue{def}, {}, false, std::nullopt};
    };
    auto integer = [](int32_t mn, int32_t mx, int32_t def) {
        return ParameterConstraints{ParameterValue{mn}, ParameterValue{mx},
                                    ParameterValue{def}, {}, false, std::nullopt};
    };
    auto boolean = [](bool def) {
        return ParameterConstraints{std::nullopt, std::nullopt,
                                    ParameterValue{def}, {}, false, std::nullopt};
    };

    return {
        ParameterDescriptor{kLambdaC, "Chroma smoothness",
            "Arbitration prior. Higher = smoother chroma (less rainbowing); "
            "lower = sharper chroma.",
            ParameterType::DOUBLE, real(0.0, 8.0, 1.0)},
        ParameterDescriptor{kCharbonnierEps, "Luma edge scale (IRE)",
            "Edge-preservation scale of the luma prior, in IRE.",
            ParameterType::DOUBLE, real(0.05, 5.0, 0.5)},
        ParameterDescriptor{kChromaEps, "Chroma edge scale (IRE)",
            "Edge-preservation scale of the chroma prior, in IRE.",
            ParameterType::DOUBLE, real(0.05, 5.0, 1.0)},
        ParameterDescriptor{kStructureCoupling, "Y->chroma edge coupling",
            "Open the chroma edge where luma has one (removes hanging dots).",
            ParameterType::DOUBLE, real(0.0, 2.0, 0.25)},
        ParameterDescriptor{kCgIterations, "Solver iterations",
            "Conjugate-gradient iterations. 0 = holographic init only "
            "(fast preview).",
            ParameterType::INT32, integer(0, 400, 60)},
        ParameterDescriptor{kNtscJ, "NTSC-J levels",
            "Japanese discs: black at 0 IRE (colour path only).",
            ParameterType::BOOL, boolean(false)},
        ParameterDescriptor{kAcc, "Automatic Color Control",
            "Calibrate saturation from burst amplitude (colour path only).",
            ParameterType::BOOL, boolean(true)},
        ParameterDescriptor{kMonochrome, "Monochrome",
            "Zero the chroma channel.", ParameterType::BOOL, boolean(false)},
        ParameterDescriptor{kSymmetryVariant, "Spectral-symmetry init",
            "Add the Transform-NTSC certified-chroma init variant.",
            ParameterType::BOOL, boolean(false)}};
}

std::map<std::string, ParameterValue>
HvdChromaDecoderStage::get_parameters() const
{
    return {
        {kLambdaC, static_cast<double>(config_.lambda_c)},
        {kCharbonnierEps, static_cast<double>(config_.charbonnier_eps)},
        {kChromaEps, static_cast<double>(config_.chroma_eps)},
        {kStructureCoupling, static_cast<double>(config_.structure_coupling)},
        {kCgIterations, static_cast<int32_t>(config_.cg_iterations)},
        {kNtscJ, config_.ntsc_j},
        {kAcc, config_.acc},
        {kMonochrome, config_.monochrome},
        {kSymmetryVariant, config_.symmetry_variant}};
}

bool HvdChromaDecoderStage::set_parameters(
    const std::map<std::string, ParameterValue>& params)
{
    auto get_double = [&](const char* key, float& dst) {
        auto it = params.find(key);
        if (it == params.end()) return;
        if (std::holds_alternative<double>(it->second))
            dst = static_cast<float>(std::get<double>(it->second));
        else if (std::holds_alternative<int32_t>(it->second))
            dst = static_cast<float>(std::get<int32_t>(it->second));
    };
    auto get_int = [&](const char* key, int& dst) {
        auto it = params.find(key);
        if (it != params.end() && std::holds_alternative<int32_t>(it->second))
            dst = std::get<int32_t>(it->second);
    };
    auto get_bool = [&](const char* key, bool& dst) {
        auto it = params.find(key);
        if (it != params.end() && std::holds_alternative<bool>(it->second))
            dst = std::get<bool>(it->second);
    };

    get_double(kLambdaC, config_.lambda_c);
    get_double(kCharbonnierEps, config_.charbonnier_eps);
    get_double(kChromaEps, config_.chroma_eps);
    get_double(kStructureCoupling, config_.structure_coupling);
    get_int(kCgIterations, config_.cg_iterations);
    get_bool(kNtscJ, config_.ntsc_j);
    get_bool(kAcc, config_.acc);
    get_bool(kMonochrome, config_.monochrome);
    get_bool(kSymmetryVariant, config_.symmetry_variant);

    cached_output_.reset();
    refresh_status();
    return true;
}

// --------------------------------------------------------------- preview

StagePreviewCapability HvdChromaDecoderStage::get_preview_capability() const
{
    // Composed by hand (rather than PreviewHelpers::make_signal_preview_capability)
    // so the plugin also builds in the header-only in-tree configuration, which
    // links no host libraries. This stage outputs a Y/C-separated signal.
    StagePreviewCapability capability{};
    if (!cached_output_ || cached_output_->frame_count() == 0) {
        return capability;
    }
    const auto params = cached_output_->get_video_parameters();
    if (!params.has_value() || !params->is_valid()) {
        return capability;
    }

    const bool is_yc = cached_output_->has_separate_channels();
    VideoDataType data_type;
    if (params->system == VideoSystem::NTSC || params->system == VideoSystem::PAL_M) {
        data_type = is_yc ? VideoDataType::YC_NTSC : VideoDataType::CompositeNTSC;
    } else {
        data_type = is_yc ? VideoDataType::YC_PAL : VideoDataType::CompositePAL;
    }

    const auto active_width =
        params->active_video_end > params->active_video_start
            ? static_cast<uint32_t>(params->active_video_end - params->active_video_start)
            : static_cast<uint32_t>(params->frame_width_nominal);
    const auto active_height =
        params->last_active_frame_line > params->first_active_frame_line
            ? static_cast<uint32_t>(params->last_active_frame_line - params->first_active_frame_line)
            : static_cast<uint32_t>(params->frame_height);

    double dar_correction = 1.0;
    if (active_width > 0 && active_height > 0) {
        const double active_ratio =
            static_cast<double>(active_width) / static_cast<double>(active_height);
        dar_correction = (4.0 / 3.0) / active_ratio;
    }

    capability.supported_data_types = {data_type};
    capability.navigation_extent.item_count = cached_output_->frame_count();
    capability.navigation_extent.granularity = 1;
    capability.navigation_extent.item_label = "frame";
    capability.geometry.active_width = active_width;
    capability.geometry.active_height = active_height;
    capability.geometry.display_aspect_ratio = 4.0 / 3.0;
    capability.geometry.dar_correction_factor = dar_correction;
    return capability;
}

}  // namespace orc::plugins::hvd
