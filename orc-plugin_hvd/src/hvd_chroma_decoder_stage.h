/*
 * File:        hvd_chroma_decoder_stage.h
 * Module:      orc-stage-plugin-hvd-chroma-decoder
 * Purpose:     Holographic-variational NTSC Y/C separator stage + wrapper repr
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2026 hvd-decode contributors
 */

#pragma once

#include <orc/plugin/orc_plugin_sdk.h>
#include <orc/stage/colour_preview_provider.h>

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "engine/engine.h"
#include "engine/hvd_config.h"
#include "frame_bridge.h"

namespace orc::plugins::hvd {

// ---------------------------------------------------------------------------
// HvdDecodedRepresentation
// ---------------------------------------------------------------------------
// Wraps a composite VideoFrameRepresentation and runs the holographic-
// variational Y/C split internally. Decoding is lazy and cached per frame.
// This is sink-internal state now: no Y/C representation is exposed
// downstream, only a colour preview carrier and a direct RGB24 file export.
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

    // No downstream Y/C output: this representation is internal to the stage
    // (colour preview + direct file export only). Nothing else should depend
    // on has_separate_channels()/get_frame_luma()/get_frame_chroma() here —
    // they used to be overridden to feed the generic Video Sink's YC
    // dual-decode path, but that path re-demodulates the chroma channel
    // through Comb's composite/adaptive-comb pipeline (is_yc forced false for
    // the "C-route" field in video_sink_stage.cpp), which is the wrong
    // pipeline for an already-separated chroma signal. Exporting directly
    // from the baseband U/V this stage already computed sidesteps that bug
    // entirely instead of trying to fix it upstream.

    // Baseband Y/U/V for colour preview (not a modulated chroma channel).
    // Returns nullptr if the frame hasn't been decoded/is invalid.
    const ::hvd::YcFrameS16* colour_planes(FrameID id) const;

    // Assembles a full ColourFrameCarrier (geometry + Y/U/V + level anchors)
    // for IColourPreviewProvider, in RASTER (woven) row order: row r is
    // display line r, alternating fields, matching what the host's own
    // preview renderer expects (see build_colour_carrier() in the .cpp for
    // why the underlying YcFrameS16 planes can't be used as-is).
    // std::nullopt if decode failed.
    std::optional<ColourFrameCarrier> build_colour_carrier(FrameID id) const;

    // Direct-to-file export, bypassing the host's Y/C sink pipeline
    // entirely. Writes one interleaved 8-bit RGB24 frame (active picture
    // only, raster order) per call, active_width * active_height * 3 bytes.
    // Returns false if the frame couldn't be decoded. Uses the shared,
    // cached engine_ (see below) — NOT safe to call concurrently from
    // multiple threads; that's what decode_and_write_rgb24() is for.
    bool write_raw_rgb24_frame(FrameID id, std::ostream& out) const;

    // Active picture size as written by write_raw_rgb24_frame(), or
    // {0, 0} if video parameters aren't available yet.
    std::pair<uint32_t, uint32_t> active_picture_size() const;

    // Parallel-export entry point: decodes frame `id` and writes RGB24 to
    // `out`, using the CALLER-OWNED `engine` rather than the shared engine_
    // decoded()/write_raw_rgb24_frame() use, and without touching yc_cache_.
    // `read_mutex` guards the one part of this that ISN'T safe to run
    // concurrently: reading this frame's raw samples out of `source_`. The
    // SDK's own contract says get_frame()/get_frame_luma()/get_frame_chroma()
    // pointers are "valid until the next call on the representation" —
    // i.e. the representation has internal single-frame state, so calling
    // it for different frame IDs from different threads at the same time
    // is not guaranteed safe. This function takes the lock, copies the raw
    // samples out immediately, releases the lock, and only then does the
    // expensive decode work (lock-free, on the caller's own `engine`) — the
    // same "sink-owned copy" pattern the host's own multi-threaded Video
    // Sink pipeline uses for exactly this reason (video_sink_stage.cpp's
    // appendSourceFields()).
    // Safe to call from multiple threads concurrently PROVIDED every
    // thread passes its own HvdEngine and all threads share the same
    // read_mutex.
    bool decode_and_write_rgb24(FrameID id, ::hvd::HvdEngine& engine,
                                std::mutex& read_mutex, std::ostream& out) const;

private:
    const ::hvd::YcFrameS16& decoded(FrameID id) const;
    ::hvd::FrameParams frame_params() const;

    // Reorders a field-sequential YcFrameS16 plane into raster (woven) row
    // order for the active picture only. Shared by build_colour_carrier(),
    // write_raw_rgb24_frame() and decode_and_write_rgb24() so none of them
    // can drift apart again.
    struct WovenActivePicture {
        uint32_t width = 0;
        uint32_t height = 0;
        std::vector<double> y, u, v;  // size width * height, or empty
    };
    // Pure: no cache/engine/source access, just the reorder math.
    static WovenActivePicture ReorderToWoven(const ::hvd::YcFrameS16& yc,
                                             const ::hvd::FrameParams& fp);
    // Cached path: decoded(id) (shared engine_) + ReorderToWoven.
    WovenActivePicture woven_active_picture(FrameID id) const;

    // Pure: converts an already-built WovenActivePicture to RGB24 and
    // writes it. No cache/engine/source access either.
    static bool WriteWovenAsRgb24(const WovenActivePicture& pic,
                                  const ::hvd::FrameParams& fp,
                                  std::ostream& out);

    ::hvd::HvdConfig config_;

    // Shared plan-cache engine for the normal (single-threaded, cached)
    // path — decoded() reuses this across every frame instead of
    // constructing a fresh HvdEngine (and re-planning FFTW) every time.
    // NOT used by decode_and_write_rgb24(): each parallel-export worker
    // thread must own its own HvdEngine instead (Fft2d holds scratch
    // buffers that aren't safe to share across concurrent calls).
    mutable ::hvd::HvdEngine engine_;

    mutable std::mutex mutex_;
    mutable std::map<FrameID, ::hvd::YcFrameS16> yc_cache_;
};

// ---------------------------------------------------------------------------
// HvdChromaDecoderStage
// ---------------------------------------------------------------------------
class HvdChromaDecoderStage : public DAGStage,
                             public ParameterizedStage,
                             public IStagePreviewCapability,
                             public IColourPreviewProvider {
public:
    HvdChromaDecoderStage();

    std::string version() const override { return "0.1.0"; }
    ORC_STAGE_INSTRUCTIONS_MD

    NodeTypeInfo get_node_type_info() const override;

    std::vector<ArtifactPtr> execute(
        const std::vector<ArtifactPtr>& inputs,
        const std::map<std::string, ParameterValue>& parameters,
        ObservationContext& observation_context) override;

    // Sink: no downstream Y/C representation. Decoding still happens on
    // execute() (to keep the colour preview live); file export is a
    // separate, explicit action (see kExportNow/kOutputPath parameters).
    size_t required_input_count() const override { return 1; }
    size_t output_count() const override { return 0; }

    std::shared_ptr<const VideoFrameRepresentation> process(
        std::shared_ptr<const VideoFrameRepresentation> source) const;

    // Writes every frame in source's frame_range() to `output_path_` as
    // raw interleaved RGB24, active picture only
    // (width * height * 3 bytes per frame, no header). Returns false on
    // I/O failure or if there's nothing decoded yet.
    // NOTE: this is a plain raw video dump, not a container — view it with
    // e.g. `ffplay -f rawvideo -pixel_format rgb24 -video_size WxH file.rgb`.
    bool export_now(std::string* error) const;

    // Human-readable result of the last export_now() call (empty until one
    // has run). Not tied to any host status-reporting interface yet — wire
    // this into whatever the GUI's node-status panel expects.
    const std::string& export_status() const { return export_status_; }

    // ParameterizedStage
    std::vector<ParameterDescriptor> get_parameter_descriptors(
        VideoSystem project_format, SourceType source_type) const override;
    using ParameterizedStage::get_parameter_descriptors;
    std::map<std::string, ParameterValue> get_parameters() const override;
    bool set_parameters(
        const std::map<std::string, ParameterValue>& params) override;

    // IStagePreviewCapability
    StagePreviewCapability get_preview_capability() const override;

    // IColourPreviewProvider
    std::optional<ColourFrameCarrier> get_colour_preview_carrier(
        uint64_t frame_index, PreviewNavigationHint hint) const override;

private:
    void refresh_status();

    ::hvd::HvdConfig config_;
    std::string output_path_;
    mutable std::string export_status_;
    mutable std::shared_ptr<const VideoFrameRepresentation> cached_output_;
};

}  // namespace orc::plugins::hvd
