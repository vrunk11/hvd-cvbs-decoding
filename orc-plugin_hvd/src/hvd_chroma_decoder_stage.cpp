/*
 * File:        hvd_chroma_decoder_stage.cpp
 * Module:      orc-stage-plugin-hvd-chroma-decoder
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2026 hvd-decode contributors
 */

#include "hvd_chroma_decoder_stage.h"
#include "video_writer.h"

#include <cstdint>
#include <orc/stage/cvbs_signal_constants.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <type_traits>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <memory>
#include <queue>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace orc::plugins::hvd {

namespace {
constexpr const char* kLambdaC = "lambda_c";
constexpr const char* kCharbonnierEps = "charbonnier_eps";
constexpr const char* kChromaEps = "chroma_eps";
constexpr const char* kStructureCoupling = "structure_coupling";
constexpr const char* kCgIterations = "cg_iterations";
constexpr const char* kFast = "fast";
constexpr const char* kParallelAcrossFields = "parallel_across_fields";
constexpr const char* kCgTol = "cg_tol";
constexpr const char* kBidirectional = "bidirectional";
constexpr const char* kSelective3d = "selective_3d";
constexpr const char* kDiagPrior = "diag_prior";
constexpr const char* kPasses = "passes";
constexpr const char* kChunkFrames = "chunk_frames";
constexpr const char* kChunkOverlap = "chunk_overlap";
constexpr const char* kFieldOrder = "field_order";
constexpr const char* kDebugDir = "debug_dir";
constexpr const char* kAcc = "acc";
constexpr const char* kChromaGain = "chroma_gain";
constexpr const char* kMonochrome = "monochrome";
constexpr const char* kCustomSubcarrier = "custom_subcarrier";
constexpr const char* kSubcarrierKhz = "subcarrier_khz";
constexpr const char* kSymmetryVariant = "symmetry_variant";
constexpr const char* kChromaPhaseDeg = "chroma_phase_deg";
constexpr const char* kPreviewFullRaster = "preview_full_raster";
constexpr const char* kPreviewFieldView = "preview_field_view";
constexpr const char* kEnableTemporal = "enable_temporal";
constexpr const char* kTemporalStrength = "temporal_strength";
constexpr const char* kMcTile = "mc_tile";
constexpr const char* kMcSearch = "mc_search";
constexpr const char* kNrAnchor = "nr_anchor";
constexpr const char* kOddGateFloor = "odd_gate_floor";
constexpr const char* kCoherenceGate = "coherence_gate";
constexpr const char* kOutputPath = "output_path";

// BT.601-ish YUV -> RGB, same matrix as engine/colour.cpp's YuvToRgb16 but
// operating directly on normalised [0,1]-ish code-domain deltas (see
// HvdDecodedRepresentation::woven_active_picture()).
std::array<uint8_t, 3> YuvToRgb8(double y, double u, double v) {
    const double r = y + 1.13983 * v;
    const double g = y - 0.39465 * u - 0.58060 * v;
    const double b = y + 2.03211 * u;
    auto to_byte = [](double x) {
        const double clipped = std::clamp(x, 0.0, 1.0);
        return static_cast<uint8_t>(clipped * 255.0 + 0.5);
    };
    return {to_byte(r), to_byte(g), to_byte(b)};
}

// variational.cpp's #pragma omp loops parallelise a SINGLE frame's solve
// across cores — the right thing when only one frame is in flight (the
// preview path). During parallel export, THIS thread is already one of
// several concurrent workers, each decoding a different frame; without
// this, each worker would also fan its own inner loops out across all
// cores, oversubscribing by roughly (worker count)x. Call this once at
// the top of each export worker thread (never on the preview path, which
// wants the full core count for its one frame). No-op if built without
// OpenMP.
void LimitOpenMpThreadsPerWorker() {
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
}
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
    // 625-line PAL family only (see FrameParams::is_pal). PAL_M stays on
    // the NTSC path deliberately — it carries PAL-type chroma on NTSC
    // line geometry and needs its own carrier derivation before the
    // engine can claim it (the effective-carrier machinery is ready; the
    // line-advance and burst model are not derived for it yet).
    fp.is_pal = (sp.system == VideoSystem::PAL);
    // Non-standard subcarrier: overrides the fsc implied by the system,
    // never the sample rate (the host still stores the standard 4fsc grid).
    fp.subcarrier_hz =
        config_.custom_subcarrier ? config_.subcarrier_khz * 1.0e3 : 0.0;
    return fp;
}

const ::hvd::YcFrameS16& HvdDecodedRepresentation::decoded(FrameID id) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = yc_cache_.find(id);
    if (it != yc_cache_.end()) return it->second;

    const ::hvd::FrameParams fp = frame_params();
    ::hvd::YcFrameS16 yc;
    if (source_ && fp.frame_width > 0 && fp.frame_height > 0) {
        // No-op in this build (single-threaded fftw3f — see hvd_config.h's
        // comment on fft_threads); kept only so the plumbing keeps compiling.
        engine_.SetFftThreads(config_.fft_threads);

        // IMPORTANT: for a source that's already Y/C separated (S-Video-
        // style captures, some hi-fi VHS formats), get_frame() (composite)
        // returns the LUMA plane with no chroma in it at all — see
        // tbc_source_stage.cpp's `result.luma = result.samples;` for is_yc_
        // sources. Feeding that into DecodeFrameBuffer used to silently
        // decode to zero chroma. Route by has_separate_channels() instead:
        // get_frame_luma()/get_frame_chroma() are the real channels there,
        // and DecodeYcFrameBuffer is the cheaper, correct path for them
        // (no separation problem to solve — see engine.h's DecodeChromaOnly
        // doc comment).
        //
        // engine_ is reused across every call here (not a fresh HvdEngine
        // per frame) so its FFTW plan cache only pays the planning cost
        // once per distinct frame size — see engine.h's own doc comment,
        // which this used to silently not honour.
        if (source_->has_separate_channels()) {
            const sample_type* luma = source_->get_frame_luma(id);
            const sample_type* chroma = source_->get_frame_chroma(id);
            if (luma && chroma) {
                yc = ::hvd::DecodeYcFrameBuffer(luma, chroma, fp, config_, engine_);
            }
            // No temporal wiring on the Y/C-native path — DecodeChromaOnly
            // has no prev_frames parameter (see its doc comment in
            // engine.h: there's no Y/C arbitration happening there for a
            // temporal term to improve).
        } else if (const sample_type* frame = source_->get_frame(id)) {
            bool done = false;
            {
                // Composite always decodes per FIELD (2D = decoupled
                // fields, 3D = a window of frame id +/- chunk_overlap) —
                // the same pipeline as the export, and now the same SIZE
                // knob too (chunk_overlap), not a separately-hardcoded
                // +/-1: what you judge in the preview is what the export
                // does, including how much temporal context it costs.
                // DecodeFrameBuffer below is the last-resort fallback only.
                // This replaces the earlier frame-level 3D preview and its
                // self-priming chain outright — the frame-level temporal
                // term remains only as the Y/C-native export fallback.
                const bool coupled =
                    config_.enable_temporal && config_.cg_iterations > 0;
                const int overlap = std::max(0, config_.chunk_overlap);
                std::vector<const sample_type*> window;
                FrameID w0 = id;
                if (coupled) {
                    for (int k = 0; k < overlap; ++k) {
                        if (w0 == 0 || !source_->get_frame(w0 - 1)) break;
                        --w0;
                    }
                }
                for (FrameID f = w0;; ++f) {
                    const sample_type* p = source_->get_frame(f);
                    if (!p) break;
                    window.push_back(p);
                    if (!coupled && f == id) break;
                    if (f >= id + static_cast<FrameID>(overlap)) break;
                }
                const int core = static_cast<int>(id - w0);
                if (core >= 0 && core < static_cast<int>(window.size())) {
                    ::hvd::HvdConfig eff = config_;
                    // -1 = explicit OFF, not 0 (= AUTO/measure). Same
                    // three-valued convention as the export path -- see
                    // the long comment in
                    // decode_sequence_chunk_and_write_rgb24().
                    if (!config_.enable_temporal) eff.temporal_strength = -1.0F;
                    std::vector<::hvd::YcFrameS16> frames =
                        ::hvd::DecodeFrameSequenceWindow(
                            window, core, core + 1, fp, eff, engine_);
                    if (frames.size() == 1 && !frames[0].luma.empty()) {
                        yc = std::move(frames[0]);
                        done = true;
                    }
                }
            }
            if (!done) {
                yc = ::hvd::DecodeFrameBuffer(frame, fp, config_, engine_);
            }
        }
    }
    auto [ins, ok] = yc_cache_.emplace(id, std::move(yc));
    (void)ok;
    return ins->second;
}

const ::hvd::YcFrameS16* HvdDecodedRepresentation::colour_planes(
    FrameID id) const
{
    const ::hvd::YcFrameS16& yc = decoded(id);
    return yc.luma.empty() ? nullptr : &yc;
}

// static — pure reorder, no cache/engine/source access, so it's exactly the
// same code path (and therefore exactly the same behaviour) whether it's
// fed by the cached decoded() or by a parallel-export worker's own decode.
HvdDecodedRepresentation::WovenActivePicture
HvdDecodedRepresentation::ReorderToWoven(const ::hvd::YcFrameS16& yc,
                                         const ::hvd::FrameParams& fp,
                                         bool full_raster, int field)
{
    WovenActivePicture out;
    if (yc.luma.empty()) return out;
    if (fp.frame_width <= 0 || fp.frame_height <= 0) return out;

    // yc.luma/u_plane/v_plane are laid out FIELD-SEQUENTIALLY (all of field
    // 1, then all of field 2) — that's the contract get_frame_luma()/
    // get_frame_chroma() used to advertise to the host's VFrameR buffer, per
    // frame_bridge.h. But nothing downstream expects that layout for a
    // *picture*: it's two flat blocks, so a naive per-row copy shows field 1
    // as the top half of the image and field 2 as the bottom half (the
    // "split field" preview bug). A real picture needs RASTER order: row r
    // is display line r, alternating fields — exactly how the host's own
    // colour decoders build their preview carrier (video_sink_stage.cpp,
    // reading componentFrame->y(y)/u(y)/v(y) after Comb's own field weave).
    const int fw = fp.frame_width;
    const int f1 = fp.field1_lines;
    // full_raster: no crop at all — the luma plane's margins carry the RAW
    // composite (frame_bridge fills the whole plane with the source samples
    // before overwriting the active region with decoded Y; chroma is zero
    // there), so sync/burst/blanking render as monochrome signal. That is
    // the point: an un-cropped view for judging geometry and burst.
    const int a0 = full_raster ? 0 : std::max(0, fp.active_video_start);
    const int a1 = full_raster
                       ? fw
                       : (fp.active_video_end > a0 ? fp.active_video_end : fw);
    const int y0 = full_raster ? 0 : std::max(0, fp.first_active_frame_line);
    const int y1 = full_raster ? fp.frame_height
                               : (fp.last_active_frame_line > y0
                                      ? fp.last_active_frame_line
                                      : fp.frame_height);
    const uint32_t width = static_cast<uint32_t>(std::max(0, a1 - a0));

    if (field == 0 || field == 1) {
        // SINGLE-FIELD view: each output row is one field line, native
        // height, no interpolation and no weave — per-field artefacts
        // (dropouts, PAL V-switch/Hanover checks, weave errors) show
        // exactly as the solver saw them. Frame line = 2*field_line+field;
        // keep those inside [y0, y1).
        std::vector<int> flats;
        const int nfl = (field == 0) ? f1 : (fp.frame_height - f1);
        for (int fl_line = 0; fl_line < nfl; ++fl_line) {
            const int frame_line = 2 * fl_line + field;
            if (frame_line < y0 || frame_line >= y1) continue;
            flats.push_back(field == 0 ? fl_line : f1 + fl_line);
        }
        const uint32_t height = static_cast<uint32_t>(flats.size());
        if (width == 0 || height == 0) return out;
        out.width = width;
        out.height = height;
        out.y.assign(static_cast<size_t>(width) * height, 0.0);
        out.u.assign(static_cast<size_t>(width) * height, 0.0);
        out.v.assign(static_cast<size_t>(width) * height, 0.0);
        for (uint32_t row = 0; row < height; ++row) {
            const int flat_line = flats[row];
            for (uint32_t col = 0; col < width; ++col) {
                const int flat_col = a0 + static_cast<int>(col);
                const size_t src =
                    static_cast<size_t>(flat_line) * fw + flat_col;
                const size_t dst = static_cast<size_t>(row) * width + col;
                out.y[dst] = static_cast<double>(yc.luma[src]);
                out.u[dst] = yc.u_plane[src];
                out.v[dst] = yc.v_plane[src];
            }
        }
        return out;
    }

    const uint32_t height = static_cast<uint32_t>(std::max(0, y1 - y0));
    if (width == 0 || height == 0) return out;

    out.width = width;
    out.height = height;
    out.y.assign(static_cast<size_t>(width) * height, 0.0);
    out.u.assign(static_cast<size_t>(width) * height, 0.0);
    out.v.assign(static_cast<size_t>(width) * height, 0.0);

    for (uint32_t row = 0; row < height; ++row) {
        const int frame_line = y0 + static_cast<int>(row);
        const int field = frame_line % 2;              // 0 = field 1, 1 = field 2
        const int field_line = frame_line / 2;
        const int flat_line = (field == 0) ? field_line : (f1 + field_line);
        if (flat_line < 0 || flat_line >= fp.frame_height) continue;
        for (uint32_t col = 0; col < width; ++col) {
            const int flat_col = a0 + static_cast<int>(col);
            if (flat_col < 0 || flat_col >= fw) continue;
            const size_t src = static_cast<size_t>(flat_line) * fw + flat_col;
            const size_t dst = static_cast<size_t>(row) * width + col;
            out.y[dst] = static_cast<double>(yc.luma[src]);
            out.u[dst] = yc.u_plane[src];
            out.v[dst] = yc.v_plane[src];
        }
    }
    return out;
}

HvdDecodedRepresentation::WovenActivePicture
HvdDecodedRepresentation::woven_active_picture(FrameID id) const
{
    const ::hvd::YcFrameS16* yc = colour_planes(id);
    if (!yc) return {};
    return ReorderToWoven(*yc, frame_params());
}

std::optional<ColourFrameCarrier> HvdDecodedRepresentation::build_colour_carrier(
    FrameID id, bool full_raster, int field) const
{
    const ::hvd::FrameParams fp = frame_params();
    if (fp.frame_width <= 0 || fp.frame_height <= 0) return std::nullopt;

    const ::hvd::YcFrameS16* yc = colour_planes(id);
    if (!yc) return std::nullopt;
    const WovenActivePicture pic = ReorderToWoven(*yc, fp, full_raster, field);
    if (pic.width == 0 || pic.height == 0) return std::nullopt;

    ColourFrameCarrier carrier;
    carrier.data_type = VideoDataType::ColourNTSC;
    carrier.frame_index = id;
    carrier.width = pic.width;
    carrier.height = pic.height;
    // Mark the WHOLE delivered image active, in every mode. The reason is
    // an inconsistency between two host analysis tools, both verified:
    //
    //  * vectorscope_analysis.cpp treats active_x/y_start as ABSOLUTE plane
    //    indices (sample_index = y * width + x, x from active_x_start);
    //  * preview_view_registry.cpp's histogram documents the opposite —
    //    plane[0] IS the first active pixel, only the DIFFERENCE is usable
    //    — because the chroma sink always delivers pre-cropped planes.
    //
    // ColourFrameCarrier has no flag to say which convention a given
    // carrier follows, so no single choice satisfies both. Marking the
    // full delivered extent makes BOTH tools iterate the whole delivered
    // image: in full-raster mode that means sync/blanking are included
    // (luma distribution reads low), which is a graceful degradation.
    // Marking the true active window instead would keep the vectorscope
    // exact but make the histogram analyse a same-sized rectangle from the
    // top-left corner — i.e. mostly sync — which is garbage, not
    // degradation. Turn "Preview: full raster" OFF for measurement work.
    carrier.active_x_start = 0;
    carrier.active_x_end = pic.width;
    carrier.active_y_start = 0;
    carrier.active_y_end = pic.height;
    carrier.y_plane = pic.y;
    carrier.u_plane = pic.u;
    carrier.v_plane = pic.v;

    carrier.cvbs_blanking = static_cast<double>(fp.blanking_level);
    carrier.cvbs_black = static_cast<double>(fp.black_level);
    carrier.cvbs_white = static_cast<double>(fp.white_level);

    return carrier.is_valid() ? std::make_optional(carrier) : std::nullopt;
}

// static — pure RGB conversion + write, no cache/engine/source access.
// Same conversion as WriteWovenAsRgb24, into an in-memory PreviewImage
// instead of a stream — used by the IStageCustomPreviewRenderer views.
PreviewImage HvdDecodedRepresentation::WovenToPreviewImage(
    const WovenActivePicture& pic, const ::hvd::FrameParams& fp)
{
    PreviewImage img;
    if (fp.white_level <= fp.black_level) return img;
    if (pic.width == 0 || pic.height == 0) return img;

    img.width = pic.width;
    img.height = pic.height;
    img.rgb_data.resize(static_cast<size_t>(pic.width) * pic.height * 3);

    const double range = fp.white_level - fp.black_level;
    for (uint32_t row = 0; row < pic.height; ++row) {
        for (uint32_t col = 0; col < pic.width; ++col) {
            const size_t i = static_cast<size_t>(row) * pic.width + col;
            const double ny = (pic.y[i] - fp.black_level) / range;
            const double nu = pic.u[i] / range;
            const double nv = pic.v[i] / range;
            const auto rgb = YuvToRgb8(ny, nu, nv);
            const size_t o = i * 3;
            img.rgb_data[o + 0] = rgb[0];
            img.rgb_data[o + 1] = rgb[1];
            img.rgb_data[o + 2] = rgb[2];
        }
    }
    return img;
}

PreviewImage HvdDecodedRepresentation::render_custom_preview(
    FrameID id, bool full_raster, int field) const
{
    const ::hvd::FrameParams fp = frame_params();
    const ::hvd::YcFrameS16* yc = colour_planes(id);
    if (!yc) return {};
    const WovenActivePicture pic = ReorderToWoven(*yc, fp, full_raster, field);
    return WovenToPreviewImage(pic, fp);
}

bool HvdDecodedRepresentation::WriteWovenAsRgb24(
    const WovenActivePicture& pic, const ::hvd::FrameParams& fp,
    std::ostream& out)
{
    if (fp.white_level <= fp.black_level) return false;
    if (pic.width == 0 || pic.height == 0) return false;

    const double range = fp.white_level - fp.black_level;
    std::vector<uint8_t> row_bytes(static_cast<size_t>(pic.width) * 3);
    for (uint32_t row = 0; row < pic.height; ++row) {
        for (uint32_t col = 0; col < pic.width; ++col) {
            const size_t i = static_cast<size_t>(row) * pic.width + col;
            const double ny = (pic.y[i] - fp.black_level) / range;
            const double nu = pic.u[i] / range;
            const double nv = pic.v[i] / range;
            const auto rgb = YuvToRgb8(ny, nu, nv);
            row_bytes[col * 3 + 0] = rgb[0];
            row_bytes[col * 3 + 1] = rgb[1];
            row_bytes[col * 3 + 2] = rgb[2];
        }
        out.write(reinterpret_cast<const char*>(row_bytes.data()),
                  static_cast<std::streamsize>(row_bytes.size()));
        if (!out) return false;
    }
    return true;
}

bool HvdDecodedRepresentation::write_raw_rgb24_frame(
    FrameID id, std::ostream& out) const
{
    const ::hvd::FrameParams fp = frame_params();
    const WovenActivePicture pic = woven_active_picture(id);
    return WriteWovenAsRgb24(pic, fp, out);
}

bool HvdDecodedRepresentation::decode_and_write_rgb24(
    FrameID id, ::hvd::HvdEngine& engine, std::mutex& read_mutex,
    std::ostream& out) const
{
    ::hvd::FrameParams fp;
    ::hvd::YcFrameS16 yc;
    {
        // Everything that touches `source_`'s per-call state lives inside
        // this lock — see the header doc comment on this function for why.
        // The actual decode (FFT + solver, the expensive part) happens
        // below, after we've released the lock, on data we now own.
        std::lock_guard<std::mutex> lock(read_mutex);
        if (!source_) return false;
        fp = frame_params();
        if (fp.frame_width <= 0 || fp.frame_height <= 0) return false;
        const size_t n = static_cast<size_t>(fp.frame_width) * fp.frame_height;

        if (source_->has_separate_channels()) {
            const sample_type* luma = source_->get_frame_luma(id);
            const sample_type* chroma = source_->get_frame_chroma(id);
            if (!luma || !chroma) return false;
            const std::vector<sample_type> luma_copy(luma, luma + n);
            const std::vector<sample_type> chroma_copy(chroma, chroma + n);
            yc = ::hvd::DecodeYcFrameBuffer(luma_copy.data(), chroma_copy.data(),
                                            fp, config_, engine);
            // No frame-level temporal wiring on the Y/C-native path — see
            // decoded()'s equivalent comment for why (DecodeChromaOnly has
            // no arbitration step for a temporal term to improve). A
            // caller chaining state across calls just won't get a neighbour
            // whenever the source happens to be Y/C-native; harmless.
        } else {
            const sample_type* frame = source_->get_frame(id);
            if (!frame) return false;
            const std::vector<sample_type> frame_copy(frame, frame + n);
            yc = ::hvd::DecodeFrameBuffer(frame_copy.data(), fp, config_, engine);
        }
    }
    if (yc.luma.empty()) return false;
    const WovenActivePicture pic = ReorderToWoven(yc, fp);
    return WriteWovenAsRgb24(pic, fp, out);
}

bool HvdDecodedRepresentation::decode_sequence_chunk_and_write_rgb24(
    FrameID t0, FrameID t1, FrameID range_first, FrameID range_last,
    ::hvd::HvdEngine& engine, std::mutex& read_mutex, std::ostream& out) const
{
    if (t1 < t0) return false;
    ::hvd::FrameParams fp;
    std::vector<std::vector<sample_type>> window;  // owned raw buffers
    FrameID w0 = t0;
    {
        // Copy the whole window's raw samples under the read lock; the
        // expensive decode happens below on data we own.
        std::lock_guard<std::mutex> lock(read_mutex);
        if (!source_) return false;
        if (source_->has_separate_channels()) return false;  // no composite
        fp = frame_params();
        if (fp.frame_width <= 0 || fp.frame_height <= 0) return false;
        const size_t n = static_cast<size_t>(fp.frame_width) * fp.frame_height;

        // Overlap provides temporal context; with the temporal terms off
        // (pure per-field 2D) the fields are decoupled and context is
        // useless weight.
        const bool coupled =
            config_.enable_temporal && config_.cg_iterations > 0;
        const FrameID ov = coupled
            ? static_cast<FrameID>(std::max(0, config_.chunk_overlap))
            : FrameID{0};
        w0 = (t0 > range_first + ov) ? (t0 - ov) : range_first;
        const FrameID w1 = std::min(t1 + ov, range_last);
        window.reserve(static_cast<size_t>(w1 - w0 + 1));
        for (FrameID id = w0; id <= w1; ++id) {
            const sample_type* frame = source_->get_frame(id);
            if (!frame) return false;
            window.emplace_back(frame, frame + n);
        }
    }

    std::vector<const int16_t*> ptrs;
    ptrs.reserve(window.size());
    for (const auto& buf : window) ptrs.push_back(buf.data());
    const int core_begin = static_cast<int>(t0 - w0);
    const int core_end = static_cast<int>(t1 - w0) + 1;

    // enable_temporal is the switch; the strength dial keeps its value.
    // The sequence driver only sees the strength, so the OFF position has
    // to be expressed through it in a local copy.
    //
    // IMPORTANT -- the engine's convention (see DecodeFrameSequenceWindow
    // in engine/sequence.cpp) is THREE-valued, not two:
    //     < 0  => 3D OFF
    //     == 0 => AUTO: measure the content's Y/C ambiguity and pick a
    //             strength, which for anything but perfectly clean
    //             content resolves to a POSITIVE value and switches the
    //             entire 3D path (motion estimation, neighbour
    //             equations, multi-pass/anchor) back on
    //     > 0  => that fixed strength
    // Writing 0.0F here therefore did NOT mean "off" -- it meant "decide
    // for me", so switching 3D off in the GUI left the export running the
    // adaptive ambiguity scan plus, usually, full 3D anyway. That is what
    // made composite exports enormously slower than the (frame-path)
    // preview while looking like the 2D path, and it also masked the
    // effect of the field-parallelism toggle, since a coupled 3D window
    // parallelises across far fewer fields than a decoupled 2D one.
    ::hvd::HvdConfig eff = config_;
    if (!config_.enable_temporal) eff.temporal_strength = -1.0F;
    const std::vector<::hvd::YcFrameS16> frames =
        ::hvd::DecodeFrameSequenceWindow(ptrs, core_begin, core_end, fp,
                                         eff, engine,
                                         static_cast<int64_t>(w0));
    if (frames.size() != static_cast<size_t>(core_end - core_begin))
        return false;
    for (const ::hvd::YcFrameS16& yc : frames) {
        if (yc.luma.empty()) return false;
        const WovenActivePicture pic = ReorderToWoven(yc, fp);
        if (!WriteWovenAsRgb24(pic, fp, out)) return false;
    }
    return true;
}

std::pair<uint32_t, uint32_t> HvdDecodedRepresentation::active_picture_size() const
{
    const ::hvd::FrameParams fp = frame_params();
    if (fp.frame_width <= 0 || fp.frame_height <= 0) return {0, 0};
    const int a0 = std::max(0, fp.active_video_start);
    const int a1 = fp.active_video_end > a0 ? fp.active_video_end : fp.frame_width;
    const int y0 = std::max(0, fp.first_active_frame_line);
    const int y1 = fp.last_active_frame_line > y0 ? fp.last_active_frame_line
                                                    : fp.frame_height;
    return {static_cast<uint32_t>(std::max(0, a1 - a0)),
            static_cast<uint32_t>(std::max(0, y1 - y0))};
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
        NodeType::SINK, "hvd_chroma_decoder", "HVD Chroma Decoder",
        "Holographic-variational NTSC/PAL Y/C separator (experimental). "
        "Colour preview only — no downstream Y/C representation; use "
        "'Export' to write a raw RGB24 file directly.",
        1, 1, 0, 0, VideoFormatCompatibility::ALL,  // NTSC + PAL (see plugin.h note)
        SinkCategory::THIRD_PARTY, "Chroma decode"};
}

std::vector<ArtifactPtr> HvdChromaDecoderStage::execute(
    const std::vector<ArtifactPtr>& inputs,
    const std::map<std::string, ParameterValue>& parameters,
    ObservationContext&)
{
    // Sink: never returns an output artifact (output_count() == 0). We still
    // decode on every execute() so the colour preview stays live. File
    // export happens through trigger() (TriggerableStage) instead — a real
    // "Export" button in the GUI, not a parameter here.
    if (inputs.empty() || !inputs[0]) {
        cached_output_.reset();
        return {};
    }
    auto vfr = std::dynamic_pointer_cast<const VideoFrameRepresentation>(inputs[0]);
    if (!vfr) {
        cached_output_.reset();
        return {};
    }
    if (!parameters.empty()) set_parameters(parameters);

    cached_output_ = process(vfr);
    return {};
}

std::shared_ptr<const VideoFrameRepresentation>
HvdChromaDecoderStage::process(
    std::shared_ptr<const VideoFrameRepresentation> source) const
{
    if (!source) return nullptr;
    return std::make_shared<HvdDecodedRepresentation>(std::move(source), config_);
}

bool HvdChromaDecoderStage::trigger(
    const std::vector<ArtifactPtr>& inputs,
    const std::map<std::string, ParameterValue>& parameters,
    IObservationContext&)
{
    trigger_in_progress_.store(true);
    auto fail = [&](const std::string& msg) {
        export_status_ = "Error: " + msg;
        trigger_in_progress_.store(false);
        return false;
    };

    if (!parameters.empty()) set_parameters(parameters);

    // Self-sufficient, per TriggerableStage's contract ("reading all fields
    // from input and writing to output file") — build the representation
    // fresh from `inputs` rather than assuming execute() already ran. Falls
    // back to whatever's already cached if inputs weren't (re-)supplied.
    if (!inputs.empty() && inputs[0]) {
        if (auto vfr = std::dynamic_pointer_cast<const VideoFrameRepresentation>(inputs[0])) {
            cached_output_ = process(vfr);
        }
    }

    // Default to stdout ("-", see ContainerFromPath()/VideoWriter::Open())
    // when nothing was ever set, rather than hard-failing: this only
    // matters for headless invocation (e.g. orc-cli without an
    // output_path=... argument at all), where omitting it is a
    // reasonable way to say "just pipe it somewhere" -- via the GUI,
    // output_path_ is essentially never empty at this point because the
    // file-browser parameter widget always writes SOME path back through
    // set_parameters() once touched. The GUI-facing default shown in
    // get_parameter_descriptors() deliberately stays an empty string, so
    // a GUI user still sees a blank field prompting an explicit choice
    // instead of a silent pipe default there too.
    if (output_path_.empty()) output_path_ = "-";

    auto repr = std::dynamic_pointer_cast<const HvdDecodedRepresentation>(cached_output_);
    if (!repr) return fail("no input (connect a video source and try again)");

    const auto [w, h] = repr->active_picture_size();
    if (w == 0 || h == 0) return fail("no active picture geometry");

    // Container is decided by output_path_'s extension EXCEPT when the
    // destination is a pipe: MP4 needs to seek back at the end to rewrite
    // its header (moov atom) with the final duration/index, which a pipe
    // fundamentally can't do, and Matroska (while more pipe-tolerant than
    // MP4) still isn't designed for it the way NUT explicitly is — every
    // NUT frame is fully self-contained as it's written, nothing deferred
    // to a trailer a pipe reader could never seek back to read anyway.
    // Two distinct things count as "a pipe" here:
    //   - output_path_ == "-": the standard ffmpeg/orc-cli convention for
    //     "stdout", e.g. `orc-cli ... output_path=- | ffplay -`. This is
    //     NOT a filesystem path at all — is_fifo() on it would just look
    //     for (and not find) a real file literally named "-", which is
    //     exactly what silently produced an empty pipe before this: bytes
    //     went into a file called "-" on disk, never into the shell pipe
    //     ffplay was reading from. VideoWriter::Open() below translates
    //     this to libav's own "pipe:1" URL.
    //   - an actual named pipe (mkfifo) on disk. is_fifo() only
    //     recognises POSIX named pipes; on Windows a named pipe
    //     (\\.\pipe\...) won't be detected here and falls through to
    //     extension-based selection instead.
    std::error_code fifo_ec;
    const bool is_pipe = output_path_ == "-" ||
        (std::filesystem::is_fifo(output_path_, fifo_ec) && !fifo_ec);
    const ::hvd::VideoContainer container = is_pipe
        ? ::hvd::VideoContainer::kNutRaw
        : ::hvd::ContainerFromPath(output_path_);
    std::ofstream raw_file;
    ::hvd::VideoWriter video_writer;
    std::unique_ptr<::hvd::FrameEncodingStreambuf> video_streambuf;
    std::unique_ptr<std::ostream> video_stream;
    std::ostream* out_ptr = nullptr;

    if (container == ::hvd::VideoContainer::kRaw) {
        raw_file.open(output_path_, std::ios::binary | std::ios::trunc);
        if (!raw_file) return fail("could not open '" + output_path_ + "' for writing");
        out_ptr = &raw_file;
    } else {
        // Frame rate for the muxer: the same is_pal test used everywhere
        // else in this file (system == NTSC or PAL_M stays on the 30000/
        // 1001 side; PAL_M genuinely runs at NTSC's field rate despite its
        // PAL-style chroma). Expressed as a rational so timestamps land on
        // the exact broadcast rate rather than accumulating a rounding
        // error from a double.
        const auto vp = cached_output_->get_video_parameters();
        const bool is_pal = vp.has_value() &&
            !(vp->system == VideoSystem::NTSC || vp->system == VideoSystem::PAL_M);
        const int fps_num = is_pal ? 25 : 30000;
        const int fps_den = is_pal ? 1 : 1001;
        // Pixel aspect ratio for the muxer: the exact same source of
        // truth as get_preview_capability()'s dar_correction_factor (see
        // that function's long comment on why it has to come from
        // standard_dar_correction(), not from the active-area pixel
        // count) -- so the exported file/pipe ends up with the same
        // corrected 4:3 shape the preview already shows, instead of a
        // player defaulting to square pixels because nothing told it
        // otherwise.
        const double sample_aspect_ratio =
            vp.has_value() ? standard_dar_correction(vp->system) : 1.0;
        if (!video_writer.Open(output_path_, static_cast<int>(w), static_cast<int>(h),
                               fps_num, fps_den, container, sample_aspect_ratio)) {
            return fail("could not open '" + output_path_ + "' for video export: " +
                        video_writer.last_error());
        }
        video_streambuf = std::make_unique<::hvd::FrameEncodingStreambuf>(
            video_writer, static_cast<size_t>(w) * static_cast<size_t>(h) * 3);
        video_stream = std::make_unique<std::ostream>(video_streambuf.get());
        out_ptr = video_stream.get();
    }
    std::ostream& out = *out_ptr;

    // Every success return below goes through this first: a video file
    // isn't valid just because the frame bytes happened to reach the
    // encoder — the container trailer (and any frames FFmpeg is still
    // holding back internally) still needs writing. No-op for the raw
    // (kRaw) path.
    auto finalize_video = [&]() -> bool {
        if (container == ::hvd::VideoContainer::kRaw) return true;
        if (!video_writer.Finalize()) {
            fail("failed finalizing '" + output_path_ + "': " + video_writer.last_error());
            return false;
        }
        return true;
    };

    const FrameIDRange range = cached_output_->frame_range();
    if (range.last < range.first) return fail("frame range was empty");
    const uint64_t total = range.last - range.first + 1;

    if (progress_callback_) progress_callback_(0, total, "Starting export...");

    {
        // Composite exports always go through the FIELD pipeline
        // (engine/sequence.h): 2D = decoupled fields (no window overlap,
        // full within-chunk parallelism), 3D = coupled windows with
        // chunk_overlap frames of temporal context. Chunks are serial
        // (in-order output is the contract) with full per-chunk
        // parallelism. A false return before anything was written means
        // the source is Y/C-native (no composite for the field pipeline
        // to run on) — fall through to the frame-based paths below.
        ::hvd::HvdEngine engine;
        engine.SetFftThreads(config_.fft_threads);
        std::mutex read_mutex;  // the chunk reader always takes one, even
                                // though nothing else touches source_
                                // concurrently on this path
        const FrameID chunk = static_cast<FrameID>(
            std::max(1, config_.chunk_frames));

        // Decode (this thread, the loop below) and I/O (write_thread)
        // run CONCURRENTLY. Before this, decode_sequence_chunk_and_write_
        // rgb24() wrote straight to `out`, so writing chunk N blocked
        // decoding chunk N+1 on the same thread -- fine for a plain file
        // (write returns almost instantly) but not for `out` when it's a
        // pipe being drained by a realtime consumer (e.g. `| ffplay -`):
        // every chunk's write then had to wait for that chunk's own
        // *playback* time, on top of its decode time, instead of the two
        // overlapping. bounded_queue_depth caps how many DECODED-BUT-NOT-
        // YET-WRITTEN chunks can pile up in memory if the writer/consumer
        // falls behind -- decode can run ahead of a slow consumer, just
        // not arbitrarily far ahead.
        constexpr size_t kBoundedQueueDepth = 4;
        std::mutex queue_mutex;
        std::condition_variable queue_cv;
        std::queue<std::string> chunk_queue;
        bool producer_done = false;
        std::atomic<bool> io_failed{false};
        std::string io_failure_message;

        std::thread writer_thread([&]() {
            for (;;) {
                std::string chunk_bytes;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    queue_cv.wait(lock, [&] {
                        return !chunk_queue.empty() || producer_done ||
                               io_failed.load(std::memory_order_relaxed);
                    });
                    if (io_failed.load(std::memory_order_relaxed)) return;
                    if (chunk_queue.empty()) {
                        if (producer_done) return;
                        continue;  // spurious wake with nothing queued yet
                    }
                    chunk_bytes = std::move(chunk_queue.front());
                    chunk_queue.pop();
                }
                queue_cv.notify_all();  // wake a producer waiting on "not full"
                out.write(chunk_bytes.data(),
                         static_cast<std::streamsize>(chunk_bytes.size()));
                if (!out) {
                    io_failure_message = "I/O error writing '" + output_path_ + "'";
                    io_failed.store(true);
                    queue_cv.notify_all();
                    return;
                }
            }
        });

        uint64_t written = 0;
        bool sequence_ok = true;
        for (FrameID t0 = range.first; sequence_ok && t0 <= range.last; ) {
            if (io_failed.load(std::memory_order_relaxed)) {
                sequence_ok = false;
                break;
            }
            const FrameID t1 = std::min(t0 + chunk - 1, range.last);
            std::ostringstream chunk_buf(std::ios::binary);
            if (!repr->decode_sequence_chunk_and_write_rgb24(
                    t0, t1, range.first, range.last, engine, read_mutex,
                    chunk_buf)) {
                sequence_ok = false;
                break;
            }
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cv.wait(lock, [&] {
                    return chunk_queue.size() < kBoundedQueueDepth ||
                           io_failed.load(std::memory_order_relaxed);
                });
                if (io_failed.load(std::memory_order_relaxed)) {
                    sequence_ok = false;
                    break;
                }
                chunk_queue.push(chunk_buf.str());
            }
            queue_cv.notify_all();
            written += t1 - t0 + 1;
            if (progress_callback_) {
                progress_callback_(written, total,
                                   "Exported frame " + std::to_string(written) +
                                       "/" + std::to_string(total) +
                                       " (field pipeline)");
            }
            t0 = t1 + 1;
        }
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            producer_done = true;
        }
        queue_cv.notify_all();
        writer_thread.join();

        if (io_failed.load()) return fail(io_failure_message);
        if (sequence_ok) {
            if (written == 0) return fail("frame range was empty");
            if (!finalize_video()) return false;
            export_status_ = "Export complete: " + std::to_string(written) +
                             " frames (field pipeline) -> " + output_path_;
            trigger_in_progress_.store(false);
            return true;
        }
        if (written > 0)
            return fail("sequence decode failed mid-export at frame " +
                        std::to_string(written));
        // Y/C-native: continue to the frame-based export paths below.
    }

    const unsigned hw = std::thread::hardware_concurrency();
    const uint64_t num_threads =
        std::max<uint64_t>(1, std::min<uint64_t>(hw ? hw : 4, total));

    if (num_threads == 1) {
        // Single frame, or hardware_concurrency() couldn't tell us anything
        // useful and we're being conservative: no point paying thread-pool
        // overhead for one worker.
        ::hvd::HvdEngine engine;
        std::mutex read_mutex;  // decode_and_write_rgb24 always takes one
        uint64_t written = 0;
        for (FrameID id = range.first; id <= range.last; ++id) {
            if (!repr->decode_and_write_rgb24(id, engine, read_mutex, out)) {
                return fail("failed decoding frame " + std::to_string(id));
            }
            ++written;
            if (progress_callback_) {
                progress_callback_(written, total,
                                   "Exported frame " + std::to_string(written) +
                                       "/" + std::to_string(total));
            }
        }
        if (written == 0) return fail("frame range was empty");
        if (!finalize_video()) return false;
        export_status_ = "Export complete: " + output_path_;
        trigger_in_progress_.store(false);
        return true;
    }

    // --- Parallel export ----------------------------------------------
    // Each worker owns its own HvdEngine: Fft2d keeps per-call scratch
    // buffers that aren't safe to share across concurrent calls, so
    // threads can't share one engine the way decoded()'s cached path does.
    // `read_mutex` is shared by all workers and guards the one part that
    // genuinely isn't safe to run concurrently — reading this frame's raw
    // samples out of the source representation (see decode_and_write_rgb24
    // for why) — the lock is held only for that copy, not for the
    // expensive decode work that follows.
    //
    // Frames decode out of order (workers just grab the next undecided
    // FrameID off an atomic counter), but this thread writes them to `out`
    // strictly in order, so a frame decoded early just waits in `pending`
    // until its turn. In practice decode (the FFT + IRLS/CG solve) is far
    // slower than writing a few hundred KB to disk, so `pending` should
    // stay small — but nothing here bounds it, so a very fast decode
    // config (e.g. cg_iterations=0) against a very slow disk could let it
    // grow; not worth the extra complexity of a bounded queue unless that
    // turns out to matter in practice.
    std::atomic<FrameID> next_to_decode{range.first};
    std::mutex read_mutex;
    std::mutex results_mutex;
    std::condition_variable results_cv;
    std::unordered_map<FrameID, std::string> pending;
    std::atomic<bool> failed{false};
    std::string failure_message;
    std::mutex failure_mutex;

    auto worker_fn = [&]() {
        LimitOpenMpThreadsPerWorker();
        ::hvd::HvdEngine engine;  // one plan cache per thread, reused across
                                  // every frame that thread decodes
        engine.SetFftThreads(1);  // see engine.h's doc comment: this worker
                                  // IS the parallelism unit, FFTW fanning out
                                  // internally too would oversubscribe
        for (;;) {
            if (failed.load(std::memory_order_relaxed)) return;
            const FrameID id = next_to_decode.fetch_add(1);
            if (id > range.last) return;

            std::ostringstream buf(std::ios::binary);
            const bool ok = repr->decode_and_write_rgb24(id, engine, read_mutex, buf);
            if (!ok) {
                std::lock_guard<std::mutex> lock(failure_mutex);
                if (!failed.exchange(true)) {
                    failure_message = "failed decoding frame " + std::to_string(id);
                }
                results_cv.notify_all();
                return;
            }
            {
                std::lock_guard<std::mutex> lock(results_mutex);
                pending.emplace(id, buf.str());
            }
            results_cv.notify_all();
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(num_threads));
    for (uint64_t i = 0; i < num_threads; ++i) workers.emplace_back(worker_fn);

    uint64_t written = 0;
    FrameID next_to_write = range.first;
    bool io_failed = false;
    while (next_to_write <= range.last) {
        std::string bytes;
        {
            std::unique_lock<std::mutex> lock(results_mutex);
            results_cv.wait(lock, [&] {
                return failed.load(std::memory_order_relaxed) ||
                       pending.find(next_to_write) != pending.end();
            });
            if (failed.load(std::memory_order_relaxed)) break;
            auto it = pending.find(next_to_write);
            bytes = std::move(it->second);
            pending.erase(it);
        }
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        if (!out) {
            io_failed = true;
            failed.store(true);
            results_cv.notify_all();
            break;
        }
        ++next_to_write;
        ++written;
        if (progress_callback_) {
            progress_callback_(written, total,
                               "Exported frame " + std::to_string(written) +
                                   "/" + std::to_string(total));
        }
    }

    for (auto& t : workers) t.join();

    if (io_failed) return fail("I/O error writing '" + output_path_ + "'");
    if (failed.load()) {
        return fail(failure_message.empty() ? "export failed" : failure_message);
    }
    if (written == 0) return fail("frame range was empty");
    if (!finalize_video()) return false;
    export_status_ = "Export complete: " + std::to_string(written) +
                     " frames -> " + output_path_;
    trigger_in_progress_.store(false);
    return true;
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

    // The SDK's ParameterDescriptor list has no notion of UI sections/
    // categories (see ParameterConstraints below: min/max/default/enum-
    // choices/bool/optional, nothing group-related) -- the host just renders
    // this vector as a flat list, in order. So the "2 parties" split lives
    // here, in code, via two things a flat list CAN express:
    //   1. Order: every BASIC parameter comes before every ADVANCED one.
    //   2. Label prefix: advanced entries are labelled "Advanced: ..." so
    //      they're visually distinguishable even in a flat list/search box.
    // Parameter KEYS (kLambdaC, kFast, ...), TYPES, RANGES and DEFAULTS are
    // all unchanged from before this reorganisation -- existing saved
    // presets/projects still round-trip identically. If the SDK grows real
    // parameter groups later, this split maps onto that directly.

    return {
        // ===================================================================
        // BASIC -- day-to-day controls. Safe to explore; defaults are sane.
        // ===================================================================
        ParameterDescriptor{kFast, "Fast mode",
            "Same algorithm, cheaper logistics (reference's --fast, THEORY "
            "9f): adaptive solver early-exit with a 2/3 iteration cap, and "
            "tile-resolution motion-confidence maps in the 3D path. "
            "Reference measurement: >=2x speed, never worse than 0.2 dB.",
            ParameterType::BOOL, boolean(true)},
        ParameterDescriptor{kFieldOrder, "Field order",
            "0 = Auto (default): the order is MEASURED from the signal -- "
            "under the true order each field's lines interpolate the "
            "other's at +0.5 line, under the inverted order at -0.5, a "
            "deterministic half-line vertical correlation vote (majority "
            "over the export window; falls back to the ld-decode format "
            "convention, field 1 = top, on flat content). 1 = force "
            "field 1 top. 2 = force field 1 bottom. Wrong-order symptoms: "
            "one-line serration on static horizontal edges, and motion "
            "that combs even through a player's deinterlacer.",
            ParameterType::INT32, integer(0, 2, 0)},
        ParameterDescriptor{kEnableTemporal, "Enable 3D (temporal)",
            "Adds six motion-gated neighbour-field equations per field "
            "(f\u00b11/\u00b12/\u00b13) to the field-granularity solve, resolving the "
            "Y/C ambiguity that a single field cannot (cross-colour, "
            "rainbow on fine detail), and from pass 2 the synth-reference "
            "anchor adds motion-compensated temporal noise reduction. "
            "Strength defaults to adaptive (see Temporal strength, in "
            "Advanced). Measured on the regression scene: +1.9 dB "
            "single-pass, +6.4 dB anchored 2-pass over per-field 2D.",
            ParameterType::BOOL, boolean(false)},
        ParameterDescriptor{kBidirectional, "Bidirectional 3D",
            "Use BOTH past and future fields as temporal neighbors "
            "(default). Off = past-only: ~1.6x faster on the 3D path but "
            "measurably worse (reference photo test, PORTING.md Sec. 19: "
            "-1.1 dB and rainbow 1.9% vs 1.5%) because the two sides' "
            "failure modes are complementary -- scene cuts and occlusions "
            "break at most one side. Only worth it when speed matters more "
            "than the last dB, e.g. previewing.",
            ParameterType::BOOL, boolean(true)},
        ParameterDescriptor{kSelective3d, "Selective 3D",
            "Full-window 2D decode plus the complete 3D machinery re-run "
            "only on the crop of the most Y/C-ambiguous tiles, blended in. "
            "Pays off on LOCALIZED ambiguity (fan grilles, blinds, one "
            "textured area in a flat scene). Handles up to 4 separate "
            "horizontal artifact bands per frame (top+middle+bottom zones "
            "measured: zones fixed to within ~0.1% of full 3D at ~55% of "
            "its wall time). On diffuse content the detector finds no "
            "worthwhile crop and the window stays plain 2D by design "
            "(PORTING.md Sec. 21/21c). Ignored when 3D is off.",
            ParameterType::BOOL, boolean(false)},
        ParameterDescriptor{kPasses, "3D passes",
            "Gauss-Seidel fixed-point passes over each export chunk "
            "(sequence pipeline only). 1 = single pass. From pass 2 the "
            "decode->NR->re-encode anchor engages: motion-compensated "
            "temporal noise reduction whose reference never gets trusted "
            "where the raw data contradicts it. 2 (default, matches the "
            "reference's anchored-mode value) is where most of the "
            "quality is; dropping to 1 loses that anchor and gives up a "
            "fair amount of quality for a faster pass.",
            ParameterType::INT32, integer(1, 4, 2)},
        ParameterDescriptor{kAcc, "Automatic Color Control",
            "Calibrate saturation from burst amplitude (colour path only).",
            ParameterType::BOOL, boolean(true)},
        ParameterDescriptor{kChromaGain, "Chroma Gain",
            "Gain applied to U/V on top of ACC, matching the classic "
            "decoder's Chroma Gain. Range 0.0-10.0.",
            ParameterType::DOUBLE, real(0.0, 10.0, 1.0)},
        ParameterDescriptor{kMonochrome, "Monochrome",
            "Zero the chroma channel.", ParameterType::BOOL, boolean(false)},
        ParameterDescriptor{kChromaPhaseDeg, "Chroma phase (deg)",
            "Rotation applied to the burst-locked phase reference before "
            "the solver runs, same idea as the classic decoder's Chroma "
            "Phase (Comb::transformIQ). Range -180 to 180. The recovered "
            "chroma has been persistently 180 deg off since the Python "
            "reference, hence the default; treat it as tunable per-capture, "
            "not as fixed.",
            ParameterType::DOUBLE, real(-180.0, 180.0, 180.0)},
        ParameterDescriptor{kPreviewFullRaster, "Preview: full raster",
            "PREVIEW ONLY (never affects the exported/written frames, which "
            "always honour the configured VideoParameters crop). ON: show "
            "the whole stored raster -- sync, blanking and the colour burst "
            "appear as monochrome signal in the margins, decoded colour "
            "stays confined to the active area. OFF: crop to the active "
            "picture, matching the export. Turn OFF if you need the "
            "histogram/vectorscope panels to read only the active picture.",
            ParameterType::BOOL, boolean(true)},
        ParameterDescriptor{kPreviewFieldView, "Preview: field view",
            "PREVIEW ONLY. Navigate per FIELD instead of per woven frame: "
            "each item is one field at native field height (one row per "
            "field line, no interpolation). The honest view for per-field "
            "artefacts -- dropouts, weave/field-order errors, PAL "
            "V-switch/Hanover checks -- that the frame weave smears across "
            "two fields.",
            ParameterType::BOOL, boolean(false)},
        ParameterDescriptor{kOutputPath, "Output file path",
            "Destination for the Export button. Left empty when the "
            "export actually runs, it defaults to STDOUT (same as '-', "
            "below) -- convenient for headless/CLI use where omitting "
            "it usually just means 'pipe it somewhere'. Set this to '-' "
            "(a single dash) explicitly to stream to STDOUT instead of "
            "a file -- e.g. "
            "for `orc-cli ... output_path=- | ffplay -`. A NAMED PIPE "
            "(mkfifo) works the same way. Either one always goes out as "
            "NUT-wrapped raw RGB24 video regardless of any extension in "
            "the path (NUT is the one streamable-by-design container of "
            "the three below -- a pipe/stdout can't be seeked back to "
            "write a trailer/index the way FFV1/MKV or MPEG-4/MP4 need "
            "to). Otherwise (a normal file path) the extension decides: "
            ".mkv -> FFV1, mathematically lossless "
            "(encoded as 10-bit planar RGB, so there's no YUV chroma-"
            "subsampling loss either -- decoded pixels round-trip "
            "exactly). .mp4 -> MPEG-4 Part 2, lossy (4:2:0 chroma + DCT "
            "compression, fixed quantiser -- avcodec's own native "
            "encoder, no external library, so noticeably less "
            "size-efficient than H.264 at the same visual quality, which "
            "is the trade for nothing extra to build). "
            "Anything else falls back to the original raw interleaved "
            "RGB24 dump, no container: view with e.g. ffplay -f rawvideo "
            "-pixel_format rgb24 -video_size WxH.",
            ParameterType::FILE_PATH,
            ParameterConstraints{std::nullopt, std::nullopt,
                                 ParameterValue{std::string{}}, {}, false,
                                 std::nullopt}},
        ParameterDescriptor{kLambdaC, "Chroma smoothness",
            "Arbitration prior. Higher = smoother chroma (less rainbowing); "
            "lower = sharper chroma.",
            ParameterType::DOUBLE, real(0.0, 8.0, 1.0)},
        ParameterDescriptor{kCgIterations, "Solver iterations",
            "Conjugate-gradient iterations. 0 = holographic init only "
            "(fast preview). Default kept small (2) so a fresh/"
            "never-configured stage decodes quickly by default; raise it "
            "when tuning for final quality, not as a first thing to try.",
            ParameterType::INT32, integer(0, 400, 2)},
        ParameterDescriptor{kSymmetryVariant, "Spectral-symmetry init",
            "Add the Transform-NTSC certified-chroma init variant.",
            ParameterType::BOOL, boolean(false)},

        // ===================================================================
        // ADVANCED / FINE-TUNING -- solver and algorithm internals. Defaults
        // are already tuned from measured regression results (see each
        // description); change these only when diagnosing a specific
        // artefact, not as a first thing to try.
        // ===================================================================
        ParameterDescriptor{kCustomSubcarrier, "Advanced: Non-standard subcarrier",
            "For sources with a deliberately lowered colour subcarrier -- "
            "notably JVC VHD at 2556.8 kHz. Tracks the frequency below "
            "instead of the standard's nominal fsc (NTSC 3579.5455 kHz, "
            "PAL 4433.61875 kHz); the sample grid is unchanged, only the "
            "carrier moves. OFF (default) uses the standard.",
            ParameterType::BOOL, boolean(false)},
        ParameterDescriptor{kSubcarrierKhz, "Advanced: Subcarrier frequency (kHz)",
            "Used only when the checkbox above is ON. VHD 2556.8 (exact "
            "line lock 2556.8182 = 162.5 x fH), NTSC 3579.5455, PAL "
            "4433.61875. Wrong by a few kHz and the hue rotates "
            "progressively along each line: sweep it while watching a flat "
            "colour area.",
            ParameterType::DOUBLE, real(500.0, 6000.0, 2556.8)},
        ParameterDescriptor{kOddGateFloor, "Advanced: Odd-offset gate floor",
            "Weight kept on opposite-parity (odd) neighbour equations where "
            "the half-line envelope says that field cannot see the feature. "
            "CEILING of an automatic per-pixel floor: the value is scaled "
            "down where the field's vertical profile is a one-line EXTREMUM "
            "(hair, fur, fine fabric -- the opposite parity never sampled the "
            "feature, so its equation is uninformed) and kept where it is a "
            "monotonic edge (both fields see it; the equation is biased but "
            "informative). Leave at 0.35 unless diagnosing.",
            ParameterType::DOUBLE, real(0.0, 1.0, 0.35)},
        ParameterDescriptor{kCoherenceGate, "Advanced: Coherence gate",
            "InSAR-style complex-coherence gating of the temporal equations. "
            "0 disables it; higher trusts the coherence measurement more.",
            ParameterType::DOUBLE, real(0.0, 1.0, 0.6)},
        ParameterDescriptor{kCharbonnierEps, "Advanced: Luma edge scale (IRE)",
            "Edge-preservation scale of the luma prior, in IRE.",
            ParameterType::DOUBLE, real(0.05, 5.0, 0.5)},
        ParameterDescriptor{kChromaEps, "Advanced: Chroma edge scale (IRE)",
            "Edge-preservation scale of the chroma prior, in IRE.",
            ParameterType::DOUBLE, real(0.05, 5.0, 1.0)},
        ParameterDescriptor{kStructureCoupling,
            "Advanced: Y->chroma edge coupling",
            "Open the chroma edge where luma has one (removes hanging dots).",
            ParameterType::DOUBLE, real(0.0, 2.0, 0.25)},
        ParameterDescriptor{kCgTol, "Advanced: Solver early-exit tolerance",
            "Relative gradient-norm at which the conjugate-gradient solve "
            "stops early. 0 = auto (0.02, or 0.10 in fast mode). Measured "
            "on real re-encoded photo content (PORTING.md Sec. 19): 0.3 "
            "combined with fast mode is ~2.3x the default-path speed at "
            "equal-or-slightly-better PSNR and flat-zone rainbow score. "
            "Iteration count above remains the hard ceiling.",
            ParameterType::DOUBLE, real(0.0, 0.9, 0.0)},
        ParameterDescriptor{kDiagPrior, "Advanced: Diagonal chroma prior",
            "Oriented +/-45 deg chroma prior weight (reference's "
            "--diag-prior), renormalised so total prior mass is unchanged. "
            "A measured trade-off, not a win: trades axis-aligned chroma "
            "sharpness (-1 dB on SMPTE bars) for diagonal cross-colour "
            "suppression (+2 dB on zoneplate torture). 0 = off (default); "
            "try ~0.5-1.0 on diagonal-artifact-heavy material such as fine "
            "weaves or venetian blinds.",
            ParameterType::DOUBLE, real(0.0, 2.0, 0.0)},
        ParameterDescriptor{kNrAnchor, "Advanced: Anchor strength",
            "Weight of the decode->NR->re-encode anchor once it engages "
            "(passes >= 2, above) -- how strongly the temporally-denoised "
            "reference pulls the solve versus the raw per-field data. "
            "This was previously silently fixed at 1.0 (the reference "
            "default) with no GUI control. 0 disables the anchor's pull "
            "even with passes >= 2 (Gauss-Seidel iteration continues, "
            "just without the NR reference).",
            ParameterType::DOUBLE, real(0.0, 3.0, 1.0)},
        ParameterDescriptor{kChunkFrames, "Advanced: 3D chunk size",
            "Frames per export window in 3D mode (plus 'Temporal context "
            "frames', below, on each side). Bounds memory; larger chunks "
            "slightly reduce edge effects at chunk boundaries.",
            ParameterType::INT32, integer(1, 24, 6)},
        ParameterDescriptor{kChunkOverlap, "Advanced: Temporal context frames",
            "Frames of 3D context added on each side of the export "
            "window (see 3D chunk size, above) -- AND, since this is the "
            "same value the live preview's own mini-3D window uses "
            "(id +/- this many frames), also how much temporal context a "
            "single previewed frame costs. Default 1 gives a 3-frame "
            "window (n-1, n, n+1): the quality difference between 2D and "
            "3D is small enough that this is plenty for most content, and "
            "keeping preview and export on the same value means what you "
            "judge in the preview is genuinely what the export costs -- "
            "raising it only in the export while the preview stayed "
            "fixed used to be why enabling 3D felt fine in preview but "
            "was disproportionately expensive on export.",
            ParameterType::INT32, integer(0, 8, 1)},
        ParameterDescriptor{kParallelAcrossFields,
            "Advanced: Parallelise across fields (2D/fast decode)",
            "Export throughput A/B toggle for the decoupled-field decode "
            "path (2D, or fast mode) -- doesn't affect image quality, "
            "only wall-clock time, and only when NOT using 3D. ON "
            "(default): every field in the current chunk decodes "
            "CONCURRENTLY, one per core, with each field's own internal "
            "solver forced single-threaded (nested OpenMP parallelism is "
            "off by default) -- wins when the chunk has at least as many "
            "fields as you have cores (raise 3D chunk size, above, to get "
            "there). OFF: fields decode ONE AT A TIME, but each one's "
            "internal solver is then free to use every core for itself -- "
            "same parallelisation shape as this stage's own live preview. "
            "Likely wins when 3D chunk size is small relative to your "
            "core count, since ON would otherwise leave cores idle. "
            "Neither setting changes the decoded pixels: fields in this "
            "path don't read each other's state, so try both and keep "
            "whichever one measures faster on your hardware.",
            ParameterType::BOOL, boolean(true)},
        ParameterDescriptor{kTemporalStrength, "Advanced: Temporal strength",
            "Weight of the motion-compensated neighbour-field equations "
            "once Enable 3D is on. 0 = ADAPTIVE (default): the strength is "
            "measured from the content per window -- same-parity fields "
            "carry the chroma identically but flip luma leakage in sign, "
            "so their demodulated difference isolates exactly the Y/C "
            "ambiguity the 3D equations exist to resolve; strong "
            "cross-colour content gets strong 3D (up to 1.5), clean "
            "content stays near the 0.15 floor instead of lifting chroma "
            "noise. Any positive value forces that fixed strength "
            "(reference --3d uses 0.5).",
            ParameterType::DOUBLE, real(0.0, 4.0, 0.0)},
        ParameterDescriptor{kMcTile, "Advanced: Motion tile size (px)",
            "Block-matching tile size for the temporal path.",
            ParameterType::INT32, integer(8, 128, 32)},
        ParameterDescriptor{kMcSearch, "Advanced: Motion search radius (px)",
            "Block-matching search radius for the temporal path.",
            ParameterType::INT32, integer(2, 64, 16)},
        ParameterDescriptor{kDebugDir, "Advanced: Diagnostics directory",
            "When set, the export also writes, per frame, a PGM map of the "
            "RESIDUAL CARRIER-BAND ENERGY in the decoded luma -- i.e. the "
            "rainbow/dot-crawl you can see, measured (bright = separation "
            "failed there) -- plus diag.txt logging the decoder's decisions "
            "per chunk (adaptive strength chosen, measured ambiguity, "
            "noise, gates, field-order vote). If an artifact persists, "
            "send the map of the bad zone and the matching diag.txt lines "
            "instead of describing it.",
            ParameterType::FILE_PATH,
            ParameterConstraints{std::nullopt, std::nullopt,
                                 ParameterValue{std::string{}}, {}, false,
                                 std::nullopt}}};
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
        {kFast, config_.fast},
        {kNrAnchor, static_cast<double>(config_.nr_anchor)},
        {kOddGateFloor, static_cast<double>(config_.odd_gate_floor)},
        {kCoherenceGate, static_cast<double>(config_.coherence_gate)},
        {kCgTol, static_cast<double>(config_.cg_tol)},
        {kBidirectional, config_.bidirectional},
        {kSelective3d, config_.selective_3d},
        {kPasses, static_cast<int32_t>(config_.passes)},
        {kChunkFrames, static_cast<int32_t>(config_.chunk_frames)},
        {kChunkOverlap, static_cast<int32_t>(config_.chunk_overlap)},
        {kParallelAcrossFields, config_.parallel_across_fields},
        {kFieldOrder, static_cast<int32_t>(config_.field_order)},
        {kDebugDir, config_.debug_dir},
        {kDiagPrior, static_cast<double>(config_.diag_prior)},
        {kAcc, config_.acc},
        {kChromaGain, static_cast<double>(config_.chroma_gain)},
        {kMonochrome, config_.monochrome},
        {kCustomSubcarrier, config_.custom_subcarrier},
        {kSubcarrierKhz, config_.subcarrier_khz},
        {kPreviewFullRaster, preview_full_raster_},
        {kPreviewFieldView, preview_field_view_},
        {kSymmetryVariant, config_.symmetry_variant},
        {kChromaPhaseDeg, static_cast<double>(config_.chroma_phase_deg)},
        {kEnableTemporal, config_.enable_temporal},
        {kTemporalStrength, static_cast<double>(config_.temporal_strength)},
        {kMcTile, static_cast<int32_t>(config_.mc_tile)},
        {kMcSearch, static_cast<int32_t>(config_.mc_search)},
        {kOutputPath, output_path_}};
}

bool HvdChromaDecoderStage::set_parameters(
    const std::map<std::string, ParameterValue>& params)
{
    // Generic in the destination type. Most config fields are float, but
    // subcarrier_khz is double: the GUI's step there (0.1 Hz) is finer than
    // float's ULP at 2556 kHz (0.24 Hz), so a float destination would swallow
    // spinbox steps. See hvd_config.h.
    auto get_double = [&](const char* key, auto& dst) {
        using T = std::decay_t<decltype(dst)>;
        auto it = params.find(key);
        if (it == params.end()) return;
        if (std::holds_alternative<double>(it->second))
            dst = static_cast<T>(std::get<double>(it->second));
        else if (std::holds_alternative<int32_t>(it->second))
            dst = static_cast<T>(std::get<int32_t>(it->second));
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
    auto get_string = [&](const char* key, std::string& dst) {
        auto it = params.find(key);
        if (it != params.end() && std::holds_alternative<std::string>(it->second))
            dst = std::get<std::string>(it->second);
    };

    get_double(kLambdaC, config_.lambda_c);
    get_double(kCharbonnierEps, config_.charbonnier_eps);
    get_double(kChromaEps, config_.chroma_eps);
    get_double(kStructureCoupling, config_.structure_coupling);
    get_int(kCgIterations, config_.cg_iterations);
    get_bool(kFast, config_.fast);
    get_double(kNrAnchor, config_.nr_anchor);
    get_double(kOddGateFloor, config_.odd_gate_floor);
    get_double(kCoherenceGate, config_.coherence_gate);
    get_double(kCgTol, config_.cg_tol);
    get_bool(kBidirectional, config_.bidirectional);
    get_bool(kSelective3d, config_.selective_3d);
    get_int(kPasses, config_.passes);
    get_int(kChunkFrames, config_.chunk_frames);
    get_int(kChunkOverlap, config_.chunk_overlap);
    get_bool(kParallelAcrossFields, config_.parallel_across_fields);
    get_int(kFieldOrder, config_.field_order);
    get_string(kDebugDir, config_.debug_dir);
    get_double(kDiagPrior, config_.diag_prior);
    get_bool(kAcc, config_.acc);
    get_double(kChromaGain, config_.chroma_gain);
    get_bool(kMonochrome, config_.monochrome);
    get_bool(kCustomSubcarrier, config_.custom_subcarrier);
    get_double(kSubcarrierKhz, config_.subcarrier_khz);
    get_bool(kPreviewFullRaster, preview_full_raster_);
    get_bool(kPreviewFieldView, preview_field_view_);
    get_bool(kSymmetryVariant, config_.symmetry_variant);
    get_double(kChromaPhaseDeg, config_.chroma_phase_deg);
    get_bool(kEnableTemporal, config_.enable_temporal);
    get_double(kTemporalStrength, config_.temporal_strength);
    get_int(kMcTile, config_.mc_tile);
    get_int(kMcSearch, config_.mc_search);
    get_string(kOutputPath, output_path_);

    // Preview view toggles change WHAT the preview shows, not what is
    // decoded — dropping cached_output_ for them would re-run the whole
    // (expensive) decode just to re-view the same frames.
    bool preview_only = !params.empty();
    for (const auto& [key, value] : params) {
        (void)value;
        if (key != kPreviewFullRaster && key != kPreviewFieldView) {
            preview_only = false;
            break;
        }
    }
    if (!preview_only) cached_output_.reset();
    refresh_status();
    return true;
}

// --------------------------------------------------------------- preview

StagePreviewCapability HvdChromaDecoderStage::get_preview_capability() const
{
    // Composed by hand (rather than PreviewHelpers::make_signal_preview_capability)
    // so the plugin also builds in the header-only in-tree configuration, which
    // links no host libraries.
    StagePreviewCapability capability{};
    if (!cached_output_ || cached_output_->frame_count() == 0) {
        return capability;
    }
    const auto params = cached_output_->get_video_parameters();
    if (!params.has_value() || !params->is_valid()) {
        return capability;
    }

    const bool is_pal =
        !(params->system == VideoSystem::NTSC || params->system == VideoSystem::PAL_M);

    // Colour type only now (ColourNTSC/ColourPAL) — this is what makes the
    // host's preview renderer take the carrier-backed colour path; it
    // requires has_colour_domain_type(capability) AND a working
    // IColourPreviewProvider (see PreviewRenderer::has_colour_domain_type).
    // We used to also advertise YC_NTSC/YC_PAL so the raw separated Y/C
    // channels stayed inspectable, but there's no separate-channel output
    // behind that anymore (has_separate_channels()/get_frame_luma()/
    // get_frame_chroma() were removed — see HvdDecodedRepresentation), so
    // advertising it would just be a dead end for the GUI.
    const VideoDataType colour_type =
        is_pal ? VideoDataType::ColourPAL : VideoDataType::ColourNTSC;
    capability.supported_data_types = {colour_type};

    // GEOMETRY SEMANTICS — matched to the SDK's own reference stage
    // (PreviewHelpers::make_signal_preview_capability), because getting
    // these two wrong is what made the preview render at a visibly
    // different aspect than tbc_source:
    //
    //  * geometry.active_* describes the ACTIVE PICTURE — metadata used
    //    for aspect handling and export dimensions. It is NOT the size of
    //    the image we deliver: the reference reports the active area
    //    (e.g. 768x483) while DELIVERING the full frame (910x525). An
    //    earlier revision reported the delivered full-raster size here,
    //    so the GUI's "4:3 (Display)" mode was reasoning about a 1.73
    //    picture instead of a 1.33 one.
    //
    //  * dar_correction_factor is the signal's PIXEL ASPECT RATIO, and
    //    cvbs_signal_constants.h is explicit that it must NOT be derived
    //    from a source's actual active-area values ("changing the active
    //    window would rescale the whole preview instead of re-framing
    //    it"). standard_dar_correction() is the canonical per-system
    //    value; using it is also what keeps this stage's preview
    //    dimensionally identical to tbc_source's.
    //
    // Both are therefore independent of the preview view toggles: those
    // change which pixels are delivered, not the shape of a pixel.
    const auto active_width =
        params->active_video_end > params->active_video_start
            ? static_cast<uint32_t>(params->active_video_end - params->active_video_start)
            : static_cast<uint32_t>(params->frame_width_nominal);
    const auto active_height =
        params->last_active_frame_line > params->first_active_frame_line
            ? static_cast<uint32_t>(params->last_active_frame_line - params->first_active_frame_line)
            : static_cast<uint32_t>(params->frame_height);

    const double dar_correction = standard_dar_correction(params->system);

    capability.navigation_extent.item_count =
        preview_field_view_ ? 2 * cached_output_->frame_count()
                            : cached_output_->frame_count();
    capability.navigation_extent.granularity = 1;
    capability.navigation_extent.item_label =
        preview_field_view_ ? "field" : "frame";
    capability.geometry.active_width = active_width;
    capability.geometry.active_height = active_height;
    capability.geometry.display_aspect_ratio = 4.0 / 3.0;
    capability.geometry.dar_correction_factor = dar_correction;
    return capability;
}

std::optional<ColourFrameCarrier> HvdChromaDecoderStage::get_colour_preview_carrier(
    uint64_t frame_index, PreviewNavigationHint hint) const
{
    (void)hint;
    auto repr = std::dynamic_pointer_cast<const HvdDecodedRepresentation>(cached_output_);
    if (!repr) return std::nullopt;
    if (preview_field_view_) {
        // Navigation items are FIELDS: item 2n is frame n's field 1, item
        // 2n+1 its field 2 — temporal order, so stepping walks fields in
        // time. This is the SDK's own documented use of navigation_extent
        // ("Field 42 of 400" vs "Frame 21 of 200"), not an invention.
        const FrameID frame = static_cast<FrameID>(frame_index / 2);
        const int field = static_cast<int>(frame_index % 2);
        return repr->build_colour_carrier(frame, preview_full_raster_, field);
    }
    return repr->build_colour_carrier(static_cast<FrameID>(frame_index),
                                      preview_full_raster_, -1);
}

namespace {
constexpr const char* kOptFrame = "frame";
constexpr const char* kOptField1 = "field1";
constexpr const char* kOptField2 = "field2";
constexpr const char* kOptFullRaster = "full_raster";
}  // namespace

// IStageCustomPreviewRenderer. See the header's doc comment on why the
// host's current dispatch does not reach this for a stage that also
// implements IStagePreviewCapability (verified against
// preview_renderer.cpp: get_available_outputs()'s "else if" on
// IStageCustomPreviewRenderer is only tried when dynamic_cast to
// IStagePreviewCapability fails, and this stage's capability path always
// succeeds). Implemented in full anyway, matching the pattern
// SourceAlignStage already uses (also currently unreachable there, for the
// identical reason) — this is the SDK-documented way to add views, and
// it's what a small, mechanical host-side merge of the two option lists
// needs in order to work: nothing here needs to change when that lands.
std::vector<PreviewOption> HvdChromaDecoderStage::get_preview_options() const
{
    std::vector<PreviewOption> options;
    auto repr = std::dynamic_pointer_cast<const HvdDecodedRepresentation>(cached_output_);
    if (!repr) return options;
    const ::hvd::FrameParams fp = repr->frame_params_public();
    if (fp.frame_width <= 0 || fp.frame_height <= 0 ||
        !cached_output_ || cached_output_->frame_count() == 0) {
        return options;
    }
    const uint64_t frames = cached_output_->frame_count();
    const auto a0 = std::max(0, fp.active_video_start);
    const auto a1 = fp.active_video_end > a0 ? fp.active_video_end : fp.frame_width;
    const auto y0 = std::max(0, fp.first_active_frame_line);
    const auto y1 = fp.last_active_frame_line > y0 ? fp.last_active_frame_line
                                                    : fp.frame_height;
    const uint32_t aw = static_cast<uint32_t>(std::max(0, a1 - a0));
    const uint32_t ah = static_cast<uint32_t>(std::max(0, y1 - y0));

    // NOTE: no "Frame" entry here. The carrier path already contributes
    // exactly one output named "Frame" (get_capability_preview_outputs()
    // hardcodes it), and after patches/0001 merges the two lists a second
    // frame entry would just be a confusing duplicate. render_preview()
    // still handles kOptFrame as its fallback, so nothing breaks if a
    // caller asks for it explicitly.
    //
    // Single-field views, native field height, no interpolation — the
    // honest view for per-field artefacts (dropouts, weave/field-order
    // errors, PAL V-switch/Hanover checks) the frame weave smears across
    // two fields.
    // Canonical per-system pixel aspect, same source of truth as the
    // capability path (see the geometry comment there).
    const auto vp = cached_output_->get_video_parameters();
    const double dar = standard_dar_correction(
        vp.has_value() ? vp->system : VideoSystem::NTSC);
    options.push_back({kOptField1, "Field 1", true, aw, (ah + 1) / 2, frames, dar});
    options.push_back({kOptField2, "Field 2", true, aw, ah / 2, frames, dar});
    // Full raster: sync/blanking/burst visible as monochrome signal in the
    // margins (decoded colour stays confined to the active area) — for
    // checking geometry and the burst window, not for measurement (see the
    // README's note on vectorscope/histogram behaviour in this mode).
    options.push_back({kOptFullRaster, "Frame (full raster)", true,
                       static_cast<uint32_t>(fp.frame_width),
                       static_cast<uint32_t>(fp.frame_height), frames, dar});
    return options;
}

PreviewImage HvdChromaDecoderStage::render_preview(
    const std::string& option_id, uint64_t index,
    PreviewNavigationHint hint) const
{
    (void)hint;
    auto repr = std::dynamic_pointer_cast<const HvdDecodedRepresentation>(cached_output_);
    if (!repr) return {};
    const FrameID id = static_cast<FrameID>(index);
    if (option_id == kOptField1)
        return repr->render_custom_preview(id, /*full_raster=*/false, 0);
    if (option_id == kOptField2)
        return repr->render_custom_preview(id, /*full_raster=*/false, 1);
    if (option_id == kOptFullRaster)
        return repr->render_custom_preview(id, /*full_raster=*/true, -1);
    return repr->render_custom_preview(id, /*full_raster=*/false, -1);  // kOptFrame + fallback
}

}  // namespace orc::plugins::hvd
