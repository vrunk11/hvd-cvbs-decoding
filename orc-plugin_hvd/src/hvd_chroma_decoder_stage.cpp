/*
 * File:        hvd_chroma_decoder_stage.cpp
 * Module:      orc-stage-plugin-hvd-chroma-decoder
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2026 hvd-decode contributors
 */

#include "hvd_chroma_decoder_stage.h"

#include <orc/stage/cvbs_signal_constants.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <fstream>
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
constexpr const char* kCgTol = "cg_tol";
constexpr const char* kBidirectional = "bidirectional";
constexpr const char* kSelective3d = "selective_3d";
constexpr const char* kDiagPrior = "diag_prior";
constexpr const char* kPasses = "passes";
constexpr const char* kChunkFrames = "chunk_frames";
constexpr const char* kFieldOrder = "field_order";
constexpr const char* kDebugDir = "debug_dir";
constexpr const char* kAcc = "acc";
constexpr const char* kChromaGain = "chroma_gain";
constexpr const char* kMonochrome = "monochrome";
constexpr const char* kSymmetryVariant = "symmetry_variant";
constexpr const char* kChromaPhaseDeg = "chroma_phase_deg";
constexpr const char* kFftThreads = "fft_threads";
constexpr const char* kEnableTemporal = "enable_temporal";
constexpr const char* kTemporalStrength = "temporal_strength";
constexpr const char* kMcTile = "mc_tile";
constexpr const char* kMcSearch = "mc_search";
constexpr const char* kNrAnchor = "nr_anchor";
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
        // Tunable live from the UI (fft_threads) rather than hardcoded —
        // FFTW's own thread-sync overhead can outweigh its benefit on an
        // image this small, and the right value is CPU/size-dependent, not
        // something to guess once in code. See hvd_config.h.
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
                // fields, 3D = a mini window of frame id +/- 1) — the same
                // pipeline as the export. DecodeFrameBuffer below is the
                // last-resort fallback only.
                // Preview through the SAME field pipeline as the export
                // (per-field 2D, or a mini 3D window of frame id +/- 1):
                // what you judge in the preview is what the export does.
                // This replaces the earlier frame-level 3D preview and its
                // self-priming chain outright — the frame-level temporal
                // term remains only as the Y/C-native export fallback.
                const bool coupled =
                    config_.enable_temporal && config_.cg_iterations > 0;
                std::vector<const sample_type*> window;
                FrameID w0 = id;
                if (coupled && id > 0 && source_->get_frame(id - 1)) {
                    w0 = id - 1;
                }
                for (FrameID f = w0;; ++f) {
                    const sample_type* p = source_->get_frame(f);
                    if (!p) break;
                    window.push_back(p);
                    if (!coupled && f == id) break;
                    if (f >= id + 1) break;
                }
                const int core = static_cast<int>(id - w0);
                if (core >= 0 && core < static_cast<int>(window.size())) {
                    ::hvd::HvdConfig eff = config_;
                    if (!config_.enable_temporal) eff.temporal_strength = 0.0F;
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
                                         const ::hvd::FrameParams& fp)
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
    const int a0 = std::max(0, fp.active_video_start);
    const int a1 = fp.active_video_end > a0 ? fp.active_video_end : fw;
    const int y0 = std::max(0, fp.first_active_frame_line);
    const int y1 = fp.last_active_frame_line > y0 ? fp.last_active_frame_line
                                                    : fp.frame_height;
    const uint32_t width = static_cast<uint32_t>(std::max(0, a1 - a0));
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
    FrameID id) const
{
    const ::hvd::FrameParams fp = frame_params();
    if (fp.frame_width <= 0 || fp.frame_height <= 0) return std::nullopt;

    const WovenActivePicture pic = woven_active_picture(id);
    if (pic.width == 0 || pic.height == 0) return std::nullopt;

    ColourFrameCarrier carrier;
    carrier.data_type = VideoDataType::ColourNTSC;
    carrier.frame_index = id;
    carrier.width = pic.width;
    carrier.height = pic.height;
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
    // The sequence driver only sees the strength, so the OFF position is
    // expressed by zeroing it in a local copy (found the hard way: for a
    // while the "2D" export was silently running 3D-lite because a
    // positive default strength leaked through, which also made toggling
    // 3D look like it changed nothing).
    ::hvd::HvdConfig eff = config_;
    if (!config_.enable_temporal) eff.temporal_strength = 0.0F;
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
        "Holographic-variational NTSC Y/C separator (experimental). "
        "Colour preview only — no downstream Y/C representation; use "
        "'Export' to write a raw RGB24 file directly.",
        1, 1, 0, 0, VideoFormatCompatibility::NTSC_ONLY,
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

    if (output_path_.empty()) return fail("no output_path configured");

    auto repr = std::dynamic_pointer_cast<const HvdDecodedRepresentation>(cached_output_);
    if (!repr) return fail("no input (connect a video source and try again)");

    const auto [w, h] = repr->active_picture_size();
    if (w == 0 || h == 0) return fail("no active picture geometry");

    std::ofstream out(output_path_, std::ios::binary | std::ios::trunc);
    if (!out) return fail("could not open '" + output_path_ + "' for writing");

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
        uint64_t written = 0;
        bool sequence_ok = true;
        for (FrameID t0 = range.first; sequence_ok && t0 <= range.last; ) {
            const FrameID t1 = std::min(t0 + chunk - 1, range.last);
            if (!repr->decode_sequence_chunk_and_write_rgb24(
                    t0, t1, range.first, range.last, engine, read_mutex,
                    out)) {
                sequence_ok = false;
                break;
            }
            written += t1 - t0 + 1;
            if (progress_callback_) {
                progress_callback_(written, total,
                                   "Exported frame " + std::to_string(written) +
                                       "/" + std::to_string(total) +
                                       " (field pipeline)");
            }
            t0 = t1 + 1;
        }
        if (sequence_ok) {
            if (written == 0) return fail("frame range was empty");
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
        ParameterDescriptor{kFast, "Fast mode",
            "Same algorithm, cheaper logistics (reference's --fast, THEORY "
            "9f): adaptive solver early-exit with a 2/3 iteration cap, and "
            "tile-resolution motion-confidence maps in the 3D path. "
            "Reference measurement: >=2x speed, never worse than 0.2 dB.",
            ParameterType::BOOL, boolean(false)},
        ParameterDescriptor{kCgTol, "Solver early-exit tolerance",
            "Relative gradient-norm at which the conjugate-gradient solve "
            "stops early. 0 = auto (0.02, or 0.10 in fast mode). Measured "
            "on real re-encoded photo content (PORTING.md Sec. 19): 0.3 "
            "combined with fast mode is ~2.3x the default-path speed at "
            "equal-or-slightly-better PSNR and flat-zone rainbow score. "
            "Iteration count above remains the hard ceiling.",
            ParameterType::DOUBLE, real(0.0, 0.9, 0.0)},
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
        ParameterDescriptor{kDiagPrior, "Diagonal chroma prior",
            "Oriented +/-45 deg chroma prior weight (reference's "
            "--diag-prior), renormalised so total prior mass is unchanged. "
            "A measured trade-off, not a win: trades axis-aligned chroma "
            "sharpness (-1 dB on SMPTE bars) for diagonal cross-colour "
            "suppression (+2 dB on zoneplate torture). 0 = off (default); "
            "try ~0.5-1.0 on diagonal-artifact-heavy material such as fine "
            "weaves or venetian blinds.",
            ParameterType::DOUBLE, real(0.0, 2.0, 0.0)},
        ParameterDescriptor{kAcc, "Automatic Color Control",
            "Calibrate saturation from burst amplitude (colour path only).",
            ParameterType::BOOL, boolean(true)},
        ParameterDescriptor{kChromaGain, "Chroma Gain",
            "Gain applied to U/V on top of ACC, matching the classic "
            "decoder's Chroma Gain. Range 0.0-10.0.",
            ParameterType::DOUBLE, real(0.0, 10.0, 1.0)},
        ParameterDescriptor{kMonochrome, "Monochrome",
            "Zero the chroma channel.", ParameterType::BOOL, boolean(false)},
        ParameterDescriptor{kSymmetryVariant, "Spectral-symmetry init",
            "Add the Transform-NTSC certified-chroma init variant.",
            ParameterType::BOOL, boolean(false)},
        ParameterDescriptor{kChromaPhaseDeg, "Chroma phase (deg)",
            "Rotation applied to the burst-locked phase reference before "
            "the solver runs, same idea as the classic decoder's Chroma "
            "Phase (Comb::transformIQ). Range -180 to 180. The recovered "
            "chroma has been persistently 180 deg off since the Python "
            "reference, hence the default; treat it as tunable per-capture, "
            "not as fixed.",
            ParameterType::DOUBLE, real(-180.0, 180.0, 180.0)},
        ParameterDescriptor{kFftThreads, "FFTW threads (preview)",
            "Threads FFTW uses internally per transform in the preview "
            "path (parallel export always forces this to 1 per worker "
            "regardless). 1 disables FFTW's own threading. There's no way "
            "to predict the best value analytically — it depends on your "
            "CPU and image size — so try 1/2/4/your core count live and "
            "keep whichever measures fastest.",
            ParameterType::INT32, integer(1, 64, 4)},
        ParameterDescriptor{kEnableTemporal, "Enable 3D (temporal)",
            "Adds six motion-gated neighbour-field equations per field "
            "(f±1/±2/±3) to the field-granularity solve, resolving the "
            "Y/C ambiguity that a single field cannot (cross-colour, "
            "rainbow on fine detail), and from pass 2 the synth-reference "
            "anchor adds motion-compensated temporal noise reduction. "
            "Strength defaults to adaptive (see Temporal strength). "
            "Measured on the regression scene: +1.9 dB single-pass, "
            "+6.4 dB anchored 2-pass over per-field 2D.",
            ParameterType::BOOL, boolean(false)},
        ParameterDescriptor{kPasses, "3D passes",
            "Gauss-Seidel fixed-point passes over each export chunk "
            "(sequence pipeline only). 1 = single pass. From pass 2 the "
            "decode->NR->re-encode anchor engages: motion-compensated "
            "temporal noise reduction whose reference never gets trusted "
            "where the raw data contradicts it. 2 is the reference's "
            "anchored-mode value.",
            ParameterType::INT32, integer(1, 4, 1)},
        ParameterDescriptor{kNrAnchor, "Anchor strength",
            "Weight of the decode->NR->re-encode anchor once it engages "
            "(passes >= 2, above) -- how strongly the temporally-denoised "
            "reference pulls the solve versus the raw per-field data. "
            "This was previously silently fixed at 1.0 (the reference "
            "default) with no GUI control. 0 disables the anchor's pull "
            "even with passes >= 2 (Gauss-Seidel iteration continues, "
            "just without the NR reference).",
            ParameterType::DOUBLE, real(0.0, 3.0, 1.0)},
        ParameterDescriptor{kDebugDir, "Diagnostics directory",
            "When set, the export also writes, per frame, a PGM map of the "
            "RESIDUAL CARRIER-BAND ENERGY in the decoded luma — i.e. the "
            "rainbow/dot-crawl you can see, measured (bright = separation "
            "failed there) — plus diag.txt logging the decoder's decisions "
            "per chunk (adaptive strength chosen, measured ambiguity, "
            "noise, gates, field-order vote). If an artifact persists, "
            "send the map of the bad zone and the matching diag.txt lines "
            "instead of describing it.",
            ParameterType::FILE_PATH,
            ParameterConstraints{std::nullopt, std::nullopt,
                                 ParameterValue{std::string{}}, {}, false,
                                 std::nullopt}},
        ParameterDescriptor{kFieldOrder, "Field order",
            "0 = Auto (default): the order is MEASURED from the signal — "
            "under the true order each field's lines interpolate the "
            "other's at +0.5 line, under the inverted order at -0.5, a "
            "deterministic half-line vertical correlation vote (majority "
            "over the export window; falls back to the ld-decode format "
            "convention, field 1 = top, on flat content). 1 = force "
            "field 1 top. 2 = force field 1 bottom. Wrong-order symptoms: "
            "one-line serration on static horizontal edges, and motion "
            "that combs even through a player's deinterlacer.",
            ParameterType::INT32, integer(0, 2, 0)},
        ParameterDescriptor{kChunkFrames, "3D chunk size",
            "Frames per export window in 3D mode (plus 2 frames of context "
            "on each side). Bounds memory; larger chunks slightly reduce "
            "edge effects at chunk boundaries.",
            ParameterType::INT32, integer(1, 24, 6)},
        ParameterDescriptor{kTemporalStrength, "Temporal strength",
            "Weight of the motion-compensated neighbour-field equations "
            "once Enable 3D is on. 0 = ADAPTIVE (default): the strength is "
            "measured from the content per window — same-parity fields "
            "carry the chroma identically but flip luma leakage in sign, "
            "so their demodulated difference isolates exactly the Y/C "
            "ambiguity the 3D equations exist to resolve; strong "
            "cross-colour content gets strong 3D (up to 1.5), clean "
            "content stays near the 0.15 floor instead of lifting chroma "
            "noise. Any positive value forces that fixed strength "
            "(reference --3d uses 0.5).",
            ParameterType::DOUBLE, real(0.0, 4.0, 0.0)},
        ParameterDescriptor{kMcTile, "Motion tile size (px)",
            "Block-matching tile size for the temporal path.",
            ParameterType::INT32, integer(8, 128, 32)},
        ParameterDescriptor{kMcSearch, "Motion search radius (px)",
            "Block-matching search radius for the temporal path.",
            ParameterType::INT32, integer(2, 64, 16)},
        ParameterDescriptor{kOutputPath, "Output file path",
            "Destination for the Export button: raw interleaved RGB24, no "
            "container. View with e.g. ffplay -f rawvideo -pixel_format "
            "rgb24 -video_size WxH.",
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
        {kCgTol, static_cast<double>(config_.cg_tol)},
        {kBidirectional, config_.bidirectional},
        {kSelective3d, config_.selective_3d},
        {kPasses, static_cast<int32_t>(config_.passes)},
        {kChunkFrames, static_cast<int32_t>(config_.chunk_frames)},
        {kFieldOrder, static_cast<int32_t>(config_.field_order)},
        {kDebugDir, config_.debug_dir},
        {kDiagPrior, static_cast<double>(config_.diag_prior)},
        {kAcc, config_.acc},
        {kChromaGain, static_cast<double>(config_.chroma_gain)},
        {kMonochrome, config_.monochrome},
        {kSymmetryVariant, config_.symmetry_variant},
        {kChromaPhaseDeg, static_cast<double>(config_.chroma_phase_deg)},
        {kFftThreads, static_cast<int32_t>(config_.fft_threads)},
        {kEnableTemporal, config_.enable_temporal},
        {kTemporalStrength, static_cast<double>(config_.temporal_strength)},
        {kMcTile, static_cast<int32_t>(config_.mc_tile)},
        {kMcSearch, static_cast<int32_t>(config_.mc_search)},
        {kOutputPath, output_path_}};
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
    get_double(kCgTol, config_.cg_tol);
    get_bool(kBidirectional, config_.bidirectional);
    get_bool(kSelective3d, config_.selective_3d);
    get_int(kPasses, config_.passes);
    get_int(kChunkFrames, config_.chunk_frames);
    get_int(kFieldOrder, config_.field_order);
    get_string(kDebugDir, config_.debug_dir);
    get_double(kDiagPrior, config_.diag_prior);
    get_bool(kAcc, config_.acc);
    get_double(kChromaGain, config_.chroma_gain);
    get_bool(kMonochrome, config_.monochrome);
    get_bool(kSymmetryVariant, config_.symmetry_variant);
    get_double(kChromaPhaseDeg, config_.chroma_phase_deg);
    get_int(kFftThreads, config_.fft_threads);
    get_bool(kEnableTemporal, config_.enable_temporal);
    get_double(kTemporalStrength, config_.temporal_strength);
    get_int(kMcTile, config_.mc_tile);
    get_int(kMcSearch, config_.mc_search);
    get_string(kOutputPath, output_path_);

    cached_output_.reset();
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

    capability.navigation_extent.item_count = cached_output_->frame_count();
    capability.navigation_extent.granularity = 1;
    capability.navigation_extent.item_label = "frame";
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
    return repr->build_colour_carrier(static_cast<FrameID>(frame_index));
}

}  // namespace orc::plugins::hvd
