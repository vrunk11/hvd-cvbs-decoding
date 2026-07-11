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
constexpr const char* kAcc = "acc";
constexpr const char* kChromaGain = "chroma_gain";
constexpr const char* kMonochrome = "monochrome";
constexpr const char* kSymmetryVariant = "symmetry_variant";
constexpr const char* kChromaPhaseDeg = "chroma_phase_deg";
constexpr const char* kOutputPath = "output_path";
constexpr const char* kExportNow = "export_now";

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
        } else if (const sample_type* frame = source_->get_frame(id)) {
            yc = ::hvd::DecodeFrameBuffer(frame, fp, config_, engine_);
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
    // decode on every execute() so the colour preview stays live, and so
    // export_now() has something to write.
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

    bool want_export = false;
    auto it = parameters.find(kExportNow);
    if (it != parameters.end() && std::holds_alternative<bool>(it->second)) {
        want_export = std::get<bool>(it->second);
    }
    if (want_export) {
        std::string error;
        export_status_ =
            export_now(&error) ? ("Export complete: " + output_path_)
                               : ("Error: " + error);
    }
    return {};
}

std::shared_ptr<const VideoFrameRepresentation>
HvdChromaDecoderStage::process(
    std::shared_ptr<const VideoFrameRepresentation> source) const
{
    if (!source) return nullptr;
    return std::make_shared<HvdDecodedRepresentation>(std::move(source), config_);
}

bool HvdChromaDecoderStage::export_now(std::string* error) const
{
    auto set_error = [&](const std::string& msg) {
        if (error) *error = msg;
        return false;
    };
    if (output_path_.empty()) return set_error("no output_path configured");

    auto repr = std::dynamic_pointer_cast<const HvdDecodedRepresentation>(cached_output_);
    if (!repr) return set_error("nothing decoded yet (run the pipeline first)");

    const auto [w, h] = repr->active_picture_size();
    if (w == 0 || h == 0) return set_error("no active picture geometry");

    std::ofstream out(output_path_, std::ios::binary | std::ios::trunc);
    if (!out) return set_error("could not open '" + output_path_ + "' for writing");

    const FrameIDRange range = cached_output_->frame_range();
    if (range.last < range.first) return set_error("frame range was empty");
    const uint64_t total = range.last - range.first + 1;

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
                return set_error("failed decoding frame " + std::to_string(id));
            }
            ++written;
        }
        return written > 0;
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
    }

    for (auto& t : workers) t.join();

    if (io_failed) return set_error("I/O error writing '" + output_path_ + "'");
    if (failed.load()) {
        return set_error(failure_message.empty() ? "export failed" : failure_message);
    }
    if (written == 0) return set_error("frame range was empty");
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
        ParameterDescriptor{kOutputPath, "Output file path",
            "Destination for 'Export': raw interleaved RGB24, no container. "
            "View with e.g. ffplay -f rawvideo -pixel_format rgb24 "
            "-video_size WxH.",
            ParameterType::FILE_PATH,
            ParameterConstraints{std::nullopt, std::nullopt,
                                 ParameterValue{std::string{}}, {}, false,
                                 std::nullopt}},
        ParameterDescriptor{kExportNow, "Export",
            "Write every frame in range to output_path right now.",
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
        {kAcc, config_.acc},
        {kChromaGain, static_cast<double>(config_.chroma_gain)},
        {kMonochrome, config_.monochrome},
        {kSymmetryVariant, config_.symmetry_variant},
        {kChromaPhaseDeg, static_cast<double>(config_.chroma_phase_deg)},
        {kOutputPath, output_path_},
        {kExportNow, false}};  // momentary action, never reflects "on"
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
    get_bool(kAcc, config_.acc);
    get_double(kChromaGain, config_.chroma_gain);
    get_bool(kMonochrome, config_.monochrome);
    get_bool(kSymmetryVariant, config_.symmetry_variant);
    get_double(kChromaPhaseDeg, config_.chroma_phase_deg);
    get_string(kOutputPath, output_path_);
    // kExportNow is read directly in execute(), not stored: it's a momentary
    // action, not persistent state.

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
