/*
 * File:        video_writer.cpp
 * Module:      orc-stage-plugin-hvd-chroma-decoder
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2026 hvd-decode contributors
 */

#include "video_writer.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <thread>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libavutil/pixfmt.h>
#include <libavutil/rational.h>
#include <libswscale/swscale.h>
}

namespace hvd {

namespace {
std::string ToLowerExt(const std::string& path)
{
    const auto dot = path.find_last_of('.');
    if (dot == std::string::npos) return {};
    std::string ext = path.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                    [](unsigned char c) { return std::tolower(c); });
    return ext;
}

std::string AvErrorString(int err)
{
    char buf[AV_ERROR_MAX_STRING_SIZE] = {0};
    av_strerror(err, buf, sizeof(buf));
    return buf;
}
}  // namespace

VideoContainer ContainerFromPath(const std::string& path)
{
    const std::string ext = ToLowerExt(path);
    if (ext == ".mkv") return VideoContainer::kMkvFfv1;
    if (ext == ".mp4") return VideoContainer::kMp4Mpeg4;
    return VideoContainer::kRaw;
}

VideoWriter::~VideoWriter()
{
    if (opened_ && !finalized_) Finalize();  // best-effort: never leak an
                                              // open AVIO handle just
                                              // because the caller forgot
                                              // (or failed before reaching)
                                              // the explicit Finalize().
    CleanUp();
}

bool VideoWriter::Open(const std::string& path, int width, int height,
                       int fps_num, int fps_den, VideoContainer container,
                       double sample_aspect_ratio)
{
    if (opened_) {
        last_error_ = "VideoWriter already open";
        return false;
    }
    if (width <= 0 || height <= 0) {
        last_error_ = "invalid frame size";
        return false;
    }

    // "-" is the standard ffmpeg/orc-cli convention for "write to
    // stdout" — NOT a real filesystem path. Passed straight to
    // avio_open() it would just try (and typically fail, or worse,
    // silently create) a file literally named "-" in the working
    // directory, which is exactly what used to make `output_path=- |
    // ffplay -` produce nothing: bytes went into that file, never into
    // the shell pipe ffplay was reading. libav's own "pipe:1" URL is the
    // portable way to actually mean stdout (its pipe protocol handles
    // both POSIX and Windows).
    const std::string avio_path = (path == "-") ? "pipe:1" : path;

    const char* format_name = nullptr;
    AVCodecID codec_id = AV_CODEC_ID_NONE;
    AVPixelFormat pix_fmt = AV_PIX_FMT_RGB24;
    switch (container) {
        case VideoContainer::kMkvFfv1:
            format_name = "matroska";
            codec_id = AV_CODEC_ID_FFV1;
            // Planar RGB, no chroma subsampling: FFV1 + GBRP10LE is
            // bit-exact lossless for what we hand it (unlike FFV1 + YUV,
            // which would be lossless only relative to an already-lossy
            // RGB->YUV step) -- upconverting our 8-bit source into the
            // low 8 bits of a 10-bit container loses nothing and rounds
            // back to the identical 8-bit values on read. 10-bit was
            // chosen over plain 8-bit GBRP for two reasons: this FFV1
            // encoder build only supports GBR from 9-bit up (see its own
            // "Supported pixel formats" list -- gbrp8 isn't on it), and
            // 10-bit 4:4:4 is itself a standard, widely-recognised
            // intermediate depth other tools expect, which matters if
            // this file/pipe is read by something other than this plugin.
            pix_fmt = AV_PIX_FMT_GBRP10LE;
            break;
        case VideoContainer::kMp4Mpeg4:
            format_name = "mp4";
            codec_id = AV_CODEC_ID_MPEG4;
            // Fixed, not user-configurable: 4:2:0 is standard for MPEG-4
            // Part 2 (and for MP4 video generally).
            pix_fmt = AV_PIX_FMT_YUV420P;
            break;
        case VideoContainer::kNutRaw:
            format_name = "nut";
            codec_id = AV_CODEC_ID_RAWVIDEO;
            // Fixed at RGB24: this stage's own internal pixel format
            // exactly, so WriteFrame()'s sws_scale below is a straight
            // copy, not a real conversion.
            pix_fmt = AV_PIX_FMT_RGB24;
            break;
        case VideoContainer::kRaw:
            last_error_ = "ContainerFromPath() returned kRaw; caller should "
                          "use the plain raw-RGB24 path instead of VideoWriter";
            return false;
    }

    // Chroma-subsampled formats need even dimensions in the subsampled
    // axis/axes (4:2:0 halves both, 4:2:2 halves only width); RGB24/BGR24/
    // GBRP10LE carry full resolution on every plane and have no such
    // constraint. CVBS active pictures are effectively always even
    // already, but this is cheap insurance against an off-by-one crop
    // producing a codec-level error that would otherwise read as a
    // confusing libav errno deep in this function.
    const bool needs_even_width =
        pix_fmt == AV_PIX_FMT_YUV420P || pix_fmt == AV_PIX_FMT_YUV422P;
    const bool needs_even_height = pix_fmt == AV_PIX_FMT_YUV420P;
    if ((needs_even_width && width % 2 != 0) ||
        (needs_even_height && height % 2 != 0)) {
        last_error_ = "frame dimensions must be even for this pixel format (got " +
                      std::to_string(width) + "x" + std::to_string(height) + ")";
        return false;
    }

    int ret = avformat_alloc_output_context2(&fmt_ctx_, nullptr, format_name,
                                             avio_path.c_str());
    if (ret < 0 || !fmt_ctx_) {
        last_error_ = "avformat_alloc_output_context2 failed: " +
                      (ret < 0 ? AvErrorString(ret) : std::string("null context"));
        CleanUp();
        return false;
    }

    const AVCodec* codec = avcodec_find_encoder(codec_id);
    if (!codec) {
        last_error_ = std::string("encoder not available for codec id ") +
                      std::to_string(static_cast<int>(codec_id)) +
                      " (unexpected -- this codec should be built into any "
                      "standard ffmpeg, no extra library required)";
        CleanUp();
        return false;
    }

    stream_ = avformat_new_stream(fmt_ctx_, nullptr);
    if (!stream_) {
        last_error_ = "avformat_new_stream failed";
        CleanUp();
        return false;
    }

    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) {
        last_error_ = "avcodec_alloc_context3 failed";
        CleanUp();
        return false;
    }

    codec_ctx_->width = width;
    codec_ctx_->height = height;
    codec_ctx_->time_base = AVRational{fps_den, fps_num};
    codec_ctx_->framerate = AVRational{fps_num, fps_den};
    stream_->time_base = codec_ctx_->time_base;
    codec_ctx_->pix_fmt = pix_fmt;

    // Pixel (sample) aspect ratio: without this, a player has no way to
    // know CVBS pixels aren't square and will display at 1:1 -- exactly
    // the "pipe doesn't get the aspect ratio" bug this fixes. Set on
    // BOTH the codec context and the stream: some readers only look at
    // the stream's codecpar (populated from codec_ctx_ a few lines
    // below via avcodec_parameters_from_context), others compare the
    // two and get suspicious if they disagree, so they need to agree
    // from the start rather than one lagging the other.
    if (sample_aspect_ratio > 0.0) {
        const AVRational sar = av_d2q(sample_aspect_ratio, 100000);
        codec_ctx_->sample_aspect_ratio = sar;
        stream_->sample_aspect_ratio = sar;
    }

    if (container == VideoContainer::kMp4Mpeg4) {
        // Fixed-quantizer ("qscale") encoding: the generic avcodec knob
        // every native encoder understands, unlike CRF which is a
        // libx264-specific AVOption. FF_QP2LAMBDA converts a target
        // quantiser (2 = very high quality/low compression, 31 = very
        // low quality/high compression, same MPEG quantiser scale ffmpeg's
        // own CLI uses for -qscale:v) into the lambda units global_quality
        // expects. qscale 4 is a reasonable "looks clean, not tuned
        // against this project's regression scene" default -- the same
        // spirit as the FFV1/lossless path's fixed choices, just lossy.
        codec_ctx_->flags |= AV_CODEC_FLAG_QSCALE;
        codec_ctx_->global_quality = FF_QP2LAMBDA * 4;
    }

    // Slice-based multithreading: without this, thread_count defaults to
    // 1 -- i.e. every encode here used to run on a SINGLE core while
    // decode itself uses every core via OpenMP, making the encoder the
    // bottleneck once the export's producer/consumer queue fills up
    // (visible on file exports too, not just pipes, since it's not an
    // I/O problem). Doubly true for FFV1 feeding it GBRP10LE: FFV1's
    // golomb-rice coder only handles up to 8-bit samples, so anything
    // 9-bit or above (like our 10-bit choice) is forced onto FFV1's
    // range coder, which is inherently slower per sample -- exactly why
    // it needs the parallelism restored, not just as a nice-to-have.
    // Not done for kNutRaw: AV_CODEC_ID_RAWVIDEO doesn't compress at all,
    // so there is nothing for extra threads to parallelise there.
    if (container == VideoContainer::kMkvFfv1 ||
        container == VideoContainer::kMp4Mpeg4) {
        const unsigned hw = std::thread::hardware_concurrency();
        const int threads = static_cast<int>(hw != 0 ? std::min(hw, 16u) : 4u);
        codec_ctx_->thread_count = threads;
        codec_ctx_->thread_type = FF_THREAD_SLICE;
        if (container == VideoContainer::kMkvFfv1) {
            // FFV1-specific: thread_count alone only sets the ceiling --
            // the frame also has to actually BE cut into that many
            // independently-codeable slices, which is what this option
            // controls (a plain int, not restricted to "nice" numbers).
            av_opt_set_int(codec_ctx_->priv_data, "slices", threads, 0);
        }
    }

    if (fmt_ctx_->oformat->flags & AVFMT_GLOBALHEADER)
        codec_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    ret = avcodec_open2(codec_ctx_, codec, nullptr);
    if (ret < 0) {
        last_error_ = "avcodec_open2 failed: " + AvErrorString(ret);
        CleanUp();
        return false;
    }

    ret = avcodec_parameters_from_context(stream_->codecpar, codec_ctx_);
    if (ret < 0) {
        last_error_ = "avcodec_parameters_from_context failed: " + AvErrorString(ret);
        CleanUp();
        return false;
    }

    if (!(fmt_ctx_->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&fmt_ctx_->pb, avio_path.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) {
            last_error_ = "could not open '" + path + "': " + AvErrorString(ret);
            CleanUp();
            return false;
        }
    }

    ret = avformat_write_header(fmt_ctx_, nullptr);
    if (ret < 0) {
        last_error_ = "avformat_write_header failed: " + AvErrorString(ret);
        CleanUp();
        return false;
    }

    frame_ = av_frame_alloc();
    if (!frame_) {
        last_error_ = "av_frame_alloc failed";
        CleanUp();
        return false;
    }
    frame_->format = codec_ctx_->pix_fmt;
    frame_->width = width;
    frame_->height = height;
    ret = av_frame_get_buffer(frame_, 32);
    if (ret < 0) {
        last_error_ = "av_frame_get_buffer failed: " + AvErrorString(ret);
        CleanUp();
        return false;
    }

    // SWS_POINT: the two sides are the same width/height, so this is a
    // pure pixel-format conversion (channel split/reorder + a lossless
    // 8->10-bit upconversion for GBRP10LE, or the standard RGB->YUV
    // matrix for YUV420P) with no resampling — there is no "better"
    // scaling filter to pick here because nothing is being scaled.
    sws_ctx_ = sws_getContext(width, height, AV_PIX_FMT_RGB24, width, height,
                              codec_ctx_->pix_fmt, SWS_POINT, nullptr, nullptr,
                              nullptr);
    if (!sws_ctx_) {
        last_error_ = "sws_getContext failed";
        CleanUp();
        return false;
    }

    pkt_ = av_packet_alloc();
    if (!pkt_) {
        last_error_ = "av_packet_alloc failed";
        CleanUp();
        return false;
    }

    width_ = width;
    height_ = height;
    next_pts_ = 0;
    opened_ = true;
    return true;
}

bool VideoWriter::WriteFrame(const uint8_t* rgb24)
{
    if (!opened_) {
        last_error_ = "WriteFrame() called before a successful Open()";
        return false;
    }

    int ret = av_frame_make_writable(frame_);
    if (ret < 0) {
        last_error_ = "av_frame_make_writable failed: " + AvErrorString(ret);
        return false;
    }

    const uint8_t* src_slices[1] = {rgb24};
    const int src_stride[1] = {width_ * 3};
    sws_scale(sws_ctx_, src_slices, src_stride, 0, height_, frame_->data,
              frame_->linesize);

    frame_->pts = next_pts_++;
    return EncodeAndMux(frame_);
}

bool VideoWriter::EncodeAndMux(AVFrame* frame)
{
    int ret = avcodec_send_frame(codec_ctx_, frame);
    if (ret < 0) {
        last_error_ = "avcodec_send_frame failed: " + AvErrorString(ret);
        return false;
    }

    while (ret >= 0) {
        ret = avcodec_receive_packet(codec_ctx_, pkt_);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
        if (ret < 0) {
            last_error_ = "avcodec_receive_packet failed: " + AvErrorString(ret);
            return false;
        }
        av_packet_rescale_ts(pkt_, codec_ctx_->time_base, stream_->time_base);
        pkt_->stream_index = stream_->index;
        const int write_ret = av_interleaved_write_frame(fmt_ctx_, pkt_);
        // av_interleaved_write_frame() takes ownership of pkt_'s buffer on
        // success; on failure it's unspecified whether it unref'd it, so
        // unref unconditionally to be safe against leaking or double-freeing.
        av_packet_unref(pkt_);
        if (write_ret < 0) {
            last_error_ = "av_interleaved_write_frame failed: " + AvErrorString(write_ret);
            return false;
        }
    }
    return true;
}

bool VideoWriter::Finalize()
{
    if (!opened_) return true;   // nothing was ever opened; trivially fine
    if (finalized_) return true;  // already done, callers can call this
                                  // unconditionally without checking first
    finalized_ = true;

    // Flush: some encoders (anything using B-frames, e.g. MPEG-4 with
    // bframes enabled) hold frames
    // back internally for reordering and only release them once told
    // there's no more input, via a nullptr "frame".
    const bool flush_ok = EncodeAndMux(nullptr);
    const int trailer_ret = av_write_trailer(fmt_ctx_);
    if (trailer_ret < 0 && last_error_.empty()) {
        last_error_ = "av_write_trailer failed: " + AvErrorString(trailer_ret);
    }
    CleanUp();
    return flush_ok && trailer_ret >= 0;
}

void VideoWriter::CleanUp()
{
    if (pkt_) {
        av_packet_free(&pkt_);
        pkt_ = nullptr;
    }
    if (frame_) {
        av_frame_free(&frame_);
        frame_ = nullptr;
    }
    if (sws_ctx_) {
        sws_freeContext(sws_ctx_);
        sws_ctx_ = nullptr;
    }
    if (codec_ctx_) {
        avcodec_free_context(&codec_ctx_);
        codec_ctx_ = nullptr;
    }
    if (fmt_ctx_) {
        if (fmt_ctx_->pb && !(fmt_ctx_->oformat->flags & AVFMT_NOFILE))
            avio_closep(&fmt_ctx_->pb);
        avformat_free_context(fmt_ctx_);
        fmt_ctx_ = nullptr;
    }
    stream_ = nullptr;  // owned by fmt_ctx_, already freed above
    opened_ = false;
}

// ----------------------------------------------------------- streambuf ---

std::streamsize FrameEncodingStreambuf::xsputn(const char* s, std::streamsize n)
{
    if (failed_ || n <= 0) return failed_ ? 0 : n;
    buffer_.insert(buffer_.end(), s, s + n);
    // Consume as many whole frames as are now available, tracking how far
    // in with `consumed` rather than erasing after each one: erasing
    // frame_bytes_ off the FRONT of a vector is O(remaining size) (every
    // trailing byte has to shift down), so doing that once per frame
    // inside this loop is O(frames_in_this_call^2) overall. That was
    // invisible while every call carried at most ~1 frame (the row-by-row
    // writers, or the parallel path's one-frame-at-a-time buffer copy),
    // but the field-pipeline's decode/write overlap now hands a whole
    // CHUNK's worth of frames to a single xsputn() call, and at the
    // default chunk size that quietly became the dominant cost of the
    // entire export -- worst of all on the plain rawvideo/NUT pipe path,
    // where there's no encoding step to distract from it.
    size_t consumed = 0;
    while (buffer_.size() - consumed >= frame_bytes_) {
        if (!writer_.WriteFrame(
                reinterpret_cast<const uint8_t*>(buffer_.data() + consumed))) {
            failed_ = true;
            return 0;  // ostream::write() treats a short sputn() as
                       // failure and sets badbit — exactly the signal the
                       // callers already check via `if (!out)`.
        }
        consumed += frame_bytes_;
    }
    if (consumed > 0) {
        buffer_.erase(buffer_.begin(), buffer_.begin() + static_cast<long>(consumed));
    }
    return n;
}

int FrameEncodingStreambuf::overflow(int c)
{
    if (c == std::char_traits<char>::eof()) return c;
    const char ch = static_cast<char>(c);
    return xsputn(&ch, 1) == 1 ? c : std::char_traits<char>::eof();
}

}  // namespace hvd
