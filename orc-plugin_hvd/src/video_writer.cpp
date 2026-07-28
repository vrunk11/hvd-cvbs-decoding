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

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/pixfmt.h>
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
                       int fps_num, int fps_den, VideoContainer container)
{
    if (opened_) {
        last_error_ = "VideoWriter already open";
        return false;
    }
    if (width <= 0 || height <= 0) {
        last_error_ = "invalid frame size";
        return false;
    }

    const char* format_name = nullptr;
    AVCodecID codec_id = AV_CODEC_ID_NONE;
    AVPixelFormat pix_fmt = AV_PIX_FMT_RGB24;
    switch (container) {
        case VideoContainer::kMkvFfv1:
            format_name = "matroska";
            codec_id = AV_CODEC_ID_FFV1;
            // Planar RGB, no chroma subsampling: FFV1 + GBRP is bit-exact
            // lossless for what we hand it (unlike FFV1 + YUV, which
            // would be lossless only relative to an already-lossy
            // RGB->YUV step). Not user-configurable: this is what makes
            // the .mkv path "lossless" at all.
            pix_fmt = AV_PIX_FMT_GBRP;
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
    // GBRP carry full resolution on every plane and have no such
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
                                             path.c_str());
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
        ret = avio_open(&fmt_ctx_->pb, path.c_str(), AVIO_FLAG_WRITE);
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
    // pure pixel-format conversion (channel split/reorder for GBRP, or
    // the standard RGB->YUV matrix for YUV420P) with no resampling —
    // there is no "better" scaling filter to pick here because nothing is
    // being scaled.
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
    while (buffer_.size() >= frame_bytes_) {
        if (!writer_.WriteFrame(reinterpret_cast<const uint8_t*>(buffer_.data()))) {
            failed_ = true;
            return 0;  // ostream::write() treats a short sputn() as
                       // failure and sets badbit — exactly the signal the
                       // callers already check via `if (!out)`.
        }
        buffer_.erase(buffer_.begin(), buffer_.begin() + static_cast<long>(frame_bytes_));
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
