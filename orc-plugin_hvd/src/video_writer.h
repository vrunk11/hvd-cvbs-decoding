/*
 * File:        video_writer.h
 * Module:      orc-stage-plugin-hvd-chroma-decoder
 * Purpose:     Optional real-container video export (MKV/FFV1 lossless,
 *              MP4/MPEG-4 Part 2 lossy, NUT/raw for pipes) on top of the
 *              existing raw-RGB24 writers.
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2026 hvd-decode contributors
 */

#pragma once

#include <cstdint>
#include <ostream>
#include <streambuf>
#include <string>
#include <vector>

// Forward-declare the libav types we hold pointers to, so this header stays
// clean for translation units that don't need <libav*/*.h> themselves
// (only video_writer.cpp includes the real headers).
struct AVFormatContext;
struct AVCodecContext;
struct AVStream;
struct AVFrame;
struct AVPacket;
struct SwsContext;

namespace hvd {

// Which container/codec to use. .mkv/.mp4 are decided purely by
// output_path_'s extension (case-insensitive) — no format sniffing. A named
// pipe overrides that and always gets kNutRaw instead, regardless of
// extension: see the caller (hvd_chroma_decoder_stage.cpp's trigger()) for
// why — MP4 needs to seek back and rewrite its header/index at the end,
// which a pipe fundamentally cannot do, and while Matroska tolerates a
// non-seekable output better than MP4 does, NUT is the one of these three
// actually DESIGNED to be written to a pipe with nothing deferred to the
// end: every frame is fully self-contained as it's written.
enum class VideoContainer {
    kRaw,      // unchanged legacy behaviour for a plain file whose
               // extension isn't recognised: raw interleaved RGB24, no
               // container, no format metadata (see the "Output file
               // path" parameter's description for the ffplay invocation
               // needed to view it).
    kMkvFfv1,  // .mkv -> FFV1 in Matroska. Lossless: FFV1 is a
               // mathematically lossless codec and we feed it planar RGB
               // (GBRP), so there is no YUV chroma-subsampling loss either
               // — the decoded pixels round-trip exactly.
    kMp4Mpeg4, // .mp4 -> MPEG-4 Part 2 (avcodec's own native encoder, no
               // external library — unlike H.264/libx264, which vcpkg
               // builds as a whole separate autotools project on the
               // side and which fails to configure on some MinGW/MSYS2
               // setups; this stays inside plain ffmpeg[avcodec,
               // avformat,swscale], nothing extra to build). Lossy:
               // standard 4:2:0 chroma subsampling plus DCT-based
               // compression, at a fixed quality (qscale) rather than a
               // measured tuning. Noticeably less efficient than H.264
               // at the same visual quality (bigger files for the same
               // look), which is the trade for not depending on x264.
               // Use .mkv if you need bit-exact output, or a separate
               // ffmpeg pass afterwards if you specifically need H.264/
               // libx264 output and can build it in your environment.
    kNutRaw,   // Named pipe -> NUT container, AV_CODEC_ID_RAWVIDEO (not
               // actually encoded, just wrapped with format/size/fps
               // metadata a downstream `ffmpeg -i pipe:` can read without
               // being told `-f rawvideo -s WxH -r ...` by hand). Always
               // RGB24 -- this stage's internal pixel format exactly, so
               // no conversion at all. (A user-selectable pixel format
               // was tried here and dropped: this SDK's ParameterDescriptor
               // has no real dropdown/enum type to expose it with, only
               // BOOL/DOUBLE/INT32/FILE_PATH, so it would have shown up as
               // a bare, unlabelled-feeling integer spinbox at best.)
};

VideoContainer ContainerFromPath(const std::string& path);

// Encodes one video stream of packed-RGB24 frames to `path`, muxing as it
// goes. Not copyable (owns non-trivial libav state); construct one per
// export, Open() once, WriteFrame() per frame in PRESENTATION order,
// Finalize() exactly once when done.
class VideoWriter {
public:
    VideoWriter() = default;
    ~VideoWriter();
    VideoWriter(const VideoWriter&) = delete;
    VideoWriter& operator=(const VideoWriter&) = delete;

    // fps is given as a rational (num/den) rather than a double so the
    // muxer's timestamps land on the exact broadcast rate (30000/1001 for
    // NTSC/PAL-M, 25/1 for PAL) instead of an accumulating rounding error.
    bool Open(const std::string& path, int width, int height, int fps_num,
              int fps_den, VideoContainer container);

    // `rgb24` must point to exactly width*height*3 bytes, row-major, no
    // padding (R,G,B interleaved per pixel) — the same layout
    // WriteWovenAsRgb24() already produces.
    bool WriteFrame(const uint8_t* rgb24);

    // Flushes any frames still buffered inside the encoder and writes the
    // container trailer. MUST be called (once) for the file to be valid —
    // unlike a plain ofstream, a video file isn't valid just because the
    // bytes written so far happened to reach disk. Safe to call at most
    // once; the destructor calls it too if it was never called explicitly,
    // so a failed export still closes its file instead of leaking one.
    bool Finalize();

    bool is_open() const { return opened_; }
    std::string last_error() const { return last_error_; }

private:
    bool EncodeAndMux(AVFrame* frame);  // frame == nullptr flushes the encoder
    void CleanUp();

    AVFormatContext* fmt_ctx_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;
    AVStream* stream_ = nullptr;
    AVFrame* frame_ = nullptr;
    AVPacket* pkt_ = nullptr;
    SwsContext* sws_ctx_ = nullptr;

    int width_ = 0;
    int height_ = 0;
    int64_t next_pts_ = 0;
    bool opened_ = false;
    bool finalized_ = false;
    std::string last_error_;
};

// Adapts VideoWriter to look like the std::ostream every existing frame
// writer (WriteWovenAsRgb24, decode_and_write_rgb24, the parallel-export
// worker's stringstream copy) already writes into — so none of that code
// needs to know whether it's writing to a raw file or an encoder. Bytes
// are accumulated until exactly one frame's worth (width*height*3) has
// arrived, which is always how much any of those callers write per frame
// (whether as many small row writes or one big buffered one), then handed
// to VideoWriter::WriteFrame() as a unit.
class FrameEncodingStreambuf : public std::streambuf {
public:
    FrameEncodingStreambuf(VideoWriter& writer, size_t frame_bytes)
        : writer_(writer), frame_bytes_(frame_bytes)
    {
        buffer_.reserve(frame_bytes);
    }

    bool failed() const { return failed_; }

protected:
    std::streamsize xsputn(const char* s, std::streamsize n) override;
    int overflow(int c) override;

private:
    VideoWriter& writer_;
    size_t frame_bytes_;
    std::vector<char> buffer_;
    bool failed_ = false;
};

}  // namespace hvd
