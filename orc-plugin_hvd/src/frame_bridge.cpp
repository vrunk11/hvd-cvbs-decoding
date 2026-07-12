// SPDX-License-Identifier: GPL-3.0-or-later

#include "frame_bridge.h"

#include <algorithm>
#include <cmath>

#include "engine/engine.h"

namespace hvd {

namespace {

int16_t ClampU10(float sample) {
  const long r = std::lround(sample);
  return static_cast<int16_t>(std::clamp<long>(r, 0, 1023));
}

// The chroma channel of a Y/C representation is a SIGNED, zero-centred
// oscillation, not an unsigned 10-bit code: the host's NTSC decoder
// (Comb::FrameBuffer::splitIQ_YC) demodulates chroma_buffer[h] directly, with
// no DC term subtracted first. So chroma must swing symmetrically about 0, and
// must NOT be clamped to [0, 1023] — that would destroy every negative
// half-cycle. Clamp to the signed 10-bit-magnitude range instead.
int16_t ClampS10(float sample) {
  const long r = std::lround(sample);
  return static_cast<int16_t>(std::clamp<long>(r, -1023, 1023));
}

}  // namespace

FieldGeometry FieldGeometryFromParams(const FrameParams& fp) {
  FieldGeometry g;
  g.field_width = fp.frame_width;
  g.field_height = fp.field1_lines;  // the larger field bounds the geometry
  g.active_video_start = fp.active_video_start;
  g.active_video_end = fp.active_video_end;
  g.colour_burst_start = fp.colour_burst_start;
  g.colour_burst_end = fp.colour_burst_end;
  // Active-line indices use the woven-frame convention: field_line = frame/2.
  g.first_active_field_line = fp.first_active_frame_line / 2;
  g.last_active_field_line =
      fp.last_active_frame_line != 0 ? fp.last_active_frame_line / 2 : 0;
  g.sample_rate = fp.sample_rate;
  return g;
}

YcFrameS16 DecodeFrameBuffer(const int16_t* frame, const FrameParams& fp,
                             const HvdConfig& cfg, HvdEngine& engine,
                             const NeighborRawState* prev_frame,
                             NeighborRawState* out_state) {
  const int fw = fp.frame_width;
  const int fh = fp.frame_height;
  const int f1 = fp.field1_lines;
  const int f2 = fh - f1;  // field 2 line count (262 NTSC)
  const FieldGeometry g = FieldGeometryFromParams(fp);
  const int field_h = g.field_height;

  // --- De-weave the field-sequential buffer into two fields (IRE) ----------
  FieldInput top;     // field 1 = flat lines [0, f1)
  FieldInput bottom;  // field 2 = flat lines [f1, fh)
  top.samples = Plane(field_h, fw);
  bottom.samples = Plane(field_h, fw);
  top.is_first_field = true;
  bottom.is_first_field = false;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int y = 0; y < field_h; ++y) {
    for (int x = 0; x < fw; ++x) {
      const float st =
          (y < f1) ? static_cast<float>(frame[(y)*fw + x]) : fp.blanking_level;
      const float sb = (y < f2) ? static_cast<float>(frame[(f1 + y) * fw + x])
                                : fp.blanking_level;
      top.samples.at(y, x) = SampleToIre(st, fp.black_level, fp.white_level);
      bottom.samples.at(y, x) = SampleToIre(sb, fp.black_level, fp.white_level);
    }
  }

  std::vector<NeighborRawState> prev_frames;
  if (prev_frame && cfg.enable_temporal) prev_frames.push_back(*prev_frame);
  const FrameYc yc = engine.DecodeFrame(top, bottom, g, cfg, prev_frames);

  if (out_state) {
    out_state->luma = yc.luma;
    out_state->composite = yc.composite;
    out_state->carrier = yc.carrier;
  }

  // --- Re-weave into a field-sequential Y/C split --------------------------
  YcFrameS16 out;
  out.width = fw;
  out.height = fh;
  out.chroma_dc = fp.chroma_dc;
  out.luma.assign(static_cast<size_t>(fw) * fh, 0);
  out.chroma.assign(static_cast<size_t>(fw) * fh, 0);
  out.u_plane.assign(static_cast<size_t>(fw) * fh, 0.0);
  out.v_plane.assign(static_cast<size_t>(fw) * fh, 0.0);

  // Default: pass the composite through as luma, chroma at zero. Outside the
  // active picture there is no colour information to carry.
  for (int i = 0; i < fw * fh; ++i) {
    out.luma[i] = frame[i];
    out.chroma[i] = 0;
  }

  // Fill the active picture. The engine returns a woven plane of
  // (2 * active_field_lines) rows x active_width cols; row r belongs to field
  // (r % 2), field line = first_active_field_line + r / 2.
  const int fal = g.first_active_field_line;
  const int a0 = fp.active_video_start;
  const int active_h = yc.luma.height();
  const int active_w = yc.luma.width();
  const float scale = (fp.white_level - fp.black_level) / 100.0F;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int r = 0; r < active_h; ++r) {
    const int field = r % 2;               // 0 = field 1 (top), 1 = field 2
    const int field_line = fal + (r / 2);  // line within that field
    const int flat_line = (field == 0) ? field_line : (f1 + field_line);
    if (flat_line < 0 || flat_line >= fh) continue;
    for (int c = 0; c < active_w; ++c) {
      const int flat_col = a0 + c;
      if (flat_col < 0 || flat_col >= fw) continue;
      const size_t idx = static_cast<size_t>(flat_line) * fw + flat_col;
      out.luma[idx] = ClampU10(fp.black_level + yc.luma.at(r, c) * scale);
      // Zero-centred, signed — see ClampS10 above.
      out.chroma[idx] = ClampS10(yc.chroma.at(r, c) * scale);
      // Baseband chroma (chroma_phasor = V - iU), not the modulated `chroma`
      // channel above: this is what a colour preview needs directly.
      // Phase correction (cfg.chroma_phase_deg) is applied upstream, on the
      // burst-locked reference theta, before the engine solves for this
      // phasor — see engine.cpp — so no rotation happens here.
      // g = chroma_gain * acc_gain, matching reference/hvd/decoder.py's
      // `g = cfg.chroma_gain * acc_gain` (decoder.py:1490). acc_gain is
      // measured per-frame (engine.cpp, burst-amplitude ACC) and was
      // previously computed but never applied — this is the fix for that.
      // Deliberately NOT applied to `out.chroma` above: that channel must
      // stay the exact residual S - Y for the lossless split invariant
      // (luma + chroma == composite) to hold; gain only belongs on the
      // baseband preview/export path.
      const float g = cfg.chroma_gain * yc.acc_gain;
      const Complex phasor = yc.chroma_phasor.at(r, c);
      out.v_plane[idx] = g * static_cast<double>(phasor.real()) * scale;
      out.u_plane[idx] = g * static_cast<double>(-phasor.imag()) * scale;
    }
  }
  return out;
}

YcFrameS16 DecodeFrameBuffer(const int16_t* frame, const FrameParams& fp,
                             const HvdConfig& cfg) {
  HvdEngine engine;
  return DecodeFrameBuffer(frame, fp, cfg, engine);
}

YcFrameS16 DecodeYcFrameBuffer(const int16_t* luma, const int16_t* chroma,
                               const FrameParams& fp, const HvdConfig& cfg,
                               HvdEngine& engine) {
  const int fw = fp.frame_width;
  const int fh = fp.frame_height;
  const int f1 = fp.field1_lines;
  const int f2 = fh - f1;
  const FieldGeometry g = FieldGeometryFromParams(fp);
  const int field_h = g.field_height;
  const float scale = (fp.white_level - fp.black_level) / 100.0F;

  // --- De-weave chroma only: luma passes straight through from the raw
  //     buffer below (it's already clean, nothing to decode), so there's no
  //     need to field-split or IRE-convert it here. Chroma is re-centred:
  //     raw code minus chroma_dc, on the SAME scale as luma so ACC's
  //     burst-amplitude assumptions and chroma_gain stay meaningful across
  //     both decode paths. -----------------------------------------------
  FieldInput chroma_top, chroma_bot;
  chroma_top.samples = Plane(field_h, fw);
  chroma_bot.samples = Plane(field_h, fw);
  chroma_top.is_first_field = true;
  chroma_bot.is_first_field = false;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int y = 0; y < field_h; ++y) {
    for (int x = 0; x < fw; ++x) {
      // Unsigned raw code -> signed, zero-mean, same numeric scale as IRE.
      const float ct = (y < f1) ? static_cast<float>(chroma[y * fw + x])
                                 : fp.chroma_dc;
      const float cb = (y < f2) ? static_cast<float>(chroma[(f1 + y) * fw + x])
                                 : fp.chroma_dc;
      chroma_top.samples.at(y, x) = (ct - fp.chroma_dc) / scale;
      chroma_bot.samples.at(y, x) = (cb - fp.chroma_dc) / scale;
    }
  }

  const FrameYc yc = engine.DecodeChromaOnly(chroma_top, chroma_bot, g, cfg);

  YcFrameS16 out;
  out.width = fw;
  out.height = fh;
  out.chroma_dc = fp.chroma_dc;
  out.luma.assign(static_cast<size_t>(fw) * fh, 0);
  out.chroma.assign(static_cast<size_t>(fw) * fh, 0);
  out.u_plane.assign(static_cast<size_t>(fw) * fh, 0.0);
  out.v_plane.assign(static_cast<size_t>(fw) * fh, 0.0);

  // Luma is already clean: pass the source's own Y channel through as-is,
  // outside the active picture included (nothing to decode either way).
  // Plain copy/fill, not a #pragma omp loop — this is memory-bandwidth-
  // bound with no per-element work, so std::copy/assign (which the
  // compiler/stdlib can turn into a vectorised memcpy/memset) beats
  // parallelising it: thread launch overhead would dwarf the actual work.
  std::copy(luma, luma + static_cast<size_t>(fw) * fh, out.luma.begin());
  std::fill(out.chroma.begin(), out.chroma.end(), 0);

  const int fal = g.first_active_field_line;
  const int a0 = fp.active_video_start;
  const int active_h = yc.chroma_phasor.height();
  const int active_w = yc.chroma_phasor.width();

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int r = 0; r < active_h; ++r) {
    const int field = r % 2;
    const int field_line = fal + (r / 2);
    const int flat_line = (field == 0) ? field_line : (f1 + field_line);
    if (flat_line < 0 || flat_line >= fh) continue;
    for (int c = 0; c < active_w; ++c) {
      const int flat_col = a0 + c;
      if (flat_col < 0 || flat_col >= fw) continue;
      const size_t idx = static_cast<size_t>(flat_line) * fw + flat_col;
      // out.luma[idx] already set above from the source's clean Y channel —
      // do NOT overwrite it with anything derived from yc (yc.luma is
      // meaningless here, see engine.h's DecodeChromaOnly doc comment).
      out.chroma[idx] = ClampS10(yc.chroma.at(r, c) * scale);
      const float gain = cfg.chroma_gain * yc.acc_gain;
      const Complex phasor = yc.chroma_phasor.at(r, c);
      out.v_plane[idx] = gain * static_cast<double>(phasor.real()) * scale;
      out.u_plane[idx] = gain * static_cast<double>(-phasor.imag()) * scale;
    }
  }
  return out;
}

YcFrameS16 DecodeYcFrameBuffer(const int16_t* luma, const int16_t* chroma,
                               const FrameParams& fp, const HvdConfig& cfg) {
  HvdEngine engine;
  return DecodeYcFrameBuffer(luma, chroma, fp, cfg, engine);
}

}  // namespace hvd
