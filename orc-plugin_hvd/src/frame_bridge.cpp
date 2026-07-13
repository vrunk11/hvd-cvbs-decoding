// SPDX-License-Identifier: GPL-3.0-or-later

#include "frame_bridge.h"

#include <algorithm>
#include <cmath>

#include "engine/engine.h"
#include "engine/lockin.h"

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

void SplitFrameFields(const int16_t* frame, const FrameParams& fp,
                      FieldInput* top, FieldInput* bottom,
                      bool field1_is_bottom) {
  const int fw = fp.frame_width;
  const int fh = fp.frame_height;
  const int f1 = fp.field1_lines;
  const int f2 = fh - f1;  // field 2 line count (262 NTSC)
  const int field_h = f1;  // the larger field bounds the geometry

  top->samples = Plane(field_h, fw);
  bottom->samples = Plane(field_h, fw);
  // NOTE on naming: `top`/`bottom` here mean "buffer block 1"/"block 2"
  // (temporal order). The SPATIAL role is what these flags carry — the
  // engine's weave honours is_first_field ("true => top/even rows"), so
  // for BFF material block 1 is flagged as NOT-first and the engine swaps.
  top->is_first_field = !field1_is_bottom;
  bottom->is_first_field = field1_is_bottom;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int y = 0; y < field_h; ++y) {
    for (int x = 0; x < fw; ++x) {
      const float st =
          (y < f1) ? static_cast<float>(frame[(y)*fw + x]) : fp.blanking_level;
      const float sb = (y < f2) ? static_cast<float>(frame[(f1 + y) * fw + x])
                                : fp.blanking_level;
      top->samples.at(y, x) = SampleToIre(st, fp.black_level, fp.white_level);
      bottom->samples.at(y, x) =
          SampleToIre(sb, fp.black_level, fp.white_level);
    }
  }
}


int DetectFieldParity(const FieldInput& block1, const FieldInput& block2,
                      const FrameParams& fp) {
  const FieldGeometry g = FieldGeometryFromParams(fp);
  const int fal = g.first_active_field_line;
  const int lal = g.last_active_line();
  const int a0 = fp.active_video_start;
  const int a1 = fp.active_video_end;
  const int lines = lal - fal;
  const int width = a1 - a0;
  if (lines < 8 || width < 32) return 0;

  // Low-pass + decimate horizontally so the 4fsc chroma carrier (pi/2
  // rad/px) can't masquerade as vertical structure: average groups of 4
  // samples (a null-adjacent comb for the carrier), sampled sparsely —
  // this is a vote, not a decode; a few thousand taps are plenty.
  const int xstep = 4;
  const int ystep = std::max(1, lines / 120);
  double e_plus = 0.0, e_minus = 0.0;
  long n = 0;
  for (int r = fal + 1; r + 1 < lal; r += ystep) {
    for (int x = a0; x + 4 <= a1; x += 4 * xstep) {
      auto lp = [&](const Plane& p, int row) {
        return 0.25F * (p.at(row, x) + p.at(row, x + 1) + p.at(row, x + 2) +
                        p.at(row, x + 3));
      };
      const float b = lp(block2.samples, r);
      const float a_c = lp(block1.samples, r);
      const float a_up = lp(block1.samples, r - 1);
      const float a_dn = lp(block1.samples, r + 1);
      // block1 = top: block2 row r sits at frame line 2r+1, i.e. BETWEEN
      // block1 rows r and r+1 -> interp at +0.5.  block1 = bottom: block2
      // row r sits at frame line 2r, between block1 rows r-1 and r ->
      // interp at -0.5.
      e_plus += std::fabs(b - 0.5F * (a_c + a_dn));
      e_minus += std::fabs(b - 0.5F * (a_c + a_up));
      ++n;
    }
  }
  if (n == 0) return 0;
  e_plus /= static_cast<double>(n);
  e_minus /= static_cast<double>(n);
  // Relative margin: flat content gives e_plus ~= e_minus (both tiny) —
  // no vote rather than a coin flip.
  const double lo = std::min(e_plus, e_minus);
  const double hi = std::max(e_plus, e_minus);
  if (hi <= 1e-9 || (hi - lo) / hi < 0.02) return 0;
  return e_plus < e_minus ? +1 : -1;
}

bool ResolveField1IsBottom(const FieldInput& block1, const FieldInput& block2,
                           const FrameParams& fp, const HvdConfig& cfg) {
  if (cfg.field_order == 1) return false;
  if (cfg.field_order == 2) return true;
  const int vote = DetectFieldParity(block1, block2, fp);
  // No confident vote: the ld-decode format convention (frames assembled
  // first-field-first, first field = even frame lines) is the default.
  return vote < 0;
}

YcFrameS16 DecodeFrameBuffer(const int16_t* frame, const FrameParams& fp,
                             const HvdConfig& cfg, HvdEngine& engine,
                             const NeighborRawState* prev_frame,
                             NeighborRawState* out_state) {
  const int fw = fp.frame_width;
  const int fh = fp.frame_height;
  const int f1 = fp.field1_lines;
  const FieldGeometry g = FieldGeometryFromParams(fp);

  FieldInput top;     // field 1 = flat lines [0, f1)
  FieldInput bottom;  // field 2 = flat lines [f1, fh)
  SplitFrameFields(frame, fp, &top, &bottom);
  {
    // Per-frame resolution is safe against flicker by construction: the
    // detector only abstains (falling back to the format convention) when
    // the frame has no vertical detail — exactly the content on which the
    // two weaves are visually identical, so a momentary fallback cannot be
    // observed. The sequence path additionally majority-votes per window.
    const bool f1_bottom = ResolveField1IsBottom(top, bottom, fp, cfg);
    top.is_first_field = !f1_bottom;
    bottom.is_first_field = f1_bottom;
  }

  std::vector<NeighborRawState> prev_frames;
  if (prev_frame && cfg.enable_temporal) prev_frames.push_back(*prev_frame);
  const FrameYc yc = engine.DecodeFrame(top, bottom, g, cfg, prev_frames);

  if (out_state) {
    out_state->luma = yc.luma;
    out_state->composite = yc.composite;
    out_state->carrier = yc.carrier;
    out_state->chroma = yc.chroma_phasor;  // for the coherence gate
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
  // Auto detection needs luma vertical detail, which the C channel lacks;
  // auto therefore resolves to the format convention here, and the forced
  // modes pass through.
  chroma_top.is_first_field = cfg.field_order != 2;
  chroma_bot.is_first_field = cfg.field_order == 2;

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

std::vector<YcFrameS16> DecodeFrameSequenceWindow(
    const std::vector<const int16_t*>& frames, int core_begin, int core_end,
    const FrameParams& fp, const HvdConfig& cfg, HvdEngine& engine) {
  std::vector<YcFrameS16> out;
  const int nframes = static_cast<int>(frames.size());
  if (nframes == 0 || core_begin < 0 || core_end > nframes ||
      core_begin >= core_end) {
    return out;
  }
  const int fw = fp.frame_width;
  const int fh = fp.frame_height;
  const int f1 = fp.field1_lines;
  const FieldGeometry g = FieldGeometryFromParams(fp);
  const float scale = (fp.white_level - fp.black_level) / 100.0F;

  // --- Prepare the window's fields (2 per frame) + measure ACC ------------
  // Parity: field1 = top (parity 0). FrameParams carries no per-field
  // metadata, so index order is the fallback — same as the reference's
  // field_parity when isFirstField isn't available.
  std::vector<FieldObs> fields;
  fields.reserve(static_cast<size_t>(nframes) * 2);
  std::vector<float> burst_amps;
  burst_amps.reserve(static_cast<size_t>(nframes) * 2);
  std::vector<std::pair<FieldInput, FieldInput>> field_pairs;
  field_pairs.reserve(nframes);
  for (int t = 0; t < nframes; ++t) {
    FieldInput top;
    FieldInput bottom;
    SplitFrameFields(frames[t], fp, &top, &bottom);
    burst_amps.push_back(BurstAmplitudeIre(top.samples, g));
    burst_amps.push_back(BurstAmplitudeIre(bottom.samples, g));
    field_pairs.push_back({std::move(top), std::move(bottom)});
  }
  // Resolve the field order ONCE for the whole window (majority of the
  // per-frame signal votes; a window must be spatially consistent — a
  // per-frame flip would be a weave error, not a property of the capture).
  bool f1_bottom = cfg.field_order == 2;
  if (cfg.field_order == 0) {
    int vote = 0;
    for (const auto& pr : field_pairs)
      vote += DetectFieldParity(pr.first, pr.second, fp);
    f1_bottom = vote < 0;
  }
  const int p_first = f1_bottom ? 1 : 0;
  for (auto& pr : field_pairs) {
    // TEMPORAL order preserved (block 1 first, as stored); the field order
    // only changes which spatial PARITY each block carries. The output
    // weave downstream places content by TRUE parity, so it handles
    // p0 = 1 natively.
    fields.push_back(PrepareFieldObs(pr.first, g, cfg, /*parity=*/p_first));
    fields.push_back(PrepareFieldObs(pr.second, g, cfg, /*parity=*/1 - p_first));
  }

  float acc_gain = 1.0F;
  if (cfg.acc && !burst_amps.empty()) {
    const size_t k = burst_amps.size() / 2;
    std::nth_element(burst_amps.begin(), burst_amps.begin() + k,
                     burst_amps.end());
    const float med = std::max(burst_amps[k], 1.0F);
    acc_gain = std::clamp(20.0F / med, 0.5F, 2.0F);
  }
  const float gain = cfg.chroma_gain * acc_gain;

  // --- The pipeline itself -------------------------------------------------
  const std::vector<DecodedField> dec =
      engine.DecodeSequenceWindow(fields, g, cfg);

  // --- Weave core frames by parity and repackage as YcFrameS16 -------------
  const int fal = g.first_active_field_line;
  const int a0 = fp.active_video_start;
  for (int t = core_begin; t < core_end; ++t) {
    const int j0 = 2 * t;
    const int j1 = j0 + 1;
    const DecodedField& d0 = dec[j0];
    const DecodedField& d1 = dec[j1];
    int p0 = fields[j0].parity;
    int p1 = fields[j1].parity;
    if (p0 == p1) {  // metadata anomaly: fall back to index order
      p0 = 0;
      p1 = 1;
    }

    YcFrameS16 yc;
    yc.width = fw;
    yc.height = fh;
    yc.chroma_dc = fp.chroma_dc;
    yc.luma.assign(static_cast<size_t>(fw) * fh, 0);
    yc.chroma.assign(static_cast<size_t>(fw) * fh, 0);
    yc.u_plane.assign(static_cast<size_t>(fw) * fh, 0.0);
    yc.v_plane.assign(static_cast<size_t>(fw) * fh, 0.0);
    // Outside the active picture: composite passthrough, zero chroma
    // (same contract as DecodeFrameBuffer).
    for (int i = 0; i < fw * fh; ++i) yc.luma[i] = frames[t][i];

    const int lines = d0.luma.height();
    const int active_w = d0.luma.width();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < 2 * lines; ++r) {
      // Frame row r (within the active picture) shows the field whose TRUE
      // parity equals r % 2 — the anti-inverted-weave weave of the
      // reference (Yf[p0::2] = Yout[j0]). The PHYSICAL output block is a
      // separate question: YcFrameS16's field-sequential contract (and
      // ReorderToWoven downstream) maps even frame rows to the field1
      // block [0, f1) and odd rows to the field2 block [f1, fh),
      // regardless of which capture field supplied the content.
      const bool row_even = (r % 2) == 0;
      const bool use_first = (r % 2) == p0;
      const DecodedField& d = use_first ? d0 : d1;
      const int field_line = r / 2;
      if (field_line >= d.luma.height()) continue;
      const int flat_line =
          row_even ? (fal + field_line) : (f1 + fal + field_line);
      if (flat_line < 0 || flat_line >= fh) continue;
      for (int c = 0; c < active_w; ++c) {
        const int flat_col = a0 + c;
        if (flat_col < 0 || flat_col >= fw) continue;
        const size_t idx = static_cast<size_t>(flat_line) * fw + flat_col;
        const float y_ire = d.luma.at(field_line, c);
        // Monochrome: zero the chroma at OUTPUT only (the solve still ran
        // and shaped Y through the arbitration), matching the reference's
        // `Cf = np.zeros_like(Cf)` in decode_sequence.
        const Complex chi =
            cfg.monochrome ? Complex{} : d.chroma.at(field_line, c);
        const Complex car =
            fields[use_first ? j0 : j1].carrier.at(field_line, c);
        yc.luma[idx] = ClampU10(fp.black_level + y_ire * scale);
        // Modulated chroma residual, zero-centred/signed (see ClampS10):
        // with cfg.output_fidelity (default) luma + chroma == composite
        // holds exactly, matching the frame path's invariant.
        yc.chroma[idx] = ClampS10((chi * car).real() * scale);
        yc.v_plane[idx] = gain * static_cast<double>(chi.real()) * scale;
        yc.u_plane[idx] = gain * static_cast<double>(-chi.imag()) * scale;
      }
    }
    out.push_back(std::move(yc));
  }
  return out;
}

}  // namespace hvd
