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
                             const HvdConfig& cfg) {
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

  HvdEngine engine;
  const FrameYc yc = engine.DecodeFrame(top, bottom, g, cfg);

  // --- Re-weave into a field-sequential Y/C split --------------------------
  YcFrameS16 out;
  out.width = fw;
  out.height = fh;
  out.chroma_dc = fp.chroma_dc;
  out.luma.assign(static_cast<size_t>(fw) * fh, 0);
  out.chroma.assign(static_cast<size_t>(fw) * fh, 0);

  // Default: pass the composite through as luma, chroma flat at its DC. Outside
  // the active picture this trivially satisfies luma + (chroma - dc) = S.
  const int16_t dc = ClampU10(fp.chroma_dc);
  for (int i = 0; i < fw * fh; ++i) {
    out.luma[i] = frame[i];
    out.chroma[i] = dc;
  }

  // Fill the active picture. The engine returns a woven plane of
  // (2 * active_field_lines) rows x active_width cols; row r belongs to field
  // (r % 2), field line = first_active_field_line + r / 2.
  const int fal = g.first_active_field_line;
  const int a0 = fp.active_video_start;
  const int active_h = yc.luma.height();
  const int active_w = yc.luma.width();
  const float scale = (fp.white_level - fp.black_level) / 100.0F;

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
      out.chroma[idx] = ClampU10(fp.chroma_dc + yc.chroma.at(r, c) * scale);
    }
  }
  return out;
}

}  // namespace hvd
