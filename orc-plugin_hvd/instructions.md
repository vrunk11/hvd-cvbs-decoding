# HVD chroma decoder

An experimental NTSC Y/C separator. Instead of a comb filter, it treats each
field as an **off-axis hologram**: the colour subcarrier is an interference
carrier, and Y/C separation is solved as a regularized inverse problem. The
stage consumes a composite `CVBS_U10_4FSC` source and produces a **lossless
Y/C split** (luma + chroma reconstruct the composite exactly), which the host's
colour path then renders.

This is a research decoder. It is slow (seconds per frame) and NTSC-only.

## When to use it

Reach for HVD on difficult material where a comb filter leaves visible dot
crawl or rainbowing — fine luma detail next to saturated colour, or diagonal
edges. On easy material a conventional 2-D/3-D comb is faster and just as good.

## Parameters

- **Chroma smoothness (`lambda_c`)** — the main knob. Higher values give
  smoother, more stable colour (less rainbowing) at the cost of chroma
  sharpness; lower values keep chroma detail but push more dot crawl into luma.
  Start at 1.0.
- **Luma / chroma edge scale** — the IRE scale at which each prior stops
  smoothing and preserves an edge. Lower = crisper edges, more sensitive to
  noise.
- **Y→chroma edge coupling (`structure_coupling`)** — lets luma edges "open"
  the chroma prior at the same location, which removes hanging dots along
  luma-only edges. 0 disables it.
- **Solver iterations (`cg_iterations`)** — conjugate-gradient steps. `0` skips
  refinement and outputs the fast holographic initialisation only (good for a
  quick preview; the configuration status turns yellow to signal reduced
  quality).
- **NTSC-J levels**, **Automatic Color Control**, **Monochrome**,
  **Spectral-symmetry init** — colour-path and initialisation options; see the
  project README.

## Notes

The default output is a lossless split, so chroma gain / saturation controls do
**not** belong here — they live in the colour render stage downstream. Only the
validated 2-D woven-frame path is implemented; 3-D/temporal modes from the
reference are not yet ported.
