# hvd-decode — Holographic-Variational NTSC decoder for ld-decode

An experimental, deliberately unconventional chroma decoder for NTSC
`.tbc` files produced by [ld-decode](https://github.com/happycube/ld-decode)
(and compatible tools such as vhs-decode). It never uses a comb filter.
Instead it treats each field as an **off-axis hologram** and solves Y/C
separation as a **regularized inverse problem**.

## The idea

Three fields of physics/math are stitched together:

**1. Digital holography (optics).** A time-base-corrected NTSC field is,
mathematically, an off-axis hologram:

```
S(x, y) = Y(x, y) + Re[ χ(x, y) · e^{iφ(x,y)} ]        χ = V − iU
```

`Y` plays the role of the zero-order/object term, `χ` is a complex
field (the chrominance phasor), and `φ` is a spatial carrier whose
fringes run *diagonally* across the frame — 90°/sample horizontally at
4fsc and 180°/line vertically, the 227.5 cycles/line geometry of NTSC.
The two classic NTSC artefacts map exactly onto the two classic
holography artefacts: **dot crawl = zero-order leakage**,
**cross-colour/rainbowing = the twin image**. Initialization is the
standard holographic reconstruction: demodulate by `e^{−iφ}` and crop
the sideband with a 2D Gaussian window in Fourier space.

**2. Lock-in detection (instrumentation physics).** The carrier phase is
recovered per line with a lock-in amplifier on the colour burst —
multiply by a local oscillator, integrate, read the complex output —
then unwrapped against the 180°/line model. This makes the decoder
robust to real-disc phase wobble without needing `fieldPhaseID` to be
trustworthy.

**3. Variational inverse problems (computational imaging / CT).** Here
is the structural insight that makes this decoder different: the data
term `‖S − Y − Re[χe^{iφ}]‖²` is *invariant* to transfers between luma
and modulated chroma — any composite sample can be "explained" by
either. So Y/C separation is not a filtering problem at all; it is an
**arbitration problem**. We enforce exact data fidelity by eliminating
Y (`Y := S − Re[χe^{iφ}]`) and solve, over χ alone:

```
argmin_χ   Σ ρ( ∇Y(χ) )  +  μ_h Σ ρc( ∂x χ )  +  μ_v Σ ρc( ∂y χ )
```

with ρ the edge-preserving Charbonnier penalty, solved by IRLS (lagged
diffusivity) + conjugate gradient. The arbitration is physical:

* dot crawl is carrier-frequency ripple in Y → enormous ∇Y penalty →
  the optimizer migrates it into χ;
* twin-image/cross-colour is a 2fsc oscillation in χ → enormous ∇χ
  penalty → it migrates back into Y.

The two artefacts *police each other*, and their trade-off collapses
into a small set of interpretable knobs (`--lambda-c`, `--chroma-eps`).

**4. Parallel level sets (multi-modal PET/MRI reconstruction).** Both
priors are made adaptive, and the chroma diffusivity is *coupled* to
the luma gradients (`--structure-coupling`): edges in Y and χ co-occur
in natural images, so wherever the residual luma sees an edge, the
chroma field is allowed a sharp transition too. Combined with decoding
at woven-frame geometry (both fields solved as one inverse problem, so
the vertical prior acts at full frame resolution), this eliminates the
classic "hanging dots"/combing at vertical chroma transitions:
line-alternation energy at colour-bar edges drops ~20× versus the
per-field quadratic version.

## The 3D mode (motion-compensated temporal data term)

The NTSC carrier flips 180° frame-to-frame, so each frame can receive a
*second data equation* built from the previous frame's raw composite:
for a static pixel, S = Y + Re[χc] and S_prev = Y − Re[χc] determine
the separation exactly — the power of a 3D comb. The classic failure of
conventional 3D combs (ghosting on motion, binary motion-detector
switching) is addressed structurally:

* the previous composite is warped by integer-pel tiled block matching
  (zero-motion margin rule, since a spurious 1-sample shift rotates
  the carrier by 90°), and its *measured* carrier phase map is warped
  with it — the 180° flip is never assumed, always measured;
* a per-tile match confidence (best-SSD vs tile variance, squared)
  pre-disengages the term on scene cuts and out-of-range motion;
* a Geman-McClure gate on the composite-domain residual makes the
  static↔motion transition continuous, per-pixel, with ε_t placed at
  motion scale (≫ noise) so the estimator stays unbiased in noise;
* because the temporal term uses raw measurements, not previous
  estimates, decoding errors do not propagate recursively.

Cross-colour leakage anti-correlates frame-to-frame (the flip), so the
3D term is strongest exactly where 2D decoders are weakest.

### Bidirectional + multi-pass (offline mode)

There is no real-time constraint, so the decoder exploits it:

* **Bidirectional**: each frame gets raw-measurement equations from
  *both* n−1 and n+1. Their failure modes are complementary — a scene
  cut or an occlusion breaks at most one side (the B-frame insight
  from video codecs). A static pixel gets three raw measurements.
* **Multi-pass fixed point** (`--passes`, default 2): each pass
  re-solves every frame using the neighbors' *refined* luma as the
  block-matching reference — better flow → better gating → better
  decode, Gauss-Seidel style. The temporal data always comes from raw
  composites, never neighbor estimates, so passes cannot drift.
* Match validation is layered and self-calibrating: per-tile energy
  confidence, median-calibrated outlier rejection (mismatches hidden
  in smooth content), and a global pair-validity gate — if even the
  median tile fails to match, the frames simply do not correspond
  (cut detector for free) and the pairing is disabled entirely.

### The re-encode loop (decode → NR → re-encode → guided re-decode)

The pass structure that unlocks the biggest gains: pass 1 decodes with
*exact* data fidelity (Y = S − Re[χc], the pure arbitration problem).
Then, between passes, a reference is manufactured from the decodes
themselves: motion-compensated temporal blending of (Y, χ) over
±`--nr-radius` frames (leakage anti-correlates at ±1 frame and cancels;
±2 frames are carrier-in-phase and average pure noise), followed by
**re-encoding** the blend to composite, Ŝ = Ŷ + Re[χ̂e^{iφ}]. The
re-encode is the honesty check: the anchor confidence is |S − Ŝ| in
the composite domain, so the reference is only trusted where it
explains the raw measurement.

Pass 2+ then re-decodes with *relaxed* fidelity — (Y, χ) both free,
soft data term, anchored to (Ŷ, χ̂). This matters because exact
fidelity forces the luma to inherit the full composite noise (a hard
PSNR ceiling at the noise floor); relaxing it is only safe with a
trustworthy prior, and the re-encode loop manufactures exactly that.
Bonus: the NR blend operates on baseband decoded fields, so it is
immune to the 4n+2 carrier degeneracy that limits raw-composite
temporal equations.

### Measured behaviour (1/f natural texture, pan + moving object + static zone, 0.8 IRE noise)

| decoder                            | static zone | moving zones |
|------------------------------------|-------------|--------------|
| naive 3D frame comb                | 39.9 dB     | 31.5 dB (ghosting) |
| hvd 2D                             | 38.8 dB     | 39.5 dB      |
| hvd 3D bidi 2-pass, exact fidelity | 42.0 dB     | 41.3 dB      |
| **hvd 3D + re-encode loop (default)** | **43.6 dB** | **42.6 dB** |

At realistic LaserDisc noise (3 IRE), where exact fidelity caps
everything at the ~30.4 dB noise floor: the re-encode loop reaches
31.4 dB static / 33.0 dB moving (4 passes), and keeps improving with
`--passes` and `--nr-radius` — the offline recursion the format
deserves.

Pathological cases vs 2D (with the default re-encode loop): hard scene
cut **+4.2 dB** (the future neighbor and the blend cover it), panning
at 14 px/frame **+2.2 dB** — the raw-composite equations are inert
there (a 4n+2-sample displacement rotates the carrier by 180° and
cancels the frame flip: a real NTSC degeneracy this formulation makes
explicit), but the baseband NR anchor is immune to it. There is no
known case where 3D falls below 2D. Enable with `--3d [strength]`.

## Field-based pipeline (real interlaced content)

Real cameras sample the two fields of a frame **1/60 s apart**. Early
versions of this decoder wove fields before decoding, which
contaminates fields across time on motion (chroma combing/ghosting on
real content — caught by user testing, reproduced with a
true-interlace synthetic encoder). The 3D pipeline now decodes **per
field**; everything cross-field is a robust, per-pixel-gated term:

* temporal neighbors of field f are f±2 (same parity, carrier flip
  180°) *and* f±1 (adjacent field, 1/60 s, carrier offset 3/4 cycle —
  |dc| = √2, non-degenerate). The adjacent-field equations give static
  content back the full woven vertical resolution *through the
  optimiser*, while motion gates them off per pixel.
* the NR/re-encode loop runs at field level (radius in fields).
* fields are woven only at output.

On the true-interlace benchmark this recovers +2.6 dB in moving zones
versus the woven pipeline and brings decoder-induced combing excess
from −0.013 to −0.003 (fields nearly faithful to their instants).
The 2D mode (no `--3d`) still uses the woven solve: fine for static
material, not recommended for motion.

## Results (synthetic ground truth, SMPTE bars + dot-crawl trap, 0.8 IRE noise)

| decoder                                     | PSNR    | edge combing RMS |
|---------------------------------------------|---------|------------------|
| classic 2D line comb (baseline)             | 25.8 dB |        —         |
| holographic reconstruction only             | 26.6 dB |        —         |
| holographic + variational (per-field, quad) | 29.1 dB |      0.133       |
| **+ woven frame, adaptive & coupled priors**| **38.8 dB** |  **0.007**   |

## Usage

```bash
# PNG frames
python3 hvd_decode.py input.tbc -s 0 -l 10 -o frames/

# raw RGB48 pipe, ld-chroma-decoder style
python3 hvd_decode.py input.tbc --pipe | ffmpeg -f rawvideo \
    -pix_fmt rgb48le -s 760x484 -r 30000/1001 -i - out.mkv

# knobs
--lambda-c 1.0            # ↑ = smoother chroma / less rainbow
--charbonnier-eps 0.5     # luma edge-preservation scale (IRE)
--chroma-eps 1.0          # chroma edge-preservation scale (IRE)
--structure-coupling 0.25 # Y->chroma joint-edge coupling (parallel level sets)
--per-field               # legacy per-field decode (more combing)
--3d [NU]                 # motion-compensated temporal mode (default 0.5)
--temporal-eps 0          # motion gate, IRE (0 = auto from measured noise)
--passes 2                # fixed-point refinement passes (3D)
--no-bidi                 # causal only (streaming-style; not recommended)
--nr-anchor 1.0           # re-encode loop strength (0 = exact fidelity only)
--nr-eps 0                # NR blend robustness (0 = auto)
--nr-radius 2             # NR temporal radius (frames)
--cg-iter 60              # 0 = pure holographic mode (fast preview)
--monochrome
```

Generate test material without hardware:

```python
from hvd.encode import make_test_pattern, write_tbc
from hvd.tbc import VideoParameters
p = VideoParameters()
write_tbc("test.tbc", [make_test_pattern(p.active_width, 484)], noise_ire=0.8)
```

## Integration with the ld-decode toolsuite

* Reads standard `.tbc` + `.tbc.json` (NTSC only; PAL is refused).
* `--pipe` emits raw `rgb48le` frames on stdout with the same active
  geometry conventions as `ld-chroma-decoder`, so it slots into
  existing ffmpeg pipelines and can be wrapped as an alternative
  backend in GUI front-ends.
* Unknown JSON keys are ignored; missing `fields` metadata is inferred
  from file size, so vhs-decode NTSC output should load too.

## Decoding vs reconstruction (the purity contract)

The DEFAULT output is a **pure decode**, and the codebase enforces it:
the delivered (Y, χ) pair satisfies Y + Re[χe^{iφ}] = S to machine
precision (verified by `run_tests.test_lossless_identity`, ~1e-15
IRE). The disc's composite is exactly reconstructible from the
output — nothing is added, removed or smoothed; the decoder only
*decides the Y/C split*, with all temporal/NR machinery serving that
decision alone. (A conventional comb filter, by contrast, is lossy.)

Two clearly-flagged opt-in modes leave that regime and say so on
stderr when used: `--soft-output` (partially denoised luma =
processing) and `--drizzle` (multi-field stacking on a synthesized
finer grid = reconstruction, not decoding — the output rows are no
longer the disc's scan lines).

## Decoding vs reconstruction — the contract

**The default output is pure decoding, in the strict sense.** Three
guarantees: (1) every output pixel satisfies Y + Re[χe^{iφ}] = S
exactly — a lossless split of that frame's own measurement, no energy
added or removed; (2) neighboring fields are used only to *arbitrate*
the Y/C ownership, never to contribute values — the same class of
temporal usage as any conventional 3D comb decoder; (3) nothing is
synthesised: no learned priors, no hallucination surface — the worst
failure of a bad arbitration is signal in the wrong component, never
signal that was not measured.

Two opt-in flags step OUTSIDE that contract and say so loudly:
`--soft-output` is temporal FILTERING (denoised luma in the
deliverable); `--drizzle` is multi-field RECONSTRUCTION (each output
frame aggregates ~10 fields' measured samples — archival stacking,
not per-frame decoding; it never invents samples, but it is no longer
"the decode of frame t"). Both print a notice when enabled.

## Scene cuts & NR semantics (FAQ)

**Scene-change detection** is built in but continuous, never binary:
per-tile energy confidence, median-calibrated outlier rejection, and a
global pair-validity gate (`conf *= q/(q+median_residual)`) — if even
the median tile fails to match, the two fields simply do not
correspond and the whole temporal pairing is disabled. At a hard cut
the past side gates off, the future side takes over (+4 dB measured on
the cut frame, zero switching artefacts).

**What the NR touches — separation only, by default.** The NR/anchor
machinery exists because it does the one thing no downstream denoiser
can: fix the Y/C *separation* before mixing errors are baked into the
image. Smoothing the *result*, by contrast, is perfectly reproducible
downstream with dedicated tools — so the decoder does not do it. The
DEFAULT output re-imposes Y = S − Re[χe^{iφ}]: the delivered pair
exactly reconstructs the composite (a lossless split — nothing is
removed, only the Y/C ownership is decided), zero temporal smoothing
in the deliverable, and the lowest measured inter-field contamination
of any mode (combing excess ≈ −0.0005). Measured value of
guide-only NR at realistic noise (3 IRE): **+5.7 dB chroma PSNR over
2D with strictly no output NR**. `--soft-output` re-enables the
anchored luma for direct-viewing workflows.

**Self-calibration.** The robust gates (`--temporal-eps`, `--nr-eps`)
default to 0 = auto: the source's noise is measured per chunk with a
carrier-cancelling estimator (stride-4 horizontal second difference —
360° of carrier, so chroma and smooth luma cancel exactly; 25th
percentile scaling for outlier immunity) and the gates scale with it.
A fixed IRE gate is 5σ on a clean disc but 1.4σ on a noisy one,
over-gating exactly where 3D helps most.

## Performance & memory (3D mode)

* Streaming: frames are processed in sliding windows of
  `--chunk-frames` (default 6) with `--chunk-overlap` (default 2)
  frames of temporal context per side, and written out as each window
  completes. Peak RAM is bounded by the window (~230 MB at defaults)
  regardless of `-l`. Overlap ≥ 2 guarantees every emitted frame has
  its full ±1-frame neighbor and NR context; larger windows slightly
  improve cross-chunk consistency at proportional memory cost.
* Motion estimation is coarse-to-fine (4× decimated exhaustive search
  + full-resolution ±3 refinement): ~3-10× faster than exhaustive at
  identical benchmark quality. Working precision is float32.
* Ballpark on CPU/NumPy: ~10 s/frame at defaults. The inner loops
  (diffs, complex multiplies, FFTs, block SSDs) are embarrassingly
  GPU-friendly if real-time-ish speed is ever wanted.

## Cross-domain toolbox (implemented + candidates)

Ideas harvested across fields, with the implemented ones marked:

* ✅ **Robust phase-trajectory smoothing** (communications/Kalman):
  the per-line burst lock-in deviations are smoothed along the field
  by a weighted tridiagonal solve + IRLS outlier rejection. Every
  equation in the decoder depends on φ, so this hardens everything at
  once. Measured: with **25% of colour bursts destroyed**, decode
  quality is unchanged (38.9 dB vs 23.6 dB without smoothing); even
  clean sources gain ~0.15 dB.
* ✅ **OBMC-style motion field** (video coding): per-tile vectors are
  median-validated (outlier-snap: a vector is replaced by its 3×3
  median only when it disagrees by >3 px) then bilinearly
  interpolated between tile centers. All consumers of motion (raw
  equations, anchor blend) benefit at once. Measured: +1.6 dB on the
  static 3D test, +0.1 static / ±0 moving on true interlace, and
  decoder-induced combing excess at −0.0001 (measurement noise).
* ✅ **InSAR complex coherence** (radar interferometry): the temporal
  equations were gated by a luma SSD; the chroma they carry now also
  answers to gamma = |<z1 z2*>|/sqrt(<|z1|^2><|z2|^2>) between the
  current and warped-neighbor chroma phasors — phase-sensitive where
  an SSD is blind. Floored so grey content keeps the equations' luma
  benefit. Consistent +0.1 dB (interlace moving, high-noise chroma),
  never negative; `coherence_gate=0` disables (saves ~30% runtime).
* ✅ **Dubois-style adaptive spectral init** (CFA demosaicing): two
  complementary anisotropic sideband crops, blended per pixel by
  which leaves the locally smoothest residual luma — the decoder's
  arbitration criterion applied at t=0. +0.15 dB init-only, +0.22 dB
  through the full 2D pipeline, +1 dB on the static 3D test.
* ✅ **Drizzle** (astronomy, HST): optional `--drizzle` output —
  vertical 2× super-resolution by robust scatter stacking onto a fine
  grid, using the intrinsic half-line parity phases + sub-pixel
  vertical motion. Vertical only, by physics: 4fsc oversamples the
  luma 1.7× horizontally (nothing aliased to recover), while 480
  unfiltered scan lines alias heavily. Measured +0.9 dB over a linear
  2× upscale on a vertically aliased tilting scene.
* Deferred with documented reasons (see THEORY.md §10): multigrid
  (naive coarsening aliases the carrier), SURE auto-λ, f-k crawl
  notch (conflicts with the purist output policy), non-local priors.
  Top remaining real-content item: **dropout maps** as per-pixel data
  weights.

* ✅ **Signal-chain calibration audit** (pure CVBS→RGB correctness):
  field parity read from `isFirstField` metadata (index assumptions
  break on real captures); **ACC** — saturation calibrated from the
  measured burst amplitude, like every analogue TV (attenuated chain:
  19.3 → 42.6 dB); `--ntsc-j` for 0 IRE black (Japanese discs);
  fields ±3 added to the temporal system (+1.3 dB static 3D — the
  geometric horizon: ±4 is carrier-in-phase and inert).

Full mathematical/engineering reference, including negative results:
**THEORY.md**.

## Honest limitations / roadmap

* Prototype-grade Python/NumPy: ~2 s/frame at 60 CG iterations. The
  operators (diffs, complex multiplies) are trivially portable to a
  GPU or to C++ for real-time-ish speeds.
* Sub-pixel motion and the half-line parity offset are now handled in
  the anchor blend: motion vectors carry parabolic half-pel
  refinement, and the decoded baseband fields are warped bilinearly
  with exact parity-grid compensation (+0.1 dB on a pure sub-pixel
  pan test; free and structurally correct). The raw temporal
  equations deliberately stay integer-pel: a negative result worth
  recording is that an envelope-domain resampling of the raw
  composite (1-line comb -> baseband -> sub-pixel warp -> re-encode
  at target phase, which also maximises |dc| to 2) was implemented
  and REJECTED — the crude comb leaks vertical luma gradients into
  its chroma estimate and the resulting equations inject cross-colour
  on textured content (−2.7 dB moving), i.e. they manufacture the
  artefact this decoder exists to remove. Feeding measurement-grade
  equations requires measurement-grade separation; anything weaker
  belongs on the (estimate-side) anchor path, where the
  composite-domain honesty check contains it.
* Dropout maps from the .tbc.json are not yet used; wiring them into
  a per-pixel data-term weight (joint solver) would let the priors
  inpaint dropouts instead of decoding garbage.
* Hue should be verified against ld-chroma-decoder on a real capture
  of colour bars before archival use.
  Note the carrier-degeneracy: displacements of 4n+2 samples carry no
  temporal chroma information at all — a fundamental NTSC property
  this formulation makes explicit and handles gracefully.
* 3D buffers the whole requested segment (~15 MB/frame in float64);
  decode long discs in chunks with a couple of frames of overlap.
* Burst lock-in assumes the TBC did its job; heavily damaged sources
  may need a phase-map smoothness prior (one more quadratic term).
* Colour convention is calibrated against the bundled encoder and
  standard burst-on-−U NTSC; real-disc hue should be verified against
  `ld-chroma-decoder` output and trimmed if needed.
