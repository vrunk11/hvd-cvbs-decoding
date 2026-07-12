# hvd-decode — Technical reference

This document consolidates the mathematics and the engineering
decisions of the decoder, including the negative results (approaches
tried and rejected, with the reason). Section numbers match the
pipeline order. Code pointers are given as `module.function`.

## 1. Signal model (the hologram identity)

A time-base-corrected NTSC field sampled at 4·fsc is modelled exactly
as an off-axis hologram:

    S(x, y) = Y(x, y) + Re[ χ(x, y) · e^{iφ(x,y)} ] + n
    χ = V − iU        (complex chrominance phasor, baseband)
    φ(x, y) = θ(y) + (π/2)·x

90° of carrier per sample horizontally (4fsc), 180° per line
vertically (227.5 cycles/line). The mapping to holography is exact:
Y is the zero-order term, χ the object wave, dot crawl = zero-order
leakage, cross-colour = the twin image. `decoder.holographic_init`.

Colour conventions: composite chroma = U·sin(ωt) + V·cos(ωt), burst
on −U (amplitude 20 IRE), hence the lock-in identity of §2. RGB↔YUV
uses BT.601 coefficients (`encode.RGB_TO_YUV`, `decoder.YUV_TO_RGB`).

## 2. Phase recovery (lock-in + robust trajectory smoothing)

Per line, a lock-in amplifier on the colour burst
(`decoder.burst_lockin_phase`):

    z = mean( burst(x) · e^{−i(π/2)x} )    ⇒    θ = arg z + π/2

The set {θ(y)} is then treated as a *trajectory* around the π-per-line
model (a communications/Kalman idea): the deviation d = θ − model is
smoothed by a weighted quadratic solve

    min_x  Σ a_y (x_y − d_y)² + λ Σ (x_{y+1} − x_y)²    (tridiagonal,
    Thomas algorithm, `decoder._tridiag_smooth`)

with amplitude-derived weights and one IRLS Huber pass to reject
outliers. Measured effect: with 25% of bursts destroyed, decoding is
unchanged (38.9 dB vs 23.6 dB without smoothing). Every equation in
the decoder depends on φ; this is the foundation layer.

## 3. Initialisation (holography + Dubois adaptive sidebands)

Demodulate: z = S·e^{−iφ} moves the chroma sideband to DC and luma to
±fsc. Two complementary anisotropic Gaussian crops are computed
(narrow-x/wide-y and wide-x/narrow-y — the two ways luma can
contaminate the sideband), and blended per pixel by whichever leaves
the locally smoothest residual luma (CFA-demosaicing idea, Dubois).
This is the decoder's arbitration criterion applied at t = 0 and only
serves as the IRLS linearisation point. +0.2 dB end-to-end.

## 4. The arbitration principle (core insight)

The data term ‖S − Y − Re[χe^{iφ}]‖² is *invariant* to transfers
between Y and modulated chroma: any sample can be explained by either.
Y/C separation is therefore not a filtering problem but an
**arbitration** problem. Fidelity is enforced by substitution
(Y := S − Re[χe^{iφ}]) and the energy is minimised over χ alone:

    E(χ) = Σ ρ(∇Y(χ))                      luma plausibility
         + μ_h Σ ρ_c(∂x χ) + μ_v Σ ρ_c(∂y χ)   chroma priors
         + ν Σ_k w_k · r_k(χ)²             temporal equations (§5)

ρ, ρ_c are Charbonnier penalties solved by IRLS (lagged diffusivity)
with normalised diffusivities (=1 in flat areas so the calibrated
luma/chroma balance is scale-free), plus a parallel-level-sets
coupling (PET/MRI reconstruction): chroma diffusivity opens wherever
the residual luma sees an edge, killing "hanging dots" at chroma
transitions. Inner solver: nonlinear CG with exact line search on the
quadratic majorant (`decoder.variational_refine`).

The two classic NTSC artefacts police each other: dot crawl is
carrier-frequency ripple in Y (huge ρ cost ⇒ migrates to χ);
cross-colour is a 2fsc oscillation in χ (huge ρ_c cost ⇒ migrates
back to Y).

## 5. Temporal equations (fields as measurements)

Decode unit = the FIELD (the two fields of a frame are 1/60 s apart;
weaving before decoding contaminates fields across time — a failure
mode found on real content and reproduced with the true-interlace
encoder, `encode.write_tbc` with per-field images).

Neighbors of field f: f±2 (same parity, carrier flip 180°, exact
3D-comb geometry) and f±1 (adjacent, carrier offset 3/4 cycle,
|dc| = √2, non-degenerate). Each contributes a *raw-measurement*
residual, with the neighbor's composite and measured phase map
motion-warped (`decoder.motion_compensate_prev`):

    r_k = (S_k^w − S) + Re[χ (c − c_k^w)]

Zeroing r_k for a static pixel with the 180° flip forces
Re[χc] = (S − S_k)/2 — the 3D comb, embedded in the arbitration.
Confidence layers (all continuous, no binary switching):

* per-tile energy confidence (best-SSD vs tile variance);
* median-calibrated outlier rejection;
* global pair validity ∝ 1/(1 + median residual) — a free scene-cut
  detector: if even the median tile fails to match, the fields do not
  correspond;
* Geman-McClure gate on r_k, with ε_t auto-calibrated to ~7× the
  measured noise (§8);
* InSAR complex coherence (§7) as a phase-sensitive chroma gate.

Carrier degeneracy worth knowing: a displacement of 4n+2 samples
rotates the carrier by 180° and cancels the frame flip (dc → 0); no
temporal chroma information exists for such motion. The formulation
makes this explicit — the term goes inert instead of corrupting.

NEGATIVE RESULT (recorded so nobody retries it naively): an
envelope-domain resampling of the raw composite (1-line comb →
baseband → sub-pixel warp → re-encode at target phase, maximising
|dc| = 2) was implemented and rejected: the crude comb leaks vertical
luma gradients into its chroma estimate and the equations inject
cross-colour on textured content (−2.7 dB moving). Measurement-grade
equations require measurement-grade separation; anything weaker
belongs on the estimate-side anchor path (§6), where the
composite-domain honesty check contains it.

## 6. The re-encode loop (guides separation; never the output)

Between passes, a reference is manufactured from the decodes:
motion-compensated robust temporal blending of (Y, χ) over
±nr_radius fields (`decoder.synth_reference`) — NTSC geometry makes
the blend powerful: leakage anti-correlates at ±1 frame (cancels),
±2 frames are carrier-in-phase (pure noise averaging). The blend is
then RE-ENCODED, Ŝ = Ŷ + Re[χ̂e^{iφ}], and the anchor confidence is
|S − Ŝ| in the composite domain: the reference is only trusted where
it explains the raw measurement.

Pass ≥ 2 solves jointly over (Y, χ) with soft data fidelity anchored
to (Ŷ, χ̂) (`decoder.variational_refine_joint`) — this is what breaks
the noise-floor ceiling of exact fidelity. Blend warps are bilinear
sub-pixel with exact parity-grid compensation (legal in baseband).

Output policy: the DEFAULT re-imposes Y = S − Re[χe^{iφ}] — a
lossless split (the delivered pair reconstructs S exactly), zero
temporal smoothing in the deliverable. The NR machinery's only
irreplaceable contribution is separation guidance (+5.7 dB chroma at
3 IRE noise with strictly no output NR); smoothing the result is
reproducible downstream and is therefore not done (opt back with
--soft-output).

## 7. Motion field (video-coding & radar imports)

Coarse-to-fine tiled block matching (4×-decimated exhaustive search,
full-res ±3 refinement, parabolic half-pel), zero-motion margin rule
(a spurious 1-sample shift rotates the carrier 90°), then:

* vector-median OUTLIER-SNAP: a vector is replaced by its 3×3 median
  only when it disagrees by > 3 px — isolated flat-tile garbage is
  squashed, coherent clusters (small objects) pass untouched;
* OBMC-style bilinear interpolation of tile vectors between tile
  centers (`decoder._vectors_per_pixel`) — removes tile-warp seams
  for every consumer of motion at once;
* InSAR complex coherence γ = |⟨z₁z₂*⟩|/√(⟨|z₁|²⟩⟨|z₂|²⟩) between
  current and warped-neighbor chroma (`decoder.complex_coherence`),
  floored so grey content keeps the equations' luma benefit.

## 8. Self-calibration

`decoder.estimate_noise_ire`: stride-4 horizontal second difference —
the carrier completes 360° over 4 samples, so chroma AND smooth luma
cancel exactly, leaving noise (variance 6σ²); 25th-percentile scaling
for outlier immunity. Gates scale from σ per chunk. Rationale: a
fixed IRE gate is 5σ on a clean disc but 1.4σ on a noisy one,
over-gating exactly where 3D helps most.

## 9. Drizzle output mode (optional — reconstruction, NOT decoding)

Boundary statement: everything up to §8 is decoding (the purity
contract of README holds: output splits S losslessly). Drizzle steps
outside — its output grid is synthesized and its values are stacked
estimates; it is an archival *enhancement* tool, opt-in, announced on
stderr, never part of the default pipeline.


`decoder.drizzle_frame`, `--drizzle`: astronomy-style scatter
stacking onto a 2× fine VERTICAL grid, using the intrinsic half-line
parity phases plus sub-pixel vertical motion, with robust weights and
graceful fallback to weave-interpolation where coverage is thin.
Vertical only, by physics: at 4fsc the ~4.2 MHz luma is oversampled
1.7× horizontally (no aliased detail exists to recover), whereas 480
unfiltered scan lines alias heavily. Measured: +0.9 dB over linear 2×
upscale on a vertically aliased tilting scene.

## 9b. Signal-chain calibration (pure decoding correctness)

Three items that dominate *correctness* on real hardware chains, all
in the CVBS→RGB path proper:

* **Field parity from metadata** (`isFirstField` in the .tbc.json):
  index-parity assumptions break on captures starting on a second
  field or with skipped fields — inverted weave and wrong half-line
  offsets everywhere. `decode_sequence.field_parity`.
* **ACC — Automatic Color Control** (`decoder.burst_amplitude_ire`):
  chroma gain is normalised by the MEASURED burst amplitude (nominal
  20 IRE), as every analogue TV does. Validated: a source chain
  attenuated to 65% decodes at 19.3 dB without ACC, 42.6 dB with —
  full saturation restoration. Clip [0.5, 2.0]; `--no-acc` disables.
* **NTSC-J setup** (`--ntsc-j`): black at 0 IRE instead of the 7.5 IRE
  NTSC-M pedestal. Wrong setup = wrong black level and contrast on
  Japanese discs.

Temporal geometry completion: fields f±3 carry a 2.25-cycle carrier
offset (90°, |dc| = √2) and contribute two more non-degenerate
equations (+1.3 dB on the static 3D test); f±4 is carrier-in-phase
(dc = 0, inert) — the temporal neighborhood is now exhausted out to
its geometric horizon.

## 9c. Pre-port audit (engineering hygiene, no new ideas)

Performed before the C++ port; found and fixed:

* Active vertical geometry read from `firstActiveFieldLine` /
  `lastActiveFieldLine` metadata instead of a hardcoded 21.
* Gradient operators switched from periodic wrap (which coupled
  opposite image borders through the priors) to Neumann boundaries,
  with the D/Dᵀ adjoint pair verified to 1e-14 (a requirement for the
  CG gradients to be true gradients — see the adjointness test).
* Hyperparameter re-sweep on the CURRENT architecture: the chroma
  anisotropy μ_h/μ_v, tuned when "vertical" meant frame lines, moved
  from 0.25 to 0.5 after the field-space switch — exactly the 2×
  predicted from the line-pitch change (+0.4 dB). λ_c stays at 1.0.
  Lesson recorded: every geometry change stales the priors' scales.

Remaining, deliberately unimplemented refinements (marginal):
asymmetric I/Q bandwidths (1.3/0.6 MHz on the 33° axes) for
broadcast-sourced material; diagonal (8-neighbour) prior terms.

## 9d. Literature review (CVBS decoding papers, borrowability audit)

Systematic pass over the published record, with conclusions:

* **BBC Transform PAL** (Easterbrook/Russell, BBC RD; ported to PAL in
  ld-chroma-decoder, never to NTSC): extracts chroma by enforcing
  spectral symmetry about the carrier. The historically documented
  NTSC blocker: I and Q share one carrier, quadrature interference
  makes the sidebands asymmetric ((kQ+kI) != (kQ-kI)), defeating
  symmetry as a *detector*. Two findings recorded here:
  (1) our forward model dissolves the blocker — a complex χ
  represents asymmetric sidebands natively, so the variational
  formulation is, in effect, the NTSC-capable generalisation
  Transform never got; (2) the symmetry test repaired as a
  *certifier* (min(|Z(+k)|,|Z(−k)|) = a chroma lower bound luma
  almost never fakes) was implemented as an init variant
  (`symmetry_variant`, the first "Transform NTSC" implementation we
  know of) and measured NEUTRAL (±0.05 dB, cross-colour −1%): the
  arbitration's residual-smoothness weighting already extracts the
  same evidence. Kept as a reference implementation, default off.
* **Consumer artifact-removal literature 2005-2007** (Lee & Le
  dot-crawl decision maps + bilateral filtering; adaptive linear +
  trained neural filters; motion-gated temporal averaging patents):
  all post-decode processing on already-separated video. Subsumed:
  their "decision maps" are hard-threshold ancestors of our
  continuous per-pixel gates, applied after the information was
  already lost. Excluded from output by the purity contract anyway.
* **Classical comb patents** (2D/3D adaptive, correlation blending,
  phase motion detectors): the machinery this decoder replaces;
  their adaptive heuristics are special cases of the IRLS weights.
* **Learning-based restoration** (2005 neural filters through modern
  approaches): excluded by design — a trained prior can synthesise
  plausible-but-unmeasured content, violating the measurement-grade
  contract of §6. The re-encode honesty check has no equivalent for
  a hallucinating prior.

* **"3-phase decoding"** (3fsc-sampling demodulators, e.g. Philips
  WO1996013127; per-pixel phase-structured sampling patents): three
  samples at 120° solve (Y, U, V) algebraically per triplet — a
  horizontal FIR in disguise, assuming (Y, χ) constant over 3
  samples. Its noble generalisation is optics' N-step
  **phase-shifting interferometry**: our field set {f, f±1..±3}
  hands every static pixel up to 7 observations under known,
  burst-measured carrier phases — literally an N-step PSI dataset.
  The explicit closed form (per-pixel 3×3 weighted LS, vectorised
  Cramer, `decoder.psi_closed_form`) was implemented as an init and
  measured: identical on static (the CG already reaches the PSI
  point), −0.3 dB on motion even with the robust gate replicated at
  the init — committing early loses to the IRLS's implicit
  navigation of the nonconvex reweighting path. Second subsumption
  result; kept as a reference implementation, `psi_init` default off.

Net result of the review: one historical gap closed (Transform NTSC),
two subsumption theorems-in-practice (spectral symmetry ⊂
arbitration; N-step PSI ⊂ the temporal equations under IRLS), zero
borrowable ideas left unharvested in the surveyed record.

## 9e. The "4th dimension" brainstorm (mapping + two additions)

A free-form ideation session (external) proposed ten "extra
dimensions" for video processing. Honest mapping: six were already
this decoder's foundations — confidence-as-dimension = the gate
stack; the complex dimension = χ itself; kinematic neighborhoods and
the "warped 4D fabric" = motion-compensated equations; between-pixel
coordinates = sub-pixel/drizzle; local frequency = the adaptive
spectral init. The semantic dimension is excluded by the purity
contract. Two ideas were genuinely uncovered and are now implemented:

* **Trajectory-coherent motion** ("the 1D filter that follows the
  matter"): the offsets f±1..±3 are six independent pairwise matches
  of ONE physical velocity. A per-tile velocity is fitted
  (confidence-masked median of d_k/k, with the known half-line parity
  term (p_k−p_j)/2 removed before fitting — sign matters, the first
  attempt doubled the bias), and pairwise vectors snap to k·v only
  under CONSENSUS (≥3 offsets agreeing within 1.5 px): structured
  motion collapses onto the trajectory, noise-dominated matching is
  left untouched. Measured: +0.08/+0.09 dB on both benches, never
  negative. Default on (`trajectory_fit`).
* **Oriented (±45°) chroma priors** (the (x,y,θ) dimension):
  diagonal Charbonnier terms with exact adjoints (verified 1e-14),
  renormalised so the total prior mass is unchanged. Measured
  trade-off, not a win: −1.0 dB on axis-aligned sharp chroma (SMPTE)
  vs +2.0 dB on diagonal cross-colour torture (zoneplate). Default
  off; a documented dial (`--diag-prior`) for diagonal-artifact-heavy
  material (fine weaves, venetian blinds).

## 9f. Fast mode (optimisation spec for the C++ port)

`--fast` keeps the algorithm and changes the logistics. Measured in
Python: ~2x wall clock, −0.2/+0.2 dB depending on bench (never worse
than 0.2). The operation-count savings are larger than the Python
wall clock shows (interpreter overhead flattens them); expect the
C++ port to realise closer to the true ratios. Components:

* **Motion cache** (biggest): one estimate per field pair per chunk,
  shared across passes AND with the anchor blend — the same (j,k)
  motion was previously computed up to 4x. Slow mode keys per pass
  (bit-exact preservation of previous behaviour).
* **Predicted+verified ME**: full pyramid search only for offsets
  {−1,+1,+2}; the rest get trajectory-predicted vectors audited by
  TWO SSD evaluations (predicted + zero) instead of ~130 — a ~60x
  reduction on those offsets (`decoder.verify_motion`).
* **Adaptive CG exit**: relative gradient-norm tolerance (0.02 slow /
  0.10 fast) instead of a fixed iteration count; fast also caps total
  CG iterations at 2/3.
* **Tile-resolution confidence maps**, bilinearly interpolated (the
  full-res blur only ever smoothed at sub-tile scale): ~256x cheaper
  on those maps.
* **Deferred coherence**: the InSAR gate starts at pass 1 (at pass 0
  the chroma fields are inits; the measurement is barely informative).
* **Exact rewrites benefiting BOTH modes**: box blur via integral
  images (two prefix-sum passes, verified to 1e-16 against the
  original INCLUDING its constant-normalisation edge semantics and
  r=2 default — an earlier rewrite silently changed the default
  radius and cost 0.75 dB, caught by the benches); per-pixel warp
  vectors computed once per warp trio instead of three times.

C++ port notes: the integral-image blur is two prefix sums (SIMD/
cache friendly); the motion cache is a per-chunk hash keyed (j,k);
FFTW plans should be created once per field geometry; tiles and
fields parallelise trivially (each field's refine is independent
within a pass).

## 10. Deferred items, with reasons

* **Multigrid / coarse-to-fine solving**: naive spatial coarsening is
  unsound here — decimating the composite aliases the carrier (90°/
  sample becomes 180°/sample at 2×). A correct multigrid needs a
  baseband reformulation of the coarse-grid operator; worthwhile for
  a production port, out of scope for the prototype.
* **SURE auto-λ**: attractive, but SURE assumes a known noise model
  acting on the *observable*; our arbitration energy scores an
  unobservable split. A principled risk estimate needs care; deferred.
* **Horizontal super-resolution**: physically empty (see §9).
* **Dropout maps** from .tbc.json: highest-value remaining item for
  real LaserDisc — wire them as a per-pixel data-term weight in the
  joint solver so the priors inpaint dropouts.
* f-k velocity notch (geophysics) for residual crawl: conflicts with
  the purist output policy; could serve the anchor path only.

## 11. Test suite map

* `run_tests.py` — regression floor: 2D arbitration, plain 3D,
  anchored joint pass.
* `bench_motion.py` — progressive motion scene (static / pan /
  object), naive-3D-comb and 2D baselines.
* `bench_interlace.py` — TRUE interlace (fields 1/60 s apart): the
  benchmark that reproduces real-content combing; per-field ground
  truth and a decoder-induced-combing excess metric.
* `tests_extra/burst_test.py` — corrupted-burst robustness (§2).
* `tests_extra/subpel_test.py` — pure sub-pixel pan (§7).
* `tests_extra/drizzle_test.py` — vertical SR vs linear upscale (§9).
* `tests_extra/acc_test.py` — ACC saturation restoration (§9b).
