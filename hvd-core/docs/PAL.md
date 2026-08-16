# PAL support in the HVD plugin

The engine now decodes 625-line PAL (B/G/I/D, fsc = 4.43361875 MHz)
alongside NTSC. The port follows the finding that closed the PAL
research phase (`research/reference-pal/THEORY-PAL.md` 7c #6): **the whole engine
is generic in the effective carrier** — NTSC is the special case
c = exp(i*phi); PAL folds the V-switch into c:

    S = Y + Re[chi * c],  chi = V - iU  (one phasor, both standards)
    c = +exp(+i*phi) on unswitched lines, -exp(-i*phi) on switched

so the standard dispatch lives in exactly three places:

* `engine/ntsc_geometry.*` — `VideoStandard` on `FieldGeometry`,
  `line_advance()` (180 vs 270 deg/line), `nominal_burst_ire()`
  (20 vs 21.43), and `MakeCarrierPal` (the effective carrier).
* `engine/lockin.*` — `BurstLockinPhasePal`: the swinging burst is a
  JOINT phase + V-switch-parity estimator (one lock-in output per line
  carries both; parity is measured, never assumed from line indices or
  metadata). Trajectory smoothing is shared with NTSC and absorbs the
  25 Hz offset drift.
* the two carrier-build sites (`engine.cpp` frame path,
  `sequence.cpp` PrepareFieldObs) dispatch on `g.standard`.

Everything else — holographic init, the variational arbitration, the
temporal equations, the anchor loop, motion, drizzle — is untouched:
it only ever sees the carrier arrays.

## PAL-specific behaviour in the sequence pipeline

* Temporal offsets extend to f+/-4: PAL's 180-deg carrier relation
  (the classic "3D comb" equation, |dc| = 2) sits at TWO frames, and
  the 135 deg/field ladder makes every offset in +/-4 non-degenerate.
  Odd offsets' oscillating |dc| (0..2 with x, mean-square 2 — a PAL
  derivation, THEORY-PAL 5c) is handled by the existing machinery:
  actual carrier arrays + the half-line envelope gate.
* PAL-swept defaults applied when the NTSC-tuned values are left in
  place: coherence_gate 0.6 -> 1.0 and nr_radius 2 -> 4 (measured in
  the reference: full-weight coherence + nu 0.5 strictly dominates;
  the anchor blend's leakage anti-correlation lives at f+/-4 on PAL).
* The adaptive-strength ambiguity discriminator pairs same-parity
  180-deg fields at stride 4 instead of 2 (same physics, longer
  baseline). The default chunk_overlap of 2 frames provides exactly
  the 4 fields of context the +/-4 offsets and nr_radius need.

## Validation status

* `tests/engine/pal_lockin_test.cpp` (hermetic, SDK-free): joint
  estimation clean + with a destroyed burst block; the effective-
  carrier hologram identity to 1e-3 IRE; dispatch helpers.
* `tests/engine/pal_aniso_group_test.cpp`: proves the standard-dependent
  leak cancellation in AUTO `chroma_aniso` — the PAL leak survives a
  2-line average with residual sqrt(2) and collapses to <1e-5 under the
  4-line average actually used, while NTSC's cancels at 2. This shipped
  wrong once (PAL was measured with the NTSC 2-line average); the test
  asserts the PAL/group-2 case still FAILS to cancel, so the dispatch
  cannot be "simplified" away again.
* Cross-validated against the Python reference on an identical
  full-geometry field (1 IRE noise, 15% burst dropout): max phase
  difference 0.007 deg, 0/313 parity mismatches.
* NTSC engine tests unaffected (lockin/colour re-run green); all
  modified translation units compile clean.

## SDK seams — now verified

Two of the three originally-unverifiable identifiers have since been
checked against the real SDK headers:

1. `VideoFormatCompatibility` — VERIFIED. The both-standards value is
   `ALL` ("Works with any format (NTSC, PAL, PAL-M, etc.)",
   `orc/stage/node_type.h`). An earlier revision guessed `ANY`; fixed
   in `plugin.h` and the descriptor.
2. `VideoSystem::PAL` — VERIFIED (`orc/stage/common_types.h`,
   documented as "625-line PAL"), and `sample_rate_from_system()` /
   `fsc_from_system()` in `cvbs_signal_constants.h` return
   `kPalSampleRate = 17 734 475` / `kPalFsc = 4 433 618.75`, matching
   the engine's own `kFs4FscPal` / `kFscPal` exactly.
3. `chroma_phase_deg` default — RESOLVED, and the answer is **0 for both
   standards**. The old 180 was never a property of any source: it
   cancelled a sign error in the NTSC lock-in (`theta = arg(z) + pi/2`
   where the correct relation is `arg(z) - pi/2`; the NTSC burst is on
   -U, so `chi_burst = +iB` and `burst = -B sin(phi)`).
   `BurstLockinPhasePal` derived the swinging-burst case correctly from
   the start — verified analytically and numerically: `arg(z) - theta`
   is `+pi/4` on unswitched lines and `+3pi/4` on switched ones, exactly
   what `theta_ref = a_meas - pi/2 + s*pi/4` implements. So the global
   180 made NTSC look right *and rotated every PAL decode by half a
   turn*. The sign is fixed in `lockin.cpp`; the default is 0 in
   `HvdConfig` **and in the GUI ParameterDescriptor** (the two are now
   cross-checked by `tests/stage_smoke_test.cpp`).

   The parameter is a RELATIVE trim added on top of the measured burst
   phase, never a replacement for it, and on the PAL family it is signed
   by the measured V-switch parity (`ApplyChromaPhase`) — otherwise a
   constant offset rotates chi by `-delta` on one line parity and
   `+delta` on the other, i.e. Hanover bars at every angle except 0 and
   180. `tests/engine/chroma_reference_test.cpp` pins all of this.

## What is ported from the research, and what is not

The Python reference (`research/reference-pal/`) is the oracle; not all of it is
decoder functionality. Current state:

| Research feature | C++ | Note |
|---|---|---|
| Effective carrier (V-switch folded into c) | yes | `MakeCarrierPal` |
| Swinging-burst joint phase+parity lock-in | yes | `BurstLockinPhasePal` |
| PAL temporal offsets to f+/-4 | yes | `sequence.cpp` |
| PAL-swept defaults (coherence 1.0, nr_radius 4) | yes | applied when NTSC defaults untouched |
| Same-parity ambiguity stride 4 | yes | `amb_stride` |
| AUTO chroma_aniso, PAL 4-line leak cancellation | yes | `HvdConfig::is_pal`, guarded by `pal_aniso_group_test` |
| Half-line envelope gate on odd offsets | yes | shared with NTSC |
| Trajectory-coherent motion | yes | shared |
| Anchor loop (decode -> NR -> re-encode) | yes | `SynthReference` |
| Transform-PAL certifier init variant | yes | `symmetry_variant` |
| PSI closed form | yes | `PsiClosedForm` |
| Drizzle | partial | see below |
| PolSAR cross-pol diagnostic map | **no** | operator-facing diagnostic, not decoding — see below |
| Delay-line baseline decoder | **no** | deliberately: a research A/B reference (the incumbent to beat), not something to ship |

**Drizzle is partial.** The C++ implementation deposits with INTEGER,
margin-snapped motion vectors, so it exploits the guaranteed half-line
parity phase between fields but not sub-pixel motion dither. The
research added parabolic sub-pixel refinement, vertex-cost confidence
and a bilinear agreement check to exploit both (THEORY-PAL 5d). Porting
those is a real improvement AND a trap: the deposit formula in
`DrizzleFrame` is coupled to the integer-vector convention, and adding
sub-pixel without correcting it lands static cross-parity content a
full frame-line off. The coupling is spelled out in a comment at the
deposit site so it cannot be changed in isolation.

**PolSAR cross-pol map is absent.** It answers "where and how badly is
this source's chain differential-phase distorted?" by decomposing
V-switch line pairs into co-/cross-polarised channels (20x
discrimination measured in the research). It consumes nothing the
decoder produces — it must read the RAW per-line demodulation, since
the vertical prior deliberately arbitrates the evidence away — so it
would be a small standalone analysis path, not a change to the solver.
Worth adding as an archivist tool; not required for decoding.

## Not yet claimed

PAL-M and PAL-N (PAL-type chroma on 525/60 and 625-line/3.582 MHz
geometries) stay on their previous paths: the effective-carrier
machinery is ready, but their line-advance/burst models are underived.
The PAL drizzle/subpel refinements from the reference (vertex-cost
confidence, bilinear agreement, trajectory-base subpel; THEORY-PAL
5d + 7c #2-3) also flag two latent issues in the NTSC drizzle port —
audit before enabling drizzle on either standard.
