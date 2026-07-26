# hvd-pal — Technical reference (research phase)

Companion architecture to the NTSC hvd decoder. Same philosophy
(hologram identity, burst lock-in trajectory, arbitration-by-
substitution, purity contract), DIFFERENT architecture folder: PAL's
structure changes the mathematics in ways that deserve their own
derivations, defaults, and tests rather than flags on the NTSC code.

## 1. The PAL hologram identity (the V-switch is phase conjugation)

A time-base-corrected PAL field at 4fsc, on the stored 1135-sample
grid:

    S(x, l) = Y + U*sin(phi) + s(l)*V*cos(phi) + n
    phi(x, l) = theta(l) + (pi/2)*x
    s(l) = +/-1                                (V-switch)

90 deg of carrier per sample (4fsc), 270 deg per line (283.75
cycles/line on the stored grid — the true line is 1135.0064 samples;
the 25 Hz offset survives as a slow +0.57 deg/line phase drift that
the lock-in MEASURES rather than assumes).

Define ONE global chroma phasor chi = V - iU. Then, exactly:

    S = Y + Re[ chi * c ]
    c(x, l) =  exp(+i*phi)    where s = +1
            = -exp(-i*phi)    where s = -1

Proof for s = -1: Re[(V - iU)(-e^{-i*phi})] = U*sin(phi) - V*cos(phi).

This is the load-bearing structural result of the port. The V-switch
is not a per-line sign to special-case everywhere — it is a
CONJUGATED REFERENCE WAVE, folded into an effective carrier of unit
modulus. Every equation of the NTSC decoder touches the carrier only
through Re[chi*c] and conj(c), so the entire variational machinery
(init, gradients, arbitration, temporal residuals) is generic in c
and ports unchanged. NTSC becomes the special case c = e^{i*phi}.

In optics this has a precise name: **two-step phase-conjugate /
phase-shifting holography**. Alternate lines record the hologram with
the conjugate reference; summing a pair cancels the twin image to
first order. PAL's designers built phase-shifting interferometry into
a 1967 broadcast standard — and the classic artefacts map exactly:

| PAL artefact          | holography artefact                        |
|-----------------------|--------------------------------------------|
| dot crawl (unchanged) | zero-order leakage                         |
| cross-colour          | twin image                                 |
| **Hanover bars**      | **fringe-parity misregistration**: pairing |
|                       | lines with the wrong conjugation sense     |
| hue error -> desat.   | phase error split +delta/-delta between    |
|                       | conjugate recordings; sum = cos(delta)     |

## 2. Swinging burst = a joint phase+parity estimator (polarimetry)

The burst phasor is (U_b, V_b) = (-B/sqrt2, +B/sqrt2) through the same
V-switched modulation, so the lock-in output per line is

    z_l ~ (B/2) * exp(i*(theta_l + pi/2 - s_l*pi/4))

One complex number carries BOTH unknowns: the mean direction is the
carrier phase, the sign of the +/-45 deg swing is the V-switch parity.
The analogy is dual-polarisation radar/polarimetry: a single coherent
measurement decomposed into a common phase and an alternating
polarisation state. Consequences:

* Parity is MEASURED per field (two-hypothesis scoring over all
  burst-bearing lines, amplitude-weighted), never derived from line
  indices, fieldPhaseID, or the Bruch sequence. Verified: exact parity
  recovery with 25% of bursts destroyed (`run_tests` §2).
* Phase uses the same robust trajectory smoothing as NTSC (tridiagonal
  weighted solve around the 270 deg/line model + one IRLS Huber pass).
  The smoother's slow-drift tolerance absorbs the 25 Hz offset
  component for free. Measured: 0.46 deg max phase error at 25%
  dropout.
* ACC carries over verbatim: |z| = B/2 regardless of swing direction
  (rotation does not change modulus). Nominal B = 21.43 IRE
  (+/-150 mV burst on the 700 mV PAL white).
* PAL has no setup pedestal: black = blank = 0 IRE (the NTSC-J flag
  has no PAL equivalent; deleted rather than defaulted).

## 3. Arbitration, and why PAL strengthens it

Same principle: the data term is invariant to Y <-> Re[chi*c]
transfers; fidelity is enforced by substitution Y := S - Re[chi*c] and
the energy is minimised over chi (IRLS + CG, Charbonnier priors,
parallel-level-sets Y->chi coupling). What PAL adds:

**Hanover bars are the highest vertical frequency chi can carry.** A
differential phase error delta makes the per-line measured chroma
chi*e^{+i*delta} on s=+1 lines and chi*e^{-i*delta} on s=-1 lines — a
line-alternating conjugation error. Under the effective carrier this
is a maximal-vertical-frequency oscillation in chi, i.e. exactly what
mu_v penalises hardest. The optimiser therefore converges to the
line-pair-coherent solution chi*cos(delta): hue errors come out as
slight desaturation, no bars.

This subsumes the delay-line: **the analogue delay-line PAL decoder
is the quadratic, confidence-free limit of this decoder's vertical
chroma prior** (a hardwired 2-tap vertical average vs an adaptive,
edge-preserving, arbitrated one). Measured (run_tests §4, 10 deg
differential phase, 0.4 IRE noise): median hue error 2.3 deg,
saturation 97.8% vs cos(10 deg) = 98.5% predicted, line-alternating
hue 1.7 deg (no bars). And §3: the variational decode beats the
delay-line baseline by ~18 dB on the EBU-bars bench (42.5/42.3 dB
Y/C vs 24.1/24.1).

**Chroma anisotropy: AUTO, PAL-adapted.** The NTSC auto measurement is
ported with one real change: the leak-cancelling pre-average must be
4 LINES on PAL, not 2. Under the effective carrier, conj(c) alternates
between e^{-i*phi} and -e^{+i*phi} line to line (sign flip AND
conjugation), so the NTSC 2-line pair average does not cancel the
cross-colour leak; the pattern repeats over 2 lines and the
270 deg/line carrier walk completes 1080 = 0 mod 360 over 4 — a 4-line
average cancels both. Map centred nearer isotropy ([0.55, 1.0]; PAL
U/V bandwidths are equal). Measured: picks 0.55 on the bars chart
(floor, matching NTSC picking its 0.5 floor on charts).

## 4. Cross-domain survey (PAL-specific, beyond the NTSC set)

Approaches examined from other fields, with a verdict each:

* **Phase-shifting interferometry with a pi-conjugate step** (optics):
  IS the V-switch — adopted as the core identity (§1). The N-step PSI
  generalisation applies to the temporal dimension exactly as in NTSC
  (§5), with PAL's richer 8-field sequence providing MORE distinct
  carrier states per pixel (135 deg granularity vs NTSC's 180).
* **Polarimetric decomposition (PolSAR, Pauli basis)**: IMPLEMENTED
  (`decoder.polarimetric_maps`, `--polsar-map`). (unswitched,
  switched) line pairs are two polarisation channels; the co-pol
  component is pair-coherent chroma, the cross-pol channel isolates
  differential-phase energy at |chi|*sin(delta) — an operator-facing
  map of WHERE the source chain is phase-distorted. Measured: 0.28 IRE
  median on a clean chain vs 5.76 IRE at 10 deg differential phase
  (20x separation). NEGATIVE RESULT recorded: computing the map on the
  DECODED chroma reads ~0 by construction (0.15 IRE measured vs 7.6
  predicted) — the vertical prior has already arbitrated the
  line-alternating evidence away, exactly as designed. The map must
  use the raw per-line demodulation (horizontal low-pass only). The
  decoder itself never consumes the map.
* **Staggered-PRF Doppler radar** (the 25 Hz offset): PAL's quarter-
  line + 25 Hz offset is a deliberate spectral interleave, the same
  trick staggered PRFs use to push ambiguities apart. Consequence
  adopted: the vertical spectral position of luma-vs-chroma interleave
  differs from NTSC, and the per-line-conjugated demodulation splits
  luma leakage across two vertical alias positions. The Dubois dual
  anisotropic crops already arbitrate this (verified by the init's
  30.5 dB on bars); a PAL-tuned third crop is a measurable TODO.
* **Fringe-projection two-frequency unwrapping** (metrology):
  considered for resolving carrier-phase ambiguity across long
  dropouts using the 135 deg/field ladder; rejected for now — the
  trajectory smoother + per-field re-lock already covers realistic
  dropout spans, and the tbc is time-base corrected. Recorded as a
  negative-by-sufficiency.

## 5. Temporal geometry (derived; implementation = roadmap)

PAL's 8-field sequence changes the neighbor table. On the stored
grid the field origin advances 135 deg/field (8 x 135 = 1080 = 0
mod 360), so relative to field f, neighbor f+k has carrier rotation
Delta_k = k*135 deg and dc = c - c_k with |dc| = 2|sin(Delta_k/2)|
for parity-aligned pixels (same-sense lines; the V-switch parity of
the sequence adds a conjugation bookkeeping term the effective-carrier
formulation makes explicit rather than hazardous):

| offset k | Delta   | |dc|      | note                                |
|----------|---------|-----------|-------------------------------------|
| +/-1     | 135 deg | 1.85      | adjacent field, strong equation     |
| +/-2     | 270 deg | 1.41      | 1 frame (same parity)               |
| +/-3     | 45 deg  | 0.77      | weak but non-degenerate             |
| +/-4     | 180 deg | 2.00      | 2 frames — the classic PAL 3D comb  |
| +/-8     | 0       | 0 (inert) | carrier-in-phase, geometric horizon |

Two structural differences from NTSC worth recording now:

* NTSC's strongest equation is at 1 frame (180 deg); PAL's is at TWO
  frames (f+/-4) — exactly why hardware PAL 3D combs needed 2 frame
  stores and were rare/expensive. Here it just means the temporal
  window wants radius 4+ fields and the motion-degeneracy table
  (displacements rotating the carrier by -Delta_k) must be re-derived
  per offset before trusting any equation.
* Every |dc| > 0 offset contributes: PAL's 135 deg ladder gives up to
  SEVEN non-degenerate carrier states within +/-4 fields (an N-step
  PSI dataset richer than NTSC's), before hitting the f+/-8 horizon.

IMPLEMENTED (this revision): the SAME-PARITY subset — f+/-2 (270 deg,
|dc| = sqrt2) and f+/-4 (180 deg, |dc| = 2, the 2-frame PAL 3D comb) —
with the ported motion stack (coarse-to-fine BM, zero-motion margin
rule, vector-median outlier-snap, OBMC per-pixel vectors, tile-energy
+ median-calibrated + scene-cut confidences), Geman-McClure gating
with eps_t auto-calibrated to 7x measured noise, and the InSAR
coherence gate. Decode unit = the FIELD (weave after; NTSC lesson
kept). One PAL-specific derivation worth recording: for same-parity
neighbors the static geometry is c_k = c*e^{+i*Delta} on s=+1 lines
but c*e^{-i*Delta} on s=-1 lines — the V-switch CONJUGATES the
rotation. |dc| is parity-independent (the table stands), and because
the equations use the actual warped carrier arrays, the bookkeeping is
exact by construction rather than assumed. Integer-pel raw warps copy
carrier VALUES, so each source line's parity travels with its samples.

## 5c. Odd offsets f+/-1, f+/-3 (derived, implemented, measured)

The PAL derivation before implementation, as promised. Odd-offset
neighbors are cross-parity AND cross-V-sense. Two consequences:

* **Half-line geometry**: identical to NTSC (interlace, not standard):
  static content appears displaced h_k = (p_k - p_j)/2 field-lines;
  raw warps cannot compensate it (integer carrier copies), so a
  per-pixel validity envelope gates thin horizontal detail — the 9h
  gate is ported as `_halfline_gate` (one-sided-max baseband envelope,
  horizontal-only smoothing, floor 0.35).
* **Oscillating equation strength (PAL-only)**: with the neighbor's
  sense conjugated, dc = c - c_w = e^{i phi} + e^{-i phi} e^{-i Delta}
  on unswitched lines, so |dc| = |2 cos((2 phi + Delta)/2)| OSCILLATES
  with x between 0 and 2 (period 2 samples at 4fsc) instead of NTSC's
  constant sqrt2. Mean-square strength <|dc|^2> = 2 — identical
  information on average, differently distributed: some pixels inert,
  some at full strength. The arbitration handles this exactly (the
  per-pixel curvature Re[dC dc]^2 goes inert at the zeros, never
  corrupts), one more payoff of never assuming a carrier relation the
  arrays don't state.

Measured (bench triplet: static 3 IRE / pan 5 px/frame / 1-line thin
detail, Y/C dB):

| config        | static      | pan         | thin        |
|---------------|-------------|-------------|-------------|
| 2D            | 32.7 / 36.0 | 37.9 / 37.8 | 29.1 / 28.5 |
| 3D even-only  | 34.8 / 38.7 | 37.9 / 37.8 | 32.4 / 31.6 |
| 3D full       | 35.5 / 38.4 | 38.4 / 38.3 | 32.6 / 31.8 |

Odd offsets buy +0.7 dB static luma (cross-parity vertical coverage)
and +0.5/+0.5 on the pan, at -0.3 static chroma (the oscillating
equations admit a little cross-parity bias). Two ablations recorded:

* **The 9h gate measures NEUTRAL on PAL** — on the exact thin-detail
  content that broke NTSC (gate off: 32.6/31.7, on: 32.6/31.8, and 3D
  full is +3.5/+3.3 dB over 2D on it either way). Explanation: PAL's
  default full-weight InSAR coherence gate (1.0 vs NTSC's 0.6) is
  phase-sensitive and already rejects the cosine-zero wrong votes the
  envelope gate exists for. The gate is kept (cheap, defensive,
  floored) but its PAL necessity is unproven — see the NTSC feedback
  section.
* **Trajectory snap costs 0.2 dB on a clean pan** (h_k reinstatement
  rounds integer warps onto half-line vectors). Kept default-on for
  its purpose — noisy matching of long offsets — with the cost
  recorded.

Streaming: decode_sequence now processes bounded chunks
(`chunk_frames`, context `chunk_overlap`), verified BIT-EXACT across
chunk sizes for pass 1. Motion is estimated once per ordered field
pair on the pass-1 inits and shared by the equations, the anchor
blend, and drizzle (the NTSC fast-mode cache, default here).

## 5a. The re-encode anchor loop (passes >= 2, IMPLEMENTED)

The full decode -> MC temporal NR -> re-encode structure is ported:
`synth_reference` blends the DECODED fields over +/-nr_radius with
robust per-pixel weights and sub-pixel baseband warps (half-line
parity offset compensated exactly on the sampling grid), re-encodes
the blend through the effective carrier, and trusts it only where
|S - S_hat| is small; `variational_refine_joint` then re-solves over
BOTH (Y, chi) with soft data fidelity, the raw temporal equations,
and the anchored blend.

PAL blend geometry (differs from NTSC, recorded): the separation-
leakage anti-correlation that makes the blend powerful lives at the
180 deg carrier relation — which PAL reaches at f+/-4 (2 frames),
not f+/-1 like NTSC. Hence nr_radius defaults to 4 fields.

Measured (EBU bars, 3 IRE noise, run_tests 9):

| output            | Y (dB) | C (dB) |
|-------------------|--------|--------|
| 1-pass 3D         | 34.8   | 38.7   |
| 2-pass PURIST     | 32.4   | 42.7   |
| 2-pass soft       | 39.8   | 42.7   |

Chroma: +4.0 dB with STRICTLY no output NR (the NTSC result, on PAL).
The purist luma DROPPING to ~the raw noise floor is correct and worth
understanding: pass-1's higher Y figure was correlated noise hiding
inside chi (flattering Y PSNR while costing C); the anchor pass
cleans chi, so the lossless split hands that noise back to Y where it
honestly belongs. Purity re-verified on the 2-pass output to 1e-14.
Soft output (--soft-output) is the NR deliverable: +5.0 dB Y.

## 5d. Drizzle (vertical 2x, implemented — a debugging story worth keeping)

Ported with the NTSC boundary statement intact (reconstruction, not
decoding; outside the purity contract; `--drizzle`). Getting it to
WIN took four findings, all recorded because each is a latent NTSC
issue or a reusable rule:

1. **Integer vectors cannot feed drizzle.** The ported matcher was
   integer-pel; drizzle's premise is sub-line offsets. Parabolic
   half-pel refinement (`subpel_refine`) is the prerequisite, with a
   5-point re-scan first: the zero-motion margin rule (correct for
   raw equations) traps genuine sub-line motion one integer off and
   the parabola then converges in the wrong basin (measured: expected
   +0.75, got -0.29).
2. **A 9g-class deposit bug, derived before it bit**: sub-pixel
   vectors measure content displacement INCLUDING h_k, so the deposit
   is 2(y + v - h_k) + p_k; using the raw vector lands static
   cross-parity content a full frame-line off. (NTSC's formula
   assumes h_k-free vectors — see feedback section.)
3. **Plateau slander**: a true half-line displacement costs the same
   at both flanking integers; per-pair confidence honestly collapses
   for EXACTLY the neighbors whose deposits fill the uncovered
   residues (measured conf 0.13 on f+/-2 while f+/-4 was exact).
   Fixes: (a) per-tile VELOCITY fitted across all offsets — the
   integer-measuring offsets constrain it — hands each neighbor a
   predicted base o*v + h_k for the parabola to polish;
   (b) confidence from the PARABOLA-VERTEX cost, not the integer
   sample; (c) the robust agreement check compares at BILINEAR
   registration (a rounded compare is blind to exactly the half-line
   registrations drizzle exists to exploit).
4. **Degenerate dither rates exist**: a vertical pan of 2 fine
   rows/field parks every field on the same fine-row residue — no
   coverage, no SR, structurally. (First bench was accidentally
   degenerate; rebuilt at 1 fine row/field.)

Measured: static +1.8 dB over weave-upscale of the decoded frame
(noise averaging + parity phases); quarter-line continuous pan
+0.3 dB on a hard periodic+ledge torture. nr_radius=4 confirmed
(6 measured slightly worse on the pan).

## 5b. Gating sweep (static/pan bench pair, measured)

EBU bars, 4 frames; static at 3 IRE noise, pan at 5 px/frame true
interlace at 1.5 IRE. Y/C PSNR (dB):

| config                  | static      | pan         |
|-------------------------|-------------|-------------|
| 2D                      | 32.8 / 36.0 | 38.1 / 37.9 |
| nu=1.0, gate 0.6        | 35.9 / 36.8 | 37.2 / 36.5 |
| nu=1.0, gate 1.0        | 35.3 / 37.8 | 37.9 / 37.8 |
| nu=0.5, gate 0.6        | 35.2 / 38.1 | 38.0 / 38.1 |
| **nu=0.5, gate 1.0**    | 34.6 / 38.7 | 38.4 / 38.8 |

The chosen default (nu=0.5, full coherence weighting) is the only
configuration that is NEVER worse than 2D — it beats 2D on the pan
too (+0.3/+0.9 dB): with the phase-sensitive gate at full weight, the
equations that survive motion are the ones that genuinely correspond.
Stronger nu buys ~1 dB static luma at the price of pan losses; the
knob is exposed for archivists who know their material is static.
Auto eps_t at 4x vs 7x noise: within 0.3 dB everywhere (recorded,
7x kept for NTSC parity).

## 6. Comparison with existing PAL techniques

* **Delay-line (analogue, universal)**: subsumed — the quadratic limit
  of the vertical prior; implemented verbatim as the in-repo baseline
  (`decoder.delayline_baseline`, `--baseline`) so every claim is A/B
  measurable. Beaten by ~18 dB on the bars bench.
* **2D/3D adaptive combs**: as in NTSC, their switching heuristics are
  special cases of the IRLS weights; PAL versions additionally fight
  the 8-field bookkeeping that the effective carrier dissolves.
* **Transform PAL (BBC RD, ld-chroma-decoder transform2d/3d)**: the
  incumbent to beat on real captures. It thresholds spectral symmetry
  about the carrier — native to PAL (no NTSC blocker).
  `symmetry_variant` (the repaired certifier form from the NTSC work)
  is now PORTED AND MEASURED on the PAL bench: -0.04 dB chroma,
  i.e. neutral — the subsumption theorem-in-practice holds on PAL as
  on NTSC even with PAL's cleaner symmetry: the arbitration's
  residual-smoothness weighting already extracts the same evidence.
  Kept as a reference implementation, default off. Transform's known
  failure mode (threshold ringing on strong diagonal luma) remains
  structurally absent here: no threshold, only continuous
  diffusivities.
* **ld-chroma-decoder pal2d/palcolour**: line-locked software
  delay-line descendants; the baseline stands in for this class.

## 7. Test suite status (run_tests.py, 30/30 passing; sections
selectable: `python3 run_tests.py 6b 9`)

1. Dx/Dy adjoint identities to 7e-15 (CG gradients are true gradients).
2. Swinging-burst joint estimation: exact parity + 0.23 deg phase
   (clean), exact parity + 0.46 deg (25% burst dropout).
3. EBU bars + cross-colour bait, 0.8 IRE noise:
   delay-line 24.1/24.1 dB, holographic init 30.5/30.7 dB,
   holo+variational 42.8/42.6 dB (Y/C). Purity contract verified to
   1e-9 (delivered pair reconstructs S losslessly).
4. 10 deg differential phase: 2.4 deg median hue error, 97.4%
   saturation (cos-delta predicted 98.5%), 1.8 deg line-alternating
   hue — the PAL guarantee emerges from the prior, no delay-line.
5. 3D static (3 IRE noise, full offsets): +2.5 dB Y, +2.3 dB C.
6. 3D motion safety (5 px/frame pan, true interlace): +0.8 dB Y,
   +1.1 dB C over 2D — the gates make 3D a strict improvement.
6b. Thin 1-line detail (the NTSC 9h torture): 3D full +3.5/+3.3 dB
   over 2D — no regression with odd offsets enabled.
6c. Chunked streaming bit-exact across chunk sizes (pass 1).
6d. Drizzle beats weave-upscale by +1.8 dB (static).
7. Auto chroma-aniso in range (0.55 on bars, the floor, as NTSC);
   Transform-PAL certifier variant measured neutral (-0.04 dB).
8. PolSAR cross-pol map: 20x separation between a clean chain and
   10 deg differential phase.
9. Anchor loop (3 IRE, full offsets): purist chroma +4.3 dB with
   zero output NR, purity re-verified to 1e-14; soft output luma
   40.3 dB (+4.9 over 1-pass).

## 7b. Cross-domain ideation sweep (the PAL edition of NTSC's 9e)

A deliberate free-form pass over other fields, with the same honesty
rule as the NTSC brainstorm: map what is ALREADY a foundation, name
what was genuinely new, implement and measure the cheap ones, record
rejections with reasons.

Already this decoder's foundations (no action):

* **Phase-conjugate / phase-shifting holography** (optics) — IS the
  core identity (§1); PAL turned out to strengthen the holographic
  base, not strain it.
* **PLL / Kalman trajectory estimation** (communications) — the burst
  phase smoother (§2), inherited.
* **CDMA / DSSS spreading codes** (comms): the V-switch is a rigid
  +/-1 spreading sequence and "despreading by the measured code" is
  exactly the effective carrier — a satisfying second derivation of
  §1, no new machinery.
* **Friedel pairs** (X-ray crystallography): centrosymmetric spectral
  magnitude of a real signal = the physics under Transform PAL's
  symmetry test; measured neutral as an init variant (§6), i.e.
  subsumed by the arbitration on PAL exactly as on NTSC.

Genuinely new for PAL, implemented and measured this revision:

* **Polarimetric (Pauli) decomposition** (PolSAR) -> the cross-pol
  differential-phase diagnostic map (§4): 20x discrimination, plus
  the recorded negative result that decoded chroma cannot carry the
  evidence (the prior already ate it — measure on raw demod).
* **InSAR complex coherence** (radar interferometry) -> ported as the
  temporal chroma gate and PROMOTED on PAL: at full weight it is what
  makes 3D strictly better than 2D even on motion (§5b). On NTSC it
  was one gate among many; on PAL's sparser same-parity equation set
  it is the decisive one.

Examined and mapped, no implementation warranted:

* **Staggered-PRF Doppler radar** (§4): names the 25 Hz offset's
  purpose; the Dubois dual crops already arbitrate the resulting
  vertical alias split (init measures 30.5 dB — no evidence of a gap).
* **Two-frequency fringe unwrapping** (metrology, §4): rejected by
  sufficiency — trajectory smoothing + per-field re-lock covers
  realistic dropout spans on TBC'd material.

Deferred with NTSC precedent as the reason:

* **N-step PSI closed form** (optics): PAL's 135 deg ladder offers up
  to 7 carrier states — a richer PSI dataset than NTSC's — but NTSC
  measured the closed form at best identical (static) and -0.3 dB on
  motion vs the IRLS path. Re-measuring is only worth it after the
  odd offsets land (they contribute most of the extra states).
* **Trajectory-coherent motion** (NTSC 9e): needs >= 3 comparable
  offsets to form a consensus; with the current even-offset set (2
  per side) the consensus rule degenerates. Lands together with the
  odd offsets.
* **Astronomy drizzle**: PAL's 576 active lines alias vertically like
  NTSC's 480; the parity-phase geometry needs its own derivation.

## 7c. Findings flowing back to NTSC

The PAL port produced results that reflect on the NTSC decoder — the
promised unexpected ideas, ordered by likely value:

1. **Full-weight coherence may subsume the 9h envelope gate.** The
   gate measured NEUTRAL on PAL's thin-detail torture with
   coherence_gate = 1.0 (§5c); the InSAR gate is phase-sensitive and
   catches the cosine-zero wrong votes itself. NTSC defaults to 0.6 —
   re-sweep toward 1.0 (the PAL sweep, §5b, found 1.0 + nu 0.5
   strictly dominating) and re-bench the 9h content; the envelope gate
   may be partly redundant there too.
2. **The 9g deposit convention is LATENT in NTSC drizzle**: its
   formula assumes h_k-free vectors, valid for margin-snapped
   integers but a double-count the moment sub-pixel refinement
   captures the parity term (derived and fixed here, §5d.2). Audit
   recommended.
3. **Plateau slander + rounded-agreement blindness** (§5d.3): NTSC
   drizzle computes confidence at integer costs and agreement at
   rounded warps; both systematically suppress the half-line
   registrations drizzle needs. Port back vertex-cost confidence,
   bilinear agreement, and trajectory-velocity as the subpel BASE
   (NTSC's trajectory fit only snaps integers).
4. **Degenerate dither diagnostic** (§5d.4): certain vertical pan
   rates make vertical SR structurally impossible; a cheap coverage
   histogram of deposit residues would tell the operator in advance,
   in both decoders.
5. **Bench-interpretation caveat**: 1-pass luma PSNR is inflated by
   noise hiding in chi; the anchor pass exposes it (§5a). Any NTSC
   1-pass luma comparison shares the caveat.
6. **The effective-carrier abstraction itself**: NTSC is the c =
   e^{i phi} special case. Migrating the NTSC engine to the generic-c
   form would let both standards share one arbitration core (and one
   C++ port), with the standard reduced to a carrier builder + burst
   model.

## 7d. Port status: what reached the C++ engine

This package is the ORACLE for the C++ implementation in
`orc-plugin_hvd/src/engine/`, not a parallel product. Cross-reference,
so neither side drifts silently (the authoritative table, kept with the
code, is `orc-plugin_hvd/docs/PAL.md`):

Ported and live in the plugin: the effective carrier, the swinging-burst
joint phase+parity lock-in, temporal offsets to f+/-4, the PAL-swept
gating defaults, the same-parity ambiguity stride of 4, the AUTO
chroma_aniso with the PAL 4-line leak cancellation of section 3 (this
one shipped WRONG at first — the NTSC 2-line average was applied to PAL,
leaving the leak in the measurement; now guarded by a dedicated
regression test), the odd-offset half-line envelope gate,
trajectory-coherent motion, the anchor loop, the Transform-PAL
certifier variant, and the PSI closed form.

Partially ported: drizzle. The C++ deposits with integer, margin-snapped
vectors, so it uses the guaranteed half-line parity phase but not the
sub-pixel motion dither that section 5d added here. Note the trap
recorded in 5d.2: the deposit formula is coupled to the vector
convention, so porting sub-pixel refinement REQUIRES the h_k correction
at the same time.

Deliberately not ported: the delay-line baseline decoder (a research A/B
reference — the incumbent to measure against, not something to ship) and
the PolSAR cross-pol diagnostic map (an operator-facing analysis tool;
it must read the RAW per-line demodulation, since the vertical prior
arbitrates the line-alternating evidence away by design, so it is a
small standalone path rather than a solver change — worth adding for
archivists).

## 8. Roadmap (ordered)

1. Real-capture validation (ld-decode PAL .tbc): geometry, burst
   window, active-line metadata, colour convention vs ld-chroma-
   decoder output.
2. DONE: auto chroma-aniso (PAL 4-line leak cancellation), temporal
   gating sweep (§5b). Still open: full hyperparameter sweep on real
   material.
3. DONE: Transform-PAL certifier benched — neutral, subsumption holds.
4. DONE: full temporal equations f+/-1..4 with the PAL odd-offset
   derivation (§5c), trajectory-coherent motion, shared motion cache,
   chunked streaming (bit-exact), the NR anchor loop (§5a). Open:
   full motion-degeneracy table per offset (formulation-safe but
   undocumented), NTSC-9f speed rewrites for the Python path.
5. DONE: PolSAR cross-pol diagnostic map (`--polsar-map`).
6. DONE: drizzle (§5d), including the subpel stack it required.
7. Fast mode + C++ port notes: all NTSC 9f rewrites (integral-image
   blur, motion cache, tile-res confidences) apply verbatim — the
   blur is already the integral-image form here.
