# hvd-pal — Holographic-Variational PAL decoder (research phase)

Sibling of the NTSC hvd decoder, in a deliberately SEPARATE
architecture folder. Same base idea — each field is an off-axis
hologram and Y/C separation is solved as a regularized inverse
problem, never a comb filter — but PAL's structure earns its own
mathematics rather than flags on the NTSC code.

## The one-line port

PAL's V-switch is phase-conjugate holography. Folding it into an
EFFECTIVE CARRIER of unit modulus,

    S = Y + Re[chi * c],   chi = V - iU
    c = +exp(+i*phi) on unswitched lines, -exp(-i*phi) on switched,

makes every equation of the NTSC decoder generic in c: NTSC is the
special case c = e^{i*phi}. The variational arbitration then
suppresses Hanover bars *structurally* (they are the highest vertical
frequency chi can carry) and reproduces the delay-line's hue-to-
saturation guarantee as a property of the vertical prior — measured:
10 deg differential phase -> 2.3 deg hue error, cos(delta)
desaturation, no bars.

## Status

2D core + same-parity 3D working on synthetic ground truth
(encoder included). Bars bench (Y / C dB):

    delay-line baseline      24.1 / 24.1
    holographic init         30.5 / 30.7
    holographic+variational  42.8 / 42.6

3D (`--3d`: raw-measurement equations from ALL fields f+/-1..4 —
odd offsets under the PAL-derived oscillating-|dc| geometry and the
half-line gate — motion-compensated with trajectory-coherent vectors,
robust + InSAR-coherence gated): +2.5/+2.3 dB over 2D static,
+0.8/+1.1 dB on a 5 px/frame pan, +3.5/+3.3 dB on 1-line thin detail
— a strict improvement on every bench (THEORY 5b/5c). Streaming is
chunked and bit-exact; motion is estimated once per field pair and
shared by equations, anchor blend, and drizzle.

Anchor loop (`--passes 2`: decode -> motion-compensated temporal NR
-> re-encode honesty check -> joint re-solve): chroma +4.0 dB at
3 IRE noise with STRICTLY no output NR (default purist output still
splits S losslessly, verified to 1e-14); `--soft-output` delivers the
denoised pair instead (+5.0 dB luma).

Swinging-burst JOINT phase+parity lock-in survives 25% burst
destruction (exact parity, 0.46 deg phase). Purity contract holds:
the delivered (Y, chi) reconstructs the composite losslessly.
Transform-PAL certifier: measured neutral (subsumed, as on NTSC).
`--polsar-map`: polarimetric cross-pol map of the source chain's
differential-phase distortion (20x discrimination).
`--drizzle`: vertical 2x super-resolution, +1.8 dB over weave-upscale
on static material (reconstruction, not decoding — outside the purity
contract). THEORY 7c records six findings flowing back to the NTSC
decoder, including two latent NTSC bugs found via the PAL derivations.

## Usage

    python3 run_tests.py                      # full self-test bench

    python3 hvd_pal_decode.py input.tbc -s 0 -l 10 -o frames/
    python3 hvd_pal_decode.py input.tbc --pipe | ffmpeg -f rawvideo \
        -pix_fmt rgb48le -s 922x582 -r 25 -i - out.mkv
    python3 hvd_pal_decode.py input.tbc --baseline ...   # delay-line A/B
    python3 hvd_pal_decode.py input.tbc --3d ...         # temporal equations
    python3 hvd_pal_decode.py input.tbc --passes 2 ...   # NR anchor loop
    python3 hvd_pal_decode.py input.tbc --drizzle ...    # vertical 2x SR
    python3 hvd_pal_decode.py input.tbc --polsar-map diag.png

Synthetic material without hardware:

    from hvd_pal.encode import make_test_pattern, write_tbc
    from hvd_pal.tbc import VideoParameters
    p = VideoParameters()
    write_tbc("test.tbc", [make_test_pattern(700, 560)], p,
              noise_ire=0.8, diff_phase_deg=10.0)

See THEORY-PAL.md for the derivations, the cross-domain survey
(phase-conjugate holography, polarimetric decomposition, staggered-PRF
spectral interleave), the 8-field temporal geometry table, and the
comparison against delay-line / adaptive combs / Transform PAL.
