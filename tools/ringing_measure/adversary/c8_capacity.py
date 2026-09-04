"""Claim 8: the capacity claims, Carson's rule, and whether an envelope
spread may be converted to a carrier-to-noise ratio the way binding_limit
does."""
import sys
import numpy as np
sys.path.insert(0, "/workspaces/vhs-decode")
from vhsdecode.models import capture_profile as cp

SPREAD_DB = 0.697

print("=== 8a. Reproduce the published numbers exactly ===")
# the profile the tool printed: 8 bit, step 256 in an int16 container,
# 50 MHz, 117 codes over 45.7% of full scale, 400 Mbit/s budget
profile = {"bits": 8.0, "sample_rate_hz": 50e6, "full_scale": 256.0 * 256.0}
# signal_rms_codes solved from the published 46.7 dB capture C/N
floor = cp.quantization_floor(8.0, 50e6, 65536.0)
band = cp.carson_bandwidth(cp.NTSC_VHS_SP_DEVIATION_HZ, cp.NTSC_VHS_SP_BASEBAND_HZ)
q = cp.in_band_noise_rms(floor, band)
srms = np.sqrt(10 ** (46.7 / 10)) * q / floor["step"]
out = cp.binding_limit(SPREAD_DB, srms, profile)
for k, v in out.items():
    print(f"  {k:28s} {v}")
print(f"  ratio {out['ratio']:.3f}  (published 2.2x)")

print("\n=== 8b. Is 1/(10^(dB/20)-1)^2 a carrier-to-noise ratio? ===")
print("""  For a carrier A cos(wt) plus bandpass noise n = n_I cos - n_Q sin,
  the envelope is sqrt((A+n_I)^2 + n_Q^2) ~= A + n_I, so
        var(envelope) = var(n_I) = sigma^2,
  while the bandpass noise POWER is E[n^2] = sigma^2 and the carrier
  power is A^2/2.  Hence

        C/N = (A^2/2)/sigma^2 = 1 / (2 * relative^2),

  where relative = sigma/A is what the envelope spread measures.
  capture_profile.binding_limit line 167 computes

        tape_snr = 1.0 / max(relative, 1e-30) ** 2

  which is 2 * C/N -- exactly 3.01 dB too high.""")
rel = 10 ** (SPREAD_DB / 20.0) - 1.0
print(f"  relative (fractional envelope sd) = {rel:.6f}")
print(f"  code's tape_snr  = {10*np.log10(1/rel**2):.2f} dB   (published 21.6)")
print(f"  true C/N         = {10*np.log10(1/(2*rel**2)):.2f} dB")

print("\n  And the CAPTURE side does apply the factor, on line 133:")
print("     in_band_noise_rms(...) / sqrt(2)   'the amplitude quadrature alone'")
print("  so capture_snr = (A/sqrt2)^2 / sigma_q^2 = (A^2/2)/sigma_q^2 = C/N,")
print("  correctly.  The two links being compared therefore use DIFFERENT")
print("  conventions, and the tape is the one flattered.")
corrected = cp.binding_limit(SPREAD_DB, srms, profile)
tape_true = 1.0 / (2 * rel ** 2)
cap_true = 10 ** (46.7 / 10)
print(f"  corrected tape capacity  {cp.channel_capacity(band, tape_true)/1e6:.1f} "
      f"Mbit/s  (published 57.4)")
print(f"  capture capacity         {cp.channel_capacity(band, cap_true)/1e6:.1f} "
      f"Mbit/s  (published 124.1)")
print(f"  corrected ratio          "
      f"{cp.channel_capacity(band, cap_true)/cp.channel_capacity(band, tape_true):.2f}x"
      f"   (published 2.2x)")

print("\n=== 8c. Carson's rule: is the deviation 1.0 MHz? ===")
print("""  capture_profile lines 44-49:
     'the carrier runs 3.4 MHz at sync tip to 4.4 MHz at peak white, so the
      deviation is 1.0 MHz'
  Carson's rule B = 2(dF + fm) takes dF as the PEAK deviation from the
  UNMODULATED carrier, not the peak-to-peak swing.  A carrier swinging
  3.4 -> 4.4 MHz has a centre of 3.9 MHz and a peak deviation of
  0.5 MHz.""")
for dev, lab in ((1.0e6, "as coded (peak-to-peak swing)"),
                 (0.5e6, "peak deviation (correct)")):
    b = cp.carson_bandwidth(dev, cp.NTSC_VHS_SP_BASEBAND_HZ)
    print(f"  {lab:34s} B = {b/1e6:.1f} MHz   tape "
          f"{cp.channel_capacity(b, 1/rel**2)/1e6:6.1f}  capture "
          f"{cp.channel_capacity(b, cap_true)/1e6:6.1f} Mbit/s")
print("  The RATIO is unchanged (both capacities are linear in B), so the")
print("  '2.2x' survives; the absolute Mbit/s figures are 14% too large.")

print("\n=== 8d. What the 0.697 dB actually contains ===")
print("""  elliptical_collapse._spread computes, per carrier-frequency bin,
        std(log(flat))   with flat = envelope flattened by the fitted
                          response curve
  and averages over bins.  That scatter contains, indistinguishably:
    * additive channel noise                       (what is wanted)
    * response mismatch WITHIN a bin (the curve is a straight line in
      ln-level against frequency: `response_curve` returns centre, level,
      SLOPE -- so all curvature lands in the spread)
    * genuine AM on the carrier from band-limiting an FM signal
    * dropouts, head-switch, mistracking, tape-position variation
    * the sweep-rate dependence this arc measures as its own AXIS -- the
      amplitude axis exists precisely because the envelope depends on how
      fast the carrier is moving, and _spread bins only by frequency
  So 0.697 dB is an UPPER bound on the channel noise, hence 21.6 dB is a
  LOWER bound on C/N, and 'the tape binds by 2.2x' is the most favourable
  reading of the evidence rather than a measurement of it.""")

print("\n=== 8e. The 7.1% correction in 7.6 ===")
print(f"  21.6 dB + 10*log10(1/0.071) = "
      f"{10*np.log10(1/rel**2) + 10*np.log10(1/0.071):.1f} dB  (published 33.1)")
print("""  ARITHMETIC CONFIRMED, PREMISE REFUTED.  The 0.071 is
  1 - resolved_fraction from the FREQUENCY-axis ellipse.  Two objections:

  (1) Its pure-noise baseline in that same construction is 0.011, not 0
      (measured: 98.9% +- 0.7% resolved on 13 noise fields over 119
      places).  A structureless ensemble would license 41.6 dB by the
      same step.  The 7.1% is therefore not a noise fraction; it is
      LARGER than what noise alone produces, i.e. the real data is LESS
      structured than noise by this measure.

  (2) The two variances are not the same variance.  0.697 dB is the
      WITHIN-FIELD, WITHIN-BIN scatter of individual envelope SAMPLES.
      The ellipse is fitted to BETWEEN-FIELD differences of BINNED
      MEDIANS, each median already formed from >= MINIMUM_POPULATION
      samples.  Scaling the first by a fraction derived from the second
      is a category error: the binning has already averaged away most of
      the sample-level scatter the 0.697 dB measures.""")
r = 10 ** (46.7 / 10)
for share, lab in ((1.0, "raw"), (0.071, "published 7.1%"),
                   (0.011, "pure-noise baseline 1.1%")):
    s = (1 / rel ** 2) / share
    print(f"  {lab:26s} C/N {10*np.log10(s):5.1f} dB  tape capacity "
          f"{cp.channel_capacity(band, s)/1e6:6.1f} Mbit/s  ratio "
          f"{cp.channel_capacity(band, r)/cp.channel_capacity(band, s):.2f}x")

print("\n=== 8f. Shannon's own premise ===")
print("""  Shannon capacity is the maximum rate over a channel whose noise is
  ADDITIVE, WHITE and GAUSSIAN, for an input chosen to maximise mutual
  information.  Neither link here satisfies that:
    * the tape's dominant impairments (dropouts, mistracking, modulation
      noise) are neither additive nor stationary, and the doc's own 7.7
      says four of five are structured;
    * the input is not free -- it is a specified FM carrier, so the
      realisable rate is far below B log2(1+SNR).
  The comparison 'converter holds 2.2x the tape' is therefore a
  comparison of two upper bounds computed under a premise neither link
  meets.  It is a legitimate ORDERING argument (both bounds use the same
  B and the same formula) but not a statement that 67 Mbit/s is 'spent on
  nothing' -- no scheme could reach either bound.""")
