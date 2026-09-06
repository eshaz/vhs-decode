"""The sync pulse's SHAPE as one component carried on all three axes.

Ethan, having had to say it twice: "the luma one about noise and it's
relationship to the sync pulse shape. the shape is a component with all
parts, amplitude, frequency, time".

And earlier, the half that says where the noise comes from: "The luma and
chroma are band limited, so the data outside this bandlimited area is
noise, that can use as the differential".

WHAT THIS CORRECTS. This lane has been treating the pulse shape as an
amplitude profile that happens to have a spectrum - fitting a relaxation
here, a ring there, each a scalar amplitude with a time constant. That is
one axis with the other two implied. The statement is that the shape is
ONE component present on amplitude, frequency and time simultaneously,
which is the same three-axis model `docs/THE_ALGORITHM.md` already states
for everything else, applied to the pulse itself.

THE BAND IS WHAT SEPARATES SHAPE FROM NOISE, and it is a FORMAT CONSTANT
rather than a fitted knee. The luma is band limited, so:

    what lies INSIDE the band is the shape
    what lies OUTSIDE it is noise, BY CONSTRUCTION

That is not a threshold anyone chooses and not a signal-to-noise judgment.
It follows from the format, and it makes the noise a MEASUREMENT rather
than a leftover - which is Ethan's "can use as the differential".

HOW THE THREE AXES COME OUT OF ONE FIT. The Laplace eigenbasis
(`hilbert_space_gp`) is the natural frame for this, because its modes are
indexed by frequency and truncating it at the format's band limit IS the
split above:

    FREQUENCY  the mode frequencies, fixed by the segment and the format,
               nothing fitted - mode j sits at j/(4L) cycles per sample
    AMPLITUDE  the coefficient on each mode, which is the shape's spectrum
    TIME       the group delay implied by that spectrum, through Bode's
               relation, plus where the shape's energy sits in the segment

They are three readings of ONE fitted object, not three fits. That is the
part the previous treatment lost: a relaxation fitted for its amplitude
and a ring fitted for its frequency were two objects describing one shape,
and they disagreed because nothing made them share a frame.

MEASURED, 2026-09-06: held out on independent lines, this basis truncated
at the 3.0 MHz VHS luma band reaches 1.4x the two-measurement floor on
both tapes and both heads, where the relaxation-plus-ring enumeration sits
at 5.4 to 12.4x. The held-out error falls monotonically to that cap and
flatlines past it, so the band constant is confirmed by the data rather
than imposed on it.
"""

from typing import Dict, Optional

import numpy as np

from vhsdecode.models import hilbert_space_gp as hsgp

__all__ = ["shape_components", "differential", "VHS_LUMA_BAND_HZ",
           "SVHS_LUMA_BAND_HZ", "UMATIC_LUMA_BAND_HZ", "BROADCAST_LUMA_BAND_HZ"]

VHS_LUMA_BAND_HZ = 3.0e6
SVHS_LUMA_BAND_HZ = 5.0e6
UMATIC_LUMA_BAND_HZ = 3.5e6
BROADCAST_LUMA_BAND_HZ = 4.2e6
"""The band limits that decide where shape ends and noise begins.

FORMAT CONSTANTS, and the whole point is that they are not fitted. The
broadcast figure is the channel's, not the signal's - BTS-3 states no
luminance bandwidth of its own, it states the I and Q channels (2.4.2)
and the chrominance sidebands (2.4.8), so 4.2 MHz belongs to the
transmission channel and is quoted as such.
"""


def shape_components(profile, sample_rate_hz: float, band_limit_hz: float,
                     boundary_factor: Optional[float] = None
                     ) -> Dict[str, object]:
    """The shape on all three axes, and the noise outside the band.

    `profile` is the accumulated pulse shape - one segment, already
    aligned, in whatever amplitude unit the caller works in (IRE here).

    Returns the three axes as readings of ONE fit, plus the out-of-band
    remainder as a measurement in its own right.
    """
    values = np.asarray(profile, dtype=np.float64).ravel()
    samples = len(values)
    if samples < 8:
        raise ValueError("a shape needs more than a handful of samples")
    # the FITTING factor, not the kernel-reconstruction one: at 52 samples
    # and 43 modes the latter conditions at 3.5e15 and rcond silently
    # discards three directions, for an identical held-out answer
    factor = (hsgp.FITTING_BOUNDARY_FACTOR if boundary_factor is None
              else float(boundary_factor))
    frame = hsgp.from_format(samples, sample_rate_hz, band_limit_hz, factor)
    positions = frame["positions"]
    basis = hsgp.eigenfunctions(positions, frame["modes"], frame["half_width"])

    # THE FIT IS PLAIN LEAST SQUARES, and that is a measured choice rather
    # than a simplification. A smoothness prior over these modes was tried
    # and earns nothing: the evidence drives its lengthscale to about 0.02
    # samples - a flat density, i.e. no prior - and plain least squares
    # scores as well or better on held-out lines. The BASIS does the work;
    # the band limit is the only regularisation, and it is a constant.
    centre = float(np.mean(values))
    coefficients, *_ = np.linalg.lstsq(basis, values - centre, rcond=None)
    in_band = basis @ coefficients + centre
    out_of_band = values - in_band

    frequencies = frame["mode_frequencies_hz"]
    amplitude = np.abs(coefficients)
    # the group delay Bode's relation implies for this magnitude. The band
    # is by construction NOT wide enough to contain the roll-off - it IS
    # the roll-off - so this is a within-band reading and the truncation
    # guard is left off deliberately rather than by oversight.
    delay = hsgp.group_delay_s(np.maximum(amplitude, 1e-12), frequencies)

    energy = amplitude ** 2
    total = float(energy.sum()) or 1.0
    centroid_hz = float(np.sum(frequencies * energy) / total)

    # THE POSITION COMES FROM THE DFT PHASE, and it has to. The
    # eigenfunctions are sines pinned at the boundary, so this basis has NO
    # SHIFT THEOREM - a translated shape is a different set of
    # coefficients, not the same set rotated - and any position read off
    # the coefficients moves unpredictably. Measured on a shape shifted
    # three, six and ten samples right, an amplitude centroid in this basis
    # moved LEFT, monotonically.
    #
    # The DFT does have the theorem: a shift of k samples is exactly a
    # phase ramp of -2 pi f k, so the slope of the unwrapped phase IS the
    # position, weighted by magnitude squared so the bins carrying the
    # shape decide it. The same rule already stands for the edge position
    # in the collapsed-model test, for the same reason.
    spectrum = np.fft.rfft(values - centre)
    bins = np.fft.rfftfreq(samples)
    phase = np.unwrap(np.angle(spectrum))
    weights = np.abs(spectrum) ** 2
    weights[0] = 0.0                     # DC carries no position
    if float(weights.sum()) > 0:
        mean_f = float(np.sum(weights * bins) / weights.sum())
        mean_p = float(np.sum(weights * phase) / weights.sum())
        spread = float(np.sum(weights * (bins - mean_f) ** 2))
        slope = (float(np.sum(weights * (bins - mean_f) * (phase - mean_p)))
                 / spread) if spread > 0 else 0.0
        position_samples = -slope / (2.0 * np.pi)
    else:
        position_samples = float("nan")

    return {
        # ---- the three axes, all read off the same fit ----
        "frequency_hz": frequencies,
        "amplitude": amplitude,
        "group_delay_s": delay,
        # ---- scalar summaries of each axis ----
        "amplitude_rms": float(np.sqrt(np.mean((in_band - centre) ** 2))),
        "centroid_hz": centroid_hz,
        "position_samples": position_samples,
        "mean_group_delay_s": float(np.nanmean(delay)),
        # ---- the split the format decides ----
        "in_band": in_band,
        "out_of_band": out_of_band,
        "noise_rms": float(np.sqrt(np.mean(out_of_band ** 2))),
        "band_limit_hz": float(band_limit_hz),
        "modes": int(frame["modes"]),
        "half_width": float(frame["half_width"]),
        "coefficients": coefficients,
        "level": centre,
        # the effective number of modes actually carrying the shape, which
        # is the honest count of its degrees of freedom rather than the
        # number the basis offers
        "effective_rank": float(total ** 2 / float(np.sum(energy ** 2)))
        if float(np.sum(energy ** 2)) > 0 else 0.0,
    }


def differential(measured: Dict[str, object],
                 reference: Dict[str, object]) -> Dict[str, object]:
    """Measured against expected, on all three axes at once.

    Ethan's process is a DIFFERENCE, never a ratio - spectral division is
    what manufactured phantoms in this tree twice - so every axis here is
    subtractive.

    The out-of-band term is carried through as a difference too, which is
    the second half of his statement: the noise is not discarded, it is
    itself a channel of the differential. Two shapes whose in-band parts
    agree and whose out-of-band parts do not have a real difference, and
    it is one a band-limited comparison would report as none.
    """
    for name, side in (("measured", measured), ("reference", reference)):
        if int(side["modes"]) != int(measured["modes"]):
            raise ValueError(
                "both sides must be measured on the same basis - %s carries "
                "%d modes against %d, and a comparison is only a comparison "
                "when both sides are computed over the same set"
                % (name, int(side["modes"]), int(measured["modes"])))
    return {
        "frequency_hz": np.asarray(measured["frequency_hz"]),
        "amplitude": np.asarray(measured["amplitude"])
        - np.asarray(reference["amplitude"]),
        "group_delay_s": np.asarray(measured["group_delay_s"])
        - np.asarray(reference["group_delay_s"]),
        "time_s": float(measured["mean_group_delay_s"])
        - float(reference["mean_group_delay_s"]),
        "position_samples": float(measured["position_samples"])
        - float(reference["position_samples"]),
        "noise_rms": float(measured["noise_rms"]) - float(reference["noise_rms"]),
        "out_of_band": np.asarray(measured["out_of_band"])
        - np.asarray(reference["out_of_band"]),
        "residual": np.asarray(measured["in_band"])
        - np.asarray(reference["in_band"]),
    }
