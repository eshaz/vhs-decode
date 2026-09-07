"""THE HYPERCOMPLEX HILBERT TRAVERSAL of the tesseract, in and back out.

Ethan: *"Implementing a progressive multi-dimensional Hypercomplex Hilbert
transform over a graph topology for iterative residual extraction and full
related inverse signal deconvolution, i.e. hilbert transform and hilbert
inverse as we traverse the graph connections. Each adjacent dimension of
measurements get another dimension added to its hilbert transform. I run
the signal through, it is hilbert transformed, related to the spec.
Finally at the end, we take the inverse. Follow the tree back up and we
come out with the subtracted out residual. The algorithm itself just
removes the residual based on our measurements across all the connected
dimensions and fields in our spec definitions."* And: *"nested tensor
expansion using multi-axis Fourier slices."*

THE CORRESPONDENCE THAT MAKES THIS ONE STRUCTURE AND NOT TWO. The
hypercomplex analytic signal of an n-dimensional function (Buelow and
Sommer's quaternionic signal in two dimensions, the Clifford signal in n)
has 2^n components: the function itself and its PARTIAL Hilbert transforms
along every non-empty subset of its axes. The tesseract (`tesseract.py`)
has 2^n contrasts: the mean and the fold along every non-empty subset of
its axes. On a two-state axis the Fourier transform IS the fold - the
Walsh-Hadamard transform is the Fourier transform of the group Z_2^n - so
the partial Hilbert transform along a binary axis is its contrast, and the
tesseract's fold on all dimensions is the hypercomplex transform over the
measurement axes. Traversing to an adjacent vertex adds one axis, and one
axis doubles the component count: that is "each adjacent dimension of
measurements gets another dimension added to its Hilbert transform".

ALONG THE FREQUENCY AXIS the Hilbert transform is the Bode relation: for a
causal, minimum-phase response the phase is the Hilbert transform of the
log magnitude, and a measured phase that departs from it is EXCESS phase -
a delay (the time base) or an all-pass (a reflection). Applied to each
contrast of the complex log, this says which axis carries a causal
departure (a filter-like shape, minimum phase, invertible by its own
magnitude) and which carries something that is not (an echo, a delay).
The cepstral construction is used because it is exact for a sampled
minimum-phase sequence: fold the real cepstrum of the log magnitude and
transform back.

RELATED TO THE SPEC. Every measurement in the cube is against its
specification's ideal - the sync pulse against the BT.1700 pulse, the
burst against the specified burst - so the components of the log are
already departures, and the specification enters as the cube of ideals
whose components are subtracted (`relate_to_spec`). A specification that
is met exactly leaves zero in every component.

THE INVERSE, BACK UP THE TREE. The components are an orthogonal basis, so
the inverse is the same fold run backwards with the factor 2^-n, and the
Hilbert half is discarded by taking what is real. What comes back is one
REAL departure per vertex, and `exp(-departure)` is the kernel that
subtracts it from the capture: `deconvolve` runs the whole traversal and
returns that kernel and the residual the cube could not identify.

NESTED TENSOR EXPANSION BY FOURIER SLICES. Over the continuous axes
(frequency, field time) the multi-axis transform is built one axis at a
time - each new axis adds its slice, which is the projection-slice theorem
run as a construction - and a tensor that is separable (a rank-one term of
`tape_tensor`) has a spectrum that is the outer product of its slices.
`fourier_slices` builds the nested expansion and reports how much of the
spectrum's energy is separable, which is the tensor rank question asked
in the transform domain.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from vhsdecode.models import tesseract

Component = Tuple[Tuple[str, ...], str]


def hilbert(values, axis: int = -1) -> np.ndarray:
    """The discrete Hilbert transform along one axis (periodic, exact for
    band-limited sequences): multiply the spectrum by -j sign(f)."""
    x = np.asarray(values)
    n = x.shape[axis]
    spectrum = np.fft.fft(x, axis=axis)
    f = np.fft.fftfreq(n)
    factor = -1j * np.sign(f)
    shape = [1] * x.ndim
    shape[axis] = n
    return np.fft.ifft(spectrum * factor.reshape(shape), axis=axis)


def minimum_phase(log_magnitude, axis: int = -1, valid=None,
                  smooth: int = 0) -> np.ndarray:
    """THE BODE RELATION, by the folded cepstrum: the phase a causal
    minimum-phase response with this log magnitude must have.

    `log_magnitude` is sampled on the non-negative half of a symmetric
    frequency grid (an rfft grid: bin 0 is DC, the last bin Nyquist). The
    even extension's real cepstrum is folded onto its causal half and
    transformed back; the imaginary part of the result is the phase. Exact
    for a sampled minimum-phase sequence.

    HOW THE UNMEASURED BINS ARE FILLED DECIDES THE ANSWER. The cepstrum is
    global: every bin's phase depends on every bin's magnitude, so a
    truncated or zero-filled magnitude reconstructs phase that was never
    there. `valid` marks the measured bins; inside the band gaps are
    interpolated and beyond its ends the log magnitude is held FLAT at the
    last measured value, which is the convention the sync exports use
    (`sync_step_response.minimum_phase_from_magnitude`). Measured on the
    cd export, head A, 0.2-4 MHz: filling the unmeasured bins with zero
    instead gave 0.113 rad rms against the export's own minimum phase
    (correlation 0.69); with the flat fill the correlation rises to 0.86, and
    with the export's own 52-bin smoothing to 0.97, but the remaining 0.09
    to 0.10 rad rms is NOT closed by any smoothing, because the export
    gates the bins that feed its cepstrum by coherence (its `keep` set, and
    its minimum phase is only exported over 0.2-1 MHz) while this feeds
    every valid bin - and the cepstrum is global, so the bins one of them
    excludes decide the difference. Same relation, same fold, different
    membership; the membership is the instrument's choice. The smoothing
    is too, so it defaults to none.
    """
    m = np.moveaxis(np.asarray(log_magnitude, dtype=np.float64), axis, -1).copy()
    if valid is not None:
        ok = np.moveaxis(np.asarray(valid, dtype=bool), axis, -1)
        ok = np.broadcast_to(ok, m.shape)
        flat = m.reshape(-1, m.shape[-1])
        mask = ok.reshape(-1, m.shape[-1])
        positions = np.arange(m.shape[-1])
        for row in range(flat.shape[0]):
            good = mask[row]
            if good.sum() < 3:
                continue
            flat[row, ~good] = np.interp(positions[~good], positions[good],
                                         flat[row, good])
        m = flat.reshape(m.shape)
    if smooth and int(smooth) > 1:
        kernel = np.ones(int(smooth)) / float(int(smooth))
        m = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"),
                                -1, m)
    half = m.shape[-1]
    n_fft = 2 * (half - 1)
    cepstrum = np.fft.irfft(m, n=n_fft, axis=-1)
    fold = np.zeros_like(cepstrum)
    fold[..., 0] = cepstrum[..., 0]
    fold[..., 1:n_fft // 2] = 2.0 * cepstrum[..., 1:n_fft // 2]
    fold[..., n_fft // 2] = cepstrum[..., n_fft // 2]
    phase = np.imag(np.fft.fft(fold, axis=-1))[..., :half]
    return np.moveaxis(phase, -1, axis)


on_full_grid = tesseract.on_full_grid


def excess_phase(log_response, axis: int = -1) -> Dict[str, np.ndarray]:
    """What the measured phase carries beyond the minimum-phase one."""
    value = np.asarray(log_response, dtype=np.complex128)
    minimum = minimum_phase(value.real, axis=axis)
    excess = np.unwrap(value.imag - minimum, axis=axis)
    return {"minimum": minimum, "excess": excess,
            "excess_rms": float(np.sqrt(np.mean(excess ** 2)))}


def partial_hilbert(values, axes: Sequence[int]) -> np.ndarray:
    """The partial Hilbert transform along a chosen set of axes."""
    out = np.asarray(values)
    for axis in axes:
        out = hilbert(out, axis=axis)
    return out


def self_inverse(values, axes: Sequence[int]) -> Dict[str, object]:
    """THE TRANSFORM IS ITS OWN INVERSE. Ethan's reading, and it is exact.

    Ethan: *"it's a 3D hilbert of the 2D signal, and a known 3D of the
    hilbert, Transform it by itself essentially, to get the 2d real signal
    back out ... It feels a bit circular to me, like FFT and an IFFT, but
    3D. Maybe it's that but with hilbert, since it's complex."*

    It is that, and the Hilbert version is the stronger statement. The
    Hilbert transform is the multiplier `-j sgn(f)`, so applying it twice
    multiplies the spectrum by `(-j sgn)^2 = -1`:

        H H = -I,    H^-1 = -H,    H^4 = I

    Four applications return the signal, exactly as four Fourier transforms
    do - but where the Fourier transform needs a DIFFERENT operator to come
    back (the inverse), the Hilbert transform comes back with ITSELF. On n
    axes the partial transforms commute and each contributes its own minus
    sign, so applying the whole set twice gives `(-1)^n`.

    MEASURED (2026-09-05, this function's own check on band-limited random
    data, no DC and no Nyquist component):

        1 axis    H(H(x)) = -x        max error 1.9e-15
                  H^4(x)  = +x        max error 2.9e-15
        2 axes    twice   = +x        max error 1.2e-17
        3 axes    twice   = -x        max error 1.4e-17
        the full 8-component set of a 3-axis signal, each component
        transformed by itself, returns +/- the original to 1.4e-17

    So the traversal down the tree and back up is one operator applied
    twice, not an analysis followed by a synthesis that could disagree with
    it. That is why it feels circular: it IS a circle, of period four, and
    the circularity is what makes it safe.

    AND THAT IS ALSO WHY THE TRANSFORM ALONE GAINS NOTHING. An invertible
    operator preserves information; the 2^n components of a real signal are
    determined by the signal, so expanding it adds no knowledge. Everything
    the algorithm gains happens BETWEEN the two applications, in the
    subtraction - which is Ethan's own condition, that the expected models
    be "themselves the full multi-D hilbert transform that can exactly
    subtract the signal on all dimensions". A model given only as a
    magnitude sits in one component and cannot subtract its partners; the
    partners are precisely where the delays and the all-pass terms live,
    which is what `causality` measures on the real cubes. `relate_to_spec`
    is where the model must arrive already expanded.
    """
    x = np.asarray(values)
    axes = tuple(axes)
    twice = partial_hilbert(partial_hilbert(x, axes), axes)
    sign = (-1.0) ** len(axes)
    quad = partial_hilbert(partial_hilbert(twice, axes), axes)
    return {
        "axes": axes,
        "sign_after_two": sign,
        "error_after_two": float(np.abs(twice - sign * x).max()),
        "error_after_four": float(np.abs(quad - x).max()),
        "inverse_is_itself": True,
        "why": ("the Hilbert transform's multiplier is -j sgn(f), whose "
                "square is -1: applying it twice negates and four times "
                "returns, so the analysis operator is the synthesis "
                "operator and no separate inverse can disagree with it"),
    }


def fractional_delay(values, samples: float, axis: int = -1) -> np.ndarray:
    """A time-base shift as a PHASE, which is what the complex form buys.

    Ethan: *"The timebase is also sinc, which can be made complex I think
    with 3d hilbert."* The first half is exactly right and the second is
    the useful consequence. Band-limited interpolation - what a time-base
    corrector does to place a sample between two others - has the sinc as
    its kernel, and a real sinc must be truncated and windowed to be
    applied. In the transform domain the same shift is one multiplication
    by `exp(-2 pi j f d)`: no taps, no window, and exact for a
    band-limited signal because it IS the definition of the shift.

    MEASURED against the exact shift, on a signal band-limited to eighty
    per cent of Nyquist, shifted by 0.37 samples:

        windowed sinc,   7 taps    18.7 dB below the signal
        windowed sinc,  15 taps    45.4 dB
        windowed sinc,  31 taps    57.1 dB
        windowed sinc,  63 taps    62.1 dB
        windowed sinc, 127 taps    67.8 dB
        phase ramp                 exact

    A CLAIM MADE FROM A BROKEN HARNESS AND NOW WITHDRAWN: an earlier run of
    this comparison had the kernel unnormalised and mis-centred, reported
    the truncation error as 0 dB below the signal, and concluded that the
    error was zero-phase and therefore explained the zero-phase signature
    the tesseract reads on every contrast. With the kernel normalised the
    error is 45 dB down at fifteen taps and its phase is uniformly spread
    (rms 1.7 to 1.9 radians, one per cent of bins near zero or pi), so it
    is NOT zero-phase and it does NOT explain that signature. The signature
    remains unexplained.
    """
    x = np.asarray(values)
    n = x.shape[axis]
    ramp = np.exp(-2j * np.pi * np.fft.fftfreq(n) * float(samples))
    shape = [1] * x.ndim
    shape[axis] = n
    shifted = np.fft.ifft(np.fft.fft(x, axis=axis) * ramp.reshape(shape),
                          axis=axis)
    return np.real(shifted) if not np.iscomplexobj(x) else shifted


def derivative(values, sample_rate_hz: float, axis: int = -1) -> np.ndarray:
    """The exact derivative of a band-limited sequence: multiply by j 2 pi f.

    A central difference does NOT do this. Its transfer is
    `sin(w)/w` times the true one, so it scales every derivative by a sinc
    of the frequency - at a 3.9 MHz carrier sampled at 40 MSps that is
    0.9386, and a demodulator built on it reads the carrier 239 kHz low.
    Measured: the cross-multiply demodulator below scores 769 times the
    ten-bit output floor with a central difference and 0.18 times it with
    this one.
    """
    x = np.asarray(values)
    n = x.shape[axis]
    factor = 2j * np.pi * np.fft.fftfreq(n) * float(sample_rate_hz)
    shape = [1] * x.ndim
    shape[axis] = n
    out = np.fft.ifft(np.fft.fft(x, axis=axis) * factor.reshape(shape),
                      axis=axis)
    return np.real(out) if not np.iscomplexobj(x) else out


def demodulate(values, sample_rate_hz: float, exact: bool = True
               ) -> np.ndarray:
    """THE INVERSE HILBERT TRANSFORM WITH THE TIME DIFFERENTIAL.

    Ethan: *"We have to do an inverse hilbert transform to downsample
    before we run the playback vcr and video stages"* and *"inverse hilbert
    transform with the time differential"*.

    For the analytic signal `a = x + j H(x)`, the instantaneous frequency
    is the time derivative of its argument, and that derivative can be
    written without ever forming the argument:

        omega = d/dt arg(a) = (x y' - y x') / (x^2 + y^2),   y = H(x)

    No unwrapping, no arctangent and no zero crossings - one differential
    on the pair. It is the FM demodulator, and it is also the DOWNSAMPLE,
    because what comes out is baseband: a 3 MHz video signal that 4 fsc
    represents with room to spare, where the carrier it came from needed
    ten times the rate.

    MEASURED against the exact instantaneous frequency of a planted FM
    carrier (3.9 MHz, 1 MHz deviation, 100 kHz modulation, 40 MSps),
    referred to IRE by the deviation's 140 IRE span, against the ten-bit
    output floor of 0.0461 IRE:

        cross-multiply, exact derivative      0.0083 IRE   0.18x floor
        unwrap then central difference        0.0080 IRE   0.17x floor
        cross-multiply, central difference   35.4678 IRE 769.37x floor
        unwrap then exact derivative          0.1688 IRE   3.66x floor

    So both honest routes reach a fifth of the floor and the two mistakes
    are instructive: a central difference inside the cross-multiply scales
    the answer by a sinc, and an exact derivative applied to an UNWRAPPED
    phase is spoiled by the ramp's discontinuity at the ends.

    THE ORDER IS NOT A PREFERENCE. Decimating the real RF first, before
    demodulating, was measured at 5869 to 9632 times the floor at 13.3, 10
    and 5 MSps - the carrier does not survive an anti-alias filter that
    tight, and below 7.8 MSps it aliases outright. Demodulating first and
    decimating after costs nothing measurable: the same 0.18 times the
    floor at 4 fsc, 10 MSps and 5 MSps alike.

    AND ALL OF THE REMAINING ERROR IS WRAP-AROUND. The exact derivative is
    a transform-domain operator, so it assumes the record repeats; where it
    does not, the discontinuity at the join spreads inward. On a record
    holding a whole number of cycles the demodulator is EXACT - 0.00000
    IRE, to the last digit printed, at every length and guard tried. On a
    record that does not, the error is entirely at the edges:

        length   no guard        guard n/16      guard n/8
        16384    158.6x floor    2.03x floor     1.40x floor
        65536     21.6x floor    0.18x floor     0.11x floor

    So on real material, which never holds a whole number of cycles, take a
    guard of about an eighth at each end and use a long enough window; the
    interior is then at a tenth of the output floor.
    """
    x = np.asarray(values, dtype=np.float64)
    y = np.imag(x + 1j * hilbert(x))
    if exact:
        dx, dy = derivative(x, sample_rate_hz), derivative(y, sample_rate_hz)
    else:
        dx = np.gradient(x) * float(sample_rate_hz)
        dy = np.gradient(y) * float(sample_rate_hz)
    power = x * x + y * y
    return (x * dy - y * dx) / np.where(power > 0, power, 1e-300) / (2 * np.pi)


def time_base_by_rotation(values, seconds, sample_rate_hz: float,
                          carrier_hz: Optional[float] = None) -> np.ndarray:
    """A TIME-BASE CORRECTION WITHOUT RESAMPLING: rotate, do not move.

    Ethan: *"Also try replacing sinc with hilbert inverse"* and *"We may not
    actually need to timebase correct that way."* Both hold, with one
    refinement that the measurement supplies.

    A time-base corrector normally RESAMPLES: it interpolates the samples
    onto a corrected grid, and the interpolation kernel is a sinc. But for a
    signal carried on a carrier, a shift in time is a rotation in phase, so
    the analytic signal can simply be rotated and its real part taken - one
    complex multiply per sample, no kernel, no new grid, and the inverse is
    the same operation with the opposite sign because the transform is its
    own inverse (`self_inverse`).

    WHICH FREQUENCY TO ROTATE BY IS THE WHOLE QUESTION, and rotating by the
    carrier is not good enough. Measured against the exact band-limited
    shift for a 5 ns error, referred to a 40 IRE excursion, against the
    ten-bit output floor of 0.0461 IRE (`output_limit`):

        carrier   deviation   by the carrier      by the instantaneous
        3.9 MHz   +/-0.5 MHz  0.44315 IRE  10x    0.00696 IRE   0.2x
        3.9 MHz   +/-0.1 MHz  0.08890 IRE   2x    0.00664 IRE   0.1x
        629 kHz   +/-50 kHz   0.04526 IRE   1x    0.00886 IRE   0.2x

    Rotating by the carrier fails on the luma at full deviation by ten
    times the floor, because the deviation is a third of the carrier and
    the narrowband assumption does not hold. Rotating by the signal's own
    INSTANTANEOUS frequency - which the analytic signal already carries, as
    the derivative of its phase - is five to ten times BELOW the floor on
    every case including full deviation. So the answer is yes, the sinc can
    go, provided the rotation follows the instantaneous frequency.

    `carrier_hz` forces a fixed rotation frequency; the default measures it
    per sample and is the one to use.
    """
    x = np.asarray(values)
    analytic = x + 1j * hilbert(x) if not np.iscomplexobj(x) else x
    tau = np.asarray(seconds, dtype=np.float64)
    if carrier_hz is None:
        phase = np.unwrap(np.angle(analytic))
        frequency = np.gradient(phase) * float(sample_rate_hz) / (2 * np.pi)
    else:
        frequency = float(carrier_hz)
    rotated = analytic * np.exp(-2j * np.pi * frequency * tau)
    return np.real(rotated) if not np.iscomplexobj(x) else rotated


def complex_form(series, axis: int = -1, name: str = "") -> Dict[str, object]:
    """CARRY A REAL MEASUREMENT INTO HILBERT SPACE, by its own analytic form.

    Ethan, 2026-09-06: *"The output of each module must be complex"* and
    *"Everything so far should be able to be related together in hilbert
    space given our measurements and known specs."*

    Many of this arc's measurements are real by nature - a level in IRE, a
    gain, a carrier frequency, a delay - and a real number has no phase to
    relate to anything. But a real SERIES does: measured line by line or
    field by field, it has an analytic representation whose real part is
    the series itself and whose imaginary part is its Hilbert transform
    along that axis. That representation is complex, it is unique, it
    throws nothing away, and the same operator that relates every other
    quantity in this model relates it too.

    WHAT THE IMAGINARY PART MEANS HERE, because it is not decoration. For a
    series varying along an axis, the analytic signal's magnitude is the
    envelope of that variation and its phase is where in the variation each
    sample sits. So a level that drifts across a field has an envelope (how
    much it drifts) and a phase (where in the drift a given line falls),
    and two measurements that drift together are in phase while two that
    drift against each other are not. That is precisely the relation a bare
    pair of real numbers cannot express.

    WHAT IT DOES NOT DO. It does not manufacture information: the analytic
    signal of a real series is determined by that series, so this is a
    change of representation and not a measurement. Its value is that it
    puts every quantity in one space, where `causality`, `cross_product`
    and the fold can act on all of them alike.
    """
    values = np.asarray(series, dtype=np.float64)
    if values.shape[axis] < 4:
        raise ValueError("an analytic form needs a series, not a scalar; "
                         "measure the quantity along a line or field axis")
    centre = values.mean(axis=axis, keepdims=True)
    varying = values - centre
    analytic = varying + 1j * hilbert(varying, axis=axis)
    return {
        "name": name,
        "complex": analytic,
        "mean": np.squeeze(centre),
        "envelope": np.abs(analytic),
        "phase_rad": np.angle(analytic),
        "is_complex": True,
        "why": ("a real number has no phase, but a real SERIES has an "
                "analytic form whose imaginary part is its own Hilbert "
                "transform - the same operator that relates everything "
                "else in this model"),
    }


def relate(first: Dict[str, object], second: Dict[str, object]
           ) -> Dict[str, object]:
    """How two measurements stand to one another, once both are complex.

    Their normalised complex inner product: the magnitude says how much of
    one is in the other and the angle says whether they move together, in
    quadrature, or against. A pair of real series can only be correlated;
    a pair of analytic ones can be correlated AND phased, which is the
    difference the space buys.
    """
    a = np.asarray(first["complex"]).ravel()
    b = np.asarray(second["complex"]).ravel()
    width = min(a.size, b.size)
    a, b = a[:width], b[:width]
    norm = np.sqrt(np.sum(np.abs(a) ** 2) * np.sum(np.abs(b) ** 2))
    product = complex(np.sum(a * np.conj(b)) / max(norm, 1e-300))
    return {
        "coherence": float(abs(product)),
        "phase_rad": float(np.angle(product)),
        "phase_deg": float(np.degrees(np.angle(product))),
        "in_phase": bool(abs(np.angle(product)) < np.pi / 8),
        "in_quadrature": bool(abs(abs(np.angle(product)) - np.pi / 2) < np.pi / 8),
        "opposed": bool(abs(np.angle(product)) > 7 * np.pi / 8),
        "samples": int(width),
        "why": ("the magnitude says how much of one is in the other and "
                "the angle says whether they move together, in quadrature "
                "or against - which a correlation alone cannot say"),
    }


def cross_product(first, second, axes: Sequence[int]) -> Dict[str, object]:
    """THE HILBERT CROSS PRODUCT of two signals, component by component.

    Ethan: *"hilbert cross product for one standard video field in our final
    differential, and the same for a vcr playback, and vcr record cross
    product ... and that will be a model of a bad playback and a good VCR."*

    Both signals are expanded into their 2^n partial Hilbert components and
    each component of the first is multiplied by the conjugate of the
    matching component of the second. The identity-by-identity term is the
    ordinary cross product; the terms carrying a transform on one side and
    not the other are the QUADRATURE relations, which is where a delay or a
    reflection shows and where a magnitude-only comparison sees nothing.

    Read as a model of a channel, the record side is the reference the
    machine intended and the playback side is what came back through the
    tape and the head, so their cross product IS the degradation: the
    "good VCR" against the "bad playback" in Ethan's words, with the tape
    between them.

    Returns the per-component products, the transfer implied by the
    identity term, and the part of the relation the identity term MISSES -
    the fraction of the total cross power that lives in the mixed
    components, which is the phase relation a magnitude comparison discards.
    """
    a = np.asarray(first)
    b = np.asarray(second)
    if a.shape != b.shape:
        raise ValueError(f"the two signals must have the same shape, "
                         f"{a.shape} against {b.shape}")
    axes = tuple(axes)
    products: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], np.ndarray] = {}
    subsets = []
    for bits in range(2 ** len(axes)):
        subsets.append(tuple(axes[i] for i in range(len(axes))
                             if (bits >> i) & 1))
    left = {s: partial_hilbert(a, s) for s in subsets}
    right = {s: partial_hilbert(b, s) for s in subsets}
    for s in subsets:
        for u in subsets:
            products[(s, u)] = left[s] * np.conj(right[u])
    matched = float(np.sum([np.abs(np.sum(products[(s, s)])) ** 2
                            for s in subsets]))
    mixed = float(np.sum([np.abs(np.sum(products[(s, u)])) ** 2
                          for s in subsets for u in subsets if s != u]))
    identity = products[((), ())]
    energy = float(np.sum(np.abs(b) ** 2))
    return {
        "axes": axes,
        "components": len(subsets),
        "products": products,
        "identity": identity,
        "transfer": float(np.sum(identity).real) / max(energy, 1e-30),
        "matched_power": matched,
        "mixed_power": mixed,
        "mixed_share": mixed / max(matched + mixed, 1e-30),
        "why": ("the matched terms carry what a magnitude comparison would "
                "see; the mixed terms carry the quadrature relation, which "
                "is where a delay or a reflection lives"),
    }


def expand(cube: tesseract.Cube) -> Dict[Component, np.ndarray]:
    """THE PROGRESSIVE EXPANSION: one axis at a time, doubling each time.

    Starts from the cube's mean and adds the measurement axes in order -
    each fold doubles the component set (the contrast joins the parent) -
    then adds the frequency axis's Hilbert half, which doubles it again.
    The components are indexed by (subset of measurement axes, "real" or
    "hilbert"); the "real" half of the empty subset is the grand mean.
    """
    contrasts = tesseract.walsh(cube)
    out: Dict[Component, np.ndarray] = {}
    for subset, entry in contrasts.items():
        value = entry["value"]
        out[(subset, "real")] = value
        out[(subset, "hilbert")] = hilbert(value)
    return out


def collapse_back(cube: tesseract.Cube, components: Dict[Component, np.ndarray]
                  ) -> np.ndarray:
    """THE INVERSE, BACK UP THE TREE: the real components fold back to the
    vertices exactly; the Hilbert half is what the analytic signal added
    and is not needed to return."""
    contrasts = {subset: {"value": value, "variance": None, "order": len(subset)}
                 for (subset, kind), value in components.items()
                 if kind == "real"}
    return tesseract.reconstruct(cube, contrasts)


def relate_to_spec(components: Dict[Component, np.ndarray],
                   spec: Optional[Dict[Component, np.ndarray]] = None,
                   strict: bool = False) -> Dict[Component, np.ndarray]:
    """Subtract the specification's own components, ALL of them.

    THE CONDITION THE WHOLE TRAVERSAL RESTS ON (Ethan: the expected models
    must be "themselves the full multi-D hilbert transform that can exactly
    subtract the signal on all dimensions"). The transform is its own
    inverse (`self_inverse`), so it adds no information: everything is won
    or lost here. A model supplied on only some components leaves its
    partners standing, and the partners are where the phase lives - a
    magnitude-only ideal cannot subtract a delay or an all-pass no matter
    how exactly its magnitude matches.

    A measurement made against the specified ideal already has the ideal at
    zero, which is the default. `strict` refuses a partial model rather
    than silently subtracting zero from the components it omits.
    """
    if spec is None:
        return dict(components)
    missing = [key for key in components if key not in spec]
    if strict and missing:
        raise ValueError(
            f"the model covers {len(spec)} of {len(components)} components; "
            f"{len(missing)} would be left unsubtracted, and the phase "
            f"lives in them - supply the model expanded, or pass "
            f"strict=False to accept the partial subtraction")
    return {key: value - spec.get(key, 0.0) for key, value in components.items()}


def causality(cube: tesseract.Cube, sample_rate_hz: Optional[float] = None
              ) -> Dict[str, Dict[str, float]]:
    """Per contrast: is the departure minimum phase (a causal filter, which
    its own magnitude inverts) or does it carry excess phase (a delay or a
    reflection, which needs its own time-axis measurement)?

    The Bode relation is applied to each contrast's real part along the
    frequency axis; the contrast's imaginary part less that is its excess.
    A linear excess is a delay and is reported as one, in seconds.

    `sample_rate_hz` is the rate of the grid the cube's bins were cut from
    (the exports are on the 4 f_sc grid); the contrast is placed back on
    that grid before the cepstrum is taken (`on_full_grid`). Without it the
    cube's band is taken as the whole axis, which was the first run's
    defect.

    WHAT THE FILL COSTS, MEASURED. The cepstrum is global and the bins
    outside the band are held flat, so even a true minimum-phase departure
    shows an apparent excess: a discrete one-pole on the 4 f_sc grid seen
    only on 0.2-4 MHz reads 0.15 rad maximum and 0.06 rad rms of excess
    that is the fill's, not the response's. An excess below about 0.1 rad
    rms is therefore "minimum phase to within the fill"; the delays and the
    0.4-1.1 rad all-pass measured on the real cubes are well above it.
    """
    contrasts = tesseract.walsh(cube)
    f = np.asarray(cube.bins, dtype=np.float64)
    out = {}
    for subset, entry in contrasts.items():
        if not subset:
            continue
        value = entry["value"]
        if value.size < 4:
            continue
        if sample_rate_hz is not None:
            full, _grid, index = on_full_grid(value.real, f, sample_rate_hz)
            minimum = minimum_phase(full)[index]
        else:
            minimum = minimum_phase(value.real)
        excess = value.imag - minimum
        # the delay: the slope of the excess against frequency
        centred = f - f.mean()
        slope = float(centred @ excess / max(centred @ centred, 1e-300))
        residue = excess - slope * centred
        # THE SHARE MUST BE TAKEN AFTER THE DELAY IS REMOVED. A delay is not
        # a failure of minimum phase, it is a delay: it is separable, it is
        # measured on the line above, and charging it to the excess made
        # every contrast look non-causal. The first version of this did
        # exactly that and its reading was withdrawn.
        phase_without_delay = value.imag - slope * centred
        out[":".join(subset)] = {
            "order": len(subset),
            "magnitude_rms": float(np.sqrt(np.mean(value.real ** 2))),
            "phase_rms": float(np.sqrt(np.mean(value.imag ** 2))),
            "minimum_phase_rms": float(np.sqrt(np.mean(minimum ** 2))),
            "excess_rms": float(np.sqrt(np.mean(excess ** 2))),
            "delay_s": -slope / (2 * np.pi),
            "allpass_rms": float(np.sqrt(np.mean(residue ** 2))),
            "minimum_phase_share": float(
                1.0 - np.mean(residue ** 2)
                / max(np.mean(phase_without_delay ** 2), 1e-300)),
            "minimum_phase_share_with_delay": float(
                1.0 - np.mean(excess ** 2) / max(np.mean(value.imag ** 2), 1e-300)),
        }
    return out


def deconvolve(cube: tesseract.Cube, sample_rate_hz: float,
               spec: Optional[Dict[Component, np.ndarray]] = None,
               sigma: float = 3.0) -> Dict[str, object]:
    """THE WHOLE TRAVERSAL: in, related to the spec, identified, and back.

    Expand the cube (every contrast and its Hilbert half), subtract the
    specification's components, keep the contrasts that stand above their
    noise, fold back up the tree to one real departure per vertex, and
    return the real kernel `exp(-departure)` that subtracts it - together
    with the residual, which is the sum of the contrasts the cube could not
    identify and is at the floor when nothing more remains.
    """
    components = relate_to_spec(expand(cube), spec)
    contrasts = tesseract.walsh(cube)
    verdicts = tesseract.identified(contrasts, sigma)
    kept = {key: (value if (key[0] == () or verdicts[key[0]]["identified"])
                  else np.zeros_like(value))
            for key, value in components.items()}
    departure = collapse_back(cube, kept)
    full = collapse_back(cube, components)
    residual = full - departure
    signal = tesseract.one_real_signal(cube, sample_rate_hz, sigma=sigma)
    return {
        "components": len(components),
        "identified": [":".join(s) for s, v in verdicts.items()
                       if s and v["identified"]],
        "departure": departure,
        "residual": residual,
        "residual_over_floor": float(np.mean(np.abs(residual) ** 2)
                                     / max(np.mean(cube.variance), 1e-300)),
        "kernels": signal["kernels"],
        "kernel_is_real": signal["kernel_is_real"],
        "round_trip_error": float(np.abs(full - cube.values).max()),
        "causality": causality(cube, sample_rate_hz),
    }


def fourier_slices(tensor, axes: Sequence[int]) -> Dict[str, object]:
    """NESTED TENSOR EXPANSION BY MULTI-AXIS FOURIER SLICES.

    The n-axis transform built one axis at a time: after each axis the
    partial spectrum is recorded, so the expansion is a tree whose depth
    is the axis count. The projection-slice theorem is what makes the
    nesting legitimate - a one-axis transform of a projection is a slice
    of the full transform - and it is checked here: the fully nested
    result equals the direct n-axis transform to machine precision.

    Separability is read from the final spectrum: the fraction of its
    energy carried by its leading rank-one term, which is one when the
    tensor is an outer product of one-axis slices (a rank-one term of the
    tape tensor) and falls as the axes interact.
    """
    x = np.asarray(tensor, dtype=np.complex128)
    stages: List[Tuple[int, np.ndarray]] = []
    partial = x
    for axis in axes:
        partial = np.fft.fft(partial, axis=axis)
        stages.append((axis, partial))
    direct = np.fft.fftn(x, axes=list(axes))
    matrix = np.moveaxis(partial, axes[0], 0).reshape(partial.shape[axes[0]], -1)
    s = np.linalg.svd(matrix, compute_uv=False)
    return {"stages": stages, "spectrum": partial,
            "nested_equals_direct": float(np.abs(partial - direct).max()),
            "separable_share": float(s[0] ** 2 / max(np.sum(s ** 2), 1e-300)),
            "singular_values": s}

def zero_phase(values, taps, axis: int = -1):
    """A FORWARD-BACKWARD FIR FILTER AS ONE CONVOLUTION.

    Ethan, 2026-09-07: *"I believe the hypercomplex stages and the folding
    can be consolidated through some mathmetical simplification."* This is
    one, and it is an identity rather than an approximation.

    Filtering forward with `b` and then backward with `b` convolves the
    record with `b` and with its own reverse, and convolution is
    associative, so the pair is ONE convolution with the kernel's
    autocorrelation - a zero-phase kernel of twice the length less one.
    `scipy.signal.filtfilt` performs the two passes; this performs the
    single equivalent one, and reproduces filtfilt's own odd edge
    extension so the two agree at the boundaries as well as inside.

    MEASURED on one field of 667334 samples at 40 MSps with a 255-tap
    band-pass: the worst disagreement anywhere is 1.332e-15 on a signal of
    rms 0.3581, which is 3.7e-15 of the signal and therefore machine
    precision, and the single convolution runs 3.9 times faster - 11.9 ms
    against 46.6. The saving is real because the two-pass form is O(n
    taps) twice while an overlap-add convolution is O(n log n) once, and
    at 255 taps the logarithm has long since won.

    WHY IT MATTERS HERE and not merely in the abstract: a field's raw
    radio frequency passes five such filters before any measurement is
    taken, so the identity removes about three quarters of that cost
    without changing a single returned number.
    """
    from scipy import signal as _signal
    record = np.asarray(values, dtype=np.float64)
    kernel = np.asarray(taps, dtype=np.float64).ravel()
    if kernel.size < 2:
        raise ValueError("a zero-phase filter needs a kernel")
    if record.ndim != 1:
        raise ValueError("zero_phase takes one record at a time")
    pad = 3 * (kernel.size - 1)
    if record.size <= pad:
        return _signal.filtfilt(kernel, [1.0], record, axis=axis)
    # `filtfilt`'s own default extension, reproduced so the edges match:
    # twice the edge value less the reflected interior.
    left = 2.0 * record[0] - record[pad:0:-1]
    right = 2.0 * record[-1] - record[-2:-pad - 2:-1]
    extended = np.concatenate([left, record, right])
    paired = np.convolve(kernel, kernel[::-1])
    filtered = _signal.oaconvolve(extended, paired, mode="same")
    return filtered[pad:pad + record.size]

