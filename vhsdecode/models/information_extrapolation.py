"""Multidimensional information extrapolation - the component model.

The restoration is three-dimensional: AMPLITUDE x FREQUENCY x TIME. Every
component of the system is measured on its own criterion along those three
axes. On each axis a component has a RESOLUTION - its measurement grid and
bounds, the information boundary on that axis - and a RESIDUAL - its
departure from the ideal, a function over the other axes. The restoration
applies the anti-residuals iteratively, component by component, until no
residual is left: the residual on the held-out lines has no stable
structure and sits at the noise floor. The noise floor is the LAST
component, and it is amplitude only: what remains is distributed over a
statistical curve, so no phase residual can be derived from it. Then the
2-D differential (frequency x time) of what remains is found.

The proposal this formulates is docs/CHANNEL_LOOP_PROPOSAL.md, in its
author's words; the laws the process obeys are listed on `Gauges`.

Two stages: first over the raw RF (the "pure RF frequency and phase cutoff
extrapolation" component), which yields the exact incoming residual curves
for the luma as a set of anti-residuals applied DURING demodulation; then
over the luma with the components already identified, each instantiated as
a `Component` with its own criterion.

The frequency axis is differentiated against the channel identifier's
response - the BASELINE - never against flat. The time axis is corrected
by the hsync and chroma-burst lock elsewhere in the decoder, and the
downscale (resampling on the distance between sync pulses) removes the
residual RF phase; the derivation is applied to the time-base correction
too, helped by the wow-factor amplitude difference, because the time-base
residual is what keeps phase transferring through the residual.

The residuals are CHANNELS. Each residual - amplitude, time, frequency -
is computed per RF block at the capture rate and carried through the
downscale to 4fsc on the FINAL time base correction, exactly as the
demodulated video and the envelope channels are ("keep the existing
residuals, but downscaled to 4fsc; each channel is a residual that we
downscale based on the final TBC"). That puts every residual on the
output grid the gauges read, time-aligned by the corrected linelocs, and
the downscale's sync-distance resampling removes the residual RF phase
from the residual channels as it does from the picture.

This module holds the model and the process; the measurements (gauges),
the feed-forward through the real chain (subset decodes) and the export
live in the offline tools under tools/ringing_measure/, and the runtime
application of the converged anti-residuals is `channel_eq`.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Protocol, Sequence, Tuple

import numpy as np

# THE CHAIN SPLITS IN TWO, and the split decides where a component's
# correction may legitimately be applied.
#
#   PLAYBACK  everything from the tape to the demodulator - heads, tape,
#             preamp, the decoder's own RF filters. Its distortion is
#             LTI on the RF and passes through a nonlinear demodulator,
#             so only a correction BEFORE the demodulator can invert it
#             (the standing law: no video-domain filter can).
#   SOURCE    everything from the source signal into the recording VCR -
#             the record-side filters, the clipper, whatever the signal
#             met before it was ever modulated. Its distortion happened
#             in the VIDEO domain, before modulation, so a video-domain
#             filter is exactly the right instrument for it - the same
#             law that forbids one for PLAYBACK permits one here.
#
# The stages are measured in order: PLAYBACK first, on the raw decode;
# then SOURCE, on a decode the playback correction has already been
# applied to, because what remains after the playback path is inverted
# is by construction what the recording put there.
# THE CHAIN HAS THREE STAGES, because the signal passes through TWO
# MACHINES and then a specification.
#
#   RF_PLAYBACK   the playback VCR's own known components - its heads, its
#                 transport, its playback electronics. It acted LAST on the
#                 signal, so it is inverted FIRST. Everything here is LTI
#                 on the RF ahead of a nonlinear demodulator, so only a
#                 pre-demodulator correction can invert it.
#   RF_RECORDING  the recording VCR's known components - its record
#                 electronics, its heads, its transport. Measured on a
#                 decode the playback correction has already cleaned, so
#                 what it sees is the recording machine's own contribution
#                 rather than the two convolved together. This is what
#                 recovers "the RF before it went into the VHS VCR".
#   PICTURE       the picture against the VIDEO SPECIFICATION, differentiated
#                 by the ELECTRONIC VARIABLES - the emphasis tuning knobs and
#                 their like. This is a video-domain stage, so a video-domain
#                 filter is legitimate here where it is forbidden at RF.
#
# The order is forced by the chain, not chosen: a machine can only be
# inverted once everything that acted after it has been. The same principle
# orders the components WITHIN a stage (see `orthogonalize`).
Stage = Literal["rf_playback", "rf_recording", "picture"]
RF_PLAYBACK, RF_RECORDING, PICTURE = "rf_playback", "rf_recording", "picture"
STAGES: Tuple[Stage, Stage, Stage] = (RF_PLAYBACK, RF_RECORDING, PICTURE)
# the earlier two-stage names, kept so existing callers and exports do not
# break: the playback half is the first RF stage, and what was called the
# source stage is the picture stage
PLAYBACK, SOURCE = RF_PLAYBACK, PICTURE

# Which MACHINE a component belongs to. The same physical part exists in
# both machines - each has heads, a transport and electronics - so a
# component must say which one it describes, or the two are pooled and the
# pair's difference is attributed to one of them.
Machine = Literal["recording", "playback", "pair"]
RECORDING, PLAYBACK_MACHINE, PAIR = "recording", "playback", "pair"

Axis = Literal["amplitude", "frequency", "time", "field"]
AXES: Tuple[Axis, Axis, Axis] = ("amplitude", "frequency", "time")
# The per-field dimension: the NON-REDUCIBLE residual - what is left at the
# floor once every component has converged - is tracked field by field, and
# its variation between fields is differentiated on as well ("this will be
# our per field dimension"). The residual channels are exported one file
# per field with the head parity for exactly this.
FIELD_AXIS: Axis = "field"

# The residual channels and what each is made of at the decoder
# (`vhsdecode.residual_channels`, exported by --residual_channels), every
# one on the 4fsc grid of the FINAL time base correction. The channels the
# demodulator emits ride the video dict to the field and are resampled on
# the corrected linelocs like the picture; the time channel is the final
# TBC's own resample map.
RESIDUAL_CHANNELS: Dict[str, str] = {
    # the carrier envelope: the amplitude reference; resampled on the
    # timing alone (the wow LEVEL adjust is a frequency-domain correction,
    # and multiplying it into the amplitude would write the very drum-rate
    # wobble the time-base witness reads back into the channel)
    "amplitude": "envelope",
    # what the RF anti-residual changed, in the demodulated luma
    "frequency": "demod - demod_noeq",
    # the final TBC's per-pixel speed deviation (the wow factor - 1)
    "time": "wowfactors - 1",
    # the color-under gain applied - the chroma channel's identified
    # component - resampled on the timing alone
    "chroma_amplitude": "chroma_envelope_correction",
}

# The capture's amplitude depth. Hard-coded for now, as directed; an input
# flag will supply it (TODO: --rf_bits).
DEFAULT_RF_BITS = 8
# The capture stage's own roll-off, as stated for this capture chain.
CAPTURE_ROLLOFF_HZ = 13.3e6
DEFAULT_SAMPLE_RATE_HZ = 40e6


@dataclass(frozen=True)
class Resolution:
    """The information boundary on one axis: what one step is, the bounds,
    and the number of steps where the axis is discrete."""

    unit: str
    minimum: Optional[float]
    maximum: Optional[float]
    total: Optional[int] = None


@dataclass
class Residual:
    """Expected minus received along one axis, as a function over the other
    axes. `value` is None until measured (or while unknown - the RF
    capture's own response needs the final encoded tape signal fed back).
    `source` states the criterion by which it is measured."""

    axis: Axis
    value: Optional[np.ndarray] = None
    over: Tuple[Axis, ...] = ()
    source: str = ""
    # The standard error of `value`, element for element, where the gauge
    # supplies one: what the hold-out test compares the residual against.
    error: Optional[np.ndarray] = None

    @property
    def measured(self) -> bool:
        return self.value is not None


@dataclass
class Component:
    """One component of the system, identified by its own criterion."""

    name: str
    ideal: Optional[Callable[..., np.ndarray]]
    resolution: Dict[Axis, Resolution]
    residual: Dict[Axis, Residual]
    criterion: str
    stage: Stage = RF_PLAYBACK      # which stage of the chain it belongs to
    machine: Machine = PAIR         # which VCR it describes, where it is
                                    # a property of a machine rather than
                                    # of the pair together
    frozen: bool = False            # "no longer differentiate over that part"
    amplitude_only: bool = False    # the noise floor: no phase is ever derived
    passes: int = 0
    # THE ACCUMULATED EXPECTED SIGNAL - the model. Each pass the measured
    # residual is fed forward INTO this, so it grows from the ideal toward
    # what the signal actually is, and the residual shrinks toward zero.
    # "The residual difference is accumulated in the expected signals."
    expected: Dict[Axis, np.ndarray] = field(default_factory=dict)

    def unmeasured_axes(self) -> List[Axis]:
        return [axis for axis, residual in self.residual.items()
                if not residual.measured]

    def accumulate(self, residual: Dict[Axis, np.ndarray]) -> None:
        """Feed the measured residual forward into the expected signal.

        The expected signal starts at the component's ideal (an empty
        accumulation) and gains the residual on every pass, so at the
        limit - the residual approaching zero - the accumulation holds the
        whole shape the ideal was missing. That accumulated shape is what
        the component knows, and the correction it implies is its
        departure from the specification's ideal."""
        for axis, value in residual.items():
            if value is None:
                continue
            value = np.asarray(value)
            if axis in self.expected and self.expected[axis].shape == value.shape:
                self.expected[axis] = self.expected[axis] + value
            else:
                self.expected[axis] = value.copy()
            self.residual[axis] = Residual(
                axis, value=value,
                over=self.residual[axis].over if axis in self.residual else (),
                source=self.residual[axis].source if axis in self.residual else "",
                error=self.residual[axis].error if axis in self.residual else None)


# --------------------------------------------------------------------------
# The components, as specified
# --------------------------------------------------------------------------


def flat_to_nyquist(frequency_hz: np.ndarray) -> np.ndarray:
    """The ideal capture: unity from DC to Nyquist."""
    return np.ones_like(np.asarray(frequency_hz, dtype=np.float64))


def rf_capture(n_samples: int, sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
               bits: int = DEFAULT_RF_BITS) -> Component:
    """Pure RF frequency and phase cutoff extrapolation - the first
    component, with the resolutions and residuals exactly as specified.

    Effective information budget = bits x sample rate (8 bits x 40 MHz).
    """
    nyquist_hz = sample_rate_hz / 2.0
    steps = 2 ** bits
    return Component(
        name="pure RF frequency and phase cutoff extrapolation",
        ideal=flat_to_nyquist,
        resolution={
            "amplitude": Resolution(
                unit=f"{bits}-bit step (hard-coded {DEFAULT_RF_BITS}; "
                     "an input flag will supply the capture depth)",
                minimum=0,                 # DC
                maximum=steps - 1,         # 255 for the initial RF 8 bits
                total=steps),
            "frequency": Resolution(
                unit="Hz (Nyquist theory: sample rate / N per bin)",
                minimum=0,                 # DC
                maximum=nyquist_hz,        # 20 MHz at 40 MSps, with the
                total=None),               # 13.3 MHz capture roll-off applied
            "time": Resolution(
                unit="sample",
                minimum=0,
                maximum=None,
                total=int(n_samples)),     # the number of amplitude
        },                                 # measurements
        residual={
            "amplitude": Residual(
                "amplitude", value=None, over=("frequency", "time"),
                source="the RF capture's own frequency response (NOT KNOWN "
                       "now - it needs the final ENCODED tape signal fed "
                       "back), and the response over time between ideal "
                       "Nyquist and the hardware roll-off filter"),
            "frequency": Residual(
                "frequency", value=None, over=("time",),
                source="the differential, over time, between that "
                       "response and the ideal-Nyquist-minus-hardware-"
                       "roll-off difference"),
            "time": Residual(
                "time", value=None, over=(),
                source="the final noise-floor residual - or none, if we "
                       "stop here"),
        },
        criterion=f"ideal Nyquist ({nyquist_hz / 1e6:.1f} MHz) against the "
                  f"{CAPTURE_ROLLOFF_HZ / 1e6:.1f} MHz capture roll-off; "
                  f"information budget {bits} bits x "
                  f"{sample_rate_hz / 1e6:.0f} MHz",
    )


def noise_floor() -> Component:
    """The last component: the noise floor of the block the FM signals were
    gathered in. Modeled exactly by integrating the sync pulse shape to
    infinity - the accumulated sync converges to its deterministic shape
    and what does not converge is the noise. Its amplitude is its level
    relative to the CORRECTED sync pulse; its residual is that remainder.
    Amplitude only: distributed over a statistical curve, no phase."""
    return Component(
        name="noise floor of the block the FM signals were gathered in",
        ideal=None,
        resolution={
            "amplitude": Resolution(unit="IRE relative to the corrected "
                                         "sync pulse", minimum=0,
                                    maximum=None),
        },
        residual={
            "amplitude": Residual(
                "amplitude", value=None, over=("frequency", "time"),
                source="integrate the sync pulse shape to infinity: the "
                       "variance of the accumulated mean per lag is the "
                       "floor; subtracted from the block; gates every other "
                       "residual by amplitude (the Wiener term)"),
        },
        criterion="the remainder after every knowable component is "
                  "exhausted: structureless on the held-out lines",
        amplitude_only=True,
    )


def channel_baseline(export_path: Optional[str] = None) -> Component:
    """Part A - the channel identifier - as the BASELINE: the expected
    frequency response every later component is differentiated against
    on the frequency axis. Produced offline by the pilot identification
    loop (tools/ringing_measure/channel_identify.py) and carried as
    channel_response.npz."""
    return Component(
        name="channel identifier (Part A): the baseline frequency response",
        ideal=None,
        resolution={
            "frequency": Resolution(unit="Hz on the RF axis", minimum=0,
                                    maximum=None),
        },
        residual={
            "frequency": Residual(
                "frequency", value=None, over=("time",),
                source="pilot identification: the sync edges per polarity "
                       "at the tip and blanking carriers for phase, the "
                       "constant envelope for magnitude, the decoder's own "
                       "filters compensated analytically"
                       + (f"; export {export_path}" if export_path else "")),
        },
        criterion="the reference response: received minus baseline is the "
                  "frequency-axis residual of every later component",
    )


def time_base(n_samples: int) -> Component:
    """The time axis - the time-base residual, the missing part. Timing is
    corrected by the hsync and chroma-burst lock; the downscale (resampling
    on the sync-to-sync distance) removes the residual RF phase. The
    derivation is applied to the time-base correction itself, witnessed by
    the wow-factor AMPLITUDE difference - the envelope's drum-rate wobble
    that correlates with head-to-tape speed - and fed to the carrier TBC as
    its flutter prior, so the time-base residual is at its floor BEFORE the
    frequency-axis differentiation runs."""
    return Component(
        # the time base correction IS the sync resampler - the downscale
        # on the sync-to-sync distance - and it is a DERIVABLE component
        # because its residual is taken from it each pass: the final
        # resample map's own deviation (the time channel, per pixel and
        # per line; hsync real, burst phase imaginary), fed back to the
        # line locations as its anti-residual
        name="time base: the sync resampler (wow and flutter)",
        ideal=None,   # the specification's line timing, held by the TBC
        resolution={
            "time": Resolution(unit="sample", minimum=0, maximum=None,
                               total=int(n_samples)),
        },
        residual={
            "time": Residual(
                "time", value=None, over=("time",),
                source="the time-base residual (wow/flutter), witnessed by "
                       "the wow-factor amplitude difference of the envelope "
                       "residual; derived and fed to the carrier TBC "
                       "(carrier_tbc.refine_linelocs) as its prior. The "
                       "witness is PER HEAD: measured, the drum-band "
                       "amplitude wobble's correlation with the period "
                       "deviation flips sign between the heads (head-contact "
                       "modulation at the drum rate with a per-head phase), "
                       "so the prior is a per-head regression weighted by "
                       "its learned r-squared, with its own standard error "
                       "in `error` - the first entry of the per-field "
                       "dimension"),
        },
        criterion="timing before anything spectral: correct the time base, "
                  "then the downscale removes the residual RF phase - what "
                  "was keeping phase transferring through the residual",
    )


def _component(name, criterion, axes, linear, stage=RF_PLAYBACK,
               machine=PAIR) -> Component:
    """A component identified by the offline gauges on the given axes:
    linear ones fold into the anti-residual set, nonlinear ones stay with
    the subtraction/restoration stage (no linear filter at any point
    absorbs them)."""
    return Component(
        name=name, ideal=None,
        resolution={axis: Resolution(unit="per the gauge", minimum=None,
                                     maximum=None) for axis in axes},
        residual={axis: Residual(axis, None, tuple(a for a in AXES
                                                    if a != axis),
                                 source=criterion) for axis in axes},
        criterion=criterion + (" [linear: anti-residual]" if linear
                               else " [nonlinear: restoration stage]"),
        stage=stage,
        machine=machine,
    )


def luma_components() -> List[Component]:
    """The luma components already identified, each with its criterion.
    Their residuals are measured by the offline gauges."""
    component = _component

    return [
        component("sync-step response, per polarity",
                  "the sync edge's derivative against the spec-shaped ideal "
                  "at the measured plateau levels: fall at the tip carrier, "
                  "rise at the blanking carrier",
                  ("frequency", "time"), linear=True),
        component("constant-envelope magnitude",
                  "envelope variation on the FM = |H| at the instantaneous "
                  "carrier; the picture sweeps the carrier",
                  ("amplitude", "frequency"), linear=True),
        component("record-side nonlinear emphasis",
                  "level-dependent by design; the polarity-common remainder "
                  "above the linear synthesis near the de-emphasis corner",
                  ("amplitude", "frequency"), linear=False),
        component("record dark clip",
                  "the recording VCR clipped the input at the sync tip: a "
                  "PERFECTLY FLAT tip (its variance collapses below the "
                  "porch's, which carries the same noise) is the criterion; "
                  "the fall's tail below the clip is not channel, so the "
                  "sync area's response is measured excluding the clipped "
                  "tip (the rise carries the edge); per-tape, not universal",
                  ("amplitude", "time"), linear=False),
        component("color-under footprint under the luma",
                  "the color-under is under the luma in the RF: its "
                  "frequency (the color-under carrier and sidebands), "
                  "amplitude (the chroma amplitude channel) and time (the "
                  "burst phase) are known from the chroma channel, so its "
                  "footprint is re-synthesized and filtered out of the luma "
                  "channel before demodulation - the same process the "
                  "chroma path applies to the luma's footprint on the "
                  "chroma during its TBC and frequency and phase correction",
                  ("amplitude", "frequency", "time"), linear=True),
        component("source-recorded ringing and roll-off",
                  "level-independent, zero-phase where the source was "
                  "linear-phase filtered; not channel", ("frequency",
                                                          "time"),
                  linear=False),
        component("head switch",
                  "the DC step the head amplifiers put on the carrier at "
                  "every switch, on the envelope residual", ("amplitude",
                                                            "time"),
                  linear=False),
        component("dropouts",
                  "envelope collapses at dropout edges", ("amplitude",
                                                          "time"),
                  linear=False),
    ]


def heterodyne_timing_scale(subcarrier_hz: float,
                            color_under_hz: float) -> float:
    """How much a TIME error is understated by reading it as a phase on
    the up-converted subcarrier: `subcarrier_hz / color_under_hz`
    (5.6875 on NTSC VHS).

    Heterodyning is multiplication by a local oscillator: a component at
    f_cu + d with amplitude A e^{j theta} lands at f_sc + d as
    A e^{j(theta + phi_LO)}. The phase is CARRIED OVER UNCHANGED, plus
    one constant common to every component - nothing multiplies it. So a
    correction expressed as a COMPLEX GAIN PER FREQUENCY crosses the
    heterodyne with no scale factor at all, magnitude and phase alike,
    and that is how a correction should be carried.

    The ratio belongs to the phase-to-TIME conversion, which is a
    modelling choice rather than a property of the mix. A delay tau is a
    phase of 2 pi f tau, so the same delay is a different phase at the
    two carriers. In the decoder the chroma is time-base corrected while
    it is still the color-under and up-converted afterwards, so a
    residual displacement of the output grid imprints its phase at
    f_cu and the up-conversion carries that phase to f_sc unchanged:
    reading the burst's phase at the output as though it had accrued at
    f_sc understates the displacement by this factor.

    The same ratio is why color-under recording exists: a time-base error
    writes 5.6875 times less chroma phase error at f_cu than a direct
    subcarrier recording would, and up-conversion carries the small phase
    forward rather than restoring the large one.

    Corollary, and the reason to carry complex gains: a correction
    expressed as a group delay at one carrier and re-applied as a group
    delay at the other is wrong by this factor, over-correcting at the
    subcarrier.
    """
    if not color_under_hz:
        raise ValueError("the color-under carrier frequency is required")
    return float(subcarrier_hz) / float(color_under_hz)


def chroma_components() -> List[Component]:
    """The chroma channel: the correction happens there as well, with the
    component identified there. The color-under was written beside the
    luma carrier and took the same tape loss, so its amplitude residual is
    the luma envelope's deviation from the amplitude model at the
    instantaneous carrier, applied as a gain on the color-under before the
    bursts are measured (the chroma stage's envelope gain, band-limited by
    the measured luma-to-chroma transfer). The burst is the chroma's pilot:
    its known amplitude and phase every line against the received burst
    give the chroma path's response and its timing (the burst lock). Its
    residual channels ride the chroma downscale on the same final time
    base as the luma's."""
    return [
        _component("color-under envelope amplitude",
                   "the luma envelope's deviation from the amplitude model "
                   "at the instantaneous carrier, as the gain the color-"
                   "under is corrected by, within the measured luma-to-"
                   "chroma transfer band",
                   ("amplitude", "frequency"), linear=True),
        _component("chroma burst pilot",
                   "the burst's known amplitude and phase every line "
                   "against the received burst: the chroma path's response "
                   "and its timing, the burst lock",
                   ("frequency", "time"), linear=True),
        _component("chroma heterodyne scale",
                   "the chroma was recorded DOWN-heterodyned, so the "
                   "channel acted on it at the color-under carrier: model "
                   "and cancel the response error THERE, and apply what "
                   "remains after up-heterodyning as a COMPLEX GAIN PER "
                   "FREQUENCY, which crosses the mix unscaled (heterodyning "
                   "translates the spectrum and carries phase over "
                   "unchanged). The ratio `heterodyne_timing_scale` "
                   "(5.6875 on NTSC VHS) belongs only where phase is read "
                   "as TIME: the chroma is time-base corrected while it is "
                   "still the color-under, so the burst's output phase "
                   "accrued at the color-under carrier and understates a "
                   "grid displacement by that factor",
                   ("frequency", "time"), linear=True),
        _component("colour-under heterodyne frequency",
                   "the heterodyne's own frequency error, read from what the "
                   "burst lock had to MOVE rather than from any residual "
                   "phase: the lock nulls the burst's phase by shifting line "
                   "positions, so the error never survives AS phase and a "
                   "phase measurement returns zero by construction. A "
                   "per-line period deviation d is a frequency offset of "
                   "f_sc x d. It separates as the recording does: a LINE-"
                   "LOCKED part, a fixed pattern repeating field to field, "
                   "and a WOBBLING part that does not - the source's own "
                   "video frequency tuning. The line-locked part peaks at "
                   "exactly half a cycle per line, which is the recording "
                   "VCR's comb-filter period, so an imbalance there is what "
                   "the amplitude wobble writes into the comb",
                   ("frequency", "time"), linear=True),
        _component("burst-to-sync phase lock",
                   "the burst's phase referred to the SYNC PULSE - the one "
                   "absolute phase reference the signal carries, because the "
                   "standard fixes the subcarrier at a half-integer multiple "
                   "of the line rate, so the burst must advance by exactly "
                   "half a cycle every line. Read at a fixed offset from the "
                   "line's origin, which IS the sync fall, the relationship "
                   "is pinned; read from the lock's own displacement it is "
                   "not, because that array's endpoints are pinned and a "
                   "constant frequency offset lies in its null space. Across "
                   "the vertical interval no burst exists at all, so the "
                   "relationship must be carried by the equalizing pulses "
                   "alone and checked at the first burst after - that check "
                   "is the long-baseline drift",
                   ("frequency", "time"), linear=True),
    ]


def vcr_components(machine: Machine) -> List[Component]:
    """One VCR's known components: its magnetics, its mechanics and its RF
    electronics. Called ONCE PER MACHINE, which is what makes the VCR stage
    run twice.

    Both machines have all of these - each has heads, a transport and
    electronics - so the same set is instantiated for each, and every
    component carries which machine it describes. Pooling them would
    attribute the pair's difference to one of them.

    They are KNOWN components, not free curves. The head parameters are
    physical (`head_model`: spacing, gap, azimuth by Wallace, gap, thickness
    and azimuth loss); the transport's disturbances land at frequencies its
    part diameters fix (`transport_model`); the electronics have the
    format's own filter forms with per-device parameters (`filter_model`).
    That is what lets a residual be attributed to a PART rather than to a
    bin of a spectrum."""
    stage = RF_PLAYBACK if machine == PLAYBACK_MACHINE else RF_RECORDING
    where = ("the playback machine, inverted FIRST because it acted last"
             if machine == PLAYBACK_MACHINE else
             "the recording machine, inverted on a playback-corrected decode "
             "so what it sees is its own contribution and not the two "
             "machines convolved")
    return [
        _component("%s head response" % machine,
                   "the physical head model fitted to the RF envelope "
                   "against the instantaneous carrier - Wallace spacing, "
                   "gap, thickness and azimuth loss, with the format's own "
                   "mechanics. Fitted at RF and never in the video domain, "
                   "where spacing loss is exactly annihilated by the "
                   "carrier-normalization law. " + where,
                   ("amplitude", "frequency"), linear=True,
                   stage=stage, machine=machine),
        _component("%s transport" % machine,
                   "every rotating part turns at a rate its diameter fixes, "
                   "so the specification predicts WHERE to look and a line "
                   "found at a predicted rate names the part. Two witnesses "
                   "separate the faults: the time base answers to speed, the "
                   "envelope to tension and contact. The tape path's ORDER "
                   "matters, because a disturbance upstream of the capstan "
                   "is partly regulated out and one downstream is not",
                   ("amplitude", "time"), linear=True,
                   stage=stage, machine=machine),
        _component("%s RF electronics" % machine,
                   "the machine's own RF chain - record amplifier and "
                   "modulator, or playback preamp and equalizer - as the "
                   "format's filter forms with per-device parameters. LTI on "
                   "the RF ahead of a nonlinear demodulator, so only a "
                   "pre-demodulator correction inverts it",
                   ("frequency", "time"), linear=True,
                   stage=stage, machine=machine),
    ]


def picture_components() -> List[Component]:
    """The picture against the VIDEO SPECIFICATION, differentiated by the
    ELECTRONIC VARIABLES.

    This stage runs once, after both machines have been inverted at RF. It
    is a video-domain stage, so a video-domain filter is legitimate here
    where the law forbids one at RF. What differentiates it is the
    recording machine's tuning: the emphasis networks, whose form the
    format fixes and whose parameters - the shelf's mid frequency, gain and
    Q - are the knobs that vary from device to device."""
    return [
        _component("emphasis tuning",
                   "the de-emphasis shelf in the DECODER'S OWN "
                   "parametrization (deemph_mid, deemph_gain, deemph_q), "
                   "fitted as a DEPARTURE from what the decoder already "
                   "assumes, so a fitted value can be handed straight back "
                   "as a decode setting. The shelf acts near 274 kHz, where "
                   "a sync pulse has almost no resolution: it needs the "
                   "vertical interval or a flat field",
                   ("amplitude", "frequency"), linear=True,
                   stage=PICTURE, machine=RECORDING),
        _component("video specification levels",
                   "the picture against the specification's own levels and "
                   "timings - sync depth, blanking, the active window - "
                   "which are the fixed references this stage measures "
                   "against rather than anything estimated",
                   ("amplitude", "time"), linear=True,
                   stage=PICTURE, machine=RECORDING),
    ]


def head_dc_components() -> List[Component]:
    """The DC response of each video head, one constant per head.

    The offset between the expected blanking level and the received one
    is a property of the head that laid the field down, and the heads are
    independent: their field-to-field wanderings do not correlate, so
    each carries its own DC response rather than sharing a path-wide one.

    It must be pooled PER HEAD. Pooled across both heads the offset
    appears to track the response tilt strongly on real video, but the
    correlation is entirely between the groups: within either head it
    vanishes. That pooled figure measures the heads sitting at different
    offsets, not a relationship between offset and tilt, and reading it as
    the latter would attribute a head-to-head difference to a frequency
    response."""
    return [
        _component("per-head DC response",
                   "the offset between the expected blanking level and the "
                   "received one, measured separately for each video head "
                   "and constant within it. The heads are independent, so "
                   "the groups are never pooled: a correlation taken across "
                   "them reports the difference BETWEEN the heads and not "
                   "anything within either. Averaging is taken WITHIN a "
                   "head, which is coherent and buys root-N; the difference "
                   "between the heads is fixed hardware and is held as a "
                   "constant, but PER RECORDING - it does not correlate "
                   "between tapes played on the same deck, so it is not the "
                   "playback path and must never be carried across tapes",
                   ("amplitude",), linear=True),
    ]


def rf_stage_drift_components() -> List[Component]:
    """The RF stage's parameters as they move ALONG a tape.

    The per-head loss is fixed hardware and is held as a constant. The RF
    stage's own parameters are not: the recording device's battery voltage
    sags and its temperature climbs while it records, and a camcorder does
    both far more than a mains-powered deck. So these are tracked over
    time, and the drift is a control variable rather than noise.

    Two things must be kept apart, because the head-model fit removes an
    overall constant and therefore CANNOT see a pure gain: a sagging
    battery reducing record current shows as a gain change, while a
    changing head-tape contact shows in the fitted shape. Both are
    measured, separately."""
    return [
        _component("RF stage drift along the tape",
                   "the head model's parameters fitted per field and "
                   "described as a time series: the slope per field with a "
                   "t statistic scaled to the EFFECTIVE sample size, since "
                   "consecutive fields are not independent draws and a naive "
                   "t on serially correlated data is badly overconfident. A "
                   "spread below the gauge's own error bar is a warning, not "
                   "a finding",
                   ("amplitude", "frequency", "time"), linear=True),
        _component("record-side gain and deviation drift",
                   "what the head-model fit is blind to, measured directly: "
                   "the overall RF envelope level (a sagging battery reduces "
                   "record current and shows here, not in the fitted shape) "
                   "and the sync-tip-to-blanking carrier spacing, which is "
                   "the FM deviation and moves with the supply voltage",
                   ("amplitude", "frequency"), linear=True),
    ]


# A matrix pencil fitted to the sync pulse's aftermath returns DAMPED
# SINUSOIDS, and a damped sinusoid is exactly this arc's three axes in
# parametric form: a FREQUENCY, a residue AMPLITUDE, and a decay rate,
# which is a TIME. So the pencil is not a separate kind of measurement -
# it is the same triple, measured on the sync pulse and expressed in a few
# parameters instead of a curve.
#
# It belongs to the VIDEO path. The sync pulse it is fitted to is
# demodulated video, past the demodulator and the de-emphasis, so it
# describes what happens after the RF stages and not within them.
#
# `sync_step_response.polarity_poles` supplies frequency and decay but NOT
# the amplitude: the magnitude it returns is |pole|, which is the decay's
# own magnitude, not how much of the residual that mode accounts for.
# `pole_residues` solves for that, which is what completes the triple.


def pole_residues(segment, poles) -> Dict[str, np.ndarray]:
    """How much of a residual each damped mode actually accounts for.

    The pencil gives the modes; this gives their weights. With modes z_k
    the residual is a sum of c_k z_k^n, linear in the c_k once the z_k are
    known, so the residues follow from one least-squares solve. Their
    magnitudes are the AMPLITUDE axis, their arguments the phase each mode
    starts at, and the share of variance each explains says which modes are
    worth keeping."""
    raw = np.asarray(segment).ravel()
    real_signal = not np.iscomplexobj(raw)
    values = raw.astype(np.complex128)
    modes = np.asarray(poles, dtype=np.complex128).ravel()
    if values.size < 4 or modes.size == 0:
        return {}
    # A REAL signal's damped sinusoids come in CONJUGATE PAIRS: cos decays
    # as z^n plus its conjugate, so fitting one side alone recovers exactly
    # half the amplitude. Each complex mode therefore contributes both
    # itself and its conjugate to the design, and the pair's residue is
    # combined afterwards.
    columns, pairs = [], []
    for index, mode in enumerate(modes):
        columns.append(mode)
        pairs.append((index, False))
        if real_signal and abs(mode.imag) > 1e-12:
            columns.append(np.conjugate(mode))
            pairs.append((index, True))
    basis = np.asarray(columns, dtype=np.complex128)
    n = np.arange(values.size)[:, None]
    design = basis[None, :] ** n
    solved, *_ = np.linalg.lstsq(design, values, rcond=None)
    fitted = design @ solved
    error = values - fitted
    residues = np.zeros(modes.size, dtype=np.complex128)
    shares = np.zeros(modes.size, dtype=np.float64)
    for column, (index, is_conjugate) in enumerate(pairs):
        if not is_conjugate:
            residues[index] = solved[column]
    for index in range(modes.size):
        take = [c for c, (i, _) in enumerate(pairs) if i == index]
        one = design[:, take] @ solved[take]
        shares[index] = float(np.var(one) / max(float(np.var(values)), 1e-30))
    # a conjugate pair sums to twice the real part, so the real sinusoid's
    # amplitude is twice the residue's magnitude
    scale = np.array([2.0 if (real_signal and abs(m.imag) > 1e-12) else 1.0
                      for m in modes])
    return {
        "residues": residues,
        "amplitude": np.abs(residues) * scale,
        "phase": np.angle(residues),
        "share": shares,
        "explained": float(1.0 - np.var(error) / max(np.var(values), 1e-30)),
    }


def poles_to_axes(frequencies_hz, decays_us, amplitudes
                  ) -> Dict[Axis, np.ndarray]:
    """One damped mode, on this arc's three axes.

    frequency  where the mode sits, in hertz
    time       its decay, in microseconds - a rate, which is a time
    amplitude  its residue's magnitude, which is what the pencil alone
               does not give
    """
    return {
        "frequency": np.asarray(frequencies_hz, dtype=np.float64),
        "time": np.asarray(decays_us, dtype=np.float64),
        "amplitude": np.asarray(amplitudes, dtype=np.float64),
    }


# A clipped sync tip damages the FALL and leaves the rise alone: the
# clipper flattens the bottom, so the falling edge arrives into a level
# that is no longer where the signal wanted to go, while the rise leaves
# that same flat floor with its own shape intact. Where clipping is
# detected the fall is therefore not a measurement, and everything that
# would have been referenced to it is referenced to the RISE instead.
#
# TWO detectors, because they can disagree and the disagreement is
# informative:
#   NOISE      the tip's own noise against the porch's. A clipper flattens
#              the tip, so its noise collapses. This is the established
#              test and the more direct one.
#   ASYMMETRY  the two edges' slopes differing. Weaker evidence, because
#              the edges land on DIFFERENT CARRIERS - the fall at the tip
#              carrier, the rise at blanking - and a demodulator's response
#              differs between them, so some asymmetry is expected with no
#              clipping at all.
CLIP_NOISE_RATIO = 0.75          # the established threshold
CLIP_ASYMMETRY_RATIO = 1.5       # slower edge over faster, before suspicion


def clipped_tip(noise_ratio=None, fall_slope_us=None, rise_slope_us=None
                ) -> Dict[str, object]:
    """Whether the sync tip was clipped, and on which evidence."""
    out: Dict[str, object] = {"by_noise": False, "by_asymmetry": False}
    if noise_ratio is not None and np.isfinite(noise_ratio):
        out["noise_ratio"] = float(noise_ratio)
        out["by_noise"] = bool(noise_ratio < CLIP_NOISE_RATIO)
    if (fall_slope_us and rise_slope_us
            and np.isfinite(fall_slope_us) and np.isfinite(rise_slope_us)):
        slower = max(float(fall_slope_us), float(rise_slope_us))
        faster = min(float(fall_slope_us), float(rise_slope_us))
        ratio = slower / max(faster, 1e-12)
        out["asymmetry_ratio"] = float(ratio)
        out["by_asymmetry"] = bool(ratio > CLIP_ASYMMETRY_RATIO)
    out["clipped"] = bool(out["by_noise"] or out["by_asymmetry"])
    out["use_fall"] = not out["clipped"]
    out["why"] = (
        "the tip's noise has collapsed: it was clipped flat, so the fall is "
        "not a measurement" if out["by_noise"] else
        "the edges differ by more than the carrier difference explains, so "
        "the fall is treated as unreliable" if out["by_asymmetry"] else
        "neither detector fires: the fall may be used")
    return out


def width_reference(fall_crossing, rise_crossing, clipped: bool
                    ) -> Dict[str, object]:
    """Where to reference the pulse from, given whether the tip was clipped.

    Unclipped, the pulse is what lies between its two edges. Clipped, the
    fall is not a measurement and the RISE alone carries the reference: the
    pulse's position is the rise's own crossing, and what is tracked is how
    that position varies rather than a width spanning a damaged edge.

    Referencing to a damaged edge does not merely add noise - a clipper
    moves the fall's crossing systematically, so a width built on it is
    biased, and a bias looks exactly like the calibration offset that is
    deliberately not corrected."""
    if not clipped:
        fall = np.asarray(fall_crossing, dtype=np.float64)
        rise = np.asarray(rise_crossing, dtype=np.float64)
        return {"reference": "width between both edges",
                "value": rise - fall, "uses_fall": True}
    rise = np.asarray(rise_crossing, dtype=np.float64)
    return {"reference": "the rise alone; the fall is clipped",
            "value": rise - np.nanmedian(rise), "uses_fall": False,
            "note": "a position relative to the population, not a width: "
                    "no width can be honest when one of its edges is not"}


def sync_width_correction(widths_us, reference_us=None) -> Dict[str, object]:
    """The per-line time-base correction the sync pulse's width implies.

    THE ABSOLUTE WIDTH IS NOT CORRECTED. A pulse that is uniformly narrow is
    the source's sync generator calibrated differently, and forcing it to the
    specification would impose a number on a machine that was simply built to
    another one. What is corrected is each line's departure from the
    population's own median, because a width that VARIES cannot be a fixed
    calibration - the same generator cannot be miscalibrated differently from
    one line to the next.

    The correction is a scale: a line whose pulse came out narrow was
    sampled fast there, so its time base wants stretching by the ratio of
    the median to its own width. The reference is the median by default,
    which is what leaves the absolute alone.

    Whether this SHOULD be applied depends on where the variation came
    from, and that is a separate measurement: if it does not follow the
    burst-locked time base, it arrived with the input signal rather than
    from the tape. It may still be worth correcting - a source that was not
    itself locked can carry real time-base error - but that is a decision
    about the recording, not a defect of this playback."""
    widths = np.asarray(widths_us, dtype=np.float64)
    good = np.isfinite(widths) & (widths > 0)
    if good.sum() < 8:
        return {"usable": False, "why": "too few measured widths"}
    reference = (float(np.median(widths[good])) if reference_us is None
                 else float(reference_us))
    scale = np.full(widths.shape, np.nan)
    scale[good] = reference / widths[good]
    return {
        "usable": True,
        "reference_us": reference,
        "scale": scale,
        "variation_rms": float(np.std(widths[good])),
        "variation_fraction": float(np.std(widths[good]) / max(reference, 1e-12)),
        "absolute_left_alone": True,
        "correction_rms_fraction": float(np.nanstd(scale[good] - 1.0)),
    }


def follows_time_base(width_deviation, burst_added, hsync_deviation
                      ) -> Dict[str, float]:
    """Whether a width variation follows the time base, or arrived with the
    input signal.

    The criterion is Ethan's: a width variation that does NOT correlate with
    what the burst lock had to do is not the tape's doing. Both witnesses
    are tested - what the burst lock added and the hsync's own period
    deviation - and the significance is scaled to the EFFECTIVE sample size,
    because consecutive lines are not independent draws."""
    from vhsdecode.models import head_model

    deviation = np.asarray(width_deviation, dtype=np.float64)
    out: Dict[str, float] = {}
    worst = 0.0
    for values, name in ((burst_added, "burst"), (hsync_deviation, "hsync")):
        other = np.asarray(values, dtype=np.float64)
        width = min(len(deviation), len(other))
        a, b = deviation[:width], other[:width]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 32 or np.std(a[ok]) <= 0 or np.std(b[ok]) <= 0:
            continue
        r = float(np.corrcoef(a[ok], b[ok])[0, 1])
        effective = head_model.effective_sample_size(a[ok])
        t = abs(r) * np.sqrt(max(effective - 2, 1) / max(1 - r * r, 1e-12))
        out["r_" + name] = r
        out["t_" + name] = float(t)
        worst = max(worst, t)
    out["follows_time_base"] = float(worst > 3.0)
    out["from_input_signal"] = float(worst <= 3.0)
    return out


def sync_pole_components() -> List[Component]:
    """The sync pulse's damped modes, as a PICTURE-stage component."""
    return [
        _component("sync-pulse damped modes",
                   "a matrix pencil on the sync pulse's settled aftermath, "
                   "per polarity: the fall rings against the tip carrier and "
                   "the rise against the blanking carrier. Each mode is this "
                   "arc's three axes at once - a frequency, a residue "
                   "amplitude and a decay time - so the pencil is the "
                   "parametric form of the same triple rather than a "
                   "different measurement. Fitted on DEMODULATED video, so "
                   "it belongs to the video path and not to either RF stage. "
                   "Below about 0.3 MHz use the nonparametric response and "
                   "never the poles: that is the witnessability bound the "
                   "pencil's own contract states",
                   ("amplitude", "frequency", "time"), linear=True,
                   stage=PICTURE, machine=RECORDING),
        _component("sync transition slope",
                   "each edge's 10-90 time against the specification's own "
                   "transition, as a parameter to fit rather than a curve. "
                   "Measured at the FIRST 90% crossing: the LAST crossing "
                   "swallows the ringing and reads the rise at more than "
                   "twice its slope. The two polarities are separate "
                   "parameters and differ in kind, not just degree - the "
                   "fall is the SLOWER edge (2.65x the ideal) and does not "
                   "ring at all, while the rise is faster (2.0x) and rings "
                   "for about 0.39 microseconds after it. The FALL needs the "
                   "previous line joined on, because the output line begins "
                   "AT the fall and that edge's top lies in the line before",
                   ("frequency", "time"), linear=True,
                   stage=PICTURE, machine=RECORDING),
        _component("sync-pulse width variation",
                   "the pulse's width, line by line, against the "
                   "population's own median - never against the "
                   "specification, because a uniformly narrow pulse is the "
                   "source's sync generator and not a defect. The time base "
                   "locks sync-to-sync SPACING, so it nulls that by "
                   "construction and never touches the pulse's own width, "
                   "which makes the width an independent witness. Measured "
                   "on real tape the variation does NOT follow the "
                   "burst-locked time base, so it arrived with the input "
                   "signal",
                   ("time",), linear=True, stage=PICTURE, machine=RECORDING),
    ]


def vertical_interval_components() -> List[Component]:
    """The vertical interval, as a low-frequency reference.

    A line lasts 63.5 microseconds, so nothing measured within one reaches
    below about 16 kHz; the whole per-line apparatus is blind there. The
    vertical sync block is a single excursion of roughly 190 microseconds
    and charges any low-frequency roll-off in the path, and the equalizing
    pulses on either side of it bracket the level it departs from. That
    makes the interval the only low-frequency probe the signal contains.

    It is a SOURCE-stage component: a droop of this kind is a video-domain
    effect ahead of modulation, and it is a CONSTANT of the path, applied
    without varying in time."""
    return [
        _component("vertical interval low-frequency response",
                   "the vertical sync block's charge on the path's low-"
                   "frequency roll-off, read by three instruments: the DROOP "
                   "within the block (weakest - the serrations dilute it), "
                   "the STEP in blanking level across it (a real roll-off "
                   "moves blanking and sync tip together; a gain change moves "
                   "only one), and the RECOVERY of the back porch over the "
                   "following lines, which reads the time constant directly "
                   "with no assumption about the block's depth and so sets "
                   "the reported corner",
                   ("amplitude", "frequency"), linear=True, stage=PICTURE,
                   machine=RECORDING),
    ]


# --------------------------------------------------------------------------
# The process
# --------------------------------------------------------------------------


class Gauges(Protocol):
    """The measurements and the feed-forward, supplied by the offline tools.

    Laws they must honour (the proposal's rules): feed forward all
    components together through the real chain and measure the residual on
    the CORRECTED output; converge TO the floor, never through it; the step
    size errs low (the half-amount law - the update diverges at twice the
    optimum); fit on the held-in lines, measure on the held-out; a residual
    with stable spectral or temporal structure on the held-out lines means
    a missing component, a structureless one means the floor; a residual on
    the sync edge that differs from one on a white transition is the
    level-dependence node, added, not cycled; the noise model is the
    constraint set, never the residual itself; a converged component is
    frozen and no longer differentiated; any linear residue appearing after
    the RF converges goes back to the RF loop, never into a baseband stage.
    """

    def measure(self, component: Component, signal) -> Dict[Axis, np.ndarray]:
        """The component's residual on each of its axes, against its ideal
        (or, on the frequency axis, against the channel baseline), measured
        on the held-out lines."""

    def at_floor(self, component: Component,
                 residual: Dict[Axis, np.ndarray]) -> bool:
        """True when the held-out residual is structureless and at the noise
        floor on every axis of the component - the exhaustion test."""

    def anti_residual(self, component: Component,
                      residual: Dict[Axis, np.ndarray]):
        """The residual IS the kernel to apply: the anti-residual, at the
        err-low step, gated by amplitude against the noise model."""

    def feed_forward(self, signal, anti):
        """Apply the anti-residual through the real chain - all components
        together - returning the corrected signal."""

    def cross_check(self, residuals: Dict[str, Dict[Axis, np.ndarray]]) \
            -> Dict[str, Dict[Axis, np.ndarray]]:
        """Reconcile the components' residuals against each other before
        any of them is accumulated.

        The components influence each other: the same physical departure
        shows in more than one of them, and a residual one component
        claims may belong to another. What they agree on is attributed to
        whichever component's criterion identifies it; what they disagree
        on stays where it was measured. Optional - a gauge set that does
        not implement it leaves every residual where it was measured."""

    def subtract_noise_floor(self, signal, floor: Dict[Axis, np.ndarray]):
        """Subtract the last component from the block (amplitude only)."""


@dataclass
class PassRecord:
    component: str
    index: int
    residual_rms: Dict[Axis, float]
    at_floor: bool


@dataclass
class Result:
    # THE RESULT IS THE MODELED SIGNAL: the accumulated expected signal
    # per component, which at the limit holds every shape the ideal was
    # missing. `signal` is the corrected data the last pass produced;
    # `model` is what the loop was for.
    signal: object
    anti_residuals: list
    differential_2d: Optional[np.ndarray]
    noise_floor: Optional[Dict[Axis, np.ndarray]]
    passes: List[PassRecord] = field(default_factory=list)
    weakly_determined: List[str] = field(default_factory=list)
    # the non-reducible residual per field (field index -> the floor on
    # each axis), and its differential between fields
    field_floor: Optional[np.ndarray] = None
    field_differential: Optional[np.ndarray] = None
    # the 3-D differential, where the floor was supplied on all three axes
    differential_3d: Optional[np.ndarray] = None
    # the accumulated expected signal per component - the MODEL, and the
    # loop's actual product - with the residual left at the limit
    model: Dict[str, Dict[Axis, np.ndarray]] = field(default_factory=dict)
    limit: Dict[str, float] = field(default_factory=dict)
    # which halves of the chain were run, in order
    stages: List[str] = field(default_factory=list)


def field_differential(field_floor: np.ndarray) -> np.ndarray:
    """The per-field dimension differentiated: the change of the
    non-reducible residual from one field to the next (axis 0 = field)."""
    return np.diff(np.asarray(field_floor, dtype=np.float64), axis=0)


# THE TERMINATION RULE. "The derivation is complete and perfect once a
# random noise begins to appear in the residual, and no amount of
# correction can remove it. We cannot estimate random noise, that is
# physically not possible, this process ends there."
#
# So completion is not a threshold on the residual's SIZE - a small
# residual with structure still has a component in it, and a large one
# without structure has none left to find. It is a test of whether any
# structure remains, and the only honest test of structure is whether it
# REPRODUCES on data that did not build it. Noise does not reproduce.
# Everything else here follows from that: a residual that reproduces has a
# component still to identify, and one that does not is the floor, which
# cannot be estimated and must not be modelled.
STRUCTURE_AGREEMENT = 0.3     # correlation on held-out data below which
                              # nothing is reproducing
SPECTRAL_FLATNESS_FLOOR = 0.5 # geometric over arithmetic mean of the power


def structureless(fitted: np.ndarray, held_out: np.ndarray) -> Dict[str, float]:
    """Whether a residual is random - the process's stopping condition.

    `fitted` and `held_out` are the same residual measured on two
    independent populations (the loop's even and odd lines). Three
    readings, of which the first is decisive:

      agreement   how well the held-out residual matches the fitted one.
                  Structure reproduces; noise does not. If this is at zero
                  there is nothing left to identify, whatever the size of
                  what remains.
      flatness    the spectrum's geometric mean over its arithmetic mean.
                  One means perfectly flat; a stable line or a shaped
                  roll-off pulls it down.
      lag one     serial correlation, which any smooth structure carries
                  and white noise does not.
    """
    first = np.asarray(fitted, dtype=np.float64).ravel()
    second = np.asarray(held_out, dtype=np.float64).ravel()
    width = min(first.size, second.size)
    first, second = first[:width], second[:width]
    good = np.isfinite(first) & np.isfinite(second)
    out: Dict[str, float] = {"population": float(good.sum())}
    if good.sum() < 16:
        out["complete"] = 0.0
        out["why"] = "too few samples to judge whether anything reproduces"
        return out
    first, second = first[good], second[good]
    if np.std(first) <= 0 or np.std(second) <= 0:
        out["complete"] = 0.0
        out["why"] = "a residual with no variation is not a measurement"
        return out
    agreement = float(np.corrcoef(first, second)[0, 1])
    power = np.abs(np.fft.rfft(first - first.mean())) ** 2
    power = power[1:]
    flatness = (float(np.exp(np.mean(np.log(power + 1e-30)))
                      / max(np.mean(power), 1e-30)) if power.size else 1.0)
    centred = first - first.mean()
    serial = float(centred[:-1] @ centred[1:] / max(centred @ centred, 1e-30))
    out.update({"agreement": agreement, "flatness": flatness,
                "lag1": serial})
    reproduces = abs(agreement) > STRUCTURE_AGREEMENT
    shaped = flatness < SPECTRAL_FLATNESS_FLOOR
    out["complete"] = float(not reproduces and not shaped)
    out["why"] = (
        "a component remains: the residual reproduces on lines that did not "
        "build it" if reproduces else
        "a component remains: the residual's spectrum is shaped, not flat"
        if shaped else
        "COMPLETE: nothing reproduces and the spectrum is flat. What is left "
        "is random noise, which cannot be estimated and must not be modelled")
    return out


# THE LOOP BACK. Random noise in ONE sample cannot be estimated, and that
# ends the derivation for that sample. It does not end the process. The
# 3-D residual taken over the SET of samples feeds back in, and the
# derivation continues on it until that residual reaches zero.
#
# The reason it works is the same square root that governs every other
# estimate here. A residual that is random within one field has no
# structure to remove there, but across N fields its mean has an error
# smaller by the square root of N, so structure lying BELOW one field's
# noise but above that reduced floor is still there to be found. The
# per-sample floor is a floor for one sample only; the ensemble's floor
# sits a factor of root-N beneath it, and the loop runs again against
# that. Each pass of the loop back lowers the floor it is measured
# against, and the process ends when the ENSEMBLE residual is
# structureless too - at which point nothing remains at any depth the data
# can reach.


def ensemble_residual(per_sample: np.ndarray) -> Dict[str, np.ndarray]:
    """The residual of the SET, from residuals that are each individually at
    their own noise floor.

    Returns the ensemble mean - the part that survived averaging, which is
    what the loop back derives on - together with the floor that mean is
    measured against, which is the single-sample scatter reduced by the
    square root of the count."""
    samples = np.asarray(per_sample, dtype=np.float64)
    if samples.ndim < 2:
        samples = samples[None, :]
    count = samples.shape[0]
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(samples, axis=0)
        scatter = np.nanstd(samples, axis=0)
    floor = scatter / np.sqrt(max(count, 1))
    return {"mean": mean, "floor": floor, "count": np.array(float(count)),
            "single_sample_floor": scatter}


def loop_back(per_sample: np.ndarray,
              held_out: Optional[np.ndarray] = None) -> Dict[str, float]:
    """One pass of the loop back: is there still structure in the SET?

    Splits the samples in half and asks whether the two halves' means agree.
    Structure common to the set reproduces between halves; what was only
    noise does not, and averaging has already taken it down by root-N. The
    process continues while they agree and ends when they do not."""
    samples = np.asarray(per_sample, dtype=np.float64)
    if samples.ndim < 2 or samples.shape[0] < 4:
        return {"complete": 0.0,
                "why": "the loop back needs at least four samples to split"}
    half = samples.shape[0] // 2
    first = np.nanmean(samples[:half], axis=0)
    second = np.nanmean(samples[half:], axis=0)
    verdict = structureless(first, second)
    summary = ensemble_residual(samples)
    ratio = float(np.nanstd(summary["mean"])
                  / max(float(np.nanmean(summary["floor"])), 1e-30))
    verdict["ensemble_over_floor"] = ratio
    verdict["gain_over_single_sample"] = float(np.sqrt(samples.shape[0]))
    # a mean that stands above its own reduced floor still holds something,
    # even where every individual sample was already structureless
    if ratio > 3.0 and verdict.get("complete", 0.0) == 1.0:
        verdict["complete"] = 0.0
        verdict["why"] = ("the SET still holds a component: its mean stands "
                          "%.1f times above the floor that averaging bought, "
                          "even though each sample alone was random"
                          % ratio)
    return verdict


# THE LIMIT THE WHOLE PROCESS RUNS TO, and why it must be BOUNDED.
#
# The loop removes a residual, remakes the model, and takes the change as a
# differential; iterating that has to stop somewhere, and where it stops is
# the noise floor of the RF sample set itself. That floor has a piece which
# is known exactly and a piece which is not:
#
#   the QUANTIZATION floor is arithmetic. An 8-bit capture has a step of
#   one LSB and a noise power of one twelfth of a step squared, spread flat
#   across the Nyquist band. Nothing about the tape or the machines enters
#   it - it is a property of the capture device's word length and rate, and
#   it is a HARD lower bound on any residual.
#   the AMBIENT floor - the capture's own analogue noise - is not known
#   without a separate measurement, so it can only be BOUNDED.
#
# That bound is not a nuisance: it is what says how many times the
# components need refining. Each pass takes the residual down by some
# factor, so the number of passes worth running is set by how far there is
# to fall - from where the residual starts to where the floor is. A floor
# known only within a range gives a pass count known only within a range,
# and running past it refines noise.
QUANTIZATION_NOISE_DIVISOR = 12.0     # a uniform step's variance is d^2/12


def quantization_floor(bits: int, sample_rate_hz: float,
                       full_scale: float = 256.0) -> Dict[str, float]:
    """The capture's own arithmetic noise floor.

    This is the one part of the limit that needs no measurement: a uniform
    quantizer of this word length has a noise variance of one twelfth of a
    step squared, and it is flat across the band because quantization is
    added at the converter, after everything analogue."""
    step = float(full_scale) / (2.0 ** int(bits))
    power = step * step / QUANTIZATION_NOISE_DIVISOR
    nyquist = float(sample_rate_hz) / 2.0
    return {
        "bits": float(bits),
        "step": step,
        "power": power,
        "rms": float(np.sqrt(power)),
        "density_per_hz": power / max(nyquist, 1e-12),
        "nyquist_hz": nyquist,
        # the dynamic range the word length allows at all
        "signal_to_noise_db": 6.02 * float(bits) + 1.76,
    }


def floor_differential(measured_power: float, bits: int,
                       sample_rate_hz: float,
                       full_scale: float = 256.0) -> Dict[str, float]:
    """The measured noise floor against the arithmetic one.

    What the difference means: quantization is the floor the capture could
    not go below, so anything above it is the capture's ANALOGUE noise plus
    whatever the chain contributed. Measured in a band the capture's
    anti-alias filter has emptied of signal, that excess IS the ambient
    noise of the capture - the baseline that otherwise needs a separate
    measurement - because nothing else can be there."""
    floor = quantization_floor(bits, sample_rate_hz, full_scale)
    measured = float(measured_power)
    excess = measured - floor["power"]
    return {
        "measured_power": measured,
        "quantization_power": floor["power"],
        "excess_power": excess,
        "ratio": measured / max(floor["power"], 1e-30),
        "excess_db": (10.0 * np.log10(max(excess, 1e-30)
                                      / max(floor["power"], 1e-30))
                      if excess > 0 else float("-inf")),
        "at_the_arithmetic_floor": bool(excess <= 0.1 * floor["power"]),
        "effective_bits": (float(bits)
                           - 0.5 * np.log2(max(measured, 1e-30)
                                           / max(floor["power"], 1e-30))),
    }


def refinement_bound(initial_residual: float, floor_low: float,
                     floor_high: float, per_pass_factor: float = 0.5
                     ) -> Dict[str, float]:
    """How many refinement passes the floor allows, given that the floor is
    known only within a range.

    The residual falls by a factor each pass, so the passes worth running
    are how many it takes to reach the floor. Because the floor is bracketed
    rather than known, the count is bracketed too - and the HIGH floor gives
    the SMALL count, which is the one to act on. Running to the low floor
    when the true floor is the high one spends passes fitting noise, and
    every pass past the floor is a pass that makes the model worse while the
    fitted residual keeps falling."""
    if not (0.0 < per_pass_factor < 1.0):
        raise ValueError("each pass must reduce the residual")
    start = float(initial_residual)
    logarithm = np.log(per_pass_factor)

    def passes(floor):
        floor = max(float(floor), 1e-30)
        if start <= floor:
            return 0.0
        return float(np.log(floor / start) / logarithm)

    low, high = passes(floor_high), passes(floor_low)
    return {
        "passes_minimum": low,           # against the HIGHER floor
        "passes_maximum": high,          # against the LOWER floor
        "passes_to_run": float(np.ceil(low)),
        "uncertainty_passes": high - low,
        "why": ("the floor is bracketed, so the pass count is too; run to the "
                "HIGHER floor, because passes beyond the true floor refine "
                "noise while the fitted residual still appears to fall"),
    }


# The luma and the colour-under are BAND LIMITED, so whatever sits outside
# their bands is noise - and it is noise measured in the same chain, at the
# same moment, beside the signal it bounds. That makes it a far better
# estimate of the in-band noise than anything measured elsewhere, and it is
# available in every capture without a separate run.
#
# TWO CAUTIONS, both measured on this capture:
#
#   USE A MEDIAN, NEVER A MEAN. The empty band is not empty of everything:
#   discrete interference lives there - on this rig a 12.0000 MHz crystal
#   with sidebands at exactly 50.0 kHz, the signature of a switching supply.
#   Those lines are capture-side, not tape noise, and they DOUBLE the mean
#   of the band they sit in (mean/median 2.03 there against 1.05 in clean
#   bands) while leaving the median untouched. A mean would read the
#   capture's own interference as the tape's noise.
#
#   CHOOSE THE BAND JUST BESIDE THE SIGNAL. Above the capture's anti-alias
#   roll-off only the converter's own noise remains; that measures the
#   CAPTURE, not the chain. The band immediately above the recorded
#   spectrum but below the roll-off still carries the head, the preamp and
#   the tape, and measures 13.8 dB above the capture's floor. On this
#   capture it agrees with the noise inside the signal band to 0.2 dB,
#   which is what makes it a valid stand-in for a floor that cannot be
#   measured under the signal directly.
OUT_OF_BAND_ESTIMATOR = "median"


def out_of_band_noise(psd, frequency_hz, signal_bands, capture_roll_off_hz,
                      nyquist_hz=None) -> Dict[str, float]:
    """The noise differential, from the bands the format leaves empty.

    `signal_bands` are (low, high) pairs the format actually occupies. The
    band between the highest of them and the capture's roll-off is the
    CHAIN's noise; the band above the roll-off is the CAPTURE's alone; and
    their difference is what the chain adds. Every estimate is a median, so
    discrete interference cannot inflate it."""
    power = np.asarray(psd, dtype=np.float64)
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    highest = max(high for _, high in signal_bands)
    chain_band = (frequency >= highest) & (frequency < float(capture_roll_off_hz))
    top = float(nyquist_hz) if nyquist_hz else float(frequency.max())
    capture_band = (frequency >= float(capture_roll_off_hz) * 1.15) & (frequency <= top)
    out: Dict[str, float] = {}
    if chain_band.sum() > 8:
        out["chain_noise"] = float(np.median(power[chain_band]))
        out["chain_mean_over_median"] = float(
            power[chain_band].mean() / max(out["chain_noise"], 1e-30))
    if capture_band.sum() > 8:
        out["capture_noise"] = float(np.median(power[capture_band]))
        out["capture_mean_over_median"] = float(
            power[capture_band].mean() / max(out["capture_noise"], 1e-30))
    if "chain_noise" in out and "capture_noise" in out:
        out["chain_over_capture_db"] = float(
            10.0 * np.log10(out["chain_noise"] / max(out["capture_noise"], 1e-30)))
        out["chain_binds"] = float(out["chain_noise"] > 2.0 * out["capture_noise"])
    for index, (low, high) in enumerate(signal_bands):
        inside = (frequency >= low) & (frequency < high)
        if inside.sum() > 8:
            out["band_%d_median" % index] = float(np.median(power[inside]))
    # a band whose mean stands well above its median is carrying discrete
    # lines, and those are a COMPONENT to identify rather than noise
    out["interference_suspected"] = float(
        out.get("chain_mean_over_median", 1.0) > 1.5)
    return out


# The colour burst is a pilot: the specification fixes its amplitude, its
# frequency and its phase, so whatever the received burst does that the
# specification does not is the colour carrier's own error. Measured AFTER
# up-heterodyning it describes the carrier the picture actually rides on.
#
# It is only present during the burst window, so a per-line estimate has to
# carry across the line - which is legitimate ONLY because the error is
# smooth. Measured on real tape the amplitude's lag-one across lines is
# +0.57 to +0.80, so neighbouring lines really do share it. A white error
# would make the burst estimate pure noise, and "correcting" with it would
# inject that noise into every line.
BURST_SMOOTH_LINES = 5           # over which the estimate is pooled
BURST_MINIMUM_CORRELATION = 0.3  # below this, do not carry it across


# THE FOUR-FIELD COLOUR SEQUENCE. Removing the standard's half-cycle-per-line
# alternation from the burst leaves a per-field CONSTANT, and that constant
# takes one of two values 180 degrees apart. Which one is the chroma phase
# polarity for that field, and the sequence of them is the colour framing.
#
# Measured on real tape it is `+ - - +` repeating: period FOUR, keyed to the
# field index, and not head-locked - the NTSC four-field sequence, intact.
# Mod 180 the constant holds to about 0.2 degrees, which is what makes it
# usable as a rotator's starting phase.
#
# What this measurement CANNOT settle is whether the sequence starts where
# the standard says. A locked pattern proves the sequence is intact, not that
# its first field is the standard's first field - and that offset is exactly
# what leaves a picture 180 degrees out. Deciding it needs a reference
# outside the burst.
COLOUR_SEQUENCE_FIELDS = 4


# COUPLED COMPONENTS. A carrier's frequency and the time base are not
# independent: a fractional timing error moves every frequency by that same
# fraction, so df/f = dt/t exactly. Fitting either alone lets it absorb the
# other, and the absorbed amount is invisible in its own residual.
#
# Measured on this tape: the colour under sits +5752 Hz above nominal in the
# RAW RF - 0.91% high, the tape running fast - and +0.87 to +3.56 Hz after
# the time base. The time base accounts for 99.94% of it. A colour-under
# frequency fitted without the timing would therefore have reported a
# heterodyne error a thousand times too large.
#
# The order follows the same rule as everything else here: TIMING FIRST,
# because the time base is measured from sync, which the colour under does
# not touch, so its estimate does not depend on the carrier's. The carrier's
# residual is then what the timing could not explain.
COUPLED_WITH_TIME_BASE = ("colour-under heterodyne frequency",
                          "chroma heterodyne scale",
                          "burst-to-sync phase lock")


def timing_implied_frequency_shift(carrier_hz: float,
                                   fractional_timing_error: float) -> float:
    """How far a fractional timing error moves a carrier.

    Exact, not approximate: a time base that runs fast by a fraction moves
    every frequency up by that same fraction. This is what must be removed
    from a measured carrier offset before any of it is called a heterodyne
    error."""
    return float(carrier_hz) * float(fractional_timing_error)


def separate_timing_from_carrier(measured_offset_hz: float,
                                 carrier_hz: float,
                                 fractional_timing_error: float,
                                 timing_error_uncertainty: float = 0.0
                                 ) -> Dict[str, float]:
    """Split a measured carrier offset into its timing and heterodyne parts.

    The timing part is not fitted - it is computed from a timing measurement
    made on sync, which is independent of the carrier. What remains is the
    heterodyne's own error, and it carries the timing measurement's
    uncertainty with it."""
    from_timing = timing_implied_frequency_shift(carrier_hz,
                                                 fractional_timing_error)
    remainder = float(measured_offset_hz) - from_timing
    uncertainty = abs(float(carrier_hz) * float(timing_error_uncertainty))
    return {
        "measured_hz": float(measured_offset_hz),
        "from_timing_hz": from_timing,
        "heterodyne_hz": remainder,
        "heterodyne_uncertainty_hz": uncertainty,
        "timing_share": (abs(from_timing) / abs(measured_offset_hz)
                         if measured_offset_hz else float("nan")),
        "resolved": float(abs(remainder) > 3.0 * uncertainty)
        if uncertainty > 0 else float("nan"),
    }


def chroma_framing(burst_phasors_per_field, field_parities=None
                   ) -> Dict[str, object]:
    """The chroma phase polarity per field, and the sequence it follows.

    `burst_phasors_per_field` is one array of per-line burst phasors for
    each field. Returns the polarity of each field, the constant it holds
    modulo 180 degrees, the period of the pattern, and whether that period
    is the standard's four."""
    polarities, constants = [], []
    for field in burst_phasors_per_field:
        phasors = np.asarray(field, dtype=np.complex128).ravel()
        lines = np.arange(len(phasors))
        # remove the standard's own alternation; what remains is the framing
        rotated = phasors * np.exp(-1j * np.pi * lines)
        angle = float(np.angle(np.nansum(rotated)))
        polarities.append(1 if abs(angle) < np.pi / 2 else -1)
        # modulo 180: the constant that does NOT depend on the polarity
        constants.append(0.5 * float(np.angle(np.exp(2j * angle))))
    polarity = np.array(polarities, dtype=int)
    constant = np.array(constants, dtype=np.float64)
    out: Dict[str, object] = {
        "polarity": polarity,
        "constant_radians": constant,
        "constant_degrees": float(np.degrees(np.median(constant))),
        "constant_coherence": float(np.abs(np.mean(np.exp(2j * constant)))),
    }
    period = None
    for candidate in (2, 3, 4, 5, 6, 8):
        if len(polarity) < 2 * candidate:
            continue
        if all(np.all(polarity[i::candidate] == polarity[i])
               for i in range(candidate)):
            period = candidate
            break
    out["period"] = period
    out["is_colour_sequence"] = float(period == COLOUR_SEQUENCE_FIELDS)
    if field_parities is not None and len(field_parities) >= len(polarity):
        parity = np.asarray(field_parities, dtype=bool)[:len(polarity)]
        if len(set(parity.tolist())) == 2:
            out["head_locked"] = float(
                np.all(polarity[parity] == polarity[parity][0])
                and np.all(polarity[~parity] == polarity[~parity][0]))
    out["locked"] = float(period is not None
                          and out["constant_coherence"] > 0.9)
    out["why"] = (
        "the framing follows the standard's four-field sequence and its "
        "constant holds: the reversal is locked" if out["is_colour_sequence"]
        and out["locked"] else
        "the polarity does not repeat on a fixed period: the reversal is not "
        "locked and a rotator started from it would drift"
        if period is None else
        "the polarity repeats on %s fields rather than the standard's four"
        % period)
    return out


def rotator_starting_phase(framing: Dict[str, object], field_index: int
                           ) -> Dict[str, float]:
    """The starting phase a chroma rotator should begin this field with.

    Two parts, and they are different in kind. The CONSTANT is a measured
    offset and is applied as measured. The POLARITY is a choice of one of
    two states 180 degrees apart, and this measurement fixes the SEQUENCE
    but not its origin - so it is returned separately, with the offset that
    would flip it, rather than folded silently into one number that hides
    which half is measured and which is convention."""
    polarity = np.asarray(framing.get("polarity", ()), dtype=int)
    if not len(polarity):
        return {}
    index = int(field_index) % len(polarity)
    sign = int(polarity[index])
    constant = float(np.median(np.asarray(framing["constant_radians"])))
    return {
        "constant_radians": constant,
        "polarity": float(sign),
        "starting_phase_radians": constant + (0.0 if sign > 0 else np.pi),
        "flip_radians": np.pi,
        "measured": "the constant and the sequence",
        "convention": "which field of the four is the first - not measured "
                      "here, and the half that decides a 180 degree error",
    }


# The luma and the colour under are separated by a front-end filter, and
# whatever that filter fails to separate appears in the other channel. The
# defect belongs to PLAYBACK: the recording put the two in their own bands,
# and it is the reading machine's filter that must keep them apart.
#
# The removal is mutual - each channel carries a little of the other - and it
# is COHERENT, not a notch: the coupling is estimated as a complex gain and
# that much of the other channel is subtracted, leaving everything else.
#
# IT IS GUARDED, and on this tape the guard declines. Fitting the colour
# under into the luma at its own carrier gives a coherence of 0.0145 - the
# coupling is not established, and subtracting an unestablished coupling
# injects the estimate's noise into a channel that did not have it. The
# framework is here for a decode where the coupling IS resolved.
CROSSTALK_MINIMUM_COHERENCE = 0.2


def crosstalk_coupling(source, victim) -> Dict[str, object]:
    """The complex coupling from one channel into the other, and whether it
    is established.

    Both arrays are the SAME measurement made on the two channels - the
    phasor at the carrier in question, line by line. The coupling is the
    least-squares complex gain, and the coherence says whether it describes
    the data or merely fits it."""
    a = np.asarray(source, dtype=np.complex128).ravel()
    b = np.asarray(victim, dtype=np.complex128).ravel()
    width = min(a.size, b.size)
    a, b = a[:width], b[:width]
    good = np.isfinite(a) & np.isfinite(b) & (np.abs(a) > 0)
    a, b = a[good], b[good]
    if a.size < 32:
        return {"established": 0.0, "why": "too few lines"}
    power = float(np.vdot(a, a).real)
    gain = np.vdot(a, b) / max(power, 1e-30)
    coherence = float(np.abs(np.vdot(a, b))
                      / np.sqrt(max(power * float(np.vdot(b, b).real), 1e-30)))
    residual = b - gain * a
    explained = float(1.0 - np.var(residual) / max(np.var(b), 1e-30))
    return {
        "gain": complex(gain),
        "magnitude": float(np.abs(gain)),
        "phase_degrees": float(np.degrees(np.angle(gain))),
        "coherence": coherence,
        "explained": explained,
        "lines": float(a.size),
        "established": float(coherence > CROSSTALK_MINIMUM_COHERENCE),
        "why": ("the coupling describes the victim channel"
                if coherence > CROSSTALK_MINIMUM_COHERENCE else
                "coherence %.4f: the coupling is not established, and "
                "subtracting it would add the estimate's own noise"
                % coherence),
    }


def remove_crosstalk(luma, chroma, into_luma, into_chroma,
                     amount: float = 0.5):
    """Subtract each channel's leakage from the other, where established.

    Mutual and simultaneous: the luma has the colour under taken out of it
    and the colour under has the luma taken out of it, each using its own
    measured coupling. A coupling whose guard has declined contributes
    nothing rather than contributing noise.

    At half strength by the standing law - the coupling's own error is
    invisible to the fit that produced it."""
    out_luma = np.asarray(luma, dtype=np.complex128).copy()
    out_chroma = np.asarray(chroma, dtype=np.complex128).copy()
    applied = {"luma": False, "chroma": False}
    if into_luma.get("established"):
        out_luma = out_luma - float(amount) * into_luma["gain"] * np.asarray(
            chroma, dtype=np.complex128)
        applied["luma"] = True
    if into_chroma.get("established"):
        out_chroma = out_chroma - float(amount) * into_chroma["gain"] * np.asarray(
            luma, dtype=np.complex128)
        applied["chroma"] = True
    return out_luma, out_chroma, applied


def crosstalk_components() -> List["Component"]:
    """The leakage, split by WHERE IT HAPPENED.

    Two different defects wear the same name and belong to different stages:

    PLAYBACK leakage is the reading machine's separation filter failing to
    keep two bands apart that the recording put in separate bands. Both
    signals are still present, each contaminating the other, so a
    correlation between the channels can find it and a subtraction can
    remove it.

    RECORD-TIME cross colour is different in kind. The recording VCR
    separated chroma from luma in the COMPOSITE domain and heterodyned the
    chroma down; whatever its decoder mistook for chroma was MOVED - taken
    out of the luma and written into the colour under. What is in the
    chroma is therefore what is MISSING from the luma, and no correlation
    between the two recorded channels can recover it, because a
    correlation sees what is present in both, not what was moved between
    them. That is why the measured coherence is 0.0005 to 0.041.

    What can be recovered is the SEPARATION FILTER that decided what moved,
    from the notch it left behind - and only then can the energy be put
    back where it came from."""
    return [
        _component("record-time cross colour",
                   "the recording VCR's own chroma decoder: what it mistook "
                   "for chroma in the composite input was MOVED out of the "
                   "luma and heterodyned down with the colour under. Not "
                   "recoverable by correlating the two recorded channels - "
                   "the energy is in one and absent from the other, and a "
                   "correlation finds only what is in both. Identified "
                   "instead from the separation filter's own notch, which "
                   "says what was moved and to where",
                   ("amplitude", "frequency"), linear=True,
                   stage=RF_RECORDING, machine=RECORDING),
        _component("luma / colour-under mutual crosstalk",
                   "what the front end's separation filter failed to keep "
                   "apart, each channel measured at the other's carrier and "
                   "subtracted coherently. A PLAYBACK defect: the recording "
                   "put the two in their own bands and the reading machine's "
                   "filter must keep them there. Guarded by coherence, "
                   "because subtracting an unestablished coupling injects "
                   "the estimate's own noise - on this tape the guard "
                   "declines at coherence 0.0145. This is the PLAYBACK "
                   "half only: both signals are still present and each "
                   "contaminates the other, which is what makes a "
                   "correlation able to see it at all",
                   ("amplitude", "frequency"), linear=True,
                   stage=RF_PLAYBACK, machine=PLAYBACK_MACHINE),
    ]


def burst_carrier_error(burst_phasors, alternation_cycles: float = 0.5
                        ) -> Dict[str, object]:
    """The colour carrier's amplitude and frequency error, from the burst.

    `burst_phasors` is one complex burst measurement per line. The
    specification's own half-cycle-per-line advance is removed first, so
    what remains is error rather than the standard.

    Returns the per-line amplitude ratio and phase residual, the frequency
    error their line-to-line change implies, and - decisively - how far
    the error is correlated between neighbouring lines. That correlation is
    what says whether the estimate may be carried across a line at all."""
    phasors = np.asarray(burst_phasors, dtype=np.complex128)
    if phasors.ndim == 1:
        phasors = phasors[None, :]
    amplitude = np.abs(phasors)
    reference = np.nanmedian(amplitude, axis=1, keepdims=True)
    ratio = amplitude / np.maximum(reference, 1e-30)
    lines = np.arange(phasors.shape[1])
    expected = 2.0 * np.pi * float(alternation_cycles) * lines
    phase = np.unwrap(np.angle(phasors) - expected[None, :], axis=1)
    phase = phase - np.nanmedian(phase, axis=1, keepdims=True)
    correlations = []
    for row in ratio:
        centred = row - np.nanmean(row)
        if np.nanstd(centred) > 0 and len(centred) > 8:
            correlations.append(float(np.corrcoef(centred[:-1], centred[1:])[0, 1]))
    carried = float(np.nanmedian(correlations)) if correlations else 0.0
    return {
        "amplitude_ratio": ratio,
        "phase_residual": phase,
        "amplitude_noise": float(np.nanmedian(np.nanstd(ratio, axis=1))),
        "phase_noise": float(np.nanmedian(np.nanstd(phase, axis=1))),
        "line_correlation": carried,
        "carries_across": float(carried > BURST_MINIMUM_CORRELATION),
        "why": ("the error is smooth between lines, so a per-line estimate "
                "describes its neighbours too"
                if carried > BURST_MINIMUM_CORRELATION else
                "the error is independent line to line: correcting with it "
                "would inject the estimate's own noise"),
    }


def burst_carrier_correction(error: Dict[str, object], amount: float = 0.5,
                             smooth_lines: int = BURST_SMOOTH_LINES):
    """The correction the burst error implies, as a complex gain per line.

    SMOOTHED before it is applied, because a per-line burst estimate carries
    the burst's own measurement noise as well as the carrier's error, and
    the error is the smooth part. Pooling over neighbouring lines keeps the
    error and drops the noise, which is only valid because the correlation
    above says the error IS smooth.

    Applied at half strength by the standing law: the model's own error is
    invisible to the fit, so a full step toward a slightly wrong optimum is
    worse than half a step toward it."""
    if not error.get("carries_across"):
        return None
    ratio = np.asarray(error["amplitude_ratio"], dtype=np.float64)
    phase = np.asarray(error["phase_residual"], dtype=np.float64)
    width = max(int(smooth_lines), 1)
    kernel = np.ones(width) / width

    def smooth(values):
        out = np.empty_like(values)
        for index, row in enumerate(values):
            filled = np.nan_to_num(row, nan=float(np.nanmedian(row)))
            padded = np.pad(filled, width, mode="edge")
            out[index] = np.convolve(padded, kernel, mode="same")[width:-width]
        return out

    gain = 1.0 / np.maximum(smooth(ratio), 1e-6)
    turn = -smooth(phase)
    blended_gain = 1.0 + float(amount) * (gain - 1.0)
    return blended_gain * np.exp(1j * float(amount) * turn)


def subtract_capture_noise(measured_power, capture_power,
                           measured_count: int = 0,
                           capture_count: int = 0) -> Dict[str, object]:
    """Take the capture's own noise out of a measured floor.

    Independent noise sources add in POWER, not amplitude, so a floor whose
    capture contribution is known is simply that contribution less:

        chain = measured - capture

    This is exact for uncorrelated sources, and quantization noise is
    uncorrelated with anything the tape or the heads produce.

    Two things it must not hide. Subtracting two noisy estimates gives a
    result NOISIER than either, so the error is reported and grows as the
    two approach each other. And the difference can come out NEGATIVE when
    the measured floor is at the capture's own, which is not a negative
    noise power but a measurement saying the chain adds nothing resolvable;
    it is clamped to zero and flagged rather than returned as a number."""
    measured = np.asarray(measured_power, dtype=np.float64)
    capture = np.asarray(capture_power, dtype=np.float64)
    chain = measured - capture
    negative = chain < 0
    clamped = np.where(negative, 0.0, chain)
    out: Dict[str, object] = {
        "chain_power": clamped if clamped.shape else float(clamped),
        "removed_fraction": float(np.mean(capture / np.maximum(measured, 1e-30))),
        "change_db": float(np.mean(
            10.0 * np.log10(np.maximum(clamped, 1e-30)
                            / np.maximum(measured, 1e-30)))),
        "exhausted": float(np.mean(negative)),
    }
    # the error of a difference of two averaged periodogram estimates: each
    # has a relative error of one over the root of its averages
    if measured_count > 0 and capture_count > 0:
        error = np.sqrt((measured / np.sqrt(measured_count)) ** 2
                        + (capture / np.sqrt(capture_count)) ** 2)
        out["error"] = error if error.shape else float(error)
        out["signal_to_error"] = float(np.mean(clamped / np.maximum(error, 1e-30)))
    if out["exhausted"]:
        out["warning"] = ("some bins measured at or below the capture's own "
                          "floor: the chain adds nothing resolvable there, "
                          "and a negative noise power is not a measurement")
    return out


def find_interference(psd, frequency_hz, band, line_rate_hz=0.0,
                      threshold: float = 6.0, background_bins: int = 401,
                      guard_hz: float = 400.0, separation_hz: float = 3e3):
    """Interference lines inside a band the signal also occupies.

    Outside the signal's bands a line is easy to spot. INSIDE them the
    signal is there too, so a line has to be distinguished by being
    NARROW against a local background and by NOT being a harmonic of the
    line rate - those harmonics are the signal's own structure, as an
    earlier pass discovered by nearly notching one out.

    The background is a wide running median, so a line is judged against
    the floor beside it rather than against the band's average, which its
    own presence would raise."""
    from scipy.ndimage import median_filter

    power = np.asarray(psd, dtype=np.float64)
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    inside = (frequency >= band[0]) & (frequency <= band[1])
    if inside.sum() < background_bins:
        return []
    f_in, p_in = frequency[inside], power[inside]
    background = median_filter(p_in, size=int(background_bins))
    ratio = p_in / np.maximum(background, 1e-30)
    found = []
    for index in np.argsort(-ratio):
        if ratio[index] < threshold:
            break
        centre = float(f_in[index])
        if line_rate_hz > 0:
            harmonic = round(centre / line_rate_hz)
            if harmonic >= 1 and abs(centre - harmonic * line_rate_hz) < guard_hz:
                continue                    # the signal's own line structure
        if any(abs(centre - other["frequency_hz"]) < separation_hz
               for other in found):
            continue
        found.append({"frequency_hz": centre,
                      "over_background": float(ratio[index]),
                      "power": float(p_in[index])})
    return sorted(found, key=lambda item: item["frequency_hz"])


def remove_interference(samples, sample_rate_hz, frequencies,
                        block: int = 1 << 16, refine: bool = True):
    """Subtract identified interference COHERENTLY from a signal.

    A notch removes whatever shares the bin, signal included. A coherent
    subtraction removes only what is genuinely periodic at that frequency:
    the line's amplitude and phase are estimated and that sinusoid alone is
    taken away, leaving everything else where it was.

    Estimated per BLOCK rather than once, because an oscillator drifts and
    a single amplitude fitted across a long record would subtract the wrong
    thing at both ends. The frequency is refined within each block by a
    local search, since a line even a fraction of a bin off would otherwise
    leave a residue as large as what was removed."""
    x = np.asarray(samples, dtype=np.float64).copy()
    rate = float(sample_rate_hz)
    removed = {float(f): 0.0 for f in frequencies}
    # THE FREQUENCY IS REFINED ONCE, over the whole record. Searching it
    # again in every block lets the search chase noise peaks and subtract
    # them: doing that perturbed unrelated signal by 3.9% where refining
    # globally costs a hundredth of that. A crystal's frequency does not
    # move between blocks; only its amplitude and phase need re-estimating.
    exact = []
    whole = np.arange(len(x))
    for target in frequencies:
        best = (0.0, float(target))
        if refine:
            step = rate / len(x)
            for k in range(-16, 17):
                f0 = float(target) + k * step
                strength = abs(x @ np.exp(-2j * np.pi * f0 * whole / rate))
                if strength > best[0]:
                    best = (strength, f0)
        exact.append(best[1])
    n_blocks = max(int(len(x) // block), 1)
    for b in range(n_blocks):
        lo = b * block
        hi = min(lo + block, len(x))
        if hi - lo < 1024:
            continue
        segment = x[lo:hi]
        n = np.arange(lo, hi)          # absolute index: the phase is continuous
        for target, f0 in zip(frequencies, exact):
            phasor = np.exp(-2j * np.pi * f0 * n / rate)
            amplitude = 2.0 * (segment @ phasor) / len(n)
            model = np.real(amplitude * np.exp(2j * np.pi * f0 * n / rate))
            segment = segment - model
            removed[float(target)] += float(np.sum(model ** 2))
        x[lo:hi] = segment
    return x, removed


def notch_interference(psd, frequency_hz, threshold: float = 8.0,
                       neighbourhood_hz: float = 200e3,
                       protect_bands=(), protect_harmonics_of: float = 0.0,
                       harmonic_tolerance_hz: float = 2e3):
    """Replace identified discrete lines with their local background.

    Interference is not noise: it is a narrowband line at a stable
    frequency, and the right treatment is to remove it rather than to
    average it in. Each bin is judged against the MEDIAN of its own
    neighbourhood, so a line is detected relative to the floor beside it
    and a sloping floor does not produce false positives.

    Returns the cleaned spectrum and where the lines were, because a line's
    frequency identifies its source - a round number is an oscillator, and
    a regular spacing about one is that oscillator's supply.

    NOT EVERY LINE IS INTERFERENCE, and this is where a notch does harm.
    Run unprotected on a real capture it flagged 78.7 kHz, which is exactly
    the fifth harmonic of the line rate - the RF envelope is modulated at
    line rate by the picture and by head switching, so that line is the
    SIGNAL's own structure. Removing it would delete what the measurement
    exists to find. So the signal's bands are protected, and so are the
    harmonics of any rate given in `protect_harmonics_of` (the line rate,
    normally). What remains eligible is only the genuinely empty spectrum."""
    power = np.asarray(psd, dtype=np.float64).copy()
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if len(power) < 16:
        return power, []
    step = float(np.median(np.diff(frequency))) if len(frequency) > 1 else 1.0
    width = max(int(neighbourhood_hz / max(step, 1e-9)), 3)
    def protected(hz):
        for low_edge, high_edge in protect_bands:
            if low_edge <= hz < high_edge:
                return True
        if protect_harmonics_of > 0:
            nearest = round(hz / protect_harmonics_of)
            if nearest >= 1 and abs(hz - nearest * protect_harmonics_of) <= harmonic_tolerance_hz:
                return True
        return False

    found = []
    for index in range(len(power)):
        if protected(float(frequency[index])):
            continue
        low = max(index - width, 0)
        high = min(index + width + 1, len(power))
        neighbours = np.concatenate([power[low:index], power[index + 1:high]])
        if len(neighbours) < 8:
            continue
        background = float(np.median(neighbours))
        if power[index] > threshold * background:
            found.append({"frequency_hz": float(frequency[index]),
                          "over_background": float(power[index] / max(background, 1e-30))})
            power[index] = background
    return power, found


def limit_reached(residual_per_pass: Sequence[float],
                  floor_low: float, floor_high: float,
                  window: int = 4) -> Dict[str, float]:
    """Whether the iteration has reached its limit, judged on the residual's
    SLOPE rather than on its size.

    A threshold on the residual's value needs to know the floor exactly, and
    the floor is only bracketed. Its SLOPE does not: the process is finished
    when the residual stops falling, and "stops" means the slope is
    indistinguishable from zero at the precision the RF capture chain
    affords. That precision is what the floor bracket supplies - the width
    between the arithmetic floor and the measured one is the margin, and a
    slope smaller than that margin per pass cannot be told from no slope.

    Judged over the LAST few passes only, because the early ones fall fast
    and would hide a flattening tail inside their own slope."""
    values = np.asarray(residual_per_pass, dtype=np.float64)
    values = values[np.isfinite(values)]
    out: Dict[str, float] = {"passes": float(len(values))}
    if len(values) < 3:
        out["reached"] = 0.0
        out["why"] = "too few passes to measure a slope"
        return out
    tail = values[-int(max(window, 3)):]
    index = np.arange(len(tail), dtype=np.float64)
    slope, intercept = np.polyfit(index, tail, 1)
    residual = tail - (slope * index + intercept)
    spread = float(np.std(residual, ddof=1)) if len(tail) > 2 else 0.0
    span = float(np.sum((index - index.mean()) ** 2))
    slope_error = spread / np.sqrt(max(span, 1e-12))
    # the margin the capture chain affords, per pass
    margin = abs(float(floor_high) - float(floor_low))
    out.update({
        "slope_per_pass": float(slope),
        "slope_error": float(slope_error),
        "capture_margin": margin,
        "current": float(values[-1]),
        "floor_low": float(floor_low),
        "floor_high": float(floor_high),
    })
    # flat when the fall per pass is inside the capture's own margin, or
    # inside the slope's own error - whichever is the weaker statement
    tolerance = max(margin, slope_error)
    out["flat"] = float(abs(slope) <= tolerance)
    out["below_floor"] = float(values[-1] <= float(floor_high))
    out["reached"] = float(out["flat"] or out["below_floor"])
    if out["below_floor"]:
        out["why"] = ("the residual is at or under the capture chain's own "
                      "floor: nothing below it is signal")
    elif out["flat"]:
        out["why"] = ("the residual's slope is %.3g per pass, inside the "
                      "%.3g the capture chain affords: it has stopped falling"
                      % (slope, tolerance))
    else:
        out["why"] = ("still falling at %.3g per pass, outside the %.3g "
                      "margin: keep going" % (slope, tolerance))
    return out


def _rms(values: Dict[Axis, np.ndarray]) -> Dict[Axis, float]:
    out: Dict[Axis, float] = {}
    for axis, value in values.items():
        array = np.asarray(value, dtype=np.complex128).ravel()
        finite = array[np.isfinite(array)]
        out[axis] = float(np.sqrt(np.mean(np.abs(finite) ** 2))) if finite.size \
            else float("nan")
    return out


def differential_2d(residual_frequency_time: np.ndarray) -> np.ndarray:
    """The 2-D differential of what remains at the floor: the (frequency x
    time) transform of the final residual."""
    return np.fft.fft2(np.asarray(residual_frequency_time, dtype=np.float64))


def differential_3d(residual_amplitude_frequency_time: np.ndarray) -> np.ndarray:
    """The 3-D differential between amplitude, frequency and time, where
    all three are available: the transform over the three axes of the
    complex residual indexed (amplitude, frequency, time) - the landing
    level, the frequency bin and the decay lag of the kernel table."""
    return np.fft.fftn(np.asarray(residual_amplitude_frequency_time,
                                  dtype=np.complex128), axes=(0, 1, 2))


def per_head_average(fields: np.ndarray,
                     parities: Sequence[bool]) -> Dict[bool, np.ndarray]:
    """The average of EACH HEAD's signal, kept apart.

    Averaging within a head is what raises the signal-to-noise ratio, and
    it is legitimate because one head laid all of those fields down with
    one response. Averaging ACROSS heads is what destroys that response,
    replacing two different paths with a mean belonging to neither. The
    two heads are independent - their field-to-field wanderings do not
    correlate - so the average is taken per head and the differential is
    then run on each, matching the per-head response rather than
    averaging it away.

    The fields share the final time base, so a sample is comparable
    across them and the average is coherent: noise falls as the square
    root of the count while the response does not."""
    fields = np.asarray(fields, dtype=np.float64)
    parities = np.asarray(parities, dtype=bool)[:len(fields)]
    out: Dict[bool, np.ndarray] = {}
    for head in (True, False):
        chosen = fields[:len(parities)][parities == head]
        if len(chosen):
            out[head] = chosen.mean(axis=0)
    return out


def per_head_differential(blocks: Dict[bool, np.ndarray]
                          ) -> Dict[bool, np.ndarray]:
    """The same amplitude, frequency and time differential, run on each
    head's own averaged signal. One response per head, never a pooled
    one."""
    return {head: differential_3d(block) for head, block in blocks.items()}


# The head difference is a CONSTANT: it is fixed hardware, so once
# measured it is held rather than re-estimated. But its SCOPE is not
# global. If it lived only in the playback deck it would be identical on
# every tape that deck plays; measured on two tapes played back on the
# same deck, the two head-difference vectors do not correlate at all
# (r = +0.11, t = 0.8 on 54 dof), and bars resolves no head difference
# while home resolves 11x its error at DC. So the dominant part is
# RECORDING-side, and the constant is held PER RECORDING - measured once
# for a tape and reused for that tape, never carried across tapes. No
# shared playback component is resolved above the split error.
HEAD_DIFFERENCE_SCOPE = "per recording"


def head_response_difference(responses: Dict[bool, np.ndarray]
                             ) -> Optional[np.ndarray]:
    """What separates the two heads: the difference of their differentials,
    which is the part of the response that belongs to the head rather than
    to the path they share.

    It is fixed hardware and does not vary over time, so it is measured
    once and held as a constant - see `HEAD_DIFFERENCE_SCOPE` for how far
    that constant may be carried. Measured on real tape the difference is
    concentrated at the LOW end, 2.5x stronger below 1.2 MHz than above
    3.5 MHz and 11x its error at DC, which is the per-head DC response
    rather than a broadband gain."""
    if len(responses) < 2:
        return None
    first, second = responses[True], responses[False]
    width = min(first.shape, second.shape)
    cut = tuple(slice(0, n) for n in width)
    return np.asarray(first[cut]) - np.asarray(second[cut])


def differential_4d(residual_field_amplitude_frequency_time: np.ndarray
                    ) -> np.ndarray:
    """The SAME amplitude, frequency and time differential, run over
    multiple fields - the field carried as a fourth axis rather than
    averaged away.

    Averaging the parameters over time throws out the very thing the
    per-field dimension exists to hold: a residual that is not reducible
    within one field, and whose CHANGE from field to field is its own
    measurement. So the fourth axis is differentiated like the other
    three, not collapsed. The array is indexed (field, amplitude,
    frequency, time), and what comes back is the joint transform: a
    component sitting at one field frequency is a departure that repeats
    on a field cadence - head parity at one half cycle per field, the
    colour sequence at a quarter - and one at zero is the part that does
    not vary over time at all."""
    return np.fft.fftn(np.asarray(residual_field_amplitude_frequency_time,
                                  dtype=np.complex128), axes=(0, 1, 2, 3))


def field_cadence(count: int) -> Dict[str, float]:
    """Where the known field-rate cadences land on the fourth axis, in
    cycles per field: head parity alternates every field, and the colour
    sequence repeats every four. A peak at one of these is structure the
    recording's own geometry explains; a peak elsewhere is not."""
    return {"head parity": 0.5, "colour sequence": 0.25,
            "constant over time": 0.0,
            "bin width": 1.0 / max(int(count), 1)}


# THE THREE COMPLEX COMPONENTS to iterate on - one per axis, each complex
# where the decode supplies both halves, so the 3-D differential can be
# taken wherever all three exist:
#   amplitude  ln(envelope) + j * (the phase the amplitude disturbance
#              converts to through the RF band-pass's asymmetry: the
#              head-switch stage's measured AM-to-PM transfer)
#   frequency  ln|H| + j * arg H from the sync edges (the sync-step
#              response, per polarity at its landing carrier)
#   time       the hsync's per-line timing deviation (real part) + j * the
#              color burst's per-line phase deviation, in samples (the two
#              timing pilots: coarse and fine, luma and chroma)
# The luma supplies amplitude and frequency, the chroma supplies the time
# component's imaginary half; where a component lacks its imaginary half
# the differential on that axis is taken on the real part alone.
COMPLEX_COMPONENTS: Dict[Axis, str] = {
    "amplitude": "ln envelope + j AM-to-PM phase",
    "frequency": "ln|H| + j arg H (sync edges)",
    "time": "hsync deviation + j burst-phase deviation (samples)",
}


def complex_time_residual(hsync_deviation_samples, burst_phase_rad,
                          subcarrier_hz: float,
                          sample_rate_hz: float) -> np.ndarray:
    """The time component as one complex residual per line: the hsync's
    timing deviation as the real part, the burst's phase deviation as the
    imaginary part, both in samples of the grid the lines are stated on."""
    hsync = np.asarray(hsync_deviation_samples, dtype=np.float64)
    phase = np.asarray(burst_phase_rad, dtype=np.float64)
    burst_samples = phase / (2.0 * np.pi * float(subcarrier_hz)) \
        * float(sample_rate_hz)
    return hsync + 1j * burst_samples


# The pass count at which continued movement means a weakly determined
# node or a model error the graph is chasing (the proposal's rule).
MAXIMUM_PASSES = 5


def multidimensional_information_extrapolation(
        signal, components: Sequence[Component], gauges: Gauges,
        noise: Optional[Component] = None,
        maximum_passes: int = MAXIMUM_PASSES) -> Result:
    """Iterate to the LIMIT of the residual, accumulating the expected
    signal; the modelled signal is the result.

    Ethan's formulation, which this implements exactly. All the components
    influence each other, so they are measured TOGETHER on every pass, not
    one to convergence and then the next. Each pass:

      1. the current model - every component's accumulated expected
         signal, all of them together - is fed forward through the real
         chain, and the residual of each component is measured against it;
      2. the residuals are CROSS-CHECKED against each other, because the
         same departure shows in more than one component and what they
         agree on belongs to whichever criterion identifies it;
      3. each residual is added back into its own component's expected
         signal - "the residual difference is accumulated in the expected
         signals";
      4. a component whose residual has reached its limit is frozen and no
         longer differentiated over.

    The iteration is done when the limit is reached, the residual
    approaching zero, and what has accumulated by then is the shape the
    specification's ideal was missing. THAT accumulation - the modelled
    signal - is the result; the correction it implies is its departure
    from the ideal, applied ONCE rather than in fractions along the way.

    The noise floor is the last component, measured on what remains,
    amplitude only. The three complex components (`COMPLEX_COMPONENTS`)
    are iterated on wherever the decode supplies both halves, and the 3-D
    differential is taken at the floor where all three exist.
    """
    result = Result(signal=signal, anti_residuals=[], differential_2d=None,
                    noise_floor=None)
    best = None       # (total residual, signal, model) of the best pass
    # THE STAGES RUN IN ORDER, and the residuals propagate through the
    # whole chain: within a stage every component is measured together and
    # cross-checked against the whole ensemble, so refining one is
    # differentiated against all of them. A stage is finished only when
    # every component in it is frozen, and the next stage then measures on
    # a signal the finished one has already corrected - which is what
    # makes the SOURCE stage's residual the recording's own, rather than
    # the playback path's leaking into it.
    applied_before = False
    for stage in STAGES:
        in_stage = [c for c in components if c.stage == stage]
        if not in_stage:
            continue
        result.stages.append(stage)
        if applied_before:
            # THE PREVIOUS STAGE'S MODEL IS APPLIED BEFORE THIS ONE BEGINS.
            # A stage whose components all reach their limit on the first
            # pass leaves its loop without ever feeding forward, and this
            # stage would then measure on a signal that correction never
            # touched. In this chain that is not a small error: the
            # recording machine's stage would still see the PLAYBACK
            # machine's contribution and attribute it to the recording
            # machine - the exact confound that running the stage twice
            # exists to avoid. It is done BETWEEN stages only, so the
            # signal the final result carries is still the one the last
            # measurement saw.
            result.signal = gauges.feed_forward(result.signal, result.model)
        _run_stage(result, in_stage, gauges, maximum_passes)
        applied_before = True
    if noise is not None:
        _measure_noise_floor(result, gauges, noise)
    return result


# Successively fitting and removing one component at a time from the
# residual is MATCHING PURSUIT, and where the components are damped
# sinusoids it is the greedy form of Prony. But this is not the blind
# version of it. Matching pursuit must SEARCH a dictionary at every step
# for whichever atom currently explains the most; here the components are
# the actual stages of the VCR's circuitry, and the sequence in which
# they acted on the signal is known in advance. The graph is traversed by
# knowing which nodes connect ahead of time, so no search is needed and
# the ordering is not a guess.
#
# That known order changes what orthogonalization means. Greedy
# subtraction is exact only while the components are orthogonal: where
# two respond to the same physical departure, each takes credit for all
# of it and the pair is removed twice. Measured on real tape, the set is
# near-orthogonal on a test pattern (largest coherence 0.27, the sync
# fall against the sync rise) but NOT on real video, where the back porch
# and the burst-to-sync phase reach 0.70 - both moved by the same
# low-frequency droop the vertical interval measures.
#
# Because the order is known, each component is orthogonalized against
# those UPSTREAM of it only. A stage that acted earlier cannot have been
# caused by a later one, so the correction is one-directional: it is
# Gram-Schmidt down the chain rather than a symmetric solve. Stages are
# separated for the same reason - each carries its own set of frequency
# responses and its own variables - which the stage loop already
# guarantees by running them in order.
# WHICH FORM, settled by measurement rather than assumption. Both were
# implemented properly and compared against a planted truth, 600 trials
# per point, and neither is universally better - it depends on HOW the
# coupling arises:
#
#   coupling is one SHARED departure both gauges project  -> symmetric
#   coupling is upstream CONTAMINATING a downstream gauge -> ordered
#
# The failure modes are not symmetric, though. Ordered beats plain greedy
# at EVERY mixture in the range; symmetric is worse than doing nothing
# below about 40% shared, and at 0% shared it is three times worse than
# ordered. Measured on real tape by partial correlation (the low-frequency
# droop as the candidate shared driver, per line where the power is):
# home is 52% shared, and bars has no per-line coupling at all to
# attribute. At 52% ordered wins, but only by 11% - the crossover sits
# near 65% shared. So the ordered form is the default because it is the
# robust one, not because it is better everywhere; `orthogonalize_
# symmetric` is kept for a path that measures above the crossover.
#
# Both forms filter strictly within the span of the components they are
# given. Neither can invent structure outside the modelled bands, which
# is the property that makes either safe to leave on.
COHERENCE_THRESHOLD = 0.3     # below this, greedy and joint agree anyway
ORTHOGONAL_RIDGE = 1e-3       # conditions the Gram solve
ORTHOGONAL_AMOUNT = 0.5       # the standing law: apply at half the optimum


def coherence(residuals: Dict[str, Dict[Axis, np.ndarray]],
              axis: Axis) -> float:
    """The largest coherence between any two components on one axis: the
    quantity that decides whether greedy subtraction is safe."""
    worst = 0.0
    for _, directions in _grouped_directions(residuals, axis,
                                             list(residuals)):
        if len(directions) < 2:
            continue
        gram = np.abs(directions @ directions.conj().T)
        np.fill_diagonal(gram, 0.0)
        if gram.size:
            worst = max(worst, float(gram.max()))
    return worst


def axes_present(residuals: Dict[str, Dict[Axis, np.ndarray]]) -> List[Axis]:
    """Every axis any component carries a residual on, in declared order.

    Not `AXES`: that tuple omits `FIELD_AXIS`, so iterating it silently
    excluded the per-field dimension from every projection - a dimension
    could be measured and accumulated and still contribute nothing to the
    descent, because nothing was ever differentiated against it.
    """
    seen = []
    for per_axis in residuals.values():
        for axis in (per_axis or {}):
            if axis not in seen:
                seen.append(axis)
    ordered = [a for a in AXES if a in seen]
    ordered += [a for a in (FIELD_AXIS,) if a in seen]
    ordered += [a for a in seen if a not in ordered]
    return ordered


def _grid_key(value, name, axis, grids):
    """What makes two residuals projectable onto one another.

    An inner product between two vectors is only a measurement if they are
    sampled on the SAME abscissa. Two residuals on different grids have no
    common basis, and the number that comes back from dotting them is an
    artefact of how they happened to be laid out in memory.

    The key is the declared grid where one is given, and the vector's own
    length otherwise. Length alone is necessary, not sufficient - two
    different quantities can share a length - so a component that knows its
    abscissa should declare it through `Residual.grid`.
    """
    declared = (grids or {}).get(name) if grids else None
    if isinstance(declared, dict):
        declared = declared.get(axis)
    if declared is not None:
        declared = np.asarray(declared).ravel()
        if declared.size:
            return (axis, declared.size,
                    float(declared[0]), float(declared[-1]))
    return (axis, int(np.asarray(value).size))


def _stack_real(value):
    """A complex vector as a REAL vector twice as long.

    When two signatures are the SAME SHAPE but one is pure amplitude and
    the other pure phase - a gain change against a delay - the Hermitian
    inner product reads them as one direction, because <r, jr> has
    magnitude |r|^2. That is correct when the complex value IS the
    measurement, which is the case for a measured RF residual: amplitude
    and phase there are two parts of one quantity.

    It is WRONG when each component's parameter is a real physical
    quantity and the complex signature is that parameter's effect, because
    then a gain and a delay are two different mechanisms that happen to
    share a magnitude profile. Stacking the real and imaginary parts makes
    them orthogonal, which they are.

    Measured: on a set of propagation mechanisms the Hermitian form
    returned 2.20 effectively distinguishable and the stacked form 3.79.
    The tape's mechanisms read 1.58 under both, because their signatures
    are real, so a comparison across the two is still fair.
    """
    value = np.asarray(value).ravel()
    if not np.issubdtype(value.dtype, np.complexfloating):
        return value.astype(np.float64, copy=False)
    return np.concatenate([value.real, value.imag])


def _grouped_directions(residuals, axis, order, grids=None, real=False):
    """The projectable groups on one axis: components that share a grid,
    IN CHAIN ORDER, with their unit directions.

    Complex is carried through. `COMPLEX_COMPONENTS` declares all three
    axes complex - amplitude as ln envelope plus j AM-to-PM phase,
    frequency as ln|H| plus j arg H, time as the hsync deviation plus j the
    burst-phase deviation - and casting those to float64 silently keeps the
    real part and throws the phase away. That defeated the complex design
    wherever a residual passed through here.
    """
    collected = {}
    for name in order:
        value = (residuals.get(name) or {}).get(axis)
        if value is None:
            continue
        value = np.asarray(value).ravel()
        if not np.issubdtype(value.dtype, np.complexfloating):
            value = value.astype(np.float64, copy=False)
        if value.size == 0 or not np.isfinite(value).any():
            continue
        key = _grid_key(value, name, axis, grids)
        if real:
            value = _stack_real(value)
        names, vectors = collected.setdefault(key, ([], []))
        names.append(name)
        vectors.append(value)
    groups = []
    for names, vectors in collected.values():
        if len(vectors) < 2:
            continue
        stack = np.array([np.nan_to_num(v) for v in vectors])
        norms = np.linalg.norm(stack, axis=1, keepdims=True)
        keep = norms.ravel() > 0
        if keep.sum() < 2:
            continue
        groups.append(([n for n, k in zip(names, keep) if k],
                       stack[keep] / norms[keep]))
    return groups


def _directions(residuals, axis, order, grids=None):
    """The single largest projectable group, for callers wanting one basis."""
    groups = _grouped_directions(residuals, axis, order, grids)
    if not groups:
        return [], np.zeros((0, 0))
    return max(groups, key=lambda g: len(g[0]))


# How far above the random-ensemble floor an asymmetry must stand before it
# counts as a direction rather than as scatter. Three sigma, the same bar
# every other admission in this arc is held to.
SPHERE_SIGMA = 3.0


def sphere_floor(count: int, length: int) -> Dict[str, float]:
    """How circular a purely random ensemble of this size looks.

    Ethan's terminating condition is "down to the circular shape of the
    complex signal": structure gives the ellipsoid a preferred direction,
    and noise has none, so the floor is reached when the shape is a sphere.
    A FINITE ensemble of random directions is never exactly spherical, so
    the threshold has to be derived rather than chosen.

    Both quantities are closed form, which is what keeps this rule free of
    a fitted constant. With `trace(G) = N` and
    `trace(G^2) = N + sum_{i != j} |G_ij|^2`, and `E|G_ij|^2 = 1/L` for
    random unit directions:

        participation = N^2 / (N + N(N-1)/L)
        asymmetry     = (N - participation) / (N - 1) = N / (L + N - 1)

    Checked against simulation over N in {3..32} and L in {64..1024}, real
    and complex: the ratio of measured to predicted sits at 0.88 to 1.18
    with no trend. The scatter about it measures as sqrt(2)/L, flat in N -
    which is why **adding components sharpens the test**: the floor rises
    with N while its scatter does not, so the margin a real asymmetry has
    to clear becomes better determined the more components are brought in.
    That is Ethan's "increase the constant, how many components, to know
    exactly when you have reached the end".
    """
    count = max(int(count), 1)
    length = max(int(length), 1)
    return {
        "asymmetry": count / (length + count - 1.0),
        "scatter": float(np.sqrt(2.0)) / length,
        "count": float(count),
        "length": float(length),
    }


def ellipsoid(residuals: Dict[str, Dict[Axis, np.ndarray]], axis: Axis,
              floor: Optional[float] = None,
              removed: int = 0,
              real_parameters: bool = False) -> Dict[str, object]:
    """THE COLLAPSE. Ethan's closed form for the whole differentiation.

    > The shapes of the dimensions are asymmetric, and we can extrapolate
    > how symmetric they are based on their fit to our 3 dimensional
    > elliptical plane which represents the final result. The constant is
    > the elipse. All I need is to fit these residuals to an elipse of how
    > ever many dimensions I have, and I have fully collapsed the
    > differentiation.

    The recursion of matched-set differentials converges on the eigenbasis
    of the ensemble's second moment. Solving that eigenproblem reaches the
    same place in ONE step, which is what makes this a collapse rather than
    an acceleration: no passes, no order, no half-steps.

    The components' unit directions form `G = D D^H`, the quadratic form
    whose level set is the ellipsoid. Its eigenvectors are the independent
    dimensions and its eigenvalues are the squared semi-axes.

    **The constant is the ellipse, exactly.** Every row of `D` is a unit
    vector, so `trace(G) = N` whatever the components do: perfectly
    independent components give N eigenvalues of 1 - a sphere - and
    perfectly collinear ones give a single eigenvalue of N and the rest
    zero. The total is conserved and only the SHAPE carries information,
    which is why the asymmetry is the measurement and the ellipse is the
    invariant.

    `rank` is how many principal axes stand above the floor, and that is
    the number of dimensions the evidence actually supports - the same
    question the pass-by-pass recursion answers slowly.

    `real_parameters` selects which inner product is the physical one, and
    the choice is decided by WHOSE PHASE IT IS.

    **The default is blind to an unknown absolute phase, and that is why it
    is the default for measured components.** The Gram is built from
    `|<u_i, u_j>|`, so multiplying a component by any global phase factor
    leaves it unchanged: measured on twelve components carrying one shared
    departure, aligning their phases or randomising them gives +167.8 and
    +168.0 sigma - a difference of two tenths. A simple FFT is therefore
    sufficient for a component whose absolute phase nobody knows, because
    the complex bin carries frequency and phase together as ONE dimension
    and the only thing the transform asks of it is its direction.

    **Setting it tells a gain from a delay, and costs that blindness.**
    Stacking real and imaginary parts makes `r` and `j r` orthogonal, which
    is correct when each component is a REAL physical parameter's signature
    - a gain is not a delay. But a rotated copy then reads as a different
    direction, and the same experiment costs 39.4 sigma when the phase is
    unknown rather than 0.2.

    So the two modes are complementary and the split is the one the method
    already draws elsewhere:

      MODELLED signatures have a phase WE CHOSE when we wrote them, so they
      take `real_parameters=True` and a gain is distinguishable from a delay;

      MEASURED components have a phase NOBODY KNOWS, so they take the
      default and the transform sees them through it.

    Getting this backwards is not a small error: it either throws away the
    gain-versus-delay distinction on the modelled key, or it lets an unknown
    phase masquerade as a missing direction in the measured residual.
    """
    groups = _grouped_directions(residuals, axis, list(residuals),
                                 real=real_parameters)
    if not groups:
        return {"names": [], "rank": 0, "semi_axes": np.zeros(0),
                "axes": np.zeros((0, 0)), "asymmetry": 0.0,
                "trace": 0.0, "sphere": True}
    names, directions = max(groups, key=lambda g: len(g[0]))
    count = len(names)
    gram = directions @ directions.conj().T
    gram = 0.5 * (gram + gram.conj().T)          # Hermitian by construction
    values, vectors = np.linalg.eigh(gram)
    values = np.clip(np.real(values)[::-1], 0.0, None)
    vectors = vectors[:, ::-1]
    # the floor: an eigenvalue below it is a direction the evidence does
    # not resolve. Given none, take the level a single component's own
    # numerical noise reaches, scaled by the ensemble size.
    if floor is None:
        floor = float(count) * float(np.finfo(np.float64).eps) ** 0.5
    rank = int(np.count_nonzero(values > floor))
    # WHICH directions are real. Under pure noise the Gram's spectrum
    # reaches the Marchenko-Pastur upper edge (1 + sqrt(N/L))^2 and no
    # further, so an eigenvalue above it is a direction and one below it is
    # scatter. Checked against simulation over N in {4..16}, L in
    # {64..1024}: the measured largest eigenvalue of a noise ensemble sits
    # at 0.83 to 0.96 of the edge and approaches it as L grows, so the edge
    # is a conservative bound - it never admits noise as a direction.
    length = int(directions.shape[1])
    mp_edge = (1.0 + np.sqrt(count / max(length, 1))) ** 2
    # (the edge uses the ENSEMBLE size, which is what sets the
    # spectrum's width; the shape below is judged on what survives)
    significant = int(np.count_nonzero(values > mp_edge))
    # How far the ellipsoid is from the sphere, on [0, 1]: zero when every
    # direction carries the same, one when a single direction carries all.
    #
    # Judged on the directions that REMAIN, which is what `removed` counts.
    # Deflating a direction drawn from the ensemble's own span leaves an
    # exact zero eigenvalue - the set then spans one dimension fewer - and
    # counting that zero as an unequal share reports the deflation's own
    # bookkeeping as structure. Measured: one planted departure removed
    # cleanly (0.892 alignment with the planted vector) read 0.224 against
    # a 0.023 floor with the zero counted, and 0.019 against a 0.019 floor
    # without it, which is the correct verdict of spherical.
    #
    # `removed` and not "drop the zeros": a genuine needle - N views of one
    # departure - also has N-1 zero eigenvalues, and that IS maximal
    # asymmetry rather than an artefact. Only the caller knows how many
    # directions it has taken out.
    effective = max(int(count) - max(int(removed), 0), 1)
    live = values[:effective]
    total = float(live.sum())
    share = live / total if total > 0 else live
    participation = (float(1.0 / np.sum(share ** 2))
                     if np.any(share) else float(effective))
    asymmetry = (float((effective - participation) / (effective - 1))
                 if effective > 1 else 0.0)
    # "Down to the circular shape of the complex signal": the floor is
    # reached when what is left is as spherical as a random ensemble of
    # this size and length would be.
    circle = sphere_floor(max(effective, 1), length)
    margin = ((asymmetry - circle["asymmetry"]) / circle["scatter"]
              if circle["scatter"] > 0 else 0.0)
    # Ethan: "the total eigenvalue is the total amount of information we
    # could possible derive from our components. I.e. this is multi
    # component radio tuning." Exactly so, and it is the invariant above:
    # every component contributes one unit to the trace whatever it
    # carries, so N components hold N units of derivable information and no
    # more. What the fit adds is HOW MUCH of that budget stands in real
    # directions - the eigenvalue mass above the noise bulk - against how
    # much is spread as scatter.
    quiet_bulk = values[values <= mp_edge]
    bulk_level = float(np.median(quiet_bulk)) if quiet_bulk.size else 1.0
    resolved = float(np.sum(np.clip(values[:significant] - bulk_level, 0.0,
                                    None))) if significant else 0.0
    return {
        "names": names,
        "rank": rank,
        "significant": significant,
        # the information budget, and the share of it that is signal
        "information": float(count),
        "resolved": resolved,
        "resolved_fraction": resolved / max(float(count), 1e-30),
        "bulk": bulk_level,
        "mp_edge": float(mp_edge),
        "semi_axes": np.sqrt(values),
        "eigenvalues": values,
        "axes": vectors,
        "participation": participation,
        "asymmetry": asymmetry,
        "trace": total,
        "sphere": bool(asymmetry < 1e-9),
        "floor": float(floor),
        "sphere_floor": circle["asymmetry"],
        "sphere_scatter": circle["scatter"],
        "sigma": float(margin),
        # the terminating test: no direction stands above what randomness
        # of this size and length produces on its own
        "at_sphere": bool(margin <= SPHERE_SIGMA),
    }


def graph_relations(declared: Dict[str, object]) -> Dict[str, List[str]]:
    """Who is related to whom, read from the pipeline's own declaration.

    Ethan's traversal: *"siblings where two residuals don't relate to each
    other, but DO relate to their parent and their children"*. So a node is
    related to its parents, to its children, and to its siblings - the
    nodes sharing a parent with it - and to nothing else. Two components
    with no path between them have no relationship for a differential to
    be taken of, and pairing them manufactures one.

    The declaration is `vhsdecode/pipeline/stages.toml`, walked by
    `residual_limit.order`, which already resolves parents from both the
    channel edges (`reads`/`writes`) and the model edges
    (`consumes_model`).
    """
    from vhsdecode import residual_limit

    nodes = residual_limit.order(declared)
    parents = {node["name"]: list(node.get("parents") or []) for node in nodes}
    children: Dict[str, List[str]] = {name: [] for name in parents}
    for name, above in parents.items():
        for parent in above:
            children.setdefault(parent, []).append(name)
    relations: Dict[str, List[str]] = {}
    for name in parents:
        near = set(parents.get(name, ())) | set(children.get(name, ()))
        for parent in parents.get(name, ()):        # siblings share a parent
            near |= {other for other in children.get(parent, ())
                     if other != name}
        relations[name] = sorted(near)
    return relations


def component_differentials(synthetic: Dict[str, np.ndarray],
                            measured: Dict[str, np.ndarray],
                            axis: Axis,
                            relations: Optional[Dict[str, List[str]]] = None
                            ) -> Dict[str, Dict[Axis, np.ndarray]]:
    """The matrix Ethan's differentiation actually forms.

    > The differentiation is the differential of our synthetic components
    > to the actual measured componnet, this has nested differentials in it
    > as well according to our graph, so this expands out to a multi
    > dimensional matrix that represents our known components.

    Two kinds of entry, and the second is what makes it a matrix rather
    than a list:

    **The diagonal** is each component's own differential, `synthetic minus
    measured` - the specification's ideal against what the tape actually
    produced. `Component.ideal` holds the synthetic side and
    `Component.expected` the model as it currently stands, so both are
    already carried.

    **The off-diagonal** is the nested differential between two components
    the graph relates. The graph states an EXPECTED relationship between
    them, `synthetic_i - synthetic_j`; the measurements state the ACTUAL
    one, `measured_i - measured_j`; and the entry is the difference of
    those two, which reduces to `d_i - d_j`. That is the "expected
    relationship vs. the actual relationship" the traversal takes the limit
    of, and it is only formed where the graph relates the pair - parents,
    children and siblings. Unrelated components are not differenced,
    because a difference between them is not a relationship anyone claimed.

    What comes back feeds straight into `ellipsoid` and
    `differentiate_to_floor`: the ellipse is fitted over the entries of
    this matrix, not over the raw components.
    """
    own: Dict[str, np.ndarray] = {}
    for name, ideal in synthetic.items():
        actual = measured.get(name)
        if ideal is None or actual is None:
            continue
        ideal = np.asarray(ideal).ravel()
        actual = np.asarray(actual).ravel()
        if ideal.shape != actual.shape or not ideal.size:
            continue
        own[name] = ideal - actual
    out: Dict[str, Dict[Axis, np.ndarray]] = {
        name: {axis: value} for name, value in own.items()}
    if relations is None:
        relations = {name: [other for other in own if other != name]
                     for name in own}
    seen = set()
    for name, near in relations.items():
        if name not in own:
            continue
        for other in near:
            if other not in own or other == name:
                continue
            pair = tuple(sorted((name, other)))
            if pair in seen:
                continue
            first, second = pair
            if own[first].shape != own[second].shape:
                continue          # different grids are not differenced
            seen.add(pair)
            out[f"{first} vs {second}"] = {axis: own[first] - own[second]}
    return out


def surrogate_null(residuals: Dict[str, Dict[Axis, np.ndarray]], axis: Axis,
                   draws: int = 24, seed: int = 0,
                   real_parameters: bool = False) -> Dict[str, object]:
    """THE ONLY VALID NULL FOR A CONSTRUCTED ENSEMBLE.

    Every analytic null in this module - the sphere floor, the
    Marchenko-Pastur edge - assumes the ensemble's rows are independent. A
    NESTED DIFFERENTIAL MATRIX has no such rows: `D_ij = d_i - d_j` is a
    linear combination of the diagonals, so `N(N+1)/2` entries span only `N`
    dimensions. Applying an analytic null there does not merely lose
    accuracy, it measures the construction.

    How badly, measured: fed structureless Gaussian noise through the same
    construction at the sizes this arc reported, the analytic statistics
    return rank 13, twelve significant directions, 99.1 per cent resolved
    and +37.9 sigma - against 92.9 per cent and +39.3 sigma from the real
    decode. **The published run was reproduced, number for number, by
    noise.** Every rank, direction count, resolved fraction and sigma taken
    on a nested matrix before this function existed is withdrawn.

    So the null is generated rather than derived: the same construction, the
    same sizes, structureless input, many draws. What comes back is the
    distribution each statistic takes when there is nothing to find, and a
    real ensemble earns a claim only by standing above it.
    """
    names = [name for name in residuals
             if (residuals.get(name) or {}).get(axis) is not None]
    if len(names) < 3:
        return {"draws": 0, "why": "too few entries to build a null"}
    lengths = {int(np.asarray(residuals[name][axis]).size) for name in names}
    if len(lengths) != 1:
        return {"draws": 0, "why": "entries do not share a length"}
    length = lengths.pop()
    complex_valued = any(
        np.issubdtype(np.asarray(residuals[name][axis]).dtype,
                      np.complexfloating) for name in names)
    # the ensemble's own rank is what says how it was constructed; the null
    # must be built the same way or it is not a null for this statistic
    observed = ellipsoid(residuals, axis, real_parameters=real_parameters)
    deficiency = max(len(observed["names"]) - int(observed["rank"]), 0)
    independent = max(len(names) - deficiency, 1)

    rng = np.random.default_rng(seed)
    columns = {"asymmetry": [], "resolved_fraction": [], "significant": [],
               "rank": [], "sigma": []}
    for _ in range(max(int(draws), 2)):
        seeds = {}
        for index in range(independent):
            value = rng.standard_normal(length)
            if complex_valued:
                value = value + 1j * rng.standard_normal(length)
            seeds[f"s{index}"] = value
        # rebuild the SAME construction: the independent draws as the
        # diagonal, and every pairwise difference beside them
        built = {name: {axis: value} for name, value in seeds.items()}
        keys = list(seeds)
        for first in range(len(keys)):
            for second in range(first + 1, len(keys)):
                built[f"{keys[first]} vs {keys[second]}"] = {
                    axis: seeds[keys[first]] - seeds[keys[second]]}
        fit = ellipsoid(built, axis, real_parameters=real_parameters)
        if not fit["names"]:
            continue
        for key in columns:
            columns[key].append(float(fit[key]))
    out: Dict[str, object] = {"draws": len(columns["asymmetry"]),
                              "length": length,
                              "independent": independent,
                              "deficiency": deficiency}
    for key, values in columns.items():
        array = np.asarray(values, dtype=np.float64)
        out[f"{key}_mean"] = float(array.mean()) if array.size else 0.0
        out[f"{key}_sd"] = float(array.std(ddof=1)) if array.size > 1 else 0.0
        seen = float(observed[key])
        spread = out[f"{key}_sd"]
        out[f"{key}_z"] = ((seen - out[f"{key}_mean"]) / spread
                           if spread > 0 else 0.0)
        out[f"{key}_observed"] = seen
    out["why"] = ("the same construction on structureless input; an "
                  "analytic null does not apply to an ensemble whose rows "
                  "are linear combinations of one another")
    return out


def guess_credibility(residuals: Dict[str, Dict[Axis, np.ndarray]],
                      axis: Axis, real_parameters: bool = False
                      ) -> Dict[str, object]:
    """THE CURVE IS A GUESS. This is what tests it.

    Ethan: *"You only 'guess' the curve, you don't know it."* That is an
    epistemic constraint on the implementation, not a remark. A fitted
    ellipsoid describes the ensemble it was fitted to by construction, so
    its directions carry no evidence until they are shown to describe
    evidence they have not seen.

    The ensemble is split into two banks on alternating members, the
    ellipse is fitted to each, and each bank's directions are asked how
    much of the OTHER bank they explain. What comes back per direction:

      `held_in`   the share of its own bank's energy the direction carries
      `held_out`  the share of the other bank's energy it carries
      `credibility` the ratio, clipped to [0, 1]

    A direction that describes the path reads near one. A direction that
    describes the fields it was measured from reads near zero, and it is a
    guess that failed - it must not be deflated, because deflating it
    removes real signal and calls the removal a correction.

    This is the same rule that governs every other admission in this arc,
    and the measurement that makes it non-negotiable is already recorded:
    fitted and judged on the same seven fields the limit reaches 0.0001 dB,
    and on other fields of the same head and tape it stands at 0.0328 dB,
    with nothing changed but which samples judged it.

    Alternating members rather than a contiguous split, so that a trend
    across the ensemble does not land wholly in one bank.
    """
    names = [name for name in residuals
             if (residuals.get(name) or {}).get(axis) is not None]
    if len(names) < 4:
        return {"names": [], "credibility": np.zeros(0),
                "held_in": np.zeros(0), "held_out": np.zeros(0),
                "why": "fewer than four components: nothing to hold out"}
    banks = (names[0::2], names[1::2])
    if min(len(bank) for bank in banks) < 2:
        return {"names": [], "credibility": np.zeros(0),
                "held_in": np.zeros(0), "held_out": np.zeros(0),
                "why": "a bank holds fewer than two components"}

    def directions_of(bank):
        fit = ellipsoid({name: residuals[name] for name in bank}, axis,
                        real_parameters=real_parameters)
        if not fit["names"] or not fit["significant"]:
            return None, fit
        stack = np.array([
            np.nan_to_num(np.asarray(residuals[name][axis]).ravel())
            for name in fit["names"]])
        norms = np.linalg.norm(stack, axis=1, keepdims=True)
        unit = np.divide(stack, norms, out=np.zeros_like(stack),
                         where=norms > 0)
        weights = np.asarray(fit["axes"])[:, :fit["significant"]]
        target = weights.conj().T @ unit
        basis = []
        for row in target:
            for earlier in basis:
                row = row - (earlier.conj() @ row) * earlier
            norm = float(np.linalg.norm(row))
            if norm > 0:
                basis.append(row / norm)
        return basis, fit

    def carried(basis, bank):
        """The share of a bank's energy each direction accounts for."""
        share = np.zeros(len(basis))
        total = 0.0
        for name in bank:
            vector = np.nan_to_num(np.asarray(residuals[name][axis]).ravel())
            energy = float(np.vdot(vector, vector).real)
            if not energy > 0:
                continue
            total += energy
            for index, direction in enumerate(basis):
                if direction.size != vector.size:
                    continue
                share[index] += abs(complex(direction.conj() @ vector)) ** 2
        return share / max(total, 1e-30)

    reported, held_in, held_out = [], [], []
    for index, bank in enumerate(banks):
        basis, _ = directions_of(bank)
        if not basis:
            continue
        other = banks[1 - index]
        inside = carried(basis, bank)
        outside = carried(basis, other)
        for position in range(len(basis)):
            reported.append(f"bank {index} direction {position}")
            held_in.append(float(inside[position]))
            held_out.append(float(outside[position]))
    held_in = np.asarray(held_in)
    held_out = np.asarray(held_out)
    credibility = np.clip(
        np.divide(held_out, held_in, out=np.zeros_like(held_out),
                  where=held_in > 0), 0.0, 1.0)
    # THE BAR, DERIVED RATHER THAN CHOSEN. A direction that explains
    # nothing it has not seen still carries its equal share of the other
    # bank's energy, and an equal share of `length` directions is `1/length`.
    # So a direction earns admission when it carries MORE than an equal
    # share, and `carries` states by how much. Measured: a real shared
    # departure returns 58 on a 256-point grid, and a departure planted in
    # one bank only returns 1.02 - sitting exactly on the null, which is
    # what a guess that failed should do.
    length = 0
    for name in names:
        value = (residuals.get(name) or {}).get(axis)
        if value is not None:
            length = max(length, int(np.asarray(value).size))
    carries = held_out * max(length, 1)
    return {
        "names": reported,
        "held_in": held_in,
        "held_out": held_out,
        "credibility": credibility,
        "carries": carries,
        "admissible": carries > 1.0,
        "length": length,
        "worst": float(credibility.min()) if credibility.size else 0.0,
        "best": float(credibility.max()) if credibility.size else 0.0,
        "mean": float(credibility.mean()) if credibility.size else 0.0,
        "why": ("each bank's directions asked how much of the other bank "
                "they explain; a guess that only describes its own bank "
                "reads near zero"),
    }


def differentiate_to_floor(residuals: Dict[str, Dict[Axis, np.ndarray]],
                           axis: Axis,
                           amount: Optional[float] = None,
                           maximum_passes: int = MAXIMUM_PASSES
                           ) -> Dict[str, object]:
    """Ethan's loop, closed: fit the ellipse, differentiate the source
    against it, repeat until the shape is a circle.

    > Then differentiation using the target curve onto the source data.
    > This loops until the information floor is reached (total possible
    > information on the source signal) 4d eliptical curve I believe.
    > Down to the circular shape of the complex signal.

    Each pass fits the ellipsoid to the components as they stand - that
    fitted shape is the TARGET CURVE - and removes its leading principal
    direction from every component, which is the differentiation of the
    target onto the source. What the ellipse explains leaves; what it does
    not stays. When the remainder is as spherical as randomness of this
    size and length would be, there is no preferred direction left to find
    and the loop is at the information floor.

    Each direction is removed at ITS OWN Wiener weight rather than at a
    flat step: `(lambda - bulk) / lambda`, where the bulk is where the
    eigenvalues that are not directions sit. That weight is identical to
    this project's correction-gain law `a* = 1/(1 + rho)` for every
    eigenvalue, so the ensemble supplies the gain the law asks for instead
    of it being chosen. Passing `amount` scales every weight by that factor
    for a deliberately conservative pass.

    This is what the collapse is, stated in signal terms: the eigenbasis is
    the Karhunen-Loeve transform of the ensemble, and the Wiener filter is
    diagonal in it. Fitting the ellipse finds the basis; the per-direction
    weight is the Wiener filter applied in it. On complex components -
    which is what RF is - both halves are complex throughout.

    Four axes are available - amplitude, frequency, time and the per-field
    dimension - which is the "4d elliptical curve". This runs one axis; the
    caller runs it per axis and reads the descent on each.
    """
    working = {name: dict(per_axis or {})
               for name, per_axis in residuals.items()}
    trace: List[Dict[str, float]] = []
    targets: List[np.ndarray] = []
    # THE RESIDUAL STOPS INCREASING. Ethan's terminating condition, and it
    # needs no threshold at all, which is what makes it better than the
    # sphere test it runs beside: "The limit is acheived when the residual
    # stops increasing, that's when you know you have approached the end of
    # this result, which gives you the exact residual."
    #
    # Each pass ATTRIBUTES some of the ensemble's energy to the directions
    # it found. That attribution accumulates. While a pass still finds
    # something the accumulation grows; when a pass adds nothing the
    # accumulation has converged, everything identifiable has been
    # identified, and the converged value IS the exact residual.
    attributed = 0.0
    accumulation: List[float] = []
    starting_energy = 0.0
    for per_axis in residuals.values():
        value = (per_axis or {}).get(axis)
        if value is not None:
            vector = np.nan_to_num(np.asarray(value).ravel())
            starting_energy += float(np.vdot(vector, vector).real)
    reason = "passes exhausted"
    deficiency = 0        # directions the ensemble never had, before any
    for _ in range(max(int(maximum_passes), 1)):   # deflation at all
        fit = ellipsoid(working, axis, removed=len(targets) + deficiency)
        if not trace and fit["names"]:
            # A nested differential matrix is rank-deficient by
            # construction: "a vs b" is the difference of "a" and "b", so
            # the entries span fewer dimensions than there are entries.
            # Charging that to the deflation makes the shape look
            # structured when it is only dependent, and the loop then
            # refuses its own first pass.
            deficiency = max(len(fit["names"]) - int(fit["rank"]), 0)
            if deficiency:
                fit = ellipsoid(working, axis, removed=deficiency)
        if not fit["names"]:
            reason = "no projectable group on this axis"
            break
        trace.append({
            "rank": float(fit["rank"]),
            "asymmetry": float(fit["asymmetry"]),
            "sphere_floor": float(fit["sphere_floor"]),
            "sigma": float(fit["sigma"]),
        })
        if fit["at_sphere"]:
            reason = "circular: at the information floor"
            break
        if not fit["significant"]:
            reason = "no direction stands above the noise edge"
            break
        names = fit["names"]
        stack = np.array([np.nan_to_num(np.asarray(working[n][axis]).ravel())
                          for n in names])
        # The eigenvector's entries weight the UNIT directions, because the
        # quadratic form was built from unit directions. Combining them
        # with the raw vectors instead lets a component with a larger norm
        # pull the target toward itself.
        norms = np.linalg.norm(stack, axis=1, keepdims=True)
        unit = np.divide(stack, norms, out=np.zeros_like(stack),
                         where=norms > 0)
        # THE TARGET CURVE IS THE WHOLE ELLIPSE, not one axis of it. Taking
        # the leading direction alone and deflating it part-way rebuilds
        # the pass-by-pass recursion this collapse replaces, and it does
        # not converge: measured on two and three planted departures the
        # asymmetry fell, then rose again, because the ensemble's leading
        # direction changes each time a share of the previous one is
        # removed. Every direction above the noise edge comes out together.
        vectors_out = np.asarray(fit["axes"])[:, :fit["significant"]]
        target = vectors_out.conj().T @ unit          # (significant, length)
        # THE WIENER WEIGHT, PER DIRECTION. The noise bulk is where the
        # eigenvalues that are not directions sit, so a direction carrying
        # signal stands at lambda = bulk + signal and the share of it that
        # is worth removing is (lambda - bulk) / lambda. That is exactly
        # this project's correction-gain law a* = 1/(1 + rho), derived
        # independently and identical for every lambda; the flat half-step
        # is its special case at a signal-to-noise ratio of one. Weighting
        # each direction by its own evidence is what makes this a Wiener
        # filter rather than a hard projection - a marginal direction is
        # barely touched, a strong one is removed almost whole.
        values_all = np.asarray(fit["eigenvalues"])
        quiet = values_all[values_all <= fit["mp_edge"]]
        bulk = float(np.median(quiet)) if quiet.size else 1.0
        weights_out = np.clip(
            (values_all[:fit["significant"]] - bulk)
            / np.maximum(values_all[:fit["significant"]], 1e-30), 0.0, 1.0)
        if amount is not None:
            weights_out = weights_out * float(amount)
        basis, gains = [], []
        for row, gain in zip(target, weights_out):
            for earlier in basis:
                row = row - (earlier.conj() @ row) * earlier
            norm = float(np.linalg.norm(row))
            if norm > 0:
                basis.append(row / norm)
                gains.append(float(gain))
        if not basis:
            reason = "the target curve is empty"
            break
        targets.extend(basis)
        before = float(fit["asymmetry"])
        before_energy = 0.0
        for name in names:
            vector = np.nan_to_num(np.asarray(working[name][axis]).ravel())
            before_energy += float(np.vdot(vector, vector).real)
        trial = {name: dict(per_axis) for name, per_axis in working.items()}
        for name in names:
            vector = np.asarray(working[name][axis]).ravel()
            cleaned = np.nan_to_num(vector)
            for direction, gain in zip(basis, gains):
                cleaned = cleaned - gain * (direction.conj() @ cleaned) * direction
            trial[name][axis] = cleaned
        after = ellipsoid(trial, axis,
                          removed=len(targets) + deficiency)
        if not after["names"]:
            reason = "refused: nothing projectable remains"
            break
        # REFUSE ONLY A PASS THAT MAKES THE SHAPE WORSE, and worse by more
        # than the scatter a random ensemble of this size and length shows
        # on its own. Requiring strict improvement instead threw away every
        # pass that made real but small progress, which is what a
        # deliberately conservative step produces - and it stopped the
        # accumulation rule below from ever being reached.
        worse = float(after["asymmetry"]) - before
        if worse > float(after.get("sphere_scatter", 0.0)):
            reason = "refused: the pass made the shape worse"
            break
        after_energy = 0.0
        for name in names:
            vector = np.nan_to_num(np.asarray(trial[name][axis]).ravel())
            after_energy += float(np.vdot(vector, vector).real)
        gained = max(before_energy - after_energy, 0.0)
        attributed += gained
        accumulation.append(attributed)
        working = trial
        # THE RESIDUAL STOPPED INCREASING - the primary termination, and the
        # only one that is not sensitive to the step size. Judged against
        # the largest single attribution so the test scales with the
        # ensemble and needs no constant.
        increments = np.diff([0.0] + accumulation)
        if len(accumulation) > 1 and increments.max() > 0 \
                and gained <= increments.max() * 1e-3:
            reason = "the residual stopped increasing: the exact residual"
            break
    else:
        reason = "passes exhausted"
    return {
        "residuals": working,
        "trace": trace,
        "targets": targets,
        "passes": len(trace),
        "reason": reason,
        "at_floor": reason.startswith("circular")
        or reason.startswith("the residual stopped"),
        # the accumulated attribution, and the exact residual it converged to
        "accumulation": accumulation,
        "attributed": attributed,
        "starting_energy": starting_energy,
        "exact_residual": max(starting_energy - attributed, 0.0),
        "attributed_fraction": (attributed / starting_energy
                                if starting_energy > 0 else 0.0),
    }


def orthogonalize_symmetric(residuals: Dict[str, Dict[Axis, np.ndarray]],
                           amount: float = ORTHOGONAL_AMOUNT
                           ) -> Dict[str, Dict[Axis, np.ndarray]]:
    """Matching pursuit orthogonalized SYMMETRICALLY: every component's
    share solved jointly, none of them privileged by position.

    Where the ordered form assumes the chain's sequence is known and
    lets only downstream components give ground, this assumes nothing
    about order and asks what set of shares best explains the ensemble.
    It is the maximum-likelihood answer when every gauge is measuring a
    projection of one shared departure. It cannot extrapolate outside the
    span of the components it was given, which is the point: it filters
    within the modelled bands rather than inventing structure beyond
    them."""
    out = {name: dict(per_axis or {}) for name, per_axis in residuals.items()}
    for axis in axes_present(residuals):
        for names, directions in _grouped_directions(residuals, axis,
                                                     list(residuals)):
            gram = np.real(directions @ directions.conj().T)
            off = np.abs(gram - np.diag(np.diag(gram)))
            if not off.size or off.max() < COHERENCE_THRESHOLD:
                continue
            # what each gauge reports is its own projection of the shared
            # departure; the joint shares solve G x = b for those
            # projections. The norm is real for a complex vector too, so
            # the solve stays real and the scale it returns applies to the
            # complex residual without touching its phase.
            projections = np.array([
                float(np.linalg.norm(np.nan_to_num(
                    np.asarray(residuals[n][axis]).ravel())))
                for n in names])
            ridge = ORTHOGONAL_RIDGE * float(np.trace(gram)) / max(len(names), 1)
            try:
                joint = np.linalg.solve(gram + ridge * np.eye(len(names)),
                                        projections)
            except np.linalg.LinAlgError:
                continue
            blended = projections + amount * (joint - projections)
            for name, before, after in zip(names, projections, blended):
                if before <= 0 or not np.isfinite(after):
                    continue
                out[name][axis] = (np.asarray(residuals[name][axis])
                                   * (after / before))
    return out


def orthogonalize(residuals: Dict[str, Dict[Axis, np.ndarray]],
                  order: Optional[Sequence[str]] = None,
                  amount: float = ORTHOGONAL_AMOUNT
                  ) -> Dict[str, Dict[Axis, np.ndarray]]:
    """Matching pursuit, orthogonalized DOWN A KNOWN ORDER.

    Greedy pursuit hands each component the whole of its own projection.
    Where two are correlated that over-counts both, and the error
    survives every later pass because the component was already
    subtracted and is never revisited.

    The order the stages acted in is known, so the removal is
    one-directional: each component keeps only the part of its residual
    that the components upstream of it cannot already explain. An
    upstream component is never adjusted by a downstream one, which a
    symmetric solve would wrongly do.

    The step is taken at half its length, by the same law that governs
    every other correction here: the model's own error is invisible to
    the fit, so a full step toward a slightly wrong optimum is worse than
    half a step toward it."""
    order = list(residuals) if order is None else [n for n in order
                                                  if n in residuals]
    order += [n for n in residuals if n not in order]
    out = {name: dict(per_axis or {}) for name, per_axis in residuals.items()}
    for axis in axes_present(residuals):
        for names, directions in _grouped_directions(residuals, axis, order):
            gram = np.abs(directions @ directions.conj().T)
            np.fill_diagonal(gram, 0.0)
            if not gram.size or gram.max() < COHERENCE_THRESHOLD:
                continue       # orthogonal enough that the step is identity
            basis = []         # the upstream directions, already made
            for position, name in enumerate(names):
                vector = np.asarray(residuals[name][axis]).ravel()
                if not np.issubdtype(vector.dtype, np.complexfloating):
                    vector = vector.astype(np.float64, copy=False)
                head = np.nan_to_num(vector)
                removed = np.zeros_like(head)
                for earlier in basis:
                    # the projection onto a unit direction conjugates the
                    # BASIS, which is the identity step for a real vector
                    # and the correct one for a complex vector
                    removed = removed + (earlier.conj() @ head) * earlier
                if position and np.any(removed):
                    head = head - amount * removed
                    out[name][axis] = head
                # this component's own direction, once what is upstream of
                # it has been taken out, joins the basis the next ones see
                direction = head
                for earlier in basis:
                    direction = direction - (earlier.conj() @ head) * earlier
                norm = float(np.linalg.norm(direction))
                if norm > 0:
                    basis.append(direction / norm)
    return out


def _run_stage(result, components, gauges, maximum_passes):
    # each stage keeps its OWN best pass: carrying one across the boundary
    # lets a revert restore a model from the other half of the chain and
    # silently discard everything this stage identified
    best = None
    entering = {name: dict(model) for name, model in result.model.items()}
    for index in range(maximum_passes):
        active = [c for c in components if not c.frozen]
        if not active:
            break
        # 1. measure every remaining component against the current model
        residuals = {c.name: gauges.measure(c, result.signal) for c in active}
        # 2. the loop is matching pursuit down a KNOWN order, so before
        # anything is attributed, each component gives up the part of its
        # residual that the stages upstream of it already explain. This
        # runs FIRST: it removes the structural double-count that comes
        # from the circuit's own ordering, leaving the cross-check to do
        # what only it can - attribute what remains by criterion.
        residuals = orthogonalize(residuals, [c.name for c in active])
        # 3. the components influence each other
        cross_check = getattr(gauges, "cross_check", None)
        if cross_check is not None:
            checked = cross_check(residuals)
            if checked:
                residuals = checked
        total = 0.0
        for component in active:
            residual = residuals.get(component.name, {})
            rms = _rms(residual)
            done = gauges.at_floor(component, residual)
            result.passes.append(PassRecord(component.name, index, rms, done))
            component.passes = index + 1
            total += float(np.nansum([v for v in rms.values()
                                      if np.isfinite(v)])) if rms else 0.0
            # 3. accumulate the residual into the expected signal
            component.accumulate(residual)
            result.model[component.name] = dict(component.expected)
            result.limit[component.name] = float(
                np.nansum([v for v in rms.values() if np.isfinite(v)])) \
                if rms else float("nan")
            # 4. a component at its limit is no longer differentiated over
            if done:
                component.frozen = True
        # judged on THIS stage's own components, so a stage is never
        # measured against the other half of the chain's residual
        stage_total = sum(v for c in components
                          for v in [result.limit.get(c.name, np.nan)]
                          if np.isfinite(v))
        if best is None or (np.isfinite(stage_total)
                            and stage_total < best[0]):
            best = (stage_total, result.signal,
                    {name: dict(m) for name, m in result.model.items()})
        if all(c.frozen for c in components):
            break
        # 1 (next pass). THE WHOLE MODEL FEEDS FORWARD TOGETHER - every
        # component's accumulation applied through the real chain at once,
        # never one component's correction on its own, because they
        # influence each other and a partial application is measured
        # against a chain the other components have not been applied to
        result.signal = gauges.feed_forward(result.signal, result.model)
        result.anti_residuals.append(("model", dict(result.model)))
    else:
        result.weakly_determined.extend(
            c.name for c in components if not c.frozen)
    # a pass that took the residual further from its limit is not kept,
    # judged on THIS stage's components alone; reverting restores the
    # model this stage built on, never the other stage's
    if best is not None and np.isfinite(best[0]):
        names = {c.name for c in components}
        last = sum(v for name, v in result.limit.items()
                   if name in names and np.isfinite(v))
        if last > best[0]:
            result.signal = best[1]
            result.model = {name: dict(model)
                            for name, model in best[2].items()}
            for name, model in entering.items():
                result.model.setdefault(name, model)
    return best


def _measure_noise_floor(result, gauges, noise):
    if noise is not None:
        floor = gauges.measure(noise, result.signal)
        if not noise.amplitude_only or set(floor) - {"amplitude"}:
            raise ValueError("the noise floor is amplitude only: no phase "
                             "residual can be derived from it")
        result.noise_floor = floor
        result.signal = gauges.subtract_noise_floor(result.signal, floor)
        remainder = floor.get("amplitude")
        if remainder is not None and np.ndim(remainder) == 2:
            result.differential_2d = differential_2d(remainder)
        elif remainder is not None and np.ndim(remainder) == 3:
            # indexed (amplitude, frequency, time): all three axes at once
            result.differential_3d = differential_3d(remainder)
            result.differential_2d = differential_2d(
                np.abs(remainder).mean(axis=0))
        # the per-field dimension: the non-reducible residual tracked
        # field by field, where the gauges can supply it
        per_field = getattr(gauges, "measure_per_field", None)
        if per_field is not None:
            track = per_field(noise, result.signal)
            if track is not None:
                result.field_floor = np.asarray(track, dtype=np.float64)
                if len(result.field_floor) > 1:
                    result.field_differential = field_differential(
                        result.field_floor)
