"""THE ONE RADIO-FREQUENCY STAGE: every raw-capture instrument a vertex, one
fold, one table per head, and the file's own hypercube as the noise.

Ethan, 2026-09-07:

    "I need to have a VHS RF stage transformed the same way. We do the full
    spherical shape, but, replace the noise with null space, which is a
    constant that we do not derive at all. We are extracting the signal we
    care about."

    "Perhaps an edge that represents exactly the cutoff for the input data,
    i.e. 20mhz of possible data in a 40mhz file. A hypercube ... That is a
    constant I think for the rf file itself ... Subtract out the hyper cube
    and discard it as noise."

    "Let's see what happens if we take this to hyperspace as the null space
    model, the exact inverse within our possible area of measure."

WHAT THIS MODULE IS. The mathematics lives in `models/single_transform`:
the live axes, the kept contrasts, the Welford banks, the latch, the
schedule and the published snapshot. Nothing there reads a field. This
module is the adapter between the decode and that mathematics: it takes
one raw copy and one pair of full-field transforms per field, fills the
eight vertices of the (band, head, polarity) cube from the instruments the
radio-frequency group already runs, hands in the capture's hypercube as the
floor, latches once, builds the tables the demodulator's equalizer site
multiplies by, and publishes them as one immutable snapshot for the worker
threads. The call sites are `model_stages.transform_rf`,
`rf_transform_live`, `rf_transform_table` and `transform_wants_redo`, which
delegate here.

THE VERTEX COORDINATES. `band` 0 is the luma frequency modulation and 1
the colour-under; `head` is `int(bool(field.isFirstField))`, the only head
label a decode has; `polarity` 0 is the carrier's deviation BELOW blanking
(the sync pulse's fall, toward the tip) and 1 ABOVE (the rise, toward
white). Polarity is declared null at the radio frequency because the
channel ahead of the demodulator is linear (`single_transform.RF_SIGNAL`
keeps no polarity contrast), and the chroma band is face-duplicated on it.

THE CHANNELS, AND WHY EACH GRID IS WHAT IT IS.

  `response`  N complex bins. On the LUMA faces it is the amplitude probe:
              the recorded luma carrier is constant in amplitude, so the
              envelope's departure across carrier frequency is the path's
              own magnitude response R(f) on the RF axis, and the sibling
              lane measures it per field and per head as a line in the
              logarithm (`luma_amplitude.measured_amplitude_deviation
              (field).response.separation`). That line is read here as a
              log-magnitude departure, zero at the middle of the specified
              deviation, weighted by the same raised-cosine belief the
              sibling's equalizer applies (full over the specified
              deviation, tapering to nothing across the margin the carrier
              only reaches in overshoot), on the block's own rfft grid
              restricted to that band. The block grid is chosen so that
              `tesseract.on_full_grid` places every bin exactly and the
              table needs no interpolation: the model is the measurement
              inside the band, and unity outside it. Its phase is zero
              because a carrier's amplitude reads no phase; the latch's
              causal split reports the minimum phase a causal channel with
              this magnitude must carry. The sync edge's departure against
              a specified edge was NOT used for this channel although the
              plan offered it: the recorded edge carries the record side's
              own pre-emphasis (SMPTE 32M clause 3.9.1.1.2) and dark clip,
              which the decoder undoes downstream and which no table ahead
              of the demodulator may invert; the edge is read here for the
              head schedule and the band delay, where it is a clock. On the
              CHROMA faces it is the colour-under's complex log departure
              across its own band: the burst's complex envelope, each line
              referred to its own burst phase so the record side's ninety
              degree rotation and the capture's drift leave, averaged, and
              its spectrum taken against the specified nine-cycle gate,
              zero at the carrier, on N points across the decoder's own
              colour-under band and held flat beyond the burst's own reach.
              Measured and reported, never realised: the chroma path
              already carries a static per-head colour-under gain and
              phase, and realising it again would count it twice.
  `levels`    one bin. Luma fall face: the sync tip carrier's departure
              from the specified tip; luma rise face: the back porch
              carrier's departure from specified blanking; both as a
              fraction of the specified deviation, so the polarity contrast
              is the deviation and the mean is the carrier centre. Chroma
              faces: the burst level against the luma carrier's porch
              level, in nepers, which is the dimensionless ratio the format
              records (the luma is the chroma's bias).
  `timing`    one bin. The sync-rise-to-burst interval's departure from
              the BTS-3 value, in seconds, the same at both band faces
              because it is the pair's difference.

THE HYPERCUBE FLOOR. Once per capture, the raw field's Welch spectrum on
the block grid up to Nyquist; occupied bands are the format's own video
band-pass and the colour-under band the decoder's chroma band-pass
defines; `single_transform.hypercube` gives the density where the data
must not exist. Its conversion into the power one contrast bin of the
response channel would carry is an ESTIMATE and is labelled so wherever it
is reported: a steady carrier of one-sided power P with white noise of
in-band power N has a log-envelope variance of N / (2 P) per sample; a
field's line pools the samples the sibling's fit walks; a contrast is the
signed mean of 2^n vertices each a mean of m fills. The estimate is refreshed
every field from the current fill count so it stays comparable with the
instrument floor the fold measures.

THE SCHEDULE. The workers demodulate ahead of assembly and are not told
which head is coming, so a block is labelled by extrapolation from a switch
the decoder located, with the field period as the known constant it is.
Absolute positions: the field's sample i is the capture's sample
`field.readloc + blockcut + i`, because `DemodCache.doread` loads block b
from `b * blocksize` for `blocklen` samples (lddecode/core.py:1279-1283),
`read` keeps each block's raw input cut by `blockcut` before concatenation
(core.py:1363-1365) and reports `startloc` as the first block's
`b * blocksize` (core.py:1446), which becomes `readloc` (core.py:4078); the
demodulator's `block_start` is the raw window's own `b * blocksize`, on the
same axis. The locator's regions are in field-sample coordinates
(`head_switch.locate` bins on `linelocs`), so the switch's absolute sample
is `readloc + blockcut + start`. When the locator refuses, the switch
is placed the window SMPTE 32M clause 3.6 allows ahead of the next vertical
sync, from the field-sync locator on the shared analytics. The period is
the format's field period in samples (`freq_hz / (2 FPS)`, the quantity
core.py:1104 rounds into `bytes_per_field`) refined by the median spacing
of successive anchors; a break in the parity alternation re-anchors. The
tables never change after the latch; the schedule is a clock and is
re-synchronised as switches are located, under the same version, so the
one-shot redemodulation is never asked for twice.
"""

import math
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import numpy as np

from vhsdecode import channel_eq, luma_amplitude, model_stages, pipeline_graph
from vhsdecode.models import band_delay, chroma_head_switch, head_switch_pair
from vhsdecode.models import single_transform, sync_geometry, tesseract

STAGE = "rf_transform"
KIND = "rf"
STORE = "_single_transform"          # rf.__dict__ key shared with the picture
OWN_STORE = "_rf_transform_store"    # this module's canonical copy of the same
PUBLISHED = "_rf_transform"          # the immutable snapshot the workers read
REDONE = "_rf_transform_redone"      # the version the one-shot redo has repaid

RESPONSE = "response"
LEVELS = "levels"
TIMING = "timing"

# The correction-gain law's half. The believed optimum for an exact inverse
# is one; a* = 1 / (1 + rho) with rho the model error, which no resampling
# of the fit's own data can see, and applying at half the believed optimum
# keeps three quarters of the benefit anywhere in the non-harmful range
# (memory `correction-gain-law`, measured on the RF luma equalizer: fitted
# rho 0.80, predicted optimum 0.56, measured best 0.5).
HALF_OPTIMUM_AMOUNT = 0.5
EXACT_INVERSE_AMOUNT = 1.0

# How many fields of redo memory the memo by read position keeps. A redo
# re-creates the field object at the same tape position within the cache's
# prefetch depth, which `DemodCache.__init__` sets at four fields of blocks
# plus four (lddecode/core.py:1105); sixty-four fields is sixteen times that
# depth and bounds the memo on a two-hour capture.
REDO_MEMORY_FIELDS = 64

# The fillers, gated by their own options, and the rule that stops each.
# Known constants of the capture are measured once behind their own latched
# or reported flags; the record head switch stops at its first mark; the
# rest run until the transform latches and then on the doubling stride.
_KNOWN_CONSTANTS = ("capture_profile", "capture_filter", "rf_stages",
                    "transport_model")
_FIRST_MARK = ("head_switch_pair",)
_REPORT_ONLY = ("filter_model", "band_delay", "interference", "head_model",
                "head_differential", "magnetic", "magnetic_circuit",
                "tape_path")


def _adapters():
    """The adapter for each option name, looked up late so a stub decoder in
    a test can replace one on `model_stages` and be honoured."""
    return {
        "capture_profile": model_stages.measure_capture_profile,
        "capture_filter": model_stages.measure_capture_filter,
        "rf_stages": model_stages.measure_rf_stages,
        "transport_model": model_stages.measure_transport,
        "head_switch_pair": model_stages.measure_head_switch_pair,
        "filter_model": model_stages.measure_filter_model,
        "band_delay": model_stages.measure_band_delay,
        "interference": model_stages.measure_interference,
        "head_model": model_stages.measure_head_model,
        "head_differential": model_stages.measure_head_differential,
        "magnetic": model_stages.measure_magnetic,
        "magnetic_circuit": model_stages.measure_magnetic_circuit,
        "tape_path": model_stages.measure_tape_path,
    }


# ---------------------------------------------------------------------------
# The contract the call sites bind to
# ---------------------------------------------------------------------------

def live(rf) -> bool:
    """Whether a latched table has been published for this decode."""
    return rf.__dict__.get(PUBLISHED) is not None


def table(rf, block_start, n_fft) -> Optional[np.ndarray]:
    """The complex multiplier for one block on its full transform grid, or
    None while nothing is published or the grid does not match.

    Runs on the worker threads: it reads one immutable snapshot and never
    mutates. `block_start` is the block's absolute start sample, or None
    for a caller that is not the cache's worker, in which case the common
    table stands (the snapshot's own rule)."""
    published = rf.__dict__.get(PUBLISHED)
    if published is None:
        return None
    return published.table(block_start, n_fft)


def wants_redo(rf) -> bool:
    """True exactly once per newly published version, so the decoder's
    one-shot redemodulation repays the blocks demodulated before the latch.
    A schedule re-synchronised under the same version does not fire it."""
    published = rf.__dict__.get(PUBLISHED)
    if published is None:
        return False
    done = rf.__dict__.get(REDONE, 0)
    if published.version <= done:
        return False
    rf.__dict__[REDONE] = published.version
    return True


# ---------------------------------------------------------------------------
# The per-decode store
# ---------------------------------------------------------------------------

def _store(rf) -> Dict[str, object]:
    """The per-decode store, at `rf.__dict__["_single_transform"]` as the
    plan directs, with this module's entries mirrored from a canonical
    copy of its own.

    THE MIRROR IS A REPAIR, AND THE DEFECT IT REPAIRS WAS MEASURED. The
    key is shared with the picture transform, whose per-decode reset
    empties the whole dictionary (`model_stages._picture_store(field)
    .clear()`, model_stages.py:9665). On the first real decode of this
    module (the SP bars capture, 2026-09-07) that happened once between
    the first and second fields: the field count restarted at one, the
    transform lost its first field of evidence and was constructed twice,
    and the latch came a field late. That clear is the picture stage's to
    scope; this module keeps its transform and state under its own key as
    well and restores the shared entries from there, so a clear costs
    nothing and the transform is constructed once per decode whatever the
    other stage does - verified on the same decode, twenty-six calls on
    one transform object.
    """
    own = rf.__dict__.setdefault(OWN_STORE, {})
    shared = rf.__dict__.setdefault(STORE, {})
    for key in (KIND, "rf_state", "rf_response_grid"):
        if key in own and shared.get(key) is not own[key]:
            shared[key] = own[key]
    return own


def _state(rf) -> Dict[str, object]:
    """This module's own bookkeeping beside the transform: field count,
    the memo by read position, the schedule's anchors, the witnesses'
    histories and the hypercube."""
    return _store(rf).setdefault("rf_state", {
        "fields": 0,
        "seen": OrderedDict(),
        "anchors": [],
        "anchor_parity": None,
        "expected_head": None,
        "dropout_history": [],
        "hypercube": None,
        "field_samples": 0,
        "latched_at": None,
        "schedule_version": 0,
    })


def _iretohz(rf, ire: float) -> float:
    """The specified carrier for an IRE level, through the decoder's own
    scale; `spec=True` where the decoder offers it, so a running `ire0`
    adjustment does not move the grid field to field (the sibling's own
    reasoning at `carrier_frequency_bins`)."""
    try:
        return float(rf.iretohz(ire, spec=True))
    except TypeError:
        return float(rf.iretohz(ire))


def _selection(rf):
    return getattr(getattr(rf, "options", None), "stage_selection", None)


def resolve_amount(rf, amount=None) -> Dict[str, object]:
    """How much of the measured departure the tables apply, and why.

    An explicit number from the caller wins. Otherwise the `exact_inverse`
    component of this stage, off unless selected with
    `--stages +rf_transform.exact_inverse`, asks for the inverse in full
    within the area of measure; and otherwise the correction-gain law's
    half stands.
    """
    if amount is not None:
        return {"applied": float(amount), "exact_inverse": False,
                "reason": "the caller passed the amount"}
    exact = pipeline_graph.component_enabled(
        _selection(rf), STAGE, "exact_inverse", default=False)
    if exact:
        return {"applied": EXACT_INVERSE_AMOUNT, "exact_inverse": True,
                "reason": ("--stages +rf_transform.exact_inverse: the exact "
                           "inverse of the measured departure within the "
                           "area of measure, the luma band on the causal "
                           "axis and the admitted contrasts on the folded "
                           "axes; what remains is by construction outside "
                           "that area")}
    return {"applied": HALF_OPTIMUM_AMOUNT, "exact_inverse": False,
            "reason": ("the correction-gain law: a* = 1/(1 + rho) with rho "
                       "the model error no split-half can see; half the "
                       "believed optimum keeps three quarters of the "
                       "benefit")}


def resolve_treatment(rf) -> str:
    """`remove` when the `remove_residuals` component is selected, else
    `substitute`, the stage's own default."""
    remove = pipeline_graph.component_enabled(
        _selection(rf), STAGE, "remove_residuals", default=False)
    return "remove" if remove else "substitute"


# ---------------------------------------------------------------------------
# The grids
# ---------------------------------------------------------------------------

def response_grid(rf, geometry) -> Dict[str, object]:
    """The fixed band of the RF grid the response channel lives on.

    The block's own rfft step, `freq_hz / blocklen`, restricted to the band
    the amplitude probe has evidence on: the specified deviation extended
    by the margin the sibling's curve reaches
    (`luma_amplitude.CURVE_RANGE_MARGIN_IRE`), which is where the carrier
    goes in overshoot. The belief is the sibling's own raised cosine: one
    over the specified deviation, falling to nothing across each margin.
    The reference is the middle of the specified deviation, so the table
    carries no overall gain. Derived once per decode and kept on the store.

    Measured on the test stub - 40 MSps, a 32768-sample block, NTSC VHS:
    2223 bins at a step of 1220.7031 Hz, from 2.5429 to 5.2571 MHz with
    the belief at one between 3.4 and 4.4 MHz and the reference at 3.9.
    """
    cached = _store(rf).get("rf_response_grid")
    if cached is not None:
        return cached
    n = int(geometry["block_len"])
    rate = float(geometry["sample_rate_hz"])
    step = rate / n
    vsync_ire = float(rf.SysParams["vsync_ire"])
    margin = float(luma_amplitude.CURVE_RANGE_MARGIN_IRE)
    inner_low = _iretohz(rf, vsync_ire)
    inner_high = _iretohz(rf, 100.0)
    outer_low = _iretohz(rf, vsync_ire - margin)
    outer_high = _iretohz(rf, 100.0 + margin)
    first = int(math.ceil(outer_low / step))
    last = int(math.floor(outer_high / step))
    last = min(last, n // 2)
    if last - first < 8:
        raise ValueError("the response band holds too few bins of the block "
                         "grid to carry a response")
    index = np.arange(first, last + 1)
    bins_hz = index * step
    belief = np.zeros(bins_hz.size)
    belief[(bins_hz >= inner_low) & (bins_hz <= inner_high)] = 1.0
    for low, high, rising in ((outer_low, inner_low, True),
                              (inner_high, outer_high, False)):
        edge = (bins_hz > low) & (bins_hz < high)
        position = (bins_hz[edge] - low) / max(high - low, 1.0)
        cosine = np.cos(np.pi * position)
        belief[edge] = 0.5 * (1.0 - cosine) if rising else 0.5 * (1.0 + cosine)
    grid = {
        "bins_hz": bins_hz,
        "index": index,
        "step_hz": step,
        "belief": belief,
        "reference_hz": 0.5 * (inner_low + inner_high),
        "inner_hz": (inner_low, inner_high),
        "outer_hz": (outer_low, outer_high),
        "n_fft": n,
        "why": ("the block's own rfft grid inside the band the amplitude "
                "probe has evidence on, so every bin lands exactly on the "
                "table and the model is the measurement inside the band"),
    }
    _store(rf)["rf_response_grid"] = grid
    _store(rf)
    return grid


def chroma_grid(rf, geometry, count: int) -> Optional[Dict[str, object]]:
    """The colour-under band on the same number of points as the luma
    response, as signed offsets from the colour-under carrier.

    The half width is the decoder's own chroma band-pass above the carrier
    (`chroma_bpf_upper - color_under_carrier`, the definition `chroma.py`
    uses for the band the colour-under can carry modulation in). A format
    whose two parameters do not describe such a band has no chroma grid
    and its chroma faces are not filled.

    Measured on the test stub for NTSC VHS: a half width of 570.6 kHz, a
    gate of 2.514 microseconds and so a reach of 198.9 kHz either side of
    the carrier.
    """
    params = rf.DecoderParams
    upper = params.get("chroma_bpf_upper")
    carrier = geometry.get("colour_under_hz")
    if upper is None or carrier is None:
        return None
    half_width = float(upper) - float(carrier)
    if not half_width > 0.0:
        return None
    offsets = np.linspace(-half_width, half_width, int(count))
    window = band_delay.specified_interval("NTSC")
    length_s = float(window["length_us"]) * 1e-6
    return {
        "offsets_hz": offsets,
        "bins_hz": float(carrier) + offsets,
        "carrier_hz": float(carrier),
        "half_width_hz": half_width,
        # the 1/T law: a gate of this length resolves its own reciprocal,
        # and the departure is credible to half of that either side
        "reach_hz": 0.5 / length_s,
        "gate_s": length_s,
    }


# ---------------------------------------------------------------------------
# The transform, constructed once per decode
# ---------------------------------------------------------------------------

def transform_for(rf, geometry) -> single_transform.Transform:
    """The radio-frequency transform, constructed once per decode and kept
    at `rf.__dict__["_single_transform"]["rf"]`."""
    store = _store(rf)
    transform = store.get(KIND)
    if transform is None:
        grid = response_grid(rf, geometry)
        transform = single_transform.Transform(
            KIND, single_transform.LIVE_AXES[KIND],
            {RESPONSE: grid["bins_hz"].size, LEVELS: 1, TIMING: 1},
            sample_rate_hz=float(geometry["sample_rate_hz"]),
            bins_hz={RESPONSE: grid["bins_hz"]})
        store[KIND] = transform
        _store(rf)
    return transform


# ---------------------------------------------------------------------------
# The context: one raw copy, one analytics, the instruments, the fillers
# ---------------------------------------------------------------------------

def rf_context(field, geometry, active: bool = True) -> Dict[str, object]:
    """Everything one field gives the transform, read once.

    One raw copy (`model_stages._raw_field`), one shared analytics
    (`model_stages._shared_rf_analytics`, the luma band's instantaneous
    frequency and the chroma band's complex envelope), one reserved-band
    levels record, one amplitude-deviation reading, the band-delay
    measurement on the shared analytics, and the fillers the options gate,
    each under its own stopping rule. The adapters that take their own raw
    copy do so inside `model_stages`; this function takes one. An inactive
    field - one off the doubling stride after the latch - reads nothing,
    because the two full-field transforms are the whole cost of this stage:
    measured on the 101680-sample synthetic field of the tests, an active
    field costs 43 to 62 milliseconds through the whole stage and an
    inactive one 2.
    """
    rf = field.rf
    context: Dict[str, object] = {
        "raw": None, "analytics": None, "levels": None, "amplitude": None,
        "band_delay": None, "adapters": [], "refusals": [],
    }
    if not active:
        context["refusals"].append("off the doubling stride after the latch")
        return context
    raw = model_stages._raw_field(field)
    context["raw"] = raw
    if raw is None:
        context["refusals"].append("no raw radio frequency on this field")
        return context
    system = geometry.get("system") or "NTSC"
    analytics = model_stages._shared_rf_analytics(
        field, raw, geometry["sample_rate_hz"], system)
    context["analytics"] = analytics
    context["levels"] = model_stages._reserved_band_levels(field, geometry)
    if context["levels"] is None:
        context["refusals"].append("the reserved intervals could not be read")
    try:
        context["amplitude"] = luma_amplitude.measured_amplitude_deviation(field)
    except Exception as error:                                # noqa: BLE001
        context["amplitude"] = None
        context["refusals"].append("amplitude deviation: %s" % (error,))
    if system == "NTSC" and analytics is not None:
        try:
            context["band_delay"] = band_delay.measure_capture(
                raw, geometry["sample_rate_hz"], "NTSC", shared=analytics)
        except Exception as error:                            # noqa: BLE001
            context["refusals"].append("band delay: %s" % (error,))
    else:
        context["refusals"].append(
            "band delay is cited from BTS-3 for the 525-line system only")
    context["adapters"] = _run_adapters(field)
    return context


def _adapter_should_run(field, name: str) -> bool:
    """The stopping rule for one filler, read from its own state."""
    state = field.rf.__dict__.get("_rf_measurements", {}).get(name)
    if name in _KNOWN_CONSTANTS:
        return not (state and (state.get("latched") or state.get("reported")))
    if name in _FIRST_MARK:
        return not (state and state.get("marks"))
    return True


def _run_adapters(field):
    """The fillers the options gate, in the order the eight-stage branch
    ran them, each attaching its own reading to the field."""
    options = field.rf.options
    ran = []
    picture = getattr(field, "dspicture", None)
    for name, adapter in _adapters().items():
        if not getattr(options, name, 0):
            continue
        if not _adapter_should_run(field, name):
            continue
        try:
            if name == "filter_model":
                if picture is None:
                    continue
                adapter(field, picture)
            else:
                adapter(field)
            ran.append(name)
        except Exception as error:                            # noqa: BLE001
            model_stages._log_once(
                field, "rf_transform_adapter_" + name,
                "rf_transform: the %s filler raised %r and is skipped for "
                "this decode's report", name, error)
    return ran


# ---------------------------------------------------------------------------
# The readings, per channel
# ---------------------------------------------------------------------------

def luma_response(grid, amplitude) -> Optional[np.ndarray]:
    """The amplitude probe's line as a log-magnitude departure on the
    response grid: slope times the distance from the reference, weighted
    by the belief, with no phase. The line's absolute level is a gain and
    is not a departure: a constant gain ahead of a limiter does nothing."""
    response = getattr(amplitude, "response", None) if amplitude else None
    line = getattr(response, "separation", None) if response else None
    if line is None:
        return None
    _centre_hz, _level, slope = line
    if not np.isfinite(slope):
        return None
    departure = float(slope) * (grid["bins_hz"] - grid["reference_hz"])
    return (grid["belief"] * departure).astype(np.complex128)


def chroma_response(analytics, geometry, chroma) -> Optional[Dict[str, object]]:
    """The colour-under band's complex log departure from the burst.

    Every line's burst envelope is cut at the specified window after its
    own sync rise and referred to its own resultant phase, so the record
    side's quarter turn a line and the capture's accumulated drift leave
    before the average is taken; the average's spectrum is divided by the
    specified nine-cycle gate's and the log taken, zero at the carrier.
    Beyond the burst's own reach the departure is held flat, because a
    gate two and a half microseconds long resolves nothing finer.
    """
    if analytics is None or chroma is None:
        return None
    rate = float(geometry["sample_rate_hz"])
    frequency = analytics["frequency"]
    envelope = analytics["envelope"]
    rises = band_delay.sync_rises(frequency, rate, "NTSC")
    window = band_delay.specified_interval("NTSC")
    low = int(round(window["start_us"] * 1e-6 * rate))
    length = int(round(window["length_us"] * 1e-6 * rate))
    if length < 4 or rises.size == 0:
        return None
    segments = []
    for start in rises:
        first = int(start) + low
        last = first + length
        if first < 0 or last > envelope.size:
            continue
        segment = envelope[first:last]
        resultant = complex(segment.sum())
        if not abs(resultant) > 0.0:
            continue
        segments.append(segment * np.exp(-1j * np.angle(resultant)))
    if len(segments) < 2:
        return None
    mean = np.mean(np.asarray(segments), axis=0)
    amplitude = float(np.abs(mean).max())
    if not amplitude > 0.0:
        return None
    padded = 1 << int(math.ceil(math.log2(max(8 * length, 64))))
    spectrum = np.fft.fftshift(np.fft.fft(mean, n=padded))
    gate = np.fft.fftshift(np.fft.fft(np.ones(length), n=padded))
    axis = np.fft.fftshift(np.fft.fftfreq(padded, 1.0 / rate))
    offsets = chroma["offsets_hz"]
    reach = float(chroma["reach_hz"])
    held = np.clip(offsets, -reach, reach)
    measured = (np.interp(held, axis, spectrum.real)
                + 1j * np.interp(held, axis, spectrum.imag))
    expected = (np.interp(held, axis, gate.real)
                + 1j * np.interp(held, axis, gate.imag))
    with np.errstate(divide="ignore", invalid="ignore"):
        departure = np.log(measured / expected)
    centre = np.log(complex(np.interp(0.0, axis, spectrum.real)
                            + 1j * np.interp(0.0, axis, spectrum.imag))
                    / complex(np.interp(0.0, axis, gate.real)
                              + 1j * np.interp(0.0, axis, gate.imag)))
    departure = departure - centre
    departure = departure.real + 1j * np.unwrap(departure.imag)
    if not np.all(np.isfinite(departure)):
        return None
    return {"value": departure, "lines": len(segments),
            "burst_amplitude": amplitude,
            "resolution_hz": 1.0 / float(chroma["gate_s"])}


def level_readings(levels, geometry) -> Optional[Dict[str, object]]:
    """The `levels` channel from the reserved-band record: the two luma
    faces as fractions of the specified deviation, the chroma faces as the
    burst-to-carrier ratio in nepers."""
    if not levels:
        return None
    deviation = float(geometry["white_hz"]) - float(geometry["sync_tip_hz"])
    if not deviation > 0.0:
        return None
    frequency = levels.get("frequency_hz", {})
    nepers = levels.get("level_nepers", {})
    if not {"tip", "porch"} <= set(frequency) or "porch" not in nepers:
        return None
    reading = {
        "fall": (float(frequency["tip"]) - float(geometry["sync_tip_hz"]))
        / deviation,
        "rise": (float(frequency["porch"]) - float(geometry["blanking_hz"]))
        / deviation,
        "lines": int(levels.get("lines", {}).get("tip", 1) or 1),
    }
    if "burst" in nepers:
        reading["chroma"] = float(nepers["burst"]) - float(nepers["porch"])
        reading["chroma_lines"] = int(levels.get("lines", {}).get("burst", 1)
                                      or 1)
    return reading


def timing_reading(measured) -> Optional[Dict[str, float]]:
    """The band delay's departure from the specified interval, in seconds,
    or None where no line carried a burst inside the specified window."""
    if not measured:
        return None
    pooled = measured.get("pooled") or {}
    lines = int(pooled.get("lines", 0) or 0)
    departure = float(pooled.get("departure_ns", float("nan")))
    if lines < 1 or not np.isfinite(departure):
        return None
    return {"seconds": departure * 1e-9, "lines": lines}


# ---------------------------------------------------------------------------
# The witnesses
# ---------------------------------------------------------------------------

def dropout_share(field) -> float:
    """The share of this field's envelope the decoder's own dropout
    criterion would mark: below `dod_threshold_p` of the field's mean
    (`doc.detect_dropouts_rf`). NaN where the decode carries no envelope."""
    video = getattr(field, "data", {}).get("video") if hasattr(field, "data") \
        else None
    if not video or "envelope" not in video:
        return float("nan")
    envelope = np.asarray(video["envelope"], dtype=np.float64)
    if envelope.size == 0:
        return float("nan")
    options = getattr(field.rf, "dod_options", None)
    threshold = float(getattr(options, "dod_threshold_p", float("nan")))
    if not np.isfinite(threshold):
        return float("nan")
    mean = float(np.mean(envelope))
    if not mean > 0.0:
        return 1.0
    return float(np.mean(envelope < threshold * mean))


def _dropout_witness(state, share: float) -> bool:
    """A field is dropout-heavy when its share stands off the running
    median of shares by more than the detection sigma the transform's own
    step gate uses, on the same consistency constant, after the same number
    of fields; nothing is chosen here that the fold has not already
    declared."""
    if not np.isfinite(share):
        return False
    history = state["dropout_history"]
    verdict = False
    if len(history) >= single_transform.STEPS_BEFORE_GATING:
        values = np.asarray(history, dtype=np.float64)
        median = float(np.median(values))
        spread = float(np.median(np.abs(values - median)))
        gate = (chroma_head_switch.DETECTION_SIGMA
                * single_transform.MAD_TO_SIGMA * spread)
        verdict = bool(gate > 0.0 and share - median > gate)
    history.append(float(share))
    del history[:-single_transform.STEP_HISTORY]
    return verdict


def _parity_witness(state, head: int) -> bool:
    """The alternation the format fixes: one head per field. A field whose
    head is not the one expected marks a dropped field or a seek, and is
    refused as evidence; the schedule re-anchors on the next switch."""
    expected = state["expected_head"]
    state["expected_head"] = 1 - head
    return expected is not None and expected != head


# ---------------------------------------------------------------------------
# The hypercube
# ---------------------------------------------------------------------------

def measure_hypercube(rf, geometry, raw) -> Optional[Dict[str, object]]:
    """The file's own null space, once per capture, from the raw field's
    Welch spectrum on the block grid up to Nyquist.

    Occupied: the format's own video band-pass (the luma frequency
    modulation) and the colour-under band the decoder's chroma band-pass
    defines. Everything else is where the data must not exist, and its
    mean density per bin is the constant. The carrier's one-sided power is
    read from the same spectrum, the density subtracted, so the floor
    conversion below uses one instrument in one set of units.
    """
    from scipy import signal as _signal

    values = np.asarray(raw, dtype=np.float64)
    n = int(geometry["block_len"])
    rate = float(geometry["sample_rate_hz"])
    nperseg = min(n, values.size)
    if nperseg < 64:
        return None
    bins_hz, density_per_hz = _signal.welch(values, fs=rate, nperseg=nperseg,
                                            noverlap=nperseg // 2)
    step = rate / nperseg
    power = density_per_hz * step
    occupied = [tuple(float(v) for v in geometry["band_hz"])]
    chroma = chroma_grid(rf, geometry, 2)
    if chroma is not None:
        occupied.append((chroma["carrier_hz"] - chroma["half_width_hz"],
                         chroma["carrier_hz"] + chroma["half_width_hz"]))
    try:
        box = single_transform.hypercube(power, bins_hz, occupied)
    except ValueError:
        return None
    low, high = geometry["band_hz"]
    inside = (bins_hz >= low) & (bins_hz <= high)
    carrier_power = float(np.sum(np.maximum(power[inside] - box["density"],
                                            0.0)))
    return {
        "density": float(box["density"]),
        "bins_outside": int(box["bins_outside"]),
        "bins_inside": int(box["bins_inside"]),
        "band_bins": int(np.count_nonzero(inside)),
        "carrier_power": carrier_power,
        "occupied": box["occupied"],
        "edge_hz": float(box["edge_hz"]),
        "step_hz": step,
        "why": box["why"],
    }


def response_floor(box, samples: int, fills: int, n_axes: int
                   ) -> Dict[str, object]:
    """The hypercube's density as the power one contrast bin of the
    response channel would carry from noise alone. AN ESTIMATE, and
    labelled: a steady carrier of one-sided power P with white noise of
    in-band power N = density x band bins has a log-envelope variance of
    N / (2 P) per sample (the in-phase noise over the amplitude); one
    field's line pools the samples the sibling's fit walks
    (`CURVE_INDEPENDENT_POINTS x CURVE_SAMPLES_PER_POINT`, or the field
    where it is shorter); a contrast bin is the signed mean of 2^n
    vertices each the mean of `fills` fills. Assumes white additive noise
    on a steady carrier and independent samples, and is refreshed as the
    fills grow so it stays comparable with the instrument floor."""
    if box is None or not box["carrier_power"] > 0.0:
        return {"value": float("nan"), "estimate": True,
                "why": "no carrier power to state the noise against"}
    in_band_noise = box["density"] * box["band_bins"]
    per_sample = in_band_noise / (2.0 * box["carrier_power"])
    points = int(luma_amplitude.CURVE_INDEPENDENT_POINTS
                 * luma_amplitude.CURVE_SAMPLES_PER_POINT)
    stride = max(1, int(samples) // points)
    pooled = max(int(samples) // stride, 1)
    per_field = per_sample / pooled
    value = per_field / ((2 ** int(n_axes)) * max(int(fills), 1))
    return {
        "value": float(value),
        "per_sample": float(per_sample),
        "per_field": float(per_field),
        "pooled_samples": int(pooled),
        "fills": int(max(fills, 1)),
        "estimate": True,
        "why": ("N/(2P) per sample for white noise on a steady carrier, "
                "over the samples one field's line pools, over 2^n vertices "
                "and the fills each holds; an estimate, not a measurement"),
    }


# ---------------------------------------------------------------------------
# The head schedule
# ---------------------------------------------------------------------------

def field_period_samples(rf, geometry) -> float:
    """The format's field period in samples, `freq_hz / (2 FPS)`: the
    quantity lddecode/core.py:1104 rounds up into `bytes_per_field`. A
    known constant of the capture, refined below by the median spacing of
    the switches the decoder locates."""
    fps = rf.SysParams.get("FPS")
    if fps:
        return float(geometry["sample_rate_hz"]) / (2.0 * float(fps))
    return float(geometry["sample_rate_hz"]) / float(geometry["field_rate_hz"])


def field_origin(field) -> float:
    """The capture's absolute sample at the field's sample zero:
    `readloc + blockcut`. `DemodCache.doread` loads block b from
    `b * blocksize` for `blocklen` samples (lddecode/core.py:1279-1283),
    `read` keeps each block's raw input cut by `blockcut` before it is
    concatenated (core.py:1363-1365) and reports `startloc` as the first
    block's `b * blocksize` (core.py:1446), which the field takes as its
    `readloc` (core.py:4078). So the field's sample i, in `rawdata` and in
    every channel `linelocs` indexes, is the capture's sample
    `readloc + blockcut + i`; the demodulator's `block_start` is the raw
    window's own `b * blocksize`, on the same axis."""
    readloc = float(getattr(field, "readloc", 0.0) or 0.0)
    return readloc + float(getattr(field.rf, "blockcut", 0) or 0)


def locate_switch(field, geometry, analytics) -> Optional[Dict[str, object]]:
    """The head switch's absolute sample for this field, and how it was
    found.

    The locator first: `head_switch.locate` returns regions in field-sample
    coordinates of which the LAST is the switch (its own record: line 260.2
    +- 0.8 over 24 fields of a colour-bar tape), so the absolute position is `readloc` plus
    the region's start. When it refuses, the window SMPTE 32M clause 3.6
    allows - five to eight lines ahead of the leading edge of vertical sync
    - is applied to the next vertical sync, taken as this field's own
    field-sync leading edge (`head_switch_pair.vertical_sync` on the shared
    analytics) plus one field period, at the window's centre with its half
    width as the uncertainty.
    """
    origin = field_origin(field)
    try:
        from vhsdecode import head_switch as _head_switch
        regions = _head_switch.locate(field)
    except Exception:                                        # noqa: BLE001
        regions = []
    if regions:
        start = float(regions[-1][0])
        return {"sample": origin + start, "source": "locator",
                "field_sample": start, "uncertainty_samples": 0.0}
    if analytics is None:
        return None
    rate = float(geometry["sample_rate_hz"])
    system = geometry.get("system") or "NTSC"
    try:
        edges = head_switch_pair.vertical_sync(analytics["frequency"], rate,
                                               system)
    except Exception:                                        # noqa: BLE001
        return None
    if edges.size == 0:
        return None
    line = sync_geometry.LINE_PERIOD_US["525" if system == "NTSC" else "625"] \
        * 1e-6 * rate
    window = model_stages._HEAD_SWITCH_WINDOW_H
    centre_h = 0.5 * (float(window[0]) + float(window[1]))
    half_h = 0.5 * (float(window[1]) - float(window[0]))
    period = field_period_samples(field.rf, geometry)
    next_vsync = float(edges[0]) + period
    return {"sample": origin + next_vsync - centre_h * line,
            "source": "vertical sync and the specified window",
            "field_sample": next_vsync - centre_h * line,
            "uncertainty_samples": half_h * line}


def _observe_switch(state, field, geometry, analytics, head: int,
                    parity_break: bool) -> Optional[Dict[str, object]]:
    """Record this field's switch as an anchor. The head of the field that
    BEGINS at the switch is the other head, because the switch at the end
    of this field hands over to the next field's head."""
    found = locate_switch(field, geometry, analytics)
    if found is None:
        return None
    if parity_break:
        state["anchors"] = []
    state["anchors"].append(float(found["sample"]))
    del state["anchors"][:-single_transform.STEP_HISTORY]
    state["anchor_parity"] = 1 - int(head)
    state["switch_source"] = found["source"]
    state["switch_uncertainty_samples"] = float(found["uncertainty_samples"])
    return found


def schedule_for(rf, geometry, state) -> Optional[single_transform.HeadSchedule]:
    """The schedule from the anchors seen so far: the latest switch as the
    anchor, the median spacing of successive anchors as the period where
    two or more exist and the format's own period otherwise."""
    anchors = state["anchors"]
    if not anchors or state["anchor_parity"] is None:
        return None
    period = field_period_samples(rf, geometry)
    if len(anchors) >= 2:
        spacing = np.diff(np.asarray(anchors, dtype=np.float64))
        # a spacing near one field period refines the constant; a gap of
        # two or more fields is a lost switch and is not evidence of a
        # different period
        near = spacing[np.abs(spacing - period) < 0.5 * period]
        if near.size:
            period = float(np.median(near))
    state["schedule_version"] = int(state.get("schedule_version", 0)) + 1
    return single_transform.HeadSchedule(
        anchor=float(anchors[-1]), period=float(period),
        parity=bool(state["anchor_parity"]),
        version=int(state["schedule_version"]))


# ---------------------------------------------------------------------------
# The tables
# ---------------------------------------------------------------------------

def _luma_departures(transform, latched) -> Dict[str, np.ndarray]:
    """The luma face's departure on the response channel, split into the
    head-common part and each head's own part, from the latched contrasts
    at their amounts. The polarity axis is dropped by taking its parent,
    because one table serves both landings; at the luma face (band 0) a
    contrast containing `head` carries the sign of the head."""
    sl = transform.channels[RESPONSE]
    bins = sl.stop - sl.start
    common = np.zeros(bins, dtype=np.complex128)
    own = {0: np.zeros(bins, dtype=np.complex128),
           1: np.zeros(bins, dtype=np.complex128)}
    for subset, entry in latched.contrasts.items():
        if "polarity" in subset:
            continue
        value = np.asarray(entry["value"], dtype=np.complex128)[sl]
        amount = np.asarray(latched.amounts[subset], dtype=np.float64)[sl]
        value = value * amount
        if "head" in subset:
            own[0] += value
            own[1] -= value
        else:
            common += value
    return {"common": common, "own": own}


def _half_table(departure, grid, geometry, amount: float, carrier_hz: float
                ) -> Dict[str, object]:
    """One departure on the response grid as a multiplier on the block's
    positive half grid: exp(-amount x departure), placed on the full rfft
    grid with `tesseract.on_full_grid` (flat outside the band), clamped in
    magnitude the way `channel_eq` clamps a table that asks for more than
    a playback channel has, unity at DC, delay-normalised at the carrier so
    the sync edges do not move."""
    rate = float(geometry["sample_rate_hz"])
    n = int(grid["n_fft"])
    bins_hz = grid["bins_hz"]
    scaled = float(amount) * np.asarray(departure, dtype=np.complex128)
    limit = float(channel_eq.MAXIMUM_LOG_MAGNITUDE)
    clamped = int(np.count_nonzero(np.abs(scaled.real) > limit))
    real = np.clip(scaled.real, -limit, limit)
    real_full, full_grid, _index = tesseract.on_full_grid(real, bins_hz, rate)
    imag_full, _grid, _index = tesseract.on_full_grid(scaled.imag, bins_hz,
                                                       rate)
    if real_full.size != n // 2 + 1:
        raise ValueError("the response grid does not lie on the block's "
                         "rfft grid")
    half = np.exp(-(real_full + 1j * imag_full))
    half[0] = 1.0 + 0.0j
    half = channel_eq._delay_normalize(half, full_grid, carrier_hz)
    half[0] = 1.0 + 0.0j
    return {"half": half, "clamped_bins": clamped}


def build_tables(rf, geometry, transform, latched, amount: float
                 ) -> Dict[str, object]:
    """The common table and each head's own multiplier, on the full
    Hermitian grid of length `blocklen`, from the latched luma-face
    departures. Each is delay-normalised separately so their product is
    too."""
    grid = response_grid(rf, geometry)
    parts = _luma_departures(transform, latched)
    carrier_hz = float(geometry["blanking_hz"])
    common = _half_table(parts["common"], grid, geometry, amount, carrier_hz)
    heads = {h: _half_table(parts["own"][h], grid, geometry, amount, carrier_hz)
             for h in (0, 1)}
    return {
        "common": channel_eq._full_table(common["half"]),
        "per_head": {bool(h): channel_eq._full_table(heads[h]["half"])
                     for h in (0, 1)},
        "clamped_bins": int(common["clamped_bins"]
                            + sum(v["clamped_bins"] for v in heads.values())),
        "departures": parts,
        "carrier_hz": carrier_hz,
    }


def _publish(rf, geometry, transform, latched, amount, state, reason: str
             ) -> single_transform.Published:
    tables = build_tables(rf, geometry, transform, latched, amount)
    schedule = schedule_for(rf, geometry, state)
    why = ("rf_transform version %d: %s, treatment %s, amount %.3f (%s); "
           "%d bins clamped; schedule %s" % (
               latched.version, latched.why, latched.treatment, amount,
               reason, tables["clamped_bins"],
               "anchored on the %s" % state.get("switch_source")
               if schedule is not None else "absent, common table only"))
    published = single_transform.publish(
        latched.version, tables["common"], tables["per_head"], schedule, why)
    rf.__dict__[PUBLISHED] = published
    state["tables"] = {"clamped_bins": tables["clamped_bins"],
                       "carrier_hz": tables["carrier_hz"]}
    return published


def _resync_schedule(rf, geometry, state) -> bool:
    """The clock re-synchronised under the same version: the tables the
    snapshot carries are frozen, the schedule is rebound with the latest
    anchor and period. Returns whether anything changed."""
    published = rf.__dict__.get(PUBLISHED)
    if published is None:
        return False
    schedule = schedule_for(rf, geometry, state)
    if schedule is None:
        return False
    old = published.schedule
    if old is not None and old.anchor == schedule.anchor \
            and old.period == schedule.period and old.parity == schedule.parity:
        return False
    rf.__dict__[PUBLISHED] = single_transform.Published(
        version=published.version, common=published.common,
        per_head=published.per_head, schedule=schedule, n_fft=published.n_fft,
        why=published.why)
    return True


# ---------------------------------------------------------------------------
# The fills
# ---------------------------------------------------------------------------

def _fill(transform, band: int, head: int, polarity: int, channel: str,
          values, weight: float, transient: bool, accepted: Dict[str, list],
          refused: Dict[str, list]) -> None:
    vertex = {"band": band, "head": head, "polarity": polarity}
    ok = transform.fill(vertex, channel, values, weight=max(float(weight), 1e-9),
                        transient=transient)
    (accepted if ok else refused).setdefault(channel, []).append(
        (band, head, polarity))


def fill_vertices(transform, field, geometry, context, head: int,
                  transient: bool) -> Dict[str, object]:
    """This field's readings into the vertices it evidences: the luma faces
    at both polarities, the chroma faces at both, under the witnesses."""
    rf = field.rf
    grid = response_grid(rf, geometry)
    accepted: Dict[str, list] = {}
    refused: Dict[str, list] = {}
    notes = []

    luma = luma_response(grid, context.get("amplitude"))
    if luma is None:
        notes.append("no amplitude-probe line on this field")
    else:
        for polarity in (0, 1):
            _fill(transform, 0, head, polarity, RESPONSE, luma, 1.0,
                  transient, accepted, refused)

    chroma = chroma_grid(rf, geometry, grid["bins_hz"].size)
    reading = chroma_response(context.get("analytics"), geometry, chroma) \
        if chroma is not None and geometry.get("system") == "NTSC" else None
    if reading is None:
        notes.append("no colour-under response on this field")
    else:
        for polarity in (0, 1):
            _fill(transform, 1, head, polarity, RESPONSE, reading["value"],
                  reading["lines"], transient, accepted, refused)

    levels = level_readings(context.get("levels"), geometry)
    if levels is None:
        notes.append("no reserved-band levels on this field")
    else:
        _fill(transform, 0, head, 0, LEVELS, [levels["fall"]], levels["lines"],
              transient, accepted, refused)
        _fill(transform, 0, head, 1, LEVELS, [levels["rise"]], levels["lines"],
              transient, accepted, refused)
        if "chroma" in levels:
            for polarity in (0, 1):
                _fill(transform, 1, head, polarity, LEVELS, [levels["chroma"]],
                      levels["chroma_lines"], transient, accepted, refused)

    timing = timing_reading(context.get("band_delay"))
    if timing is None:
        notes.append("band delay refused: no line carried a burst inside the "
                     "specified window")
    else:
        for band in (0, 1):
            for polarity in (0, 1):
                _fill(transform, band, head, polarity, TIMING,
                      [timing["seconds"]], timing["lines"], transient,
                      accepted, refused)

    return {"accepted": accepted, "refused": refused, "notes": notes,
            "readings": {"luma_response": luma, "chroma_response": reading,
                         "levels": levels, "timing": timing}}


def _remainders(transform, readings, head: int, amount: float
                ) -> Dict[str, object]:
    """This field's own readings against the departure the transform would
    remove, under both treatments, at the exact inverse and at the amount
    applied. Available once the fold has evidence at every vertex."""
    out: Dict[str, object] = {}
    items = []
    if readings.get("luma_response") is not None:
        items.append((RESPONSE, 0, readings["luma_response"]))
    if readings.get("chroma_response") is not None:
        items.append((RESPONSE, 1, readings["chroma_response"]["value"]))
    if readings.get("levels") is not None:
        items.append((LEVELS, 0, [readings["levels"]["fall"]]))
    if readings.get("timing") is not None:
        items.append((TIMING, 0, [readings["timing"]["seconds"]]))
    for treatment in single_transform.TREATMENTS:
        rows = {}
        for channel, band, values in items:
            vertex = {"band": band, "head": head, "polarity": 0}
            try:
                exact = transform.remainder(vertex, channel, values, treatment)
                sl = transform.channels[channel]
                expected = transform.departure(vertex, treatment)[sl]
            except (ValueError, IndexError):
                continue
            x = np.asarray(values, dtype=np.complex128).reshape(-1)
            applied = float(np.sqrt(np.mean(np.abs(x - amount * expected) ** 2)))
            rows["%s:band%d" % (channel, band)] = {
                "before": exact["before"], "exact_inverse": exact["after"],
                "applied": applied, "amount": float(amount),
                "share_explained_exact": exact["share_explained"]}
        if rows:
            out[treatment] = rows
    return out


def _vertex_vectors(transform, readings) -> Dict[Tuple[int, int], np.ndarray]:
    """One vector per (band, polarity) covering every channel in the
    transform's own bin order, from this field's readings, or nothing for a
    face the readings do not fill. The wave takes the whole vertex."""
    out: Dict[Tuple[int, int], np.ndarray] = {}
    levels = readings.get("levels") or {}
    timing = readings.get("timing") or {}
    luma = readings.get("luma_response")
    chroma = readings.get("chroma_response")
    seconds = timing.get("seconds")
    for band, response, faces in (
            (0, luma, {0: levels.get("fall"), 1: levels.get("rise")}),
            (1, None if chroma is None else chroma["value"],
             {0: levels.get("chroma"), 1: levels.get("chroma")})):
        if response is None or seconds is None:
            continue
        for polarity, level in faces.items():
            if level is None:
                continue
            parts = {RESPONSE: np.asarray(response, dtype=np.complex128).reshape(-1),
                     LEVELS: np.asarray([level], dtype=np.complex128),
                     TIMING: np.asarray([seconds], dtype=np.complex128)}
            vector = np.zeros(transform.bins, dtype=np.complex128)
            complete = True
            for name, sl in transform.channels.items():
                piece = parts.get(name)
                if piece is None or piece.size != sl.stop - sl.start:
                    complete = False
                    break
                vector[sl] = piece
            if complete:
                out[(band, polarity)] = vector
    return out


def _record_wave(transform, fills, head: int, field, geometry
                 ) -> Optional[Dict[str, object]]:
    """Record this field's remainder into each filled vertex's wave and
    report what the wave holds for it."""
    if transform.latched is None or fills is None:
        return None
    number = getattr(field, "field_number", None)
    rate = float(geometry.get("field_rate_hz") or 0.0)
    if number is None or not rate > 0.0:
        return None
    out: Dict[str, object] = {"recorded": [], "folded": {}, "why": (
        "the remainder after the latch, per vertex, folded on the bits of "
        "the field index; recorded and reported here, applied by the "
        "picture stage which holds the field")}
    for (band, polarity), vector in _vertex_vectors(transform, fills["readings"]).items():
        vertex = {"band": band, "head": head, "polarity": polarity}
        try:
            ordinal = transform.record_remainder(vertex, int(number), vector, rate)
        except ValueError:
            continue
        name = "band%d:head%d:polarity%d" % (band, head, polarity)
        out["recorded"].append(name)
        folded = transform.wave(vertex)
        if folded is not None:
            value = transform.wave_at(vertex, int(number))
            # THE COMPARISON HE ASKED FOR: the remainder after the latched
            # constant alone (the null-space treatment) against the
            # remainder after the constant and the admitted wave, on this
            # field, per channel, in each channel's own units.
            index = transform.vertex_index(vertex)
            after_constant = vector - transform.latched.departures[index]
            after_wave = after_constant if value is None else after_constant - value
            comparison = {}
            for cname, sl in transform.channels.items():
                comparison[cname] = {
                    "after_constant": float(np.sqrt(np.mean(np.abs(after_constant[sl]) ** 2))),
                    "after_wave": float(np.sqrt(np.mean(np.abs(after_wave[sl]) ** 2))),
                }
            out["folded"][name] = {
                "ordinal": ordinal,
                "fields_used": folded["fields_used"],
                "admitted": folded["admitted"],
                "noise_floor": folded["noise_floor"],
                "value_rms": (None if value is None else
                              float(np.sqrt(np.mean(np.abs(value) ** 2)))),
                "null_space_against_wave": comparison,
            }
    return out


def _on_stride(count: int, latched_at: Optional[int]) -> bool:
    """Before the latch every field; after it the doubling stride, fields
    8, 16, 32 and so on, for the report only."""
    if latched_at is None:
        return True
    return count >= 8 and (count & (count - 1)) == 0


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------

def transform_rf(field, amount=None) -> Optional[Dict[str, object]]:
    """The one radio-frequency stage, once per field on the main thread.

    Bails while the stage is off or the field carries no raw radio
    frequency. Memoised per decode by the field's read position, so a
    field the decoder redoes at the same tape position - the one-shot
    redemodulation, the levels and head-switch redos - is not counted twice
    and cannot bump the published version.

    MEASURED on the SP bars capture (2026-09-07, twelve frames at 50 MSps
    resampled to 40, every filler on by declaration): eight active fields
    at 0.70 to 0.90 seconds each with the fillers, one at 7.0 where the
    fillers pool their reports; the latch at field 8 under substitute at
    the half, common table 0.0261 nepers rms log magnitude over the half
    grid and a head ratio of 0.0023; the one-shot redemodulation then
    re-created field 8 at the same read position and was answered from
    the memo in one millisecond without a count or a version; inactive
    fields cost 0.012 seconds; the locator anchored 23 of 25 fields, the
    schedule's period settling at 666048.5 samples against the format's
    667333.7; the hypercube density 8.48 codes squared a bin gave a
    contrast floor estimate of 2.4e-10 nepers squared against an
    instrument floor of 2.0e-7, so the field-to-field variation of the
    probe's line is structure and not the capture's noise; field 16, held
    out of the latch, was explained by the frozen departures to 99.97 per
    cent on the luma response and 99.98 on the chroma.
    """
    rf = field.rf
    options = getattr(rf, "options", None)
    if not getattr(options, STAGE, 0):
        return None
    if getattr(field, "rawdata", None) is None:
        return None
    geometry = model_stages.rf_geometry(field)
    if geometry is None:
        model_stages._log_once(
            field, "rf_transform_geometry",
            "rf_transform declined: this format's definition carries no "
            "video band-pass corners, so no radio-frequency geometry can be "
            "derived and nothing is measured")
        return None

    state = _state(rf)
    readloc = float(getattr(field, "readloc", float("nan")))
    key = int(round(readloc)) if np.isfinite(readloc) else None
    if key is not None and key in state["seen"]:
        reading = _redone_reading(rf, state, state["seen"][key])
        field.rf_transform = reading
        return reading

    transform = transform_for(rf, geometry)
    treatment = resolve_treatment(rf)
    chosen = resolve_amount(rf, amount)
    applied = float(chosen["applied"])
    count = int(state["fields"]) + 1
    state["fields"] = count
    head = int(bool(getattr(field, "isFirstField", False)))
    active = _on_stride(count, state["latched_at"])

    context = rf_context(field, geometry, active)
    parity_break = _parity_witness(state, head)
    share = dropout_share(field)
    dropout_heavy = _dropout_witness(state, share)
    sync_refused = not getattr(field, "valid", True)
    witnesses = [name for name, flag in (
        ("parity break", parity_break), ("dropout-heavy", dropout_heavy),
        ("sync confidence refused the field", sync_refused)) if flag]
    transient = bool(witnesses)

    fills = None
    if active and context["raw"] is not None:
        fills = fill_vertices(transform, field, geometry, context, head,
                              transient)

    if context["raw"] is not None:
        state["field_samples"] = int(context["raw"].size)
        if state["hypercube"] is None:
            state["hypercube"] = measure_hypercube(rf, geometry, context["raw"])
    floor = None
    if state["hypercube"] is not None:
        fewest = transform.evidence()[RESPONSE]["fewest"]
        floor = response_floor(state["hypercube"],
                               int(state.get("field_samples", 0)),
                               fewest, transform.n)
        if np.isfinite(floor["value"]):
            transform.set_floor(RESPONSE, floor["value"])

    # The switch is observed on every field: the locator reads the
    # amplitude deviation the field already carries and is cheap, while the
    # fallback needs the analytics and so exists on active fields only.
    switch = _observe_switch(state, field, geometry, context["analytics"],
                             head, parity_break)

    published_now = False
    if transform.latched is None:
        latched = transform.latch(treatment, model_stages._LOCK_DECISION_FIELDS)
        if latched is not None:
            state["latched_at"] = count
            _publish(rf, geometry, transform, latched, applied, state,
                     chosen["reason"])
            published_now = True
            model_stages._log_once(
                field, "rf_transform_latched",
                "rf_transform: latched at field %d under %s, amount %.2f "
                "(%s); the tables are published and the blocks in flight "
                "are repaid by one redemodulation", count, treatment,
                applied, chosen["reason"])
    elif switch is not None:
        _resync_schedule(rf, geometry, state)

    floors = None
    try:
        if transform.ready(1, "substitute"):
            floors = transform.floors()
    except ValueError:
        floors = None

    remainders = {}
    if fills is not None:
        try:
            if transform.ready(1, "substitute"):
                remainders = _remainders(transform, fills["readings"], head,
                                         applied)
        except ValueError:
            remainders = {}

    # THE WAVE UNDER THE CONVERGENCE. Ethan: "I believe there is still a
    # wave underneath the convergence that itself can be differentialed as
    # random noise and subtracted out." After the latch, this field's own
    # readings less the latched departure go into the vertex's wave, which
    # folds them on the bits of the field index. It is RECORDED and
    # REPORTED here and not applied: the workers demodulate ahead of the
    # field thread, so a per-field term cannot reach the block it belongs
    # to from here; the picture stage, which holds the field, applies it.
    wave = _record_wave(transform, fills, head, field, geometry)

    reading = {
        "stage": STAGE,
        "field": count,
        "readloc": readloc,
        "head": head,
        "active": active,
        "witnesses": witnesses,
        "dropout_share": share,
        "treatment": treatment,
        "amount": chosen,
        "wave": wave,
        "context": {"raw": context["raw"] is not None,
                    "analytics": context["analytics"] is not None,
                    "adapters": context["adapters"],
                    "refusals": context["refusals"]},
        "fills": None if fills is None else {
            "accepted": fills["accepted"], "refused": fills["refused"],
            "notes": fills["notes"]},
        "hypercube": state["hypercube"],
        "floor": floor,
        "floors": floors,
        "remainders": remainders,
        "switch": switch,
        "schedule": _schedule_summary(rf, state),
        "published": _published_summary(rf),
        "published_now": published_now,
        "chroma": {"realised": False,
                   "why": ("the chroma path already carries a static per-head "
                           "colour-under gain and phase; realising the "
                           "chroma faces here would count it twice")},
        "report": transform.report(),
        "redone": False,
    }
    if key is not None:
        state["seen"][key] = {k: reading[k] for k in (
            "stage", "field", "readloc", "head", "active", "witnesses",
            "dropout_share", "treatment", "amount", "fills")}
        while len(state["seen"]) > REDO_MEMORY_FIELDS:
            state["seen"].popitem(last=False)
    field.rf_transform = reading
    return reading


def _redone_reading(rf, state, previous: Dict[str, object]) -> Dict[str, object]:
    """The reading for a field the decoder re-created at a read position
    already seen: the stored summary of the first pass, the current
    snapshot and schedule, and every key a first-pass reading carries, so
    a consumer of `field.rf_transform` meets one shape. Nothing is read,
    filled or counted."""
    transform = _store(rf).get(KIND)
    reading = dict(previous)
    reading.update({
        "redone": True,
        "context": {"raw": False, "analytics": False, "adapters": [],
                    "refusals": ["redone at a read position already seen; "
                                 "nothing read or counted twice"]},
        "hypercube": state["hypercube"],
        "floor": None,
        "floors": None,
        "remainders": {},
        "wave": None,
        "switch": None,
        "schedule": _schedule_summary(rf, state),
        "published": _published_summary(rf),
        "published_now": False,
        "chroma": {"realised": False,
                   "why": ("the chroma path already carries a static per-head "
                           "colour-under gain and phase; realising the "
                           "chroma faces here would count it twice")},
        "report": transform.report() if transform is not None else None,
    })
    return reading


def _schedule_summary(rf, state) -> Optional[Dict[str, object]]:
    published = rf.__dict__.get(PUBLISHED)
    schedule = published.schedule if published is not None else None
    return {
        "anchors": len(state["anchors"]),
        "source": state.get("switch_source"),
        "uncertainty_samples": state.get("switch_uncertainty_samples"),
        "published": None if schedule is None else {
            "anchor": schedule.anchor, "period": schedule.period,
            "parity": schedule.parity, "version": schedule.version},
    }


def _published_summary(rf) -> Optional[Dict[str, object]]:
    published = rf.__dict__.get(PUBLISHED)
    if published is None:
        return None
    common = published.common
    inside = np.abs(np.log(np.abs(common[:common.size // 2 + 1])))
    per_head = published.per_head or {}
    return {
        "version": int(published.version),
        "n_fft": int(published.n_fft),
        "per_head": bool(per_head),
        "common_log_magnitude_rms": float(np.sqrt(np.mean(inside ** 2))),
        "head_ratio_log_magnitude_rms": (float(np.sqrt(np.mean(np.log(
            np.abs(per_head[True][:common.size // 2 + 1])
            / np.abs(per_head[False][:common.size // 2 + 1])) ** 2)))
            if len(per_head) == 2 else float("nan")),
        "why": published.why,
    }
