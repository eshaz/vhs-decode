"""The model-based chroma stages, wired into the decode.

`vhsdecode/models/` holds the arc's measurement modules, and until now almost
none of them could be reached from a decode: two runtime importers in the whole
tree. This file is the glue for those that carry an explicit
`RUNTIME_CONTRACT`, and it exists as its own module rather than as more code
inside `vhsdecode/chroma.py` so that the call sites there stay two lines each
and the geometry, the gating and the per-decode state have one home.

NO FLAG OF ITS OWN, AND THAT IS THE DESIGN. Ethan: *"Let's retire all the
one-off options and have everything we have done so far live in the graph and
have a consistently modeling structure."* Each stage below is a NODE in
`vhsdecode/pipeline/stages.toml` and nothing else: the declaration carries its
default - ON, seeded through `declared_default = true` - and the graph
selector that already exists turns it off or on by name -

    vhs-decode ... --stages -chroma_head_switch,-colour_free_luma

There is no `--chroma_head_switch`, no umbrella, and no second gating
mechanism beside the selector. `--debug_plot pipeline_graph` draws what a
given selection resolves to.

WHAT IS WIRED HERE, AND WHAT EACH ONE IS.

  `colour_lock`            THE MEASUREMENT the other two stand on. The decoded
                           chroma and the colour-under replica the demodulated
                           luma carries are the same physical vector in two
                           frames; their difference is a constant that repeats
                           every four lines, and the departure from it is a
                           residual channel. Applies nothing by itself, so it
                           is not a node - it is what the two nodes measure
                           with, and it is taken once a field and shared.

  `chroma_head_switch`     The colour-under phase event at the head switch,
                           found in that residual and compensated on the four
                           or five lines it occupies.

  `source_correction`      THE LUMA RESPONSE THE SIGNAL CARRIED INTO THE
                           RECORDING VCR, taken back off. The only stage here
                           that is not a chroma stage and the only one on the
                           source side of the record head: everything else in
                           this file, and every VCR model in the arc, acts on
                           what the tape and the decks did. Measured on the
                           sync pulse, head-split so the per-head playback
                           term is excluded, and referred to SMPTE 170M
                           clause 7.1's flat luma.

  `colour_free_luma`       The up-heterodyned residual chroma, locked and
                           subtracted from the luma over the active area. The
                           same coupling `luma_beat` removes, estimated in the
                           channel the picture is actually made of, so it
                           SUPERSEDES that node rather than running beside it -
                           declared in the graph and refused at startup.

AND FIVE THAT MEASURE AND APPLY NOTHING, wired in the same shape and
described where they stand, at the foot of this file:

  `burst_instrument`       The burst read as the chroma's own instrument, in
                           the three dimensions the sync pulse gives the luma,
                           plus the per-line and per-field readings of one
                           specification that do not agree.
  `chroma_leakage`         Chroma left in the delivered luma, on both of the
                           channels Ethan names, pooled under the colour
                           frame and judged against a control measured on the
                           same material.
  `colour_framing`         The specified four-field sequence, the decoder's
                           shipped map checked against it, and an explicit
                           refusal to measure a framing this medium does not
                           carry.
  `colour_under`           The low carrier itself, measured in the burst
                           window of the down-converted chroma before the
                           heterodyne puts it back at the subcarrier.
  `iq_imbalance`           The chroma's quadrature imbalance, from the same
                           burst - which is where a sheared vectorscope
                           comes from, and the only place in the decode
                           where an image can be told from its signal at
                           all.
  `vectorscope`            The colour difference coordinate system, read off
                           the directions the transitions between colours
                           run in, with the shear a rotation cannot fix.

EVERY DIMENSION CARRIES. Ethan, in the same breath: *"All of the things that
we are modeling all need to be analyzed in hilbert space so all dimensions
carry through the entire graph."* Both channels are handled as analytic
envelopes from end to end and nothing here reduces one to a magnitude at a
boundary: `measure` hands on the complex chroma envelope, the complex luma
replica, the complex line product AND its angle; the lock offset is a table of
complex numbers; the transfer is one complex number a field; and the pooled
head-switch table is used on BOTH its axes - its angle is the phase applied and
its magnitude is the agreement between fields that decides whether to apply it
at all. The one deliberate reduction is `colour_lock`'s normalisation of each
line product to unit modulus, and that is the module's own finding rather than
an oversight: on a matched pair of captures the amplitude at the colour-under
carrier is the same to four parts in a thousand while the LOCK collapses from
0.98 to 0.13, so amplitude carries no information about this quantity and
weighting by it would let a saturated field outvote a pale one.

THE GEOMETRY IS DERIVED, NEVER DECLARED. Every span below comes from the
system parameters or from the specification, because a line number written
into this file would be a decode-specific constant and the arc's standing rule
forbids one:

  * the active area from `SysParams["activeVideoUS"]`;
  * the first picture line from the vertical interval's own structure - three
    sections of `numPulses` half-lines, pre-equalizing, serration and
    post-equalizing (SMPTE 170M clause 8.3, RS-170A figure 1) - taken no
    earlier than the decoder's own `burst_detected_line`;
  * the last quiet line from SMPTE 32M-2004 clause 3.6, which permits the
    playback head switch to fall between five and eight lines ahead of the
    leading edge of vertical sync, so the quiet span must end before the
    widest of those;
  * the colour-under carrier and the subcarrier from `DecoderParams` and
    `SysParams`, so the decode's own measured carrier is used and not the
    format's nominal one.

THE LOCK IS GATED, AND THE GATE IS A STATEMENT ABOUT NOISE. `colour_lock`
records that its resultant collapses from 0.98 on a tape carrying colour to
0.13 on a matched capture with nothing in the chroma band, and it declines to
choose the threshold: *"Where the threshold belongs is a decision for the
runtime and is not taken here."* This is that decision, and it is taken from
the null distribution rather than from the measured populations. The resultant
of the mean of `n` independent unit phasors drawn at random exceeds `r` with
probability `exp(-n r^2)` (Rayleigh); requiring that probability to be no
larger than the two-sided Gaussian tail at `chroma_head_switch.DETECTION_SIGMA`
- the same significance the head-switch detector already uses, so one number
governs both - gives

    r_min = sqrt(-ln(P(|Z| > 4)) / n)   =   sqrt(9.67 / n)

which on a field of roughly two hundred picture lines, fifty per four-line
residue, is 0.44. Against the figures `colour_lock` reports that admits every
tape carrying colour (0.90 to 0.99) and refuses both the y-only control (0.07
to 0.20) and the home recording (0.11 to 0.37), with no threshold taken from
either population.

AND THE DECISION IS PER DECODE, NOT PER FIELD. Whether the luma carries a
usable colour-under replica is a property of the TAPE. Gating field by field
alone let a marginal recording flicker between corrected and uncorrected
fields, which steps the picture where the estimate lands - the failure
`luma_beat` records for a model that arrives part way through a decode. Run
here on `/testdata/home.flac` the per-field resultant straddles the floor:
0.411 on the first field against a floor of 0.398, and 0.379 four fields
later. `_believable` therefore latches the refusal once a complete colour
frame has been seen and the median of those fields is below the floor.

NTSC ONLY, AND SAID RATHER THAN ASSUMED. The four-line period both stages fit
is derived for NTSC: the colour-under heterodyne advances ninety degrees a
line and the four-times-subcarrier sampling frame a hundred and eighty, so
their difference closes after four lines. PAL's line-alternating subcarrier
does not give the same closure and no PAL evidence was taken, so these stages
decline on anything but NTSC rather than fit a period they have not shown.
"""

import math

import numpy as np

import lddecode.core as ldd

from vhsdecode.models import chroma_head_switch, colour_free_luma, colour_lock
from vhsdecode.models import profiles, source_correction
from vhsdecode.models import single_transform
from vhsdecode.models import spec_signal

# SMPTE 32M-2004 clause 3.6: "The switching position between the two heads
# during playback shall lie between 5 and 8 horizontal lines ahead of the
# leading edge of the vertical sync signal." The quiet span the lock is fitted
# on has to end before the earliest line the switch may reach, so the WIDEST
# of the two is what bounds it. Read through the specification module so the
# clause travels with the number.
from vhsdecode.models import vhs_specification

_HEAD_SWITCH_WINDOW_H = vhs_specification.value_of("head_switch_window_h")

# The vertical interval is three sections of `numPulses` half-lines -
# pre-equalizing, serration, post-equalizing (SMPTE 170M clause 8.3). Stated
# as a count of sections rather than as a line number so it follows the
# system parameters into 625-line and other systems unchanged.
_VERTICAL_SYNC_SECTIONS = 3

# The Gaussian two-sided tail at the significance `chroma_head_switch` already
# uses for its detector. One significance governs both the lock's gate and the
# detector's threshold, which is why this is computed from that constant
# rather than written down beside it.
_LOCK_TAIL_PROBABILITY = math.erfc(chroma_head_switch.DETECTION_SIGMA
                                   / math.sqrt(2.0))

_LOG_ONCE = "_model_stages_logged"


def _system_supported(field):
    """The four-line lock period is derived for NTSC and shown only there."""
    return getattr(field.rf, "color_system", None) == "NTSC"


def _log_once(field, key, message, *args):
    """One line per decode per subject, so a per-field event does not fill
    the log with two hundred and sixty identical sentences."""
    seen = field.rf.__dict__.setdefault(_LOG_ONCE, set())
    if key in seen:
        return
    seen.add(key)
    ldd.logger.info(message, *args)


def geometry(field):
    """The active area, the picture lines and the quiet span, all derived.

    Returns a dictionary rather than a tuple because five of the six entries
    are consumed by name at two different call sites, and a positional tuple
    that three functions unpack differently is how a span quietly becomes the
    wrong one.
    """
    sys_params = field.rf.SysParams
    lines = int(field.outlinecount)
    active = (
        int(math.ceil(field.usectooutpx(sys_params["activeVideoUS"][0]))),
        int(math.floor(field.usectooutpx(sys_params["activeVideoUS"][1]))),
    )
    vertical_interval = int(math.ceil(
        _VERTICAL_SYNC_SECTIONS * sys_params["numPulses"] / 2.0))
    first_picture = int(field.lineoffset) + vertical_interval
    # The decoder's own colour killer has already decided where colour starts;
    # below that line the chroma is set to zero by the automatic gain, so a
    # lock fitted there would be fitting nothing.
    detected = int(getattr(field, "burst_detected_line", 0) or 0)
    first_picture = max(first_picture, detected)
    # The buffer's final row spans the field boundary and carries the next
    # field's equalizing pulse, so it is never a picture line.
    last_quiet = lines - 1 - int(math.ceil(max(_HEAD_SWITCH_WINDOW_H)))
    return {
        "lines": lines,
        "width": int(field.outlinelen),
        "active": active,
        "picture": (first_picture, lines - 1),
        "quiet": (first_picture, last_quiet),
        "sample_rate_hz": float(sys_params["outfreq"]) * 1e6,
        "subcarrier_hz": float(sys_params["fsc_mhz"]) * 1e6,
        "carrier_hz": float(field.rf.DecoderParams["color_under_carrier"]),
        # `hz_to_output_array` maps the sync tip onto `outputZero`, so black
        # sits one sync depth above it and `out_scale` is output units an IRE.
        "black_level": (float(sys_params["outputZero"])
                        - float(field.rf.DecoderParams["vsync_ire"])
                        * float(field.out_scale)),
    }


def _minimum_resultant(offset):
    """The lock strength a field has to reach before it is believed.

    Derived in the module docstring from the Rayleigh null distribution at the
    detector's own significance. The smallest residue population is used, so
    the gate is set by the weakest of the four estimates rather than by their
    average.
    """
    counts = np.asarray(offset["lines_per_residue"], dtype=np.float64)
    counts = counts[counts > 0]
    if counts.size == 0:
        return 1.0
    return float(np.sqrt(-np.log(_LOCK_TAIL_PROBABILITY) / counts.min()))


# One complete NTSC colour frame. The smallest span over which the lock can be
# judged without being biased by field parity or by where in the four-field
# sequence the decode happened to start, so it is the smallest honest basis for
# a decision about the TAPE rather than about one field.
_LOCK_DECISION_FIELDS = 4


def _believable(field, offset):
    """Whether this decode's colour lock may be believed, decided once.

    Judged per field AND per decode, and the second half is the point.
    Whether the luma carries a usable colour-under replica is a property of
    the TAPE, not of one field; a per-field gate alone lets a marginal
    recording flicker between corrected and uncorrected fields, which steps
    the picture where the estimate lands - the failure `luma_beat` records
    for a model that arrives part way through a decode.

    So: a field is refused if its own lock is below the floor, and the whole
    decode is refused once a complete colour frame has been seen and the
    MEDIAN of those fields is below it. The decision latches, so it is taken
    once and reported once.
    """
    verdict = field.rf.__dict__.get("_colour_lock_verdict")
    if verdict is not None:
        return verdict

    minimum = _minimum_resultant(offset)
    history = field.rf.__dict__.setdefault("_colour_lock_history", [])
    history.append(float(offset["resultant"]))

    if len(history) >= _LOCK_DECISION_FIELDS:
        median = float(np.median(history))
        field.rf.__dict__["_colour_lock_verdict"] = median >= minimum
        if median < minimum:
            _log_once(field, "weak",
                      "model stages: declined for this decode - the colour "
                      "lock's median over %d fields is %.3f against the %.3f a "
                      "random phase would reach at %.1f sigma, so the luma "
                      "carries no usable colour-under replica on this tape",
                      len(history), median, minimum,
                      chroma_head_switch.DETECTION_SIGMA)
            return False
        # ACCEPTED FOR THE DECODE, and the acceptance latches as firmly as
        # the refusal. A field whose own lock happens to sit below the floor
        # is not refused after this point, because the fit is already
        # self-limiting where the evidence is thin: a least squares transfer
        # against a model the data does not contain comes out at nearly zero,
        # which is what `colour_free_luma` demonstrates on its y-only control
        # - 0.07 per cent removed with no gate and no burst test. Refusing
        # those fields instead would make the correction come and go field to
        # field, which steps the picture where it lands.
        return True

    # Before the decode has been judged - the first colour frame - a field is
    # taken on its own lock, so a weak opening field cannot seed a table.
    if offset["resultant"] < minimum:
        ldd.logger.debug(
            "model stages: field declined, colour lock %.3f below %.3f",
            offset["resultant"], minimum)
        return False
    return True


def measure(field, uphet):
    """The colour lock for this field, measured once and cached on it.

    Both stages need the same two envelopes and the same per-line residual,
    and the measurement is the expensive half of either, so it is taken once.
    The cache is the FIELD and not the decoder: the quantity is per field and
    a stale one would be applied to the wrong picture.

    Returns None where the field cannot support the measurement - a system
    whose four-line period is not derived, a killed colour field, a picture
    with no time base corrected luma yet, or a lock too weak to believe.
    """
    cached = field.__dict__.get("_colour_lock")
    if cached is not None:
        return cached["lock"] if cached else None
    field.__dict__["_colour_lock"] = False

    if not _system_supported(field):
        _log_once(field, "system",
                  "model stages: declined, the four-line colour lock is "
                  "derived for NTSC and this decode is %s",
                  getattr(field.rf, "color_system", "unknown"))
        return None

    picture = getattr(field, "dspicture", None)
    if picture is None or uphet is None:
        return None

    geom = geometry(field)
    expected = geom["lines"] * geom["width"]
    if len(picture) != expected or len(uphet) != expected:
        return None
    if geom["quiet"][1] - geom["quiet"][0] < colour_lock.OFFSET_PERIOD_LINES:
        return None

    luma_field = np.asarray(picture, dtype=np.float64).reshape(
        geom["lines"], geom["width"])
    chroma_field = np.asarray(uphet, dtype=np.float64).reshape(
        geom["lines"], geom["width"])

    # The carriers are the DECODE'S OWN, not the format's nominal pair, and
    # the chroma is centred on zero here because `chroma_to_u16` has not run
    # yet - the up-converted chroma is still signed at this point in the
    # chain, and the module's 32768 default is the offset it acquires later.
    chroma_envelope = colour_lock.chroma_envelope(
        chroma_field, geom["sample_rate_hz"],
        subcarrier_hz=geom["subcarrier_hz"], centre=0.0)
    luma_envelope = colour_lock.luma_colour_under(
        luma_field, geom["sample_rate_hz"], carrier_hz=geom["carrier_hz"],
        black_level=geom["black_level"])
    product = colour_lock.line_product(chroma_envelope, luma_envelope,
                                       geom["active"])
    offset = colour_lock.lock_offset(product, geom["quiet"])

    if not _believable(field, offset):
        return None

    deviation = colour_lock.residual(product, offset)
    found = chroma_head_switch.detect(deviation, geom["quiet"])
    quiet = geom["quiet"]
    if found["lines"]:
        try:
            quiet = chroma_head_switch.quiet_span(
                geom["lines"], geom["picture"], found["lines"])
        except ValueError:
            quiet = geom["quiet"]

    lock = {
        "geometry": geom,
        "offset": offset,
        # BOTH AXES CARRY. `residual_deg` is the angle a consumer usually
        # wants, but the quantity it came from is complex and its magnitude is
        # the strength of that line's evidence, so the product travels with
        # it rather than being re-derived - or, worse, not being available to
        # be re-derived. The arc has lost a measurement's rank at exactly
        # this kind of boundary before.
        "line_product": product,
        "residual_deg": deviation,
        "chroma_envelope": chroma_envelope,
        "luma_colour_under": luma_envelope,
        "detected": found,
        "quiet": (int(quiet[0]), int(min(quiet[1], geom["quiet"][1]))),
    }
    field.__dict__["_colour_lock"] = {"lock": lock}
    _log_once(field, "locked",
              "model stages: colour lock %.3f (needs %.3f), head switch at "
              "line %s, %.1f sigma",
              offset["resultant"], _minimum_resultant(offset),
              found["peak_line"], found["peak_sigma"])
    return lock


def correct_chroma_head_switch(field, uphet, amount):
    """The colour-under phase event at the head switch, compensated.

    Follows `chroma_head_switch.RUNTIME_CONTRACT`: detect in the luma-chroma
    residual, accumulate the chroma's OWN departure across fields of the same
    parity, and apply once the store holds more than one field. The detector
    says where; the phase is not the detector's own quantity, because both
    channels depart at the switch and their difference over-corrects by about
    a factor of two - a table the module records with its measurement.

    Returns True where a rotation was applied.
    """
    lock = measure(field, uphet)
    if lock is None or not amount:
        return False
    reading = _read_chroma_head_switch(field, lock)
    if reading is None:
        return False
    _realise_chroma_head_switch(uphet, reading, amount)
    phases = reading["phases"]
    strength = reading["strength"]
    _log_once(field, "head_switch_applied",
              "model stages: chroma_head_switch correcting lines %s by %s "
              "degrees at gain %.2f, fields agreeing %s",
              sorted(phases),
              ["%.1f" % np.degrees(v) for _, v in sorted(phases.items())],
              float(amount),
              ["%.2f" % strength[line] for line in sorted(phases)])
    return True


def _read_chroma_head_switch(field, lock):
    """THE MEASUREMENT HALF OF `correct_chroma_head_switch`: this field's
    departure folded into the per-parity table once, and the table's
    verdict - the phases to apply, or None while the store holds fewer than
    two fields or the fields disagreed.
    """
    geom = lock["geometry"]
    parity = bool(field.isFirstField)
    store = field.rf.__dict__.setdefault("_chroma_head_switch", {})
    entry = store.setdefault(parity, {"table": None, "fields": 0,
                                      "counts": {}})

    # One field folds in once. `downscale` can be re-run on the same field
    # object when the decoder redoes a field, and a table that counted the
    # same field twice would weight it twice.
    if not field.__dict__.get("_chroma_head_switch_folded"):
        field.__dict__["_chroma_head_switch_folded"] = True
        if lock["detected"]["lines"]:
            entry["table"] = chroma_head_switch.accumulate(
                entry["table"], lock["chroma_envelope"], geom["active"],
                lock["detected"]["lines"], lock["quiet"])
            for line in lock["detected"]["lines"]:
                entry["counts"][int(line)] = entry["counts"].get(
                    int(line), 0) + 1
            entry["fields"] += 1

    if entry["fields"] < 2 or not entry["table"]:
        return None

    # THE TABLE'S OTHER AXIS, USED RATHER THAN DISCARDED. Each entry is a sum
    # of unit phasors, so its ANGLE is the phase to apply and its MAGNITUDE
    # over the count is how well the fields agreed - `chroma_head_switch`
    # calls that the confidence and says plainly what it is for: "Use it to
    # refuse a table built on fields that disagreed." Taking the angle alone
    # would drop half of what was measured at this boundary.
    #
    # The refusal is the null distribution's own median: `k` unit phasors
    # drawn at random reach a resultant of sqrt(ln 2 / k) half the time, so a
    # table whose best line sits below that is more likely noise than event.
    # A median rather than a tail because this is a shrinkage decision on a
    # measured quantity, not a detection - `detect` already did the detecting,
    # at four sigma, on a channel this one does not use.
    strength = {line: abs(value) / max(entry["counts"].get(int(line), 1), 1)
                for line, value in entry["table"].items()}
    best = max(strength.values()) if strength else 0.0
    floor = math.sqrt(math.log(2.0)
                      / max(max(entry["counts"].values(), default=1), 1))
    if best < floor:
        _log_once(field, "head_switch_refused",
                  "model stages: chroma_head_switch declined, the pooled "
                  "table's best line agrees at %.2f against the %.2f a random "
                  "phase reaches half the time over %d fields",
                  best, floor, max(entry["counts"].values(), default=0))
        return None

    phases = chroma_head_switch.phases(entry["table"])
    if not phases:
        return None
    return {"geometry": geom, "phases": phases, "strength": strength}


def _realise_chroma_head_switch(uphet, reading, amount):
    """THE REALISATION HALF: the table's rotation on the up-converted
    chroma, written in place at the caller's gain."""
    geom = reading["geometry"]
    corrected = chroma_head_switch.correct_field(
        np.asarray(uphet, dtype=np.float64).reshape(geom["lines"],
                                                    geom["width"]),
        reading["phases"], gain=float(amount), centre=0.0)
    uphet[:] = corrected.reshape(-1).astype(uphet.dtype, copy=False)


def correct_colour_free_luma(field, uphet, amount):
    """The up-heterodyned residual chroma, locked and taken off the luma.

    Follows `colour_free_luma.RUNTIME_CONTRACT`: the site is where
    `luma_beat.correct` already stands, the transfer is one complex number a
    field fitted on the active area alone, and the picture lines exclude the
    head-switch lines the detector flagged - the lock is not fitted there.

    THE PICTURE IS WRITTEN IN PLACE. `field.dspicture` and the `dsout` the
    caller is about to return are the same array, so rebinding the attribute
    would silently correct nothing.

    Returns the share of the colour-under band removed, held out on the lines
    the transfer was not fitted on, or None where nothing was applied.
    """
    lock = measure(field, uphet)
    if lock is None or not amount:
        return None
    reading = _read_colour_free(lock, amount)
    if reading is None:
        return None

    geom = lock["geometry"]
    picture = field.dspicture
    luma_field = np.asarray(picture, dtype=np.float64).reshape(
        geom["lines"], geom["width"])
    corrected = _realise_colour_free(luma_field, reading)
    np.clip(corrected, 0.0, 65535.0, out=corrected)
    picture[:] = corrected.reshape(-1).astype(picture.dtype, copy=False)

    transfer = reading["transfer"]
    share = reading["share"]
    field.colour_free_luma = {
        "transfer": complex(transfer),
        "removed": float(share["removed"]),
        "lock_resultant": float(lock["offset"]["resultant"]),
    }
    _log_once(field, "colour_free_applied",
              "model stages: colour_free_luma transfer |h| %.4f, %.2f%% of "
              "the colour-under band removed (held out), lock %.3f",
              abs(transfer), 100.0 * share["removed"],
              lock["offset"]["resultant"])
    ldd.logger.debug(
        "colour_free_luma: |h| %.4f, removed %.2f%%, lock %.3f",
        abs(transfer), 100.0 * share["removed"],
        lock["offset"]["resultant"])
    return share


def _read_colour_free(lock, amount):
    """THE MEASUREMENT HALF OF `correct_colour_free_luma`: the transfer, one
    complex number a field fitted on the active area alone, and its held-out
    share - measured, not applied.

    Fitted on even picture lines and judged on odd ones, which is the split
    every figure in `colour_free_luma`'s docstring was taken under. The
    SUBTRACTION uses every line; the split is a property of the measurement.
    """
    geom = lock["geometry"]
    lines = lock["quiet"]
    if lines[1] - lines[0] < 2 * colour_lock.OFFSET_PERIOD_LINES:
        return None
    predicted = colour_free_luma.prediction(lock["chroma_envelope"],
                                            lock["offset"])
    measured = lock["luma_colour_under"]
    transfer = colour_free_luma.transfer(predicted, measured, geom["active"],
                                         lines, stride=2)
    share = colour_free_luma.removed_share(predicted, measured, transfer,
                                           geom["active"], lines, stride=2,
                                           offset_line=1)
    return {
        "geometry": geom,
        "predicted": predicted,
        "transfer": complex(transfer),
        "gain": complex(transfer) * float(amount),
        "share": share,
        "lock_resultant": float(lock["offset"]["resultant"]),
    }


def _realise_colour_free(luma_field, reading):
    """THE REALISATION HALF: the locked estimate re-modulated and taken off
    the active area. Returns a NEW float field, as the module does."""
    geom = reading["geometry"]
    return colour_free_luma.colour_free(
        luma_field, reading["predicted"], reading["gain"], geom["active"],
        geom["sample_rate_hz"], carrier_hz=geom["carrier_hz"])


# ==========================================================================
# THE CHROMA MEASUREMENT STAGES
#
# Ethan, 2026-09-06: *"Every stage that we have designed must be called and
# used. Exhaustively go through the stages and make sure they are being
# used. All of them need to be used and all of them need to model all
# dimensions."*
#
# Five modules of the colour group were designed and could not be reached
# from a decode. They are wired below in the same shape as the two
# corrections above - a node in `pipeline/stages.toml`, a default of ON
# carried by the declaration (`declared_default = true`), and
# `--stages -<name>` as the only control -
# with one difference that is deliberate and is the whole of their design:
#
#   THEY MEASURE AND THEY APPLY NOTHING. Not one of them writes a sample.
#   `burst_instrument`'s specified separator tilt is not confirmed on the SP
#   tapes, `chroma_leakage` is the assertion that a leak EXISTS rather than
#   a way to remove it, `colour_framing` has established that its own
#   subject is not on the tape, `colour_under` reads a channel the decoder
#   has already filtered, and `vectorscope` reads a coordinate system whose
#   correction is a two-by-two map nobody has yet ruled on. A stage that
#   measures is still a stage that must be callable, and a measurement
#   shipped as a correction is how an unconfirmed model reaches a picture.
#
# So each of the five, enabled, leaves the decode BYTE-IDENTICAL and writes
# what it found to the log: the pooled verdict once per decode at INFO, and
# the per-field reading at DEBUG so the pooling can be checked afterwards.
#
# AND EACH DECLINES RATHER THAN GUESSING. Every gate below is a null
# distribution evaluated on the field's own sample count, never a chosen
# threshold, and every refusal says in the log what it refused and why.
# ==========================================================================

# ONE IMPORT A LINE, and that is not a style choice. `stage_inventory` reads
# these files as text to answer "is this module reached from a decode at
# all", and its pattern for an import does not cross a newline - so a
# parenthesised import spanning two lines would leave the modules on the
# second line still counted as unwired, which is the exact measurement this
# work exists to move.
from vhsdecode.models import burst_instrument
from vhsdecode.models import chroma_leakage
from vhsdecode.models import colour_framing
from vhsdecode.models import colour_under
from vhsdecode.models import iq_imbalance
from vhsdecode.models import tape_bias
from vhsdecode.models import vectorscope

# One significance for the whole chroma path. `chroma_head_switch` detects at
# it, the colour lock is gated on it, and the three stages below take their
# floors from it, so a decode has one number governing what counts as
# evidence rather than five.
_SIGMA = chroma_head_switch.DETECTION_SIGMA
_TAIL = _LOCK_TAIL_PROBABILITY

# One complete NTSC colour frame, which is the smallest span over which
# anything carrying the four-field sequence can be pooled without being
# biased by where in that sequence the decode happened to start.
_COLOUR_FRAME_FIELDS = _LOCK_DECISION_FIELDS


def _measurement_state(field, key):
    """The per-decode store for one measurement stage.

    On `field.rf`, because these pool ACROSS fields and a per-field cache
    would defeat the pooling; and keyed by name so five stages do not share
    one dictionary and overwrite each other.
    """
    return field.rf.__dict__.setdefault("_chroma_measurements", {}).setdefault(
        key, {"fields": 0, "reported": False})


def _rayleigh_floor(count):
    """The resultant `count` independent random unit phasors reach at the
    chroma path's own significance. `exp(-n r^2)` is the Rayleigh tail, so
    this is the same derivation the colour lock's gate uses, applied to
    whatever population the caller has."""
    if count < 1:
        return 1.0
    return float(np.sqrt(-np.log(_TAIL) / float(count)))


# THE OUTPUT'S OWN ACTIVE SPAN, and why it is not `geometry()`'s.
#
# `lddecode/core.py`'s `build_json` writes `activeVideoStart` and
# `activeVideoEnd` as `round(activeVideoUS * outfreq + badj)` with `badj`
# of -1.4 samples - the burst adjustment it has carried since 2019 - while
# `geometry()` above takes the ceiling and floor of the same microseconds
# without it. On a 4fsc NTSC field that is 134 to 894 against 136 to 895,
# and the difference is not cosmetic: it is the window every consumer of
# the written file uses.
#
# THE STATISTIC IS SENSITIVE TO IT, MEASURED. On three fields dumped from a
# real decode of `zaroff-75bars-NTSC-SP`, the same leakage resultant read
# over five plausible active spans moved between 0.0009 and 0.0275 - a
# factor of thirty - because it is a single transform bin and its Dirichlet
# response to the picture's own low-frequency content depends on the
# window's length. A runtime figure and an offline figure taken on
# different windows are therefore not comparable at all, which is exactly
# the runtime-versus-offline trap this arc has recorded before. So the
# measurement stages use the OUTPUT's span, and a reading taken here can be
# reproduced from the written pair.
#
# AND IT IS TRIMMED TO WHOLE SUBCARRIER CYCLES. At four times the
# subcarrier a cycle is four samples exactly (SMPTE ST 244), so a window
# that is a multiple of four puts the subcarrier on an exact transform bin
# and the estimator's response to a constant is identically zero. The
# output's own 134 to 894 is 760 samples, which is 190 whole cycles - it
# already satisfies this - but that is a property to be checked rather than
# relied on, because nothing in the output rule guarantees it.
_OUTPUT_BURST_ADJUST_SAMPLES = -1.4
_SAMPLES_PER_SUBCARRIER_CYCLE = 4


def _output_active_span(field, geom):
    """The active span as the written file declares it, on whole cycles."""
    spu = float(field.rf.SysParams["outfreq"])
    active_us = field.rf.SysParams["activeVideoUS"]
    start = int(round(active_us[0] * spu + _OUTPUT_BURST_ADJUST_SAMPLES))
    stop = int(round(active_us[1] * spu + _OUTPUT_BURST_ADJUST_SAMPLES))
    start = max(start, 0)
    stop = min(stop, geom["width"])
    cycles = (stop - start) // _SAMPLES_PER_SUBCARRIER_CYCLE
    if cycles < 1:
        return None
    return start, start + cycles * _SAMPLES_PER_SUBCARRIER_CYCLE


def _bin_leakage(hz, samples, sample_rate_hz):
    """The estimator's response to a CONSTANT over this window.

    A single-frequency sum over `samples` places the frequency on an exact
    transform bin when this is zero, and the further it is from zero the
    more of the picture's own low-frequency content the reading carries.
    Returned with every leakage figure so the reading arrives with its own
    conditioning rather than being read as though it had none.
    """
    turn = np.pi * float(hz) / float(sample_rate_hz)
    denominator = abs(math.sin(turn))
    if denominator <= 0.0:
        return float("inf")
    return float(abs(math.sin(turn * float(samples))) / denominator)


def _chroma_lines(field, uphet):
    """The up-converted chroma as a COMPLEX envelope, one row per line.

    `chroma._complex_envelope` is the instrument, not `hypot`: taking the
    magnitude at this boundary is what made an earlier burst instrument rank
    two of four, and every stage below reads an angle. Imported late because
    `chroma` imports this module, and cached on the field because four of
    the five stages want the same array.

    THE ENVELOPE IS REFERENCED TO EACH LINE'S OWN ORIGIN, and that puts a
    known 180 degrees of rotation between one line and the next: a 4fsc line
    is 227.5 subcarrier cycles, so restarting the column index every line
    drops half a cycle. Measured on `/output/g4_base_bars_sp` the burst
    phase alternates 145.5, -33.8, 146.7, -33.2 degrees down the field -
    exactly 180 apart, on every field of the decode. The two stages that
    read orientations are immune to it by construction, because a squared
    phasor cannot tell a vector from its reverse; the burst instrument is
    not, and de-rotates by the specified model before it folds.
    """
    cached = field.__dict__.get("_chroma_envelope_lines")
    if cached is not None:
        return cached if cached is not False else None
    field.__dict__["_chroma_envelope_lines"] = False
    from vhsdecode import chroma as chroma_module

    geom = geometry(field)
    expected = geom["lines"] * geom["width"]
    if uphet is None or len(uphet) != expected:
        return None
    block = np.asarray(uphet, dtype=np.float64).reshape(geom["lines"],
                                                        geom["width"])
    envelope = chroma_module._complex_envelope(block, 0)
    field.__dict__["_chroma_envelope_lines"] = envelope
    return envelope


def composite_subcarrier_hz(field):
    """The subcarrier the decode is working to, from its own parameters."""
    return float(field.rf.SysParams["fsc_mhz"]) * 1e6


def _burst_window(field):
    """The burst's columns, from the decoder's own geometry.

    `chroma.get_burst_area` derives them from `SysParams["colorBurstUS"]`,
    so the window follows the system parameters rather than being written
    down here - which is the arc's standing rule and also the reason this
    is a call rather than two numbers.
    """
    from vhsdecode import chroma as chroma_module

    try:
        start, stop = chroma_module.get_burst_area(field)
    except Exception:                                        # noqa: BLE001
        return None
    if stop - start < 8:
        return None
    return int(start), int(stop)


def _picture_rows(field, geom):
    """The picture lines, excluding the head-switch span at the foot.

    The same rows `colour_lock` fits on: no earlier than the decoder's own
    burst-detected line and ending before the earliest line SMPTE 32M
    clause 3.6 permits the playback head switch to reach.
    """
    first, last = geom["quiet"]
    if last - first < _COLOUR_FRAME_FIELDS:
        return None
    return int(first), int(last)


# --------------------------------------------------------------------------
# burst_instrument - the chroma's own three dimensions, from the burst
# --------------------------------------------------------------------------

def measure_burst_instrument(field, uphet, amount):
    """THE BURST AS THE CHROMA'S INSTRUMENT, per field and pooled.

    `burst_instrument`'s own docstring names the site: *"The instrument
    measures it after the up-conversion, where it is comparable with the
    specification."* That is here, at the end of the chroma path, on the
    same complex envelope every other stage in this section reads.

    ALL THREE DIMENSIONS, and the module carries all three already:
    `envelope_response` for frequency - the burst's spread, which IS the
    chroma band's shape - `amplitude` with the record-side doubler undone,
    and `timing` for the phase, which is the finest clock in the signal at
    0.776 nanoseconds a degree.

    AND THE TWO AXES ACROSS LINES AND FIELDS. `per_line` tests the
    specification's own per-line model - a constant amplitude and a fixed
    rotation - and `field_constant` asks whether the per-line and per-field
    readings of that one specification agree. They do not, by 1.6 to 2.7
    times, which is the module's recorded finding and is what this makes
    visible on any decode rather than on the four it was measured on.

    A MEASUREMENT AND NOT A CORRECTION, said plainly because the temptation
    is the other way. `amplitude_differential` is the specified separator
    tilt, and it is odd about the burst's centre - so it moves no bulk phase
    and shows only as a ramp across the burst - and it is NOT confirmed on
    the SP tapes. It is reported here and applied nowhere.
    """
    if not amount:
        return None
    state = _measurement_state(field, "burst_instrument")
    envelope = _chroma_lines(field, uphet)
    window = _burst_window(field)
    if envelope is None or window is None:
        _log_once(field, "burst_instrument_declined",
                  "model stages: burst_instrument declined, the field has no "
                  "burst window or no chroma envelope to read it from")
        return None
    geom = geometry(field)
    rows = _picture_rows(field, geom)
    if rows is None:
        return None
    start, stop = window
    burst = envelope[rows[0]:rows[1], start:stop]
    if burst.shape[0] < 4:
        return None

    # THE SPECIFIED PER-LINE ROTATION IS TAKEN OFF BEFORE THE FOLD, and it
    # has to be. The composite burst turns 180 degrees a line (SMPTE 170M
    # clause 8.4, 227.5 cycles), and folding without removing it averages a
    # vector against its own reverse: on `/output/g4_base_bars_sp` the raw
    # median fold is at the noise while the de-rotated one is the burst.
    turn = burst_instrument.LINE_ROTATION_RAD["composite"]
    index = np.arange(burst.shape[0], dtype=np.float64)
    aligned = burst * np.exp(-1j * turn * index)[:, np.newaxis]
    profile = (np.median(aligned.real, axis=0)
               + 1j * np.median(aligned.imag, axis=0))
    if not np.all(np.isfinite(profile)):
        return None

    three = burst_instrument.three_dimensions(profile, geom["sample_rate_hz"])
    amplitudes = 2.0 * np.abs(burst).mean(axis=1)
    phases = np.angle(burst.sum(axis=1))
    lines = burst_instrument.per_line(amplitudes, phases, domain="composite")
    differential = burst_instrument.amplitude_differential(
        profile, geom["sample_rate_hz"])

    # THE WINDOW TRAVELS WITH THE READING. The module's own recorded table
    # was taken over samples 74 to 110, which is 9.00 subcarrier cycles -
    # the specified number - and `get_burst_area` pads that span by four
    # samples ahead and eight behind before rounding to a multiple of four,
    # so a figure from here and a figure from there are not taken on the
    # same burst. Logging the window is what makes the difference
    # attributable instead of mysterious.
    ldd.logger.debug(
        "burst_instrument: window %d-%d (%.2f subcarrier cycles), stretch "
        "%.4f, chroma-referred %.1f, phase %.2f deg, per-line departure "
        "%+.6f deg/line (%s), scatter %.2f deg",
        start, stop, (stop - start) / 4.0,
        three["frequency"]["stretch"], three["amplitude"]["chroma_referred"],
        three["time"]["phase_deg"],
        lines["rotation_departure_deg_per_line"],
        "confirmed" if lines["rotation_confirmed"] else "refused",
        lines["phase_residual_sd_deg"])

    state.setdefault("levels", []).append(np.asarray(amplitudes,
                                                     dtype=np.float64))
    state["fields"] += 1
    reading = {
        "three_dimensions": three,
        "per_line": lines,
        "amplitude_differential": differential,
        # The pooled reading needs two fields of the same line count, so it
        # arrives later than the per-field one and is stored rather than
        # returned twice.
        "field_constant": None,
    }
    levels = [row for row in state["levels"]
              if row.size == state["levels"][0].size]
    if len(levels) >= 2:
        pooled = burst_instrument.field_constant(np.vstack(levels))
        reading["field_constant"] = pooled
        state["pooled"] = pooled
        ldd.logger.debug(
            "burst_instrument pooled over %d fields: within %.4f, between "
            "%.4f, correlated %.4f, lag one %+.4f, excess %.3fx, %s",
            pooled["fields"], pooled["within_field_sd"],
            pooled["between_field_sd"], pooled["correlated_prediction_sd"],
            pooled["lag_one"], pooled["excess_over_correlated"],
            "consistent" if pooled["consistent"] else "NOT consistent")
        if not state["reported"] and len(levels) >= _COLOUR_FRAME_FIELDS:
            state["reported"] = True
            _log_once(
                field, "burst_instrument_pooled",
                "model stages: burst_instrument over %d fields x %d lines - "
                "within field %.4f, between fields %.4f, correlated "
                "prediction %.4f (lag one %+.3f), excess %.2fx, the two "
                "readings of one specification are %s; burst stretch %.3f, "
                "phase %.2f deg",
                pooled["fields"], pooled["lines"], pooled["within_field_sd"],
                pooled["between_field_sd"], pooled["correlated_prediction_sd"],
                pooled["lag_one"], pooled["excess_over_correlated"],
                "consistent" if pooled["consistent"] else "NOT consistent",
                three["frequency"]["stretch"], three["time"]["phase_deg"])
    field.burst_instrument = reading
    return reading


# --------------------------------------------------------------------------
# chroma_leakage - chroma left in the luma, on both of Ethan's channels
# --------------------------------------------------------------------------

def measure_chroma_leakage(field, uphet, amount):
    """CHROMA LEFT IN THE LUMA, measured per field on the delivered picture.

    Ethan: *"I am still seeing chroma leakage in the luma ... The test
    should be written that asserts no chroma leakage from the color under
    and the upconverted color exists in the luma channel."* The module holds
    that assertion and it FAILS on every decode measured - the leak stands
    at 3.5 to 4.4 times its own control - and the failure is recorded as an
    expected failure rather than weakened. This is the same measurement made
    where the decoder can take it: per field, on the picture the decode is
    about to write.

    LAST IN THE CHROMA LIST, AFTER THE CORRECTIONS, and that placement is
    the measurement's meaning. Ethan's question is about the luma he is
    looking at, not about the luma before `luma_beat` or `colour_free_luma`
    have had their turn, so this reads the delivered picture and a decode
    with a correction enabled should show the figure fall.

    ALL THREE DIMENSIONS. Frequency is the three places the statistic is
    taken - the subcarrier, the colour-under carrier and a control between
    them that carries neither; amplitude is the resultant's modulus, as a
    ratio to that control rather than as a level; and time is the per-field
    series, which is the axis the colour framing acts on and the reason the
    pooling has to be done under each of the sequence's four rotations
    rather than blindly.

    THE FRAMING IS THE TRAP THIS STAGE EXISTS TO STEP OVER.
    `colour_framing.colour_under_field_advance` shows the colour-under
    advancing exactly 10500 cycles a field, fractional part zero, so it
    carries no field sequence - and that is true of the colour-under and of
    NOTHING ELSE. The up-converted channel does carry it, and reading the
    first result as though it settled both is how a blind pool reports a
    clean channel that is not clean: on the chroma-noise decode the blind
    figure is 0.0225 and the framed one 0.2080, a factor of nine.
    """
    if not amount:
        return None
    state = _measurement_state(field, "chroma_leakage")
    geom = geometry(field)
    picture = getattr(field, "dspicture", None)
    if picture is None or uphet is None:
        return None
    expected = geom["lines"] * geom["width"]
    if len(picture) != expected or len(uphet) != expected:
        return None
    rows = _picture_rows(field, geom)
    if rows is None:
        _log_once(field, "chroma_leakage_declined",
                  "model stages: chroma_leakage declined, the field has too "
                  "few picture lines to carry the statistic")
        return None

    active = _output_active_span(field, geom)
    if active is None:
        return None
    luma = np.asarray(picture, dtype=np.float64).reshape(
        geom["lines"], geom["width"])[rows[0]:rows[1]]
    chroma = np.asarray(uphet, dtype=np.float64).reshape(
        geom["lines"], geom["width"])[rows[0]:rows[1]]
    # NEITHER CHANNEL IS CENTRED HERE, and that is deliberate: `_amplitude_at`
    # removes each line's own mean over the same span it sums on, so the
    # window's response to a constant multiplies a constant that is exactly
    # zero. Doing it a second time in the caller would be harmless but would
    # put the same rule in two places, and the version that then drifted
    # would be this one.
    system = getattr(field.rf, "color_system", "NTSC")
    rates = chroma_leakage.channels(geom["sample_rate_hz"], system)
    per_field, leakage = {}, {}
    for name, hertz in rates.items():
        found = chroma_leakage.resultant(luma, chroma, hertz,
                                         geom["sample_rate_hz"], active)
        per_field[name] = complex(np.asarray(found["per_field"]).ravel()[0])
        leakage[name] = _bin_leakage(hertz, active[1] - active[0],
                                     geom["sample_rate_hz"])
    state["leakage"] = leakage
    history = state.setdefault("history", {name: [] for name in rates})
    for name, value in per_field.items():
        history[name].append(value)
    state["fields"] += 1

    # THE TWO LEVELS TRAVEL WITH THE RATIO, so a reading taken here can be
    # matched against one taken offline on the written pair. The statistic
    # is normalised and therefore says nothing about which arrays it was
    # computed on; a runtime figure that disagrees with an offline one is
    # otherwise impossible to attribute, which is a trap this arc has been
    # caught by before.
    ldd.logger.debug(
        "chroma_leakage: subcarrier %.5f at %+.1f deg, colour-under %.5f at "
        "%+.1f deg, control %.5f, over lines %d-%d, luma rms %.1f, chroma "
        "rms %.1f",
        abs(per_field["subcarrier"]),
        math.degrees(np.angle(per_field["subcarrier"])),
        abs(per_field["colour-under"]),
        math.degrees(np.angle(per_field["colour-under"])),
        abs(per_field["control"]), rows[0], rows[1],
        float(np.sqrt((luma[:, active[0]:active[1]] ** 2).mean())),
        float(np.sqrt((chroma[:, active[0]:active[1]] ** 2).mean())))

    if state["fields"] < _COLOUR_FRAME_FIELDS:
        return {"per_field": per_field, "pooled": None}

    pooled = {name: chroma_leakage.pool(values, system)
              for name, values in history.items()}
    floor = max(pooled["control"]["framed_magnitude"],
                pooled["control"]["blind_magnitude"])
    verdict = {}
    for name in ("subcarrier", "colour-under"):
        level = max(pooled[name]["framed_magnitude"],
                    pooled[name]["blind_magnitude"])
        verdict[name] = {"level": level,
                         "over_control": level / max(floor, 1e-300),
                         "best_state": pooled[name]["best_state"],
                         "blind": pooled[name]["blind_magnitude"],
                         "framed": pooled[name]["framed_magnitude"]}
    state["pooled"] = {"verdict": verdict, "control": floor,
                       "fields": state["fields"]}
    # THE RUNNING POOL, every field. The INFO line below fires once, at the
    # first colour frame, so that a short decode reports something at all -
    # but the pooled figure keeps moving as fields arrive, and an offline
    # instrument reading the written pair pools all of them. Logging the
    # running value is what lets the two be compared at the same field count
    # instead of at whichever count each happened to stop on, which is the
    # runtime-versus-offline parity trap this arc has been caught by before.
    ldd.logger.debug(
        "chroma_leakage pooled over %d fields: subcarrier %.5f (%.3fx), "
        "colour-under %.5f (%.3fx), control %.5f",
        state["fields"], verdict["subcarrier"]["level"],
        verdict["subcarrier"]["over_control"],
        verdict["colour-under"]["level"],
        verdict["colour-under"]["over_control"], floor)
    if not state["reported"]:
        state["reported"] = True
        _log_once(
            field, "chroma_leakage_pooled",
            "model stages: chroma_leakage over %d fields - subcarrier %.4f "
            "(%.2fx control, blind %.4f framed %.4f, state %d), colour-under "
            "%.4f (%.2fx control, blind %.4f framed %.4f, state %d), control "
            "%.4f; the luma is %s",
            state["fields"],
            verdict["subcarrier"]["level"], verdict["subcarrier"]["over_control"],
            verdict["subcarrier"]["blind"], verdict["subcarrier"]["framed"],
            verdict["subcarrier"]["best_state"],
            verdict["colour-under"]["level"],
            verdict["colour-under"]["over_control"],
            verdict["colour-under"]["blind"], verdict["colour-under"]["framed"],
            verdict["colour-under"]["best_state"], floor,
            "clean" if max(v["over_control"] for v in verdict.values()) <= 2.0
            else "NOT clean")
        _log_once(
            field, "chroma_leakage_window",
            "model stages: chroma_leakage read over samples %d-%d, the span "
            "the written file declares, trimmed to %d whole subcarrier "
            "cycles; the window's response to a constant is %.3g at the "
            "subcarrier, %.3g at the colour-under and %.3g at the control, "
            "and the two larger of those are the conditioning this "
            "statistic carries",
            active[0], active[1],
            (active[1] - active[0]) // _SAMPLES_PER_SUBCARRIER_CYCLE,
            leakage["subcarrier"], leakage["colour-under"],
            leakage["control"])
    field.chroma_leakage = {"per_field": per_field, "pooled": state["pooled"]}
    return field.chroma_leakage


# --------------------------------------------------------------------------
# tape_bias - the residual of recording the chroma on the luma's own bias
# --------------------------------------------------------------------------

def measure_tape_bias(field, picture, amount):
    """THE TAPE-BIAS RESIDUAL in the delivered luma, per field.

    The chrominance is recorded THROUGH the luminance: SMPTE 32M 7.5.1.2.2
    says "The chrominance signal shall be recorded with the luminance FM
    signal acting as bias" and the JVC guide says the same four times for
    baseline VHS. The bias is a FEW-CYCLE bias rather than the textbook
    ideal - `tape_bias.bias_cycles` measures three tenths of a cycle across
    the switching band - so it linearises the chroma's own transfer only
    partly, and what it leaves is a third-order product of the two carriers
    at `f_y - 2 f_c`. That product demodulates into the picture at `2 f_c`,
    80 f_H, 1.2587 MHz, and the guide names it as the reason the colour
    under sits at 40 f_H at all.

    THE PRODUCT IS SEPARABLE FROM THE PICTURE BECAUSE IT ALTERNATES. The
    colour under is rotated a quarter turn a line, so the carrier is
    40 f_H +- f_H/4 and the product at twice it is 80 f_H +- f_H/2 - half a
    line rate off the luminance's own harmonics, which is what the guide
    means by "interleaved (1/2 line offset)" and which no picture content
    shares. This reads the per-line COMPLEX amplitude at 80 f_H, splits it
    into its still and alternating parts, and reports the alternating part
    against the same statistic in the neighbouring transform bins.

    THE WINDOW IS CHOSEN SO THE PRODUCT LANDS ON AN EXACT BIN. The output
    runs at 910 f_H, so 91 output samples hold exactly eight cycles of
    80 f_H and a span that is a whole multiple of 91 puts the product on a
    bin with the neighbouring bins 1.25 f_H away carrying none of it. The
    control is therefore LOCAL - the same picture, a few kilohertz off -
    which is a stronger control than a remote frequency because it does not
    assume the picture's spectrum is flat over a wide span.

    MEASURED on the decodes already in this tree, alternating peak against
    the control's, in IRE through the field's own output scale: the
    chroma-noise SP decode reads 2.2651 against 0.0323, a ratio of 70, and
    its measured line offset is -0.4978 cycles a line on twelve fields of
    twelve - the nearest bin a 231-line transform has to exactly one half,
    which is the guide's interleave read back off the tape. The `home`
    decode, which carries no steady chroma, reads 0.0283 against 0.0241 and
    an offset of exactly zero.

    IT MEASURES AND DOES NOT CORRECT, and the reason is in
    `tape_bias.differential_gain_refused`: the same bias mechanism imposes a
    luma-dependent chroma gain of +2.084 +- 0.048 dB per MHz of luma
    carrier, measured between this deck's record and playback taps, and a
    DECODE HOLDS ONE TAP. The tape's share is a difference between two, so
    a decode cannot separate it from the source's own differential gain and
    correcting the sum would take out the picture's colour as though it
    were an artefact.
    """
    if not amount:
        return None
    state = _measurement_state(field, "tape_bias")
    geom = geometry(field)
    if picture is None:
        _log_once(field, "tape_bias_no_picture",
                  "model stages: tape_bias declined - the delivered luma "
                  "does not exist at this point in the chroma path")
        return None
    expected = geom["lines"] * geom["width"]
    if len(picture) != expected:
        _log_once(field, "tape_bias_shape",
                  "model stages: tape_bias declined - the picture holds %d "
                  "samples where the geometry wants %d",
                  len(picture), expected)
        return None
    rows = _picture_rows(field, geom)
    if rows is None:
        _log_once(field, "tape_bias_declined",
                  "model stages: tape_bias declined, the field has too few "
                  "picture lines to carry the statistic")
        return None

    carrier = float(geom["carrier_hz"])
    sample_rate = geom["sample_rate_hz"]
    start, stop = geom["active"]
    # THE SPAN IS TRIMMED TO WHOLE GROUPS, so the product lands on an exact
    # transform bin and the window's response to a constant is zero at it.
    # The group is derived from the rates rather than typed: it is the
    # smallest number of output samples holding a whole number of cycles of
    # the product, which for 910 samples a line and 80 f_H is 91.
    per_line = int(round(sample_rate / colour_under.line_rate_hz(
        getattr(field.rf, "color_system", "NTSC"))))
    harmonic = int(round(2.0 * carrier * per_line / sample_rate))
    group = per_line // math.gcd(max(harmonic, 1), max(per_line, 1))
    span = ((stop - start) // max(group, 1)) * max(group, 1)
    if span < group or span <= 0:
        _log_once(field, "tape_bias_declined_span",
                  "model stages: tape_bias declined - the active span of %d "
                  "samples does not hold one group of %d",
                  stop - start, group)
        return None

    luma = np.asarray(picture, dtype=np.float64).reshape(
        geom["lines"], geom["width"])[rows[0]:rows[1]]
    found = tape_bias.product_in_the_luma(luma, sample_rate, start,
                                          start + span,
                                          getattr(field.rf, "color_system",
                                                  "NTSC"),
                                          carrier_hz=carrier)
    scale = float(field.out_scale) or 1.0
    reading = {
        "alternating_ire": found["alternating_peak"] / scale,
        "control_ire": found["control_alternating_peak"] / scale,
        "still_ire": 2.0 * found["still_magnitude"] / scale,
        "control_still_ire": 2.0 * found["control_still"] / scale,
        "ratio": found["ratio"],
        "offset_cycles_per_line": found["offset_cycles_per_line"],
        "interleaved": found["interleaved"],
        "lands_on_a_bin": found["lands_on_a_bin"],
        "frequency_hz": found["frequency_hz"],
        "lines": found["lines"],
        "span": span,
        "group": group,
    }
    history = state.setdefault("history", [])
    history.append(reading)
    state["fields"] += 1

    ldd.logger.debug(
        "tape_bias: %.5f IRE alternating at %.0f Hz against %.5f IRE "
        "control (%.2fx), still %.5f against %.5f, offset %+.4f cycles a "
        "line over %d lines, span %d samples in groups of %d",
        reading["alternating_ire"], reading["frequency_hz"],
        reading["control_ire"], reading["ratio"], reading["still_ire"],
        reading["control_still_ire"], reading["offset_cycles_per_line"],
        reading["lines"], span, group)

    # THE OFFSET IS NOT AVERAGED, and that is not fastidiousness. It lives on
    # a circle whose two ends are +0.5 and -0.5 and it lands on both, so a
    # mean of twelve readings of one half comes out at a third - which is
    # exactly what a first pass at this reported. The pooled figure is how
    # many fields landed on the half, not where the mean of them sits.
    interleaved = sum(1 for row in history if row["interleaved"])
    state["pooled"] = {
        "fields": state["fields"],
        "alternating_ire": float(np.median([r["alternating_ire"]
                                            for r in history])),
        "control_ire": float(np.median([r["control_ire"] for r in history])),
        "ratio": float(np.median([r["ratio"] for r in history])),
        "interleaved_fields": interleaved,
    }
    if state["fields"] >= _COLOUR_FRAME_FIELDS and not state["reported"]:
        state["reported"] = True
        pooled = state["pooled"]
        _log_once(
            field, "tape_bias",
            "model stages: tape_bias over %d fields - the %.0f Hz product "
            "stands at %.4f IRE against a neighbouring-bin control of %.4f "
            "(%.2fx), interleaved on %d of %d fields; the chroma is "
            "recorded on the luma's own bias (SMPTE 32M 7.5.1.2.2, JVC "
            "VTG82063 1.1.6) and this is what that bias does not linearise. "
            "Measured and NOT corrected: %s",
            pooled["fields"], reading["frequency_hz"],
            pooled["alternating_ire"], pooled["control_ire"], pooled["ratio"],
            pooled["interleaved_fields"], pooled["fields"],
            tape_bias.differential_gain_refused()["because"])
    field.tape_bias = {"per_field": reading, "pooled": state["pooled"]}
    return field.tape_bias


# --------------------------------------------------------------------------
# colour_framing - the specified sequence, and what the decoder does
# --------------------------------------------------------------------------

def measure_colour_framing(field, uphet, amount):
    """THE COLOUR FRAME: the specification checked, and the tape declined.

    Ethan asked for the framing to be derived from the carrier's own
    relationship to hsync rather than counted. The module holds the
    arithmetic that answers him and it answers NO for this medium, exactly:
    the subcarrier and the record heterodyne oscillator both advance a
    fractional three quarters of a cycle a field, so the 270 degrees that
    carries the four-field sequence cancels in the down-conversion and the
    colour-under advances 10500 whole cycles a field. A quantity that is not
    on the tape cannot be measured off it, and a stage that cannot measure
    its subject must DECLINE rather than guess - which this one does, in the
    log, on every decode.

    WHAT IT DOES INSTEAD, and it is not nothing. Two contracts that ARE
    checkable are checked here rather than assumed:

      THE MAP. `validate_mapping` puts the decoder's shipped
      `(isFirstField, parity) -> (fieldPhaseID, base phase)` table against
      the sequence the specification fixes AT THE SYNC EDGE, which is the
      only place a burst-to-sync phase is defined. The shipped NTSC map
      passes on all four states.

      THE SEQUENCE. `fieldPhaseID` must ascend and wrap, and 45 of 683
      shipped decodes did not. That is the counter's own failure mode -
      it carries the framing forward from wherever it started and no
      counter survives a splice - and this makes it visible per decode
      instead of leaving it to be found by counting output files.

    THE DIMENSIONS, AND THE ONE THAT IS ABSENT. Frequency and time are both
    here and both derived: the sequence LENGTH comes from the subcarrier's
    ratio to the line rate, and the advance is 270 degrees a field, which
    the module converts to seconds through the subcarrier. There is no
    AMPLITUDE axis and there cannot be one - the quantity is a state index
    out of four, and an index has no magnitude. What would carry an
    amplitude is the CONFIDENCE of a measured framing, and that is
    unavailable for the reason above rather than for want of an estimator.
    """
    if not amount:
        return None
    system = getattr(field.rf, "color_system", None)
    if system != "NTSC":
        _log_once(field, "colour_framing_system",
                  "model stages: colour_framing declined, the four-field "
                  "sequence is derived for NTSC and this decode is %s",
                  system or "unknown")
        return None
    from vhsdecode import chroma as chroma_module

    state = _measurement_state(field, "colour_framing")
    spec = colour_framing.sequence(system)
    advance = colour_framing.colour_under_field_advance(system)
    if not state["reported"]:
        state["reported"] = True
        mapping = colour_framing.validate_mapping(
            chroma_module.ntsc_color_framing_map, system)
        state["mapping"] = mapping
        _log_once(
            field, "colour_framing_map",
            "model stages: colour_framing - the shipped map %s the specified "
            "sequence (%d states mapped against %d specified, worst error "
            "%.2f deg); and DECLINES to measure the framing from this tape, "
            "because the colour-under advances %.0f cycles a field with a "
            "fractional part of %.3g and so carries no field sequence at all",
            "agrees with" if mapping["consistent"] else "DISAGREES with",
            mapping["states_mapped"], mapping["states_specified"],
            mapping["worst_error_deg"], advance["cycles_per_field"],
            advance["fractional_advance"])

    identifier = getattr(field, "fieldPhaseID", None)
    if identifier is None:
        return None
    # IN FIELD ORDER, NOT ARRIVAL ORDER, and that is not a detail. The
    # chroma path runs on the decoder's field threads, so with `-t 4` the
    # fields reach this stage in whatever order they finish; judged in
    # arrival order the sequence appeared to break on four of five decodes,
    # including two whose counter is demonstrably intact. Keyed on the
    # field's own number and sorted before the contract is checked, the
    # reading is the same whatever the thread count - which is the property
    # a measurement has to have before its verdict means anything.
    history = state.setdefault("ids", {})
    history[int(getattr(field, "field_number", len(history)))] = int(identifier)
    state["fields"] += 1
    states = int(spec["states"])
    # The contract the decoder's own metadata check asserts: the identifier
    # ascends by one and wraps at the sequence length. Counted rather than
    # asserted, because a stage that measures does not stop a decode. Only
    # CONSECUTIVE field numbers are compared, so a gap in the arrival order
    # is not read as a break in the sequence.
    numbers = sorted(history)
    steps = [(history[numbers[i]] - history[numbers[i - 1]]) % states
             for i in range(1, len(numbers))
             if numbers[i] == numbers[i - 1] + 1]
    # A UNIFORM STEP AND AN IRREGULAR ONE ARE DIFFERENT FINDINGS, and
    # counting departures from +1 alone conflates them. A sequence that
    # steps by the same amount every time is INTACT whichever way it runs -
    # it is a cyclic order with an origin and a direction, and a period-four
    # two-state pattern is cyclically identical to its own rotation, which
    # `settle_the_question` already warns of. What a splice or a re-record
    # produces is an IRREGULAR step, and that is what has to be reported.
    #
    # MEASURED, AND THE DIRECTION IS DECIDED BY THE SEEK. On a decode of
    # `zaroff-75bars-NTSC-SP` the identifiers run 4,1,2,3 and on
    # `zaroff-75bars-NTSC-EP` they run 1,4,3,2 - both perfectly uniform,
    # in opposite directions. The shipped map sends (isFirstField, parity)
    # to the identifiers 1,2,3,4 with parity `(field_number // 2) % 2`, so
    # a decode whose first field is a FIRST field walks the map one way and
    # one whose first field is a second field walks it the other. The
    # direction is therefore a property of where the seek landed and not of
    # the tape, which is the counter being seek-relative exactly as
    # `colour_frame_parity` says it must be.
    modal = max(set(steps), key=steps.count) if steps else None
    irregular = sum(1 for step in steps if step != modal)
    breaks = sum(1 for step in steps if step != 1)
    state["breaks"] = breaks
    state["irregular"] = irregular
    state["modal_step"] = modal
    state["steps"] = len(steps)
    ldd.logger.debug("colour_framing: fieldPhaseID %d, modal step %s, %d of "
                     "%d steps irregular", identifier, modal, irregular,
                     len(steps))
    if len(steps) >= states and not state.get("sequence_reported"):
        state["sequence_reported"] = True
        if irregular:
            _log_once(field, "colour_framing_break",
                      "model stages: colour_framing - the counter's sequence "
                      "BREAKS, %d of the first %d steps depart from its own "
                      "modal step of %+d; the framing is counted and not "
                      "measured, so a splice or a re-record is enough to do "
                      "this", irregular, len(steps), modal)
        elif modal == 1:
            _log_once(field, "colour_framing_intact",
                      "model stages: colour_framing - the counter's sequence "
                      "is intact over the first %d steps, ascending by one "
                      "and wrapping at %d", len(steps), states)
        else:
            _log_once(field, "colour_framing_direction",
                      "model stages: colour_framing - the counter's sequence "
                      "is uniform over the first %d steps but runs the other "
                      "way, stepping %+d of %d each field rather than +1. "
                      "That is not a break: the shipped map is walked in the "
                      "direction the seek's first field parity chooses, so "
                      "the direction is a property of where this decode "
                      "started and not of the tape",
                      len(steps), modal, states)
    field.colour_framing = {"sequence": spec, "advance": advance,
                            "field_phase_id": int(identifier),
                            "breaks": breaks, "steps": len(steps),
                            "measured": False}
    return field.colour_framing


# --------------------------------------------------------------------------
# colour_under - the low carrier, measured on the channel itself
# --------------------------------------------------------------------------

def _colour_under_burst(field):
    """THE COLOUR-UNDER BURST, one complex phasor a line, measured once.

    Two stages want the same thing and it is the expensive half of either,
    so it is taken once and cached on the FIELD - the quantity is per field
    and a stale one would describe the wrong picture.

    `field.chroma_under_tbc` is the down-converted chroma on the output time
    base: the only point in the decode where the signal the tape holds
    exists on a regular grid, and the only place an imbalance can be
    measured at all. `iq_imbalance` proves why in one table - at the
    composite burst's 180 degrees a line the signal and its image are the
    same sequence, coherence 1.0000 and condition 1.3e14, and no fit can
    separate them; at the colour-under's 90 degrees a line they
    counter-rotate, coherence 0.0000 and condition 1.0. The record-side
    rotation of SMPTE 32M clause 3.9.2.1.5 is not an obstacle to the
    measurement, it is the whole of what makes it possible.

    Returns None where the field cannot support it, and the reason is
    logged by the caller rather than here, so one refusal is not reported
    twice when both stages ask.
    """
    cached = field.__dict__.get("_colour_under_burst")
    if cached is not None:
        return cached if cached is not False else None
    field.__dict__["_colour_under_burst"] = False

    under = getattr(field, "chroma_under_tbc", None)
    window = _burst_window(field)
    if under is None or window is None:
        return None
    geom = geometry(field)
    values = np.asarray(under, dtype=np.float64)
    if values.size != geom["lines"] * geom["width"]:
        return None
    rows = _picture_rows(field, geom)
    if rows is None:
        return None
    system = getattr(field.rf, "color_system", "NTSC")
    if system not in ("NTSC", "PAL"):
        return None

    # THE COMPLEX ENVELOPE, AND MIXING ALONE IS NOT ONE. The down-converted
    # chroma is a REAL signal, so multiplying it by a complex exponential
    # leaves the difference AND the sum term - and on a real block the sum
    # term makes the lag-one product's angle come out at exactly minus the
    # mixing rate whatever the signal is doing, which reported the carrier
    # as 0.0 Hz on every field of a real decode. That is what the first
    # version of this did. `colour_lock.chroma_envelope` mixes and then
    # low-passes away the image, in one call.
    #
    # THE LOW-PASS IS THE CHANNEL'S OWN HALF-WIDTH, not a chosen number:
    # `modulation_half_width_hz` is how far either side of the carrier the
    # decoder's band-pass reaches, so the envelope keeps the whole
    # colour-under band and nothing beyond it.
    ceiling = field.rf.DecoderParams.get("chroma_bpf_upper")
    half_width = colour_under.modulation_half_width_hz(
        system, band_upper_hz=float(ceiling) if ceiling else None)
    envelope = colour_lock.chroma_envelope(
        values.reshape(geom["lines"], geom["width"]),
        geom["sample_rate_hz"], subcarrier_hz=geom["carrier_hz"],
        half_width_hz=half_width, centre=0.0)

    # THE BURST PROPER, CENTRED IN THE DECODER'S WINDOW. `get_burst_area`
    # pads the specified span by four samples ahead and eight behind before
    # rounding, and those samples are blanking rather than burst: with the
    # padding included the phase slope is taken partly across the burst's
    # own switch-on, which is an envelope event and not the carrier. The
    # width used instead is ITU-R BT.1700's nine cycles - 2.514 microseconds
    # - centred in that window, so the number comes from the standard and
    # the position from the decoder.
    span = int(round(burst_instrument.BURST_DURATION_S
                     * geom["sample_rate_hz"]))
    span = min(span, window[1] - window[0])
    first = window[0] + (window[1] - window[0] - span) // 2
    block = envelope[rows[0]:rows[1], first:first + span]
    if block.shape[0] < 4 or block.shape[1] < 8:
        return None

    out = {
        "geometry": geom,
        "system": system,
        "block": block,
        # ONE COMPLEX PHASOR A LINE, which is the quantity both stages
        # want and the one a magnitude would destroy: an imbalance is
        # invisible to `abs()` because a phasor and its conjugate have the
        # same modulus.
        "phasors": block.sum(axis=1),
        "window": (first, first + span),
        "half_width_hz": half_width,
        "rows": rows,
    }
    field.__dict__["_colour_under_burst"] = out
    return out


def measure_colour_under(field, uphet, amount):
    """THE COLOUR-UNDER CHANNEL, read where it still exists as itself.

    `field.chroma_under_tbc` is the down-converted chroma on the output time
    base, before the heterodyne puts it back at the subcarrier - the only
    point in the decode where the signal the TAPE holds is available on a
    regular grid. `luma_beat` consumes and clears it, so this stage stands
    ahead of that one and the declaration says so.

    MEASURED IN THE RESERVED INTERVAL, which is what keeps it inside the
    arc's standing rule. The burst is nine cycles of a known tone sitting in
    the colour-under band at 40 times the line rate, so every reading below
    comes from the burst window and none of it from the picture:

      FREQUENCY   the carrier's own position, from the burst's phase slope
                  across the window, against the specification's 40 f_H -
                  and the sideband fold headroom, which is how close the
                  lower sideband comes to folding through zero, evaluated
                  against the decoder's OWN band-pass ceiling rather than a
                  nominal one.
      AMPLITUDE   the burst's level in the colour-under channel, reported
                  as a RATIO and never as a level, because
                  `burst_observables` establishes that three unknown gains
                  multiply it - the two automatic gain controls and
                  head-to-tape contact - and one measurement cannot separate
                  three scalars.
      TIME        the same phase slope read the other way, as the group
                  delay across the burst window, which is the quantity
                  `envelope_phase` parameterises and the one that was shown
                  to be a delay rather than the amplitude's causal partner.

    AND THE RESOLUTION IS STATED WITH THE READING. Nine cycles is 2.514
    microseconds, so by the 1/T law this window resolves 398 kHz and no
    better; it describes where the carrier sits and how the band is shaped,
    and it cannot locate a narrow feature inside that band. Saying so is the
    difference between a measurement and a number.
    """
    if not amount:
        return None
    state = _measurement_state(field, "colour_under")
    burst = _colour_under_burst(field)
    if burst is None:
        _log_once(field, "colour_under_declined",
                  "model stages: colour_under declined, the down-converted "
                  "chroma is not available - this stage must run before "
                  "luma_beat, which releases it")
        return None
    geom = burst["geometry"]
    system = burst["system"]
    block = burst["block"]
    first, last = burst["window"]
    span = last - first
    rate = geom["sample_rate_hz"]
    specified = colour_under.carrier_hz(system)

    # THE CARRIER, FROM THE BURST'S OWN PHASE SLOPE. The envelope is in the
    # decoder's carrier's rotating frame, so what is left rotates at the
    # difference between that carrier and the true one; the per-sample
    # increment is the angle of the lag-one product, which is bounded by
    # construction and needs no unwrapping.
    lag = complex((block[:, 1:] * np.conj(block[:, :-1])).sum())
    increment = float(np.angle(lag))
    measured_hz = geom["carrier_hz"] + increment * rate / (2.0 * np.pi)
    # The delay the same slope reads as: a phase advancing linearly with
    # frequency is a delay, and one sample of it is one sample period.
    delay_s = -increment / (2.0 * np.pi * max(geom["carrier_hz"], 1.0))

    level = float(2.0 * np.abs(block).mean())
    reference = float(2.0 * np.abs(block).mean(axis=1).mean())
    resolution = burst_instrument.probe_resolution_hz()
    # The DECODER'S OWN band-pass ceiling, not the module's default, because
    # `band_upper_for` is explicit that the ceiling is a setting rather than
    # a specification and a caller with a measurement of its own overrides
    # it. Passing the decode's value is what makes the headroom a statement
    # about this decode.
    ceiling = field.rf.DecoderParams.get("chroma_bpf_upper")
    headroom = colour_under.sideband_fold_headroom_hz(
        system, band_upper_hz=float(ceiling) if ceiling else None)
    provenance = colour_under.carrier_provenance(system)

    state.setdefault("carriers", []).append(measured_hz)
    state.setdefault("levels", []).append(level)
    state["fields"] += 1
    ldd.logger.debug(
        "colour_under: carrier %.1f Hz measured against %.1f specified and "
        "%.1f in the decoder, delay %.1f ns, burst level %.1f, over samples "
        "%d-%d (%.2f subcarrier cycles of the specified nine)",
        measured_hz, specified, geom["carrier_hz"], 1e9 * delay_s, level,
        first, first + span,
        span / (rate / composite_subcarrier_hz(field)))

    if state["fields"] >= 2:
        running = np.asarray(state["carriers"], dtype=np.float64)
        ldd.logger.debug(
            "colour_under pooled over %d fields: carrier %.2f +/- %.2f Hz, "
            "departure from the specified %+.2f Hz (%+.1f ppm)",
            running.size, running.mean(), running.std(ddof=1),
            running.mean() - specified,
            1e6 * (running.mean() - specified) / specified)
    if state["fields"] >= _COLOUR_FRAME_FIELDS and not state["reported"]:
        state["reported"] = True
        carriers = np.asarray(state["carriers"], dtype=np.float64)
        levels = np.asarray(state["levels"], dtype=np.float64)
        _log_once(
            field, "colour_under_pooled",
            "model stages: colour_under over %d fields - carrier %.1f +/- "
            "%.1f Hz against the specified %.1f (%s, %s), decoder %.1f; "
            "burst level %.1f +/- %.1f (a ratio only, three unknown gains "
            "multiply it); sideband fold headroom %+.1f kHz; the window "
            "resolves %.1f kHz and no better",
            carriers.size, carriers.mean(),
            carriers.std(ddof=1) if carriers.size > 1 else 0.0, specified,
            provenance["relation"],
            "printed" if provenance["printed"] else "derived",
            geom["carrier_hz"], levels.mean(),
            levels.std(ddof=1) if levels.size > 1 else 0.0,
            1e-3 * float(headroom["headroom_hz"]),
            1e-3 * resolution["resolution_hz"])

    field.colour_under = {
        "measured_carrier_hz": measured_hz,
        "specified_carrier_hz": float(specified),
        "decoder_carrier_hz": geom["carrier_hz"],
        "delay_s": delay_s,
        "burst_level": level,
        "burst_level_per_line": reference,
        "headroom": headroom,
        "resolution_hz": resolution["resolution_hz"],
        "provenance": provenance,
    }
    return field.colour_under


# --------------------------------------------------------------------------
# vectorscope - the colour difference coordinate system, from the streaks
# --------------------------------------------------------------------------

def measure_vectorscope(field, uphet, amount):
    """THE COLOUR'S COORDINATE SYSTEM, from the transitions between colours.

    Ethan, 2026-09-06, with a composite vectorscope in front of him: *"I
    believe these lines pointing to I.Q.-I,-Q need to be corrected and
    represent a measureable shape that we can use for correcting the color's
    coordinate system."*

    The module's standing finding - that no BAR TARGET sits on an I or Q
    line, the closest being 19.54 degrees away - is unchanged and is about
    the six targets. His screenshot is of a scope set to whole line rather
    than to burst, so it shows the TRANSITIONS as well, and a transition is
    a trajectory with a direction. The six targets fix six points; the
    transition directions fix the AXES those points are expressed in, which
    is the coordinate system itself and is the stronger correction because
    it moves every colour at once.

    WHAT IS MEASURED, exactly the five things he named, and all of them
    complex rather than as angles:

      * the direction of each streak, as the orientation of a complex
        difference along a line;
      * whether they CLUSTER on two axes, from the fourfold resultant,
        against the floor random orientations reach at the chroma path's
        own significance;
      * the angle between the two axes, from the twofold resultant, which
        is exactly the cosine of that angle and so is zero for the right
        angle the encoding specifies;
      * the pair's orientation against the BURST, measured on the same
        field, which is the signal's only absolute phase reference;
      * and the whole departure as one linear map, `alpha d + beta
        conj(d)`, whose second half is the shear a rotation cannot fix.

    READING THE ACTIVE PICTURE, UNDER THE EXCEPTION AND WITH ITS
    DISCIPLINE. The streaks are in the picture. This stands where
    `colour_lock` stands and keeps the same rule: a derived null
    distribution decides whether the shape is there, and the stage declines
    in the log rather than reporting a coordinate system it has not
    established. It applies nothing.

    THE LINE-ORIGIN ROTATION CANNOT REACH IT, which is worth stating because
    it is what makes the statistic usable here at all. The complex envelope
    is referenced to each line's own origin, so consecutive lines differ by
    180 degrees; both statistics square the phasor, and a squared phasor
    cannot tell a vector from its reverse. The burst reference is taken
    modulo the same 90 degrees for the same reason.
    """
    if not amount:
        return None
    state = _measurement_state(field, "vectorscope")
    envelope = _chroma_lines(field, uphet)
    window = _burst_window(field)
    geom = geometry(field)
    rows = _picture_rows(field, geom)
    if envelope is None or window is None or rows is None:
        return None
    span = _output_active_span(field, geom)
    if span is None:
        return None
    # The same window the delivered picture declares, so a reading taken
    # here can be reproduced from the written file. The statistic is a
    # difference along the line and so is a high-pass, which makes it far
    # less sensitive to the window than the leakage bin is - but taking the
    # two on different spans for no reason is how they stop being
    # comparable, and there is no reason.
    active = envelope[rows[0]:rows[1], span[0]:span[1]]
    if active.shape[0] < 4 or active.shape[1] < 8:
        return None

    steps = np.diff(active, axis=1).ravel()
    magnitude = np.abs(steps)
    scale = float(np.median(magnitude))
    if not np.isfinite(scale) or scale <= 0.0:
        return None
    # THE GATE IS THE SAME SIGNIFICANCE AS EVERYTHING ELSE HERE, DERIVED.
    # Between two settled bars the difference is the channel's noise, whose
    # modulus is Rayleigh; its median is `sigma sqrt(2 ln 2)` and it exceeds
    # `x` with probability `exp(-x^2 / 2 sigma^2)`. Requiring that to be no
    # larger than the two-sided Gaussian tail at the detector's sigma puts
    # the gate at `sqrt(-ln(tail) / ln 2)` medians - 3.73 at four sigma - so
    # what is kept is a transition and not the settled colour between two.
    gate = scale * math.sqrt(-math.log(_TAIL) / math.log(2.0))
    keep = magnitude >= gate
    kept = int(keep.sum())
    if kept < 4 * _COLOUR_FRAME_FIELDS:
        _log_once(field, "vectorscope_declined",
                  "model stages: vectorscope declined, only %d transitions "
                  "cleared the %.1f-median gate on this field - a picture "
                  "with too few colour changes cannot show the axes",
                  kept, gate / scale)
        return None

    # The burst, from the same envelope and the same field, as the absolute
    # reference. Its own 180 degrees a line is immaterial to a quantity read
    # modulo ninety, which is what the axial statistic returns.
    burst = envelope[rows[0]:rows[1], window[0]:window[1]]
    burst_deg = float(math.degrees(np.angle(
        (burst * np.exp(-1j * burst_instrument.LINE_ROTATION_RAD["composite"]
                        * np.arange(burst.shape[0])[:, np.newaxis])).sum())))

    found = vectorscope.transition_axes(steps[keep], sigma=_SIGMA,
                                        burst_phase_deg=burst_deg)
    mapping = vectorscope.coordinate_map(steps[keep])
    state.setdefault("fourfold", []).append(found["fourfold"])
    state.setdefault("twofold", []).append(found["twofold"])
    state.setdefault("counts", []).append(found["effective_count"])
    state.setdefault("shear", []).append(mapping["shear_ratio"])
    state["fields"] += 1

    ldd.logger.debug(
        "vectorscope: %d transitions, clustering %.4f against %.4f, pair "
        "%.2f deg (%.2f from the burst, %.2f from the specified %.0f "
        "%s), separation %.2f deg %s, shear %.4f, stretch %.3f along "
        "%.1f deg",
        kept, found["clustering"], found["clustering_floor"],
        found["pair_angle_deg"], found["versus_burst_deg"],
        found["versus_specified_deg"], found["specified_rotation_deg"],
        found["rotation_sense"], found["separation_deg"],
        "orthogonal" if found["orthogonal"] else "SHEARED",
        mapping["shear_ratio"], mapping["stretch"],
        mapping["stretch_axis_deg"])

    if not found["clustered"]:
        _log_once(field, "vectorscope_scattered",
                  "model stages: vectorscope declined, the transitions "
                  "scatter - fourfold resultant %.4f against the %.4f random "
                  "orientations reach at %.1f sigma over %d effective "
                  "samples, so there is no two-axis shape to report",
                  found["clustering"], found["clustering_floor"], _SIGMA,
                  int(found["effective_count"]))

    if state["fields"] >= _COLOUR_FRAME_FIELDS:
        # POOLED AS THE COMPLEX RESULTANTS THEY ARE, weighted by each
        # field's own effective count, so a field with few transitions does
        # not vote as loudly as one full of them. Pooling the ANGLES would
        # be the same mistake the axial doubling exists to avoid.
        weight = np.asarray(state["counts"], dtype=np.float64)
        four = complex((np.asarray(state["fourfold"]) * weight).sum()
                       / max(weight.sum(), 1e-300))
        two = complex((np.asarray(state["twofold"]) * weight).sum()
                      / max(weight.sum(), 1e-300))
        floor = _rayleigh_floor(float(weight.sum()))
        separation = math.degrees(math.acos(min(abs(two), 1.0)))
        state["pooled"] = {"fourfold": four, "twofold": two, "floor": floor,
                           "separation_deg": separation,
                           "fields": state["fields"]}
        ldd.logger.debug(
            "vectorscope pooled over %d fields: clustering %.4f against "
            "%.5f, pair %.3f deg, separation %.2f deg, shear %.4f",
            state["fields"], abs(four), floor,
            math.degrees(np.angle(four)) / 4.0 % 90.0, separation, abs(two))
    if state["fields"] >= _COLOUR_FRAME_FIELDS and not state["reported"]:
        state["reported"] = True
        _log_once(
            field, "vectorscope_pooled",
            "model stages: vectorscope over %d fields - clustering %.4f "
            "against %.5f, pair at %.2f deg (%.2f from the burst, %.2f from "
            "the specified %.0f %s), axes %.2f deg apart against the "
            "specified 90 (%s at this sample count), shear %.4f, mean "
            "per-field shear ratio %.4f",
            state["fields"], abs(four), floor,
            math.degrees(np.angle(four)) / 4.0 % 90.0,
            found["versus_burst_deg"], found["versus_specified_deg"],
            found["specified_rotation_deg"], found["rotation_sense"],
            separation,
            "a right angle" if abs(two) < floor else "SHEARED",
            abs(two), float(np.mean(state["shear"])))

    field.vectorscope = {"axes": found, "map": mapping,
                         "transitions": kept, "burst_deg": burst_deg}
    return field.vectorscope


# --------------------------------------------------------------------------
# THE SOURCE SIDE OF THE RECORD HEAD
#
# Everything above this line corrects what the tape and the two decks did.
# What follows corrects what had already happened to the signal BEFORE the
# recording VCR ever saw it, which is the half of the chain the arc had
# modelled and never applied. See `models/source_correction.py` for the
# specification it is referred to, the record-tap measurement that says the
# departure is added pre-tape, and the two controls that decide which axes
# are measurable and which are not.
# --------------------------------------------------------------------------

# One complete NTSC colour frame before the correction is formed, and the
# SAME span `_LOCK_DECISION_FIELDS` uses, for the same reason: it is the
# smallest basis on which a property of the TAPE can be judged without being
# biased by field parity or by where in the four-field sequence the decode
# began. Both heads must also have been seen, because a split with one head
# is not a split.
_SOURCE_DECISION_FIELDS = _LOCK_DECISION_FIELDS

_SOURCE_STATE = "_source_correction"


def sync_segment_geometry(field):
    """The sync pulse's own segment, derived from the decoder's tables.

    NAMED FOR THE SEGMENT, not for the geometry, because this file already
    reaches `vhsdecode.models.sync_geometry` by that name from another
    stage's function scope and two different things one import apart is how
    a module-level name gets quietly shadowed.

    Framed in blanking on both sides and clear of the colour burst: it runs
    from the end of one line's active video to the start of the next line's
    burst, so it contains that line's front porch, the whole sync pulse and
    the part of the back porch the burst has not reached. Every boundary is
    a system parameter and none of them is written down here.

    Where the format states no colour burst window the back porch is bounded
    by the start of active video instead, which is the same statement for a
    system that has no burst to avoid.
    """
    sys_params = field.rf.SysParams
    width = int(field.outlinelen)
    active_end = int(math.floor(field.usectooutpx(
        sys_params["activeVideoUS"][1])))
    burst = sys_params.get("colorBurstUS")
    stop = (int(math.floor(field.usectooutpx(float(burst[0]))))
            if burst is not None
            else int(math.floor(field.usectooutpx(
                sys_params["activeVideoUS"][0]))))
    return {
        "start": min(max(active_end, 0), width),
        "stop": min(max(stop, 0), width),
        "tail": width - min(max(active_end, 0), width),
        "width": width,
    }


def _sync_profile(field, geom, segment):
    """This field's mean sync profile, in IRE, over the picture lines.

    Averaged over the picture lines of the field, which is the only
    averaging done here: the pulse is the same specified shape on every one
    of them, so the mean is a measurement of the channel rather than of the
    content, and the scatter across them is what `sync_shape` reads as the
    segment's noise.
    """
    picture = getattr(field, "dspicture", None)
    if picture is None:
        return None
    lines, width = geom["lines"], geom["width"]
    if len(picture) != lines * width:
        return None
    first, last = geom["picture"]
    # the segment spans a line boundary, so the last usable row is the one
    # before the buffer's final row
    last = min(last, lines - 2)
    if last - first < 1:
        return None
    field_view = np.asarray(picture, dtype=np.float64).reshape(lines, width)
    head = field_view[first:last + 1, segment["start"]:]
    tail = field_view[first + 1:last + 2, :segment["stop"]]
    if head.shape[1] + tail.shape[1] < 8:
        return None
    joined = np.concatenate([head, tail], axis=1)
    scale = float(field.out_scale) or 1.0
    return joined.mean(axis=0) / scale


def _capture_name(field):
    """Whatever name this decode was given, or None.

    Tried rather than plumbed: the profile is a REPORT here and not a gate
    (see `_report_source_profile`), so a decode that cannot be named must
    lose the report and nothing else.
    """
    decoder = getattr(field.rf, "decoder", None)
    for value in (getattr(decoder, "infile", None),
                  getattr(decoder, "fname_out", None)):
        name = getattr(value, "name", value)
        if isinstance(name, str) and name:
            return name
    return None


def _report_source_profile(field):
    """Say which source terms this capture's profile admits, once.

    THE PROFILE PRUNES NOTHING ON VHS TODAY, AND THAT IS A FINDING RATHER
    THAN AN OMISSION. `source_correction.admissible` exists to stop a
    broadcast channel being attached to a tape that never saw one, and on a
    VHS decode neither television-only term reaches the picture: the 4.2 MHz
    transmission limit sits above the 3.0 MHz the tape returns, and
    multipath is withheld because it does not hold out. So the correction is
    the same whatever the source, the profile is REPORTED rather than
    applied, and on a wider format the same call would gate.
    """
    name = _capture_name(field)
    if name is None:
        return
    try:
        source = profiles.for_capture(name)["source"]["name"]
        verdict = source_correction.admissible(source)
    except ValueError:
        _log_once(field, "source_profile",
                  "model stages: source_correction has no recorded history "
                  "for %r, so the source's chain is not reported; the "
                  "correction itself does not depend on it - no "
                  "television-only term is observable on VHS", name)
        return
    _log_once(field, "source_profile",
              "model stages: source_correction reads this capture as a %r "
              "source - %d chain entries present, %d observable, %s "
              "withheld; the broadcast limit is %s",
              source, len(verdict["present"]), len(verdict["observable"]),
              sorted(verdict["withheld"]) or "none",
              "observable" if verdict["broadcast_limit_observable"]
              else verdict["why_not_observable"])


def correct_source_response(field, amount):
    """THE SOURCE CHANNEL, measured on the sync pulse and taken off the luma.

    Follows `source_correction.RUNTIME_CONTRACT`: accumulate the field's
    mean sync profile by head parity, form the head split once a complete
    colour frame has been seen on both heads, refer the head-common part to
    the specification's flat luma, and filter the active area with the
    resulting kernel.

    THE CORRECTION IS FORMED ONCE AND LATCHED. Re-fitting it as fields
    arrive would change the picture part way through a decode, which steps
    where the estimate lands - the failure `luma_beat` records and the
    reason `colour_free_luma` latches its own decision. The fields ahead of
    the latch are left uncorrected rather than corrected differently.

    THE PICTURE IS WRITTEN IN PLACE. `field.dspicture` and the `dsout` the
    caller is about to return are the same array, so rebinding the attribute
    would silently correct nothing.

    Returns the report for this field, or None where nothing was applied.
    """
    if not amount:
        return None
    if getattr(field, "dspicture", None) is None:
        return None

    store = field.rf.__dict__.setdefault(
        _SOURCE_STATE, {"profiles": {True: [], False: []}, "kernel": None,
                        "report": None, "refused": False})
    if store["refused"]:
        return None
    _report_source_profile(field)

    try:
        geom = geometry(field)
        segment = sync_segment_geometry(field)
    except Exception as error:                                  # noqa: BLE001
        _log_once(field, "source_geometry",
                  "model stages: source_correction declined, the segment "
                  "geometry could not be derived (%s)", error)
        store["refused"] = True
        return None

    _fold_source_profile(field, store, geom, segment)
    if store["kernel"] is None and not _form_source_kernel(field, store, geom,
                                                           amount):
        return None

    picture = field.dspicture
    luma = np.asarray(picture, dtype=np.float64).reshape(geom["lines"],
                                                         geom["width"])
    applied = _realise_source_kernel(luma, geom, store["kernel"])
    if not applied["lines"]:
        return None
    np.clip(luma, 0.0, 65535.0, out=luma)
    picture[:] = luma.reshape(-1).astype(picture.dtype, copy=False)

    field.source_correction = dict(store["report"], changed_rms=
                                   float(applied["changed_rms"]))
    ldd.logger.debug("source_correction: %d lines changed by %.1f output "
                     "units rms", applied["lines"], applied["changed_rms"])
    return field.source_correction


def _fold_source_profile(field, store, geom, segment):
    """THE MEASUREMENT HALF OF `correct_source_response`, first part: this
    field's mean sync profile into the per-parity store, once.

    One field folds in once. `downscale` can be re-run on the same field
    object when the decoder redoes a field, and a profile counted twice
    would weight that field twice.
    """
    if field.__dict__.get("_source_correction_folded"):
        return
    profile = _sync_profile(field, geom, segment)
    if profile is not None:
        field.__dict__["_source_correction_folded"] = True
        store["profiles"][bool(field.isFirstField)].append(profile)


def _form_source_kernel(field, store, geom, amount):
    """THE MEASUREMENT HALF, second part: the head split, the channel and
    the kernel, formed once a complete colour frame has been seen on both
    heads. Returns whether a kernel now stands in the store."""
    counts = {k: len(v) for k, v in store["profiles"].items()}
    if min(counts.values()) < 1 or sum(counts.values()) < _SOURCE_DECISION_FIELDS:
        return False
    try:
        split = source_correction.head_split(store["profiles"])
        measured = source_correction.channel(
            split["common"], geom["sample_rate_hz"])
        correction = source_correction.corrector(measured,
                                                 amount=float(amount))
        taps = source_correction.kernel(correction, geom["width"],
                                        geom["sample_rate_hz"])
    except Exception as error:                              # noqa: BLE001
        _log_once(field, "source_refused",
                  "model stages: source_correction declined - the sync "
                  "segment does not support the measurement (%s)", error)
        store["refused"] = True
        return False
    store["kernel"] = taps
    store["report"] = {
        "departure_db_rms": float(measured["departure_db_rms"]),
        "head_difference_rms": float(split["difference_rms"]),
        "bins": int(measured["bins"]),
        "fraction": float(correction["fraction"]),
        "maximum_gain_db": float(correction["maximum_gain_db"]),
        "capped_bins": int(correction["capped_bins"]),
        "resolution_hz": float(measured["resolution_hz"]),
        "fields": counts,
    }
    report = store["report"]
    _log_once(field, "source_formed",
              "model stages: source_correction formed on %d fields - the "
              "head-common luma departs from the specified flat by %.2f "
              "dB rms over %d bins at %.0f kHz resolution, the head "
              "difference is %.4f IRE, correcting at %.2f of it with a "
              "peak gain of %.2f dB (%d bins capped at their own "
              "evidence)",
              sum(counts.values()), report["departure_db_rms"],
              report["bins"], report["resolution_hz"] / 1e3,
              report["head_difference_rms"], report["fraction"],
              report["maximum_gain_db"], report["capped_bins"])
    return True


def _realise_source_kernel(luma, geom, taps):
    """THE REALISATION HALF: the kernel over the picture lines of a float
    field, in place, and over THE WHOLE LINE, not the active span.

    Ethan, 2026-09-07: "I am still not seeing the source -> recording VCR
    correction being applied. The cd sample will have a flat sync pulse
    when it is working." Measured, the active-span form changed only
    columns 136 to 894 of 910 and left the sync tip's within-pulse ripple
    identical to four decimal places - so the pulse could not flatten
    however right the correction was. The channel distorted the whole line
    and its inverse belongs on the whole line, which also makes the sync
    pulse the correction's own check.
    """
    return source_correction.apply_to_active(
        luma, (0, geom["width"]), geom["picture"], taps)


# ==========================================================================
# THE SYNC AND TIMING STAGES
#
# Ethan, 2026-09-06: *"Every stage that we have designed must be called and
# used. Exhaustively go through the stages and make sure they are being
# used. All of them need to be used and all of them need to model all
# dimensions."*
#
# Six modules of the sync and timing group were designed and could not be
# reached from a decode. They are wired below in the shape the two chroma
# corrections above already have - a node in `pipeline/stages.toml`, a
# default of ON carried by the declaration (`declared_default = true`),
# `--stages -<name>` as the only control, and no flag of their own.
#
# THE SUBJECT IS THE SYNC PULSE AND THE RESERVED INTERVALS, WHICH IS THE
# STANDING CONSTRAINT AND NOT A COINCIDENCE. Every reading below is taken on
# the horizontal sync pulses of the lines the SPECIFICATION names - the
# population comes from `precursor.sync_rows`, which excludes the vertical
# interval and the lines SMPTE 32M lets the head switch reach - on the burst,
# or on the reserved lines between the two. Nothing here reads the picture.
# The picture is used only as an OFFLINE gauge, in the figures quoted below,
# which is what this arc's rule permits it for.
#
# WHAT EACH ONE IS, AND WHAT IT DOES TO THE DECODE:
#
#   `sync_shape`        MEASURES. The pulse as one component on all three
#                       axes at once, fitted in the band-limited Laplace
#                       eigenbasis: the coefficient magnitudes are the
#                       amplitude axis, the mode frequencies the frequency
#                       axis, and the delay against the other head's pooled
#                       pulse - taken from the PHASE of the complex response
#                       between them, split into a common delay and a width
#                       change - is the time axis. What falls outside the
#                       format's luma band is the noise, by construction,
#                       and it is handed on rather than discarded: it is what
#                       sets `precursor`'s support window.
#
#   `precursor`         CORRECTS. The decoder's own video low-pass is a
#                       zero-phase filter, so half its impulse response acts
#                       BEFORE the input arrives and every sync edge carries
#                       a ripple ahead of itself that was never on the tape.
#                       It lands in the front porch, which is the level
#                       reference the rest of this arc rests on. Predicted
#                       from the decoder's OWN kernel and this field's own
#                       measured edge, and subtracted ahead of the edge only.
#
#   `sync_depth`        CORRECTS. Blanking is the specified zero and the
#                       synchronizing level is 40 IRE beneath it, so the
#                       measured spacing over 40 is the luma channel's own
#                       gain - the only absolute scale the signal can supply,
#                       because everything else in the composite waveform is
#                       specified as a fraction. Applied as a gain about the
#                       measured blanking, with the within-field drift
#                       removed on the same fit, and never as a level shift.
#
#   `burst_sync_lock`   MEASURES. The sync edge is unambiguous and coarse;
#                       the burst is precise and wrapped. Together they place
#                       the line absolutely, and the vernier's answer is
#                       compared with the displacement the decoder's own
#                       burst refinement applied - two estimates of one
#                       quantity that share no arithmetic.
#
#   `vertical_interval` MEASURES, AND ON THIS MATERIAL IT DECLINES. Its
#                       subject is an insertion test signal, whose synthetic
#                       side is printed in a standard to a tenth of an IRE.
#                       None of the three tapes carries one; the stage says
#                       so in the log rather than fitting a channel to a
#                       blank line.
#
#   `tape_speed`        MEASURES. The line period read from the sync
#                       positions BEFORE the time base correction, which is
#                       the only place it survives - the corrector nulls it
#                       by construction - and what that speed error does to
#                       the frequency axis, where it is exactly degenerate
#                       with a common scaling of every magnetic length and
#                       therefore cannot be measured at all.
#
# EVERY DIMENSION CARRIES, AND WHERE ONE IS ABSENT IT IS NAMED. `sync_shape`
# and `sync_depth` carry all three by measurement; `precursor` carries all
# three by construction, its kernel being a frequency response, its ripple an
# amplitude and its support a time; `burst_sync_lock` is one relation in three
# units, phi = 2 pi f t, which is the module's own point; `tape_speed` carries
# time by measurement, frequency and amplitude by derivation, and states that
# the frequency axis cannot witness it. `vertical_interval` carries all three
# and reaches none of them here, because its subject is absent.
#
# AND NOTHING IS REDUCED TO A MAGNITUDE AT A BOUNDARY. The pulse is handed on
# as its complex spectrum, the head-to-head reading as the complex response
# between two pulses, the depth's frequency axis as a complex Wiener transfer
# against the standard's own pulse, and the burst as a complex phasor. This
# arc has lost a measurement's rank at exactly this kind of boundary before.
# ==========================================================================

# One name per line rather than a wrapped tuple, deliberately: the stage
# inventory reads imports as text and only sees the names sharing a line with
# `models import`, so a continuation line would leave three of these six
# reported as designed and never called - which is the exact gap this group
# exists to close.
from vhsdecode.models import burst_sync_lock, precursor, standard_levels
from vhsdecode.models import sync_depth, sync_shape, tape_speed
from vhsdecode.models import vertical_interval

# The luma band that separates the pulse's SHAPE from its NOISE, per format.
# A format constant in every case - `sync_shape` states the citation for each
# - and a format not named here has no stated luma band in this tree, so the
# stages that need one decline rather than borrow VHS's.
_LUMA_BAND_HZ = {
    "VHS": sync_shape.VHS_LUMA_BAND_HZ,
    "VHSHQ": sync_shape.VHS_LUMA_BAND_HZ,
    "SVHS": sync_shape.SVHS_LUMA_BAND_HZ,
    "SVHS_ET": sync_shape.SVHS_LUMA_BAND_HZ,
    "UMATIC": sync_shape.UMATIC_LUMA_BAND_HZ,
    "UMATIC_HI": sync_shape.UMATIC_LUMA_BAND_HZ,
    "UMATIC_SP": sync_shape.UMATIC_LUMA_BAND_HZ,
}
# The other fourteen formats `main.py` accepts are absent deliberately.
# `sync_shape` states a luma band for three families and cites each; a band
# borrowed from a neighbouring format would be a constant invented at a call
# site, which is what the standing rule forbids, so those decodes decline and
# say which format they declined for.

# How many independent complex dimensions a band-limited segment of duration
# T holds: 2 B T. Used to turn a correlation into a significance without
# choosing a threshold - the same role the Rayleigh tail plays for the
# chroma stages above.
_DIMENSIONS_PER_SECOND = 2.0


def _sync_system(field):
    """The system name in the form the specification modules use, or None
    where this tree states no vertical-interval geometry for it."""
    system = getattr(field.rf, "color_system", None)
    if system in ("NTSC", "PAL", "PAL_M", "NLINHA", "SECAM"):
        return "NTSC" if system in ("NTSC", "PAL_M") else "PAL"
    return None


def sync_spans(field):
    """Every span the sync group reads, derived and never declared.

    Returns None where the decode cannot supply one - a system with no
    stated interval geometry, a raw export whose samples are not on an IRE
    scale, or a field too short to hold the specified population.

      * the sync line POPULATION from `precursor.sync_rows`, which takes the
        vertical interval's own length from `sync_geometry.interval_geometry`
        and drops the lines SMPTE 32M-2004 clause 3.6 permits the head switch
        to reach. Specification line numbers, never waveform-derived, which
        is this lane's standing rule.
      * the front porch from `precursor.timing`, so the segment below holds
        the porch the precursor lands in as well as the pulse.
      * the segment stops at the burst, from `SysParams["colorBurstUS"]`, so
        no chrominance enters a luma shape.
      * the tip's flat INTERIOR from `sync_depth.TIP_INTERIOR`, because the
        container's clip pins the pulse's downward overshoot at the edges.
      * the back porch from the decoder's OWN `level_windows`, which is
        already guarded off both transitions and already shown to be the
        content-independent anchor - a second derivation here would be a
        second answer to a question the decoder has answered.
      * the units per IRE from `hz_to_output` itself, evaluated at 0 and 100
        IRE exactly as `build_json` does, so the runtime and the offline
        instruments read the same scale.
    """
    cached = field.__dict__.get("_sync_spans")
    if cached is not None:
        return cached or None
    field.__dict__["_sync_spans"] = False

    if getattr(field.rf.options, "export_raw_tbc", False):
        return None
    system = _sync_system(field)
    if system is None:
        return None
    sys_params = field.rf.SysParams
    rate = float(sys_params["outfreq"]) * 1e6
    width = int(field.outlinelen)
    lines = int(field.outlinecount)
    try:
        timing = precursor.timing({"system": system, "sample_rate_hz": rate})
        rows = precursor.sync_rows(lines, system)
    except (ValueError, KeyError):
        return None
    rows = np.asarray(rows[rows >= 1], dtype=np.intp)
    if rows.size < 8:
        return None

    front_porch = int(math.ceil(timing["porch"]))
    burst_start = int(math.floor(field.usectooutpx(
        sys_params["colorBurstUS"][0])))
    segment = front_porch + burst_start
    if segment < 16 or front_porch < 2 or burst_start <= timing["sync"]:
        return None

    low, high = sync_depth.TIP_INTERIOR
    tip = (int(math.ceil(low * timing["sync"])),
           int(math.floor(high * timing["sync"])) + 1)
    _, porch = field.level_windows()

    blanking = float(field.hz_to_output(field.rf.iretohz(0)))
    white = float(field.hz_to_output(field.rf.iretohz(100)))
    units_per_ire = (white - blanking) / 100.0
    if not np.isfinite(units_per_ire) or units_per_ire <= 0:
        return None

    spans = {
        "system": system,
        "lines": lines,
        "width": width,
        "rows": rows,
        "sample_rate_hz": rate,
        "front_porch": front_porch,
        "segment": segment,
        "sync_samples": float(timing["sync"]),
        "edge_samples": float(timing["edge"]),
        "line_samples": float(timing["line"]),
        "tip": tip,
        "porch": (int(porch[0]), int(porch[1])),
        "blanking": blanking,
        "units_per_ire": units_per_ire,
        "band_limit_hz": _LUMA_BAND_HZ.get(
            str(getattr(field.rf.options, "tape_format", "")).upper()),
    }
    field.__dict__["_sync_spans"] = spans
    return spans


def sync_pulses(field, picture):
    """This field's sync pulses, pooled, on all three axes and COMPLEX.

    One measurement shared by every stage in the group, taken once and
    cached on the FIELD - the quantity is per field and a stale one would be
    applied to the wrong picture.

    The segment runs from the front porch of the preceding row through the
    pulse to the start of the burst, so it holds the porch the precursor
    lands in, the whole pulse the shape is fitted on, and no chrominance.
    The porch belongs to the PRECEDING row because a time base corrected
    line begins at the sync edge, which puts each pulse's own front porch at
    the tail of the row before it.

    Returns None where the field cannot support the measurement, and the
    reason is logged once per decode by the caller that needed it.
    """
    cached = field.__dict__.get("_sync_pulses")
    if cached is not None:
        return cached or None
    field.__dict__["_sync_pulses"] = False

    spans = sync_spans(field)
    if spans is None or picture is None:
        return None
    if len(picture) != spans["lines"] * spans["width"]:
        return None
    rows = spans["rows"]
    field_lines = np.asarray(picture, dtype=np.float64).reshape(
        spans["lines"], spans["width"])

    porch_n = spans["front_porch"]
    pulses = np.empty((rows.size, spans["segment"]), dtype=np.float64)
    pulses[:, :porch_n] = field_lines[rows - 1, spans["width"] - porch_n:]
    pulses[:, porch_n:] = field_lines[rows, :spans["segment"] - porch_n]
    pulses = (pulses - spans["blanking"]) / spans["units_per_ire"]
    profile = pulses.mean(axis=0)

    band = spans["band_limit_hz"]
    if band is None:
        return None
    try:
        shape = sync_shape.shape_components(profile, spans["sample_rate_hz"],
                                            band)
        transients = sync_shape.locate_transients(profile,
                                                  spans["sample_rate_hz"])
    except (ValueError, np.linalg.LinAlgError):
        return None

    # THE PULSE AS A COMPLEX OBJECT. The eigenbasis coefficients are real by
    # construction, so a stage handed only those would have lost the phase -
    # which is where the whole time axis lives. The spectrum is carried
    # beside them and every consumer that wants a delay reads it.
    spectrum = np.fft.rfft(profile - profile.mean())
    frequencies = np.fft.rfftfreq(profile.size, 1.0 / spans["sample_rate_hz"])

    measured = {
        "spans": spans,
        "pulses": pulses,
        "profile": profile,
        "spectrum": spectrum,
        "frequency_hz": frequencies,
        "shape": shape,
        "transients": transients,
        # the 10 per cent point of the leading transient, which is where the
        # drive's own increments begin and therefore where the precursor's
        # window ends (see `precursor.precursor`)
        "onset": _transition_onset(profile, transients, porch_n),
        "noise_rms_ire": float(shape["noise_rms"]),
    }
    field.__dict__["_sync_pulses"] = measured
    return measured


def _transition_onset(profile, transients, porch_n):
    """Where the leading transient's own increments begin: its 10 per cent
    point, measured on the profile rather than assumed at the row origin."""
    values = np.asarray(profile, dtype=np.float64)
    fall = int(transients["fall"])
    before = float(np.median(values[:max(fall - 1, 1)]))
    after = float(np.median(values[fall + 2:int(transients["rise"])]))
    threshold = before + 0.10 * (after - before)
    crossed = np.nonzero(values[:fall + 2] <= threshold)[0]
    if crossed.size:
        return int(max(1, crossed[0]))
    return int(max(1, min(fall, porch_n)))


def _head_reference(field, measured, key):
    """The pooled pulse of the OTHER head, against which a delay means
    something. Field parity is the only head label a decode has, and which
    parity is head A is not knowable from a time base corrected field, so
    the labels are a convention - `precursor` says the same.

    Returns None until both parities have been seen, because a delay
    measured against a shape's own head is zero by construction.
    """
    store = field.rf.__dict__.setdefault(key, {})
    parity = bool(field.isFirstField)
    entry = store.setdefault(parity, {"sum": None, "count": 0})
    if not field.__dict__.get(key + "_folded"):
        field.__dict__[key + "_folded"] = True
        profile = np.asarray(measured["profile"], dtype=np.float64)
        entry["sum"] = (profile.copy() if entry["sum"] is None
                        else entry["sum"] + profile)
        entry["count"] += 1
    other = store.get(not parity)
    if not other or not other["count"] or other["sum"] is None:
        return None
    if other["sum"].size != measured["profile"].size:
        return None
    return other["sum"] / float(other["count"])


def _write_back(picture, corrected):
    """A corrected field written back into its own buffer, ROUNDED.

    `lddecode.utils.hz_to_output_array` adds a half before its cast and says
    why: without it `np.uint16` truncates and every output pixel comes out
    about half a least significant bit low. A correction that writes the
    whole field back through a plain `astype` reintroduces exactly that bias
    over the entire picture - a systematic 0.5 LSB, which at 358.4 units to
    the IRE is 0.0014 IRE, small but present on every sample and in one
    direction. Adding the same half here keeps the two paths agreeing, and
    an untouched sample - an exact integer - is unchanged by it.
    """
    np.clip(corrected + 0.5, 0.0, 65535.0, out=corrected)
    picture[:] = corrected.reshape(-1).astype(picture.dtype, copy=False)


def measure_sync_shape(field, picture):
    """THE PULSE AS ONE COMPONENT ON ALL THREE AXES.

    Node `sync_shape`. Applies nothing: it is the measurement the other two
    luma stages in this group stand on, and it is declared as a node of its
    own so that it can be turned on for its log without turning on a
    correction - `--stages +sync_shape`.

    THE THIRD AXIS IS MEASURED, NOT DERIVED, and that distinction is the
    module's own. `shape_components` reports a `group_delay_s` which is
    Bode's relation applied to the shape's own magnitude - the delay a
    minimum-phase channel of that magnitude would have - so it carries
    nothing the amplitude does not. `edge_pair` is the time axis measured
    instead: the delay this field's pulse carries against the OTHER head's
    pooled pulse, taken from the phase of the complex response between them,
    and split into the part a delay explains (both transients moving
    together) and the part a width change explains (the two moving apart).
    One number from the whole pulse cannot tell those apart, and this lane
    has already been caught by exactly that confusion.

    MEASURED BY THIS FUNCTION, 2026-09-06, on the 75 per cent bar SP decode
    (26 fields, 245 sync pulses a field, the segment 96 samples from the
    preceding row's front porch to the start of the burst):

        60 modes below 3.0 MHz, of which 8.90 are effectively occupied
        shape 17.587 IRE rms, energy centroid 595 kHz
        out-of-band remainder 0.0532 IRE rms - the noise, by construction
        head against head: common delay -3.35 ns, width change +1.49 ns,
            whole pulse -3.73 ns, over 19 carried bins

    AND THE PAIR EARNS ITS PLACE ON THIS TAPE. The whole-pulse figure of
    -3.73 ns is neither of the two terms it is made of, and `is_a_delay`
    reads False for it - what survives the linear phase fit is larger than
    the delay itself - so quoting the one number would have been quoting a
    projection of a dispersive response onto a straight line.
    """
    measured = sync_pulses(field, picture)
    if measured is None:
        _log_once(field, "sync_shape_declined",
                  "model stages: sync_shape declined - this field cannot "
                  "supply the specified sync population on an IRE scale "
                  "(system %s, format %s)",
                  getattr(field.rf, "color_system", "unknown"),
                  getattr(field.rf.options, "tape_format", "unknown"))
        return None

    spans = measured["spans"]
    shape = measured["shape"]
    reading = {
        # ---- the three axes, all read off one fit ----
        "frequency_hz": shape["frequency_hz"],
        "amplitude": shape["amplitude"],
        "amplitude_rms_ire": float(shape["amplitude_rms"]),
        "centroid_hz": float(shape["centroid_hz"]),
        "effective_rank": float(shape["effective_rank"]),
        # ---- the split the format decides ----
        "noise_rms_ire": float(shape["noise_rms"]),
        "band_limit_hz": float(shape["band_limit_hz"]),
        # ---- and the pulse itself, COMPLEX ----
        "spectrum": measured["spectrum"],
        "profile": measured["profile"],
        "pulses": int(measured["pulses"].shape[0]),
    }

    reference = _head_reference(field, measured, "_sync_shape_head")
    if reference is not None:
        try:
            pair = sync_shape.edge_pair(measured["profile"], reference,
                                        spans["sample_rate_hz"],
                                        float(shape["band_limit_hz"]))
        except (ValueError, np.linalg.LinAlgError):
            pair = None
        if pair is not None:
            reading["pair"] = pair
            reading["common_delay_s"] = float(pair["common_delay_s"])
            reading["width_change_s"] = float(pair["width_change_s"])
            # THE COMPLEX RESPONSE ITSELF travels on, not its magnitude.
            reading["response"] = pair["whole"]["response"]
            _log_once(field, "sync_shape_pair",
                      "model stages: sync_shape - head to head, common "
                      "delay %+.2f ns, width change %+.2f ns, whole pulse "
                      "%+.2f ns (a delay: %s), %d bins below %.1f MHz",
                      pair["common_delay_s"] * 1e9,
                      pair["width_change_s"] * 1e9,
                      pair["whole_delay_s"] * 1e9,
                      pair["whole"]["is_a_delay"],
                      int(pair["whole"]["bins"]),
                      float(shape["band_limit_hz"]) / 1e6)

    _log_once(field, "sync_shape",
              "model stages: sync_shape - %d pulses, %d modes below %.1f "
              "MHz carrying %.2f effective, shape %.3f IRE rms, out-of-band "
              "noise %.4f IRE, energy centroid %.0f kHz",
              reading["pulses"], int(shape["modes"]),
              float(shape["band_limit_hz"]) / 1e6,
              reading["effective_rank"], reading["amplitude_rms_ire"],
              reading["noise_rms_ire"], reading["centroid_hz"] / 1e3)
    field.sync_shape = reading
    return reading


def correct_precursor(field, picture):
    """THE DECODER'S OWN PRECURSOR, TAKEN OUT OF THE FRONT PORCH.

    Node `precursor`. Follows the module's convention exactly: the kernel is
    recovered from the decoder's OWN constructor rather than rebuilt, the
    drive is the MEASURED edge rather than an ideal step - because the tape
    channel has already rolled the edge off and an ideal step has full
    energy where the ring is largest - and the subtraction touches only the
    samples AHEAD of the transition.

    WHY IT IS WORTH DOING AT ALL. The decoder's video low-pass is applied by
    multiplying a real, non-negative response, and a real response is a
    zero-phase filter whose impulse response is symmetric about lag zero. So
    half of it acts before its input arrives. Convolved with a 40 IRE sync
    edge the anticausal half predicts about 2.6 IRE at 0.2 microseconds
    ahead of the edge and still 0.08 IRE at 1.5, and the standard's front
    porch is 1.5 microseconds long - the whole porch sits inside the
    precursor's support, so there is no precursor-free part of it to retreat
    to.

    THE KERNEL IS CACHED ON THE DECODER, not the field: it is a property of
    the decoder's filter table and does not change from field to field, and
    building it involves an inverse transform of the whole block.

    APPLIED IN FULL rather than at half. The correction-gain law exists for
    an estimator whose model error is unknown; here the kernel is the
    decoder's own filter, known exactly, and the only estimated quantity is
    the drive - which is this field's own pooled edge over two hundred and
    forty-five lines.

    MEASURED, 2026-09-06, on three decodes. The kernel's split is verified
    on every one of them - the peak at lag zero, the two halves agreeing to
    6.0e-17 of it, each weighing 0.2716 - and what is removed is:

        tape         peak     rms      window   support above its own noise
        75 bars SP  0.580   0.1561   22 samp.   24 samples (1.68 us)
        home        0.682   0.1941   21 samp.   50 samples (3.49 us)
        countdown   0.468   0.1429   21 samp.   26 samples (1.82 us)

    all in IRE. AND THE GAUGE IS THE FRONT PORCH AGAINST THE SAME LINE'S
    SETTLED BACK PORCH, which this stage never reads: over the two samples
    nearest the edge that departure goes -2.0794 to -1.9739 IRE on the bar
    tape, -4.1553 to -3.8293 on home and -6.6114 to -6.3673 on countdown,
    and on the single nearest sample of the bar tape -3.0615 to -2.4745, a
    fifth of it. What remains is the relaxation the porch carries from the
    preceding active line, which is a different term and not this stage's to
    remove.
    """
    measured = sync_pulses(field, picture)
    if measured is None:
        _log_once(field, "precursor_declined",
                  "model stages: precursor declined - no sync population on "
                  "an IRE scale in this field")
        return None
    reading = _read_precursor(field, measured)
    if reading is None:
        return None
    spans = measured["spans"]
    corrected = np.array(picture, dtype=np.float64).reshape(spans["lines"],
                                                            spans["width"])
    _realise_precursor(corrected, reading)
    _write_back(picture, corrected)
    # THE POOLED PULSE IS NOW STALE. `sync_depth` reads its levels from the
    # picture directly but takes its frequency axis from this cache, and a
    # cached pulse taken before this subtraction would put the precursor
    # back into a measurement of the corrected signal.
    field.__dict__.pop("_sync_pulses", None)

    report = _precursor_report(reading)
    kernel = reading["kernel"]
    _log_once(field, "precursor",
              "model stages: precursor - kernel split verified %s, "
              "anticausal weight %.4f, symmetry %.1e; removed %.3f IRE peak "
              "and %.4f IRE rms from the %d samples ahead of the edge on %d "
              "lines, support %d samples (%.2f us) above %.4f IRE of noise",
              kernel["split_point_verified"], kernel["anticausal_weight"],
              kernel["symmetry_error"], report["peak_ire"],
              report["rms_ire"], report["onset"], report["rows"],
              report["support_samples"], report["support_us"],
              reading["noise_rms_ire"])
    field.precursor = report
    return report


def _read_precursor(field, measured):
    """THE MEASUREMENT HALF OF `correct_precursor`: the ripple the decoder's
    own zero-phase kernel put ahead of this field's pooled edge, predicted
    and not yet subtracted.

    Factored out so the picture transform can take the same reading on the
    field as it arrived and realise it inside one write with everything
    else; `correct_precursor` composes this with `_realise_precursor` and
    behaves exactly as it did.

    Returns None where the kernel cannot be recovered or the onset falls
    outside the segment, else the reading in IRE with its spans.
    """
    spans = measured["spans"]
    kernel = _precursor_kernel(field, spans)
    if kernel is None:
        return None

    onset = int(measured["onset"])
    porch_n = spans["front_porch"]
    if onset <= 1 or onset > spans["segment"]:
        return None

    predicted = precursor.precursor(measured["profile"], kernel, onset=onset)
    # HOW FAR AHEAD IT STANDS ABOVE THIS FIELD'S OWN NOISE, from the module's
    # own function and this field's own out-of-band remainder - which is
    # what `sync_shape` measured and why the two stages are one group.
    depth_ire = abs(float(np.median(
        measured["profile"][porch_n + spans["tip"][0]:
                            porch_n + spans["tip"][1]]))
        - float(np.median(measured["profile"][:max(porch_n - 2, 1)])))
    support = precursor.support_samples(
        kernel, depth_ire * spans["units_per_ire"],
        float(measured["noise_rms_ire"]) * spans["units_per_ire"])

    removed = np.asarray(predicted[:onset], dtype=np.float64)
    if not np.isfinite(removed).all() or not removed.size:
        return None
    return {
        "spans": spans,
        "kernel": kernel,
        "onset": onset,
        "removed_ire": removed,
        "support_samples": int(support),
        "noise_rms_ire": float(measured["noise_rms_ire"]),
    }


def _realise_precursor(lines, reading):
    """THE REALISATION HALF: the predicted ripple taken off the samples
    ahead of the edge, on a float field of the decode's own shape, in place.

    THE WINDOW STRADDLES TWO ROWS, AND ITS SPLIT IS NOT FIXED. Segment index
    0 is the first sample of the preceding row's front porch and index
    `front_porch` is this row's first sample, so where the onset falls
    decides how much of each row is touched - and the onset is MEASURED on
    the pulse, not assumed at the row origin. A unit test on a specified
    pulse put it one sample inside the porch, where taking the whole porch
    broadcast a 21-sample correction onto 22 columns.
    """
    spans = reading["spans"]
    onset = int(reading["onset"])
    porch_n = spans["front_porch"]
    in_units = np.asarray(reading["removed_ire"], dtype=np.float64) \
        * spans["units_per_ire"]
    rows = spans["rows"]
    start = spans["width"] - porch_n
    tail = int(min(onset, porch_n))
    head = int(max(0, min(onset - porch_n, spans["width"])))
    lines[rows - 1, start:start + tail] -= in_units[:tail]
    if head > 0:
        lines[rows, :head] -= in_units[porch_n:porch_n + head]


def _precursor_report(reading):
    """The reading `correct_precursor` has always attached to the field."""
    removed = reading["removed_ire"]
    spans = reading["spans"]
    kernel = reading["kernel"]
    support = reading["support_samples"]
    return {
        "predicted_ire": removed,
        "peak_ire": float(np.abs(removed).max()),
        "rms_ire": float(np.sqrt(np.mean(removed ** 2))),
        "support_samples": int(support),
        "support_us": float(support) / spans["sample_rate_hz"] * 1e6,
        "onset": int(reading["onset"]),
        "kernel": {k: kernel[k] for k in
                   ("anticausal_weight", "causal_weight", "centre_weight",
                    "symmetry_error", "split_point_verified",
                    "resample_deviation")},
        "rows": int(spans["rows"].size),
    }


def _precursor_kernel(field, spans):
    """The decoder's own zero-phase kernel on the OUTPUT grid, once a decode.

    `filter_kernel` regenerates the response on the new rate from the
    decoder's own builder with the decoder's own parameters and reports how
    far that is from the decoder's response interpolated onto the same grid;
    the filter passes nothing above about 4 MHz, well inside either
    Nyquist, so the two are the same continuous filter sampled twice.
    """
    cached = field.rf.__dict__.get("_precursor_kernel")
    if cached is not None:
        return cached or None
    field.rf.__dict__["_precursor_kernel"] = False
    try:
        lowpass = precursor.decoder_video_lowpass(rf=field.rf)
        # A power of two spanning the same duration as the decoder's block,
        # which is what `filter_kernel` chooses for itself when asked.
        kernel = precursor.filter_kernel(lowpass, spans["sample_rate_hz"])
    except (KeyError, ValueError, TypeError, AttributeError) as error:
        ldd.logger.warning("model stages: precursor could not recover the "
                           "decoder's own low-pass (%s) - declining", error)
        return None
    if not kernel.get("split_point_verified"):
        ldd.logger.warning(
            "model stages: precursor declined - the recovered kernel is not "
            "symmetric about lag zero (peak at lag %d, symmetry %.2e), so it "
            "is not the zero-phase filter this correction inverts",
            int(kernel["peak_lag"]), float(kernel["symmetry_error"]))
        return None
    field.rf.__dict__["_precursor_kernel"] = kernel
    return kernel


def correct_sync_depth(field, picture):
    """THE BLANKING-TO-TIP SPACING AS THE LUMA CHANNEL'S ABSOLUTE SCALE.

    Node `sync_depth`. Blanking is the specified zero and the synchronizing
    level sits 40 IRE beneath it (ITU-R BT.1700), so the measured spacing
    over 40 is the luma channel's gain - and it is the only absolute scale
    the signal can supply, because every other level in the composite
    waveform is specified as a fraction of something.

    THREE AXES, AND EACH IS USED FOR WHAT IT CAN BE:

      AMPLITUDE   the spacing itself, read on the pulse's flat interior with
                  the container's clip handled by censored maximum
                  likelihood rather than by masking - masking is the worst of
                  the three treatments and the module measures by how much.
      TIME        the spacing's drift down the field, which must be zero
                  because both levels are specified constants, fitted on the
                  same lines and removed with the level.
      FREQUENCY   `depth_spectrum`, the same deficit against the standard's
                  own pulse as a complex Wiener transfer. REPORTED, NOT
                  APPLIED: a frequency-dependent luma scale belongs to the
                  equaliser lane by Ethan's own ruling, and this stage owns
                  the flat scale the standard fixes.

    IT REFUSES TO RUN BESIDE `--ire0_adjust hsync`. That mode derives the
    output scale from the same two levels inside `hz_to_output`, so the
    spacing it produces is 40 IRE by construction and this stage would be
    fitting a correction to its own output. Declined, and said in the log.

    THE GAIN IS APPLIED ABOUT THE MEASURED BLANKING AND MOVES NOTHING ELSE.
    A shift of the zero is a different defect and belongs to
    `standard_levels`; a spacing is immune to where the zero sits, and so is
    this correction.

    MEASURED, 2026-09-06, on three decodes:

        tape        spacing   scatter   gain      drift/field   verdict
        75 bars SP   39.071    0.264   0.9768      +0.094       applied
        home         37.962    0.358   0.9490      -0.281       applied
        countdown       -         -       -           -         DECLINED

    AND THE PICTURE CONFIRMS IT WITHOUT HAVING BEEN USED TO FIT IT. On the
    bar tape the eight bar plateaus regress on SMPTE 170M table A.3's
    specified luminance levels; their root-mean-square departure from those
    levels falls from 1.546 IRE to 0.567 IRE, and the gain through the
    origin goes 0.9728 to 0.9966. The sync said 0.9768 and the picture said
    0.9728 before the correction and 0.9966 after, which is the whole claim
    of this stage demonstrated on a channel it never reads.

    ON HOME THERE IS NO SUCH GAUGE AND THE RESULT IS REPORTED WITHOUT ONE. A
    home recording carries no reference white, and that decode's picture
    already stands above 100 IRE on 9.3 per cent of its active samples
    before any correction; scaling it up puts 20.7 per cent there. Whether
    that source was over-driven is a separate question this stage cannot
    answer, so the 0.9490 is quoted as what the sync pulse says and nothing
    more.
    """
    if _sync_depth_superseded(field):
        return None

    measured = sync_pulses(field, picture)
    if measured is None:
        _log_once(field, "sync_depth_declined",
                  "model stages: sync_depth declined - no sync population on "
                  "an IRE scale in this field")
        return None
    reading = _read_sync_depth(field, picture, measured)
    if reading is None:
        return None

    spans = measured["spans"]
    corrected = np.array(picture, dtype=np.float64).reshape(spans["lines"],
                                                            spans["width"])
    _realise_sync_depth(corrected, reading)
    _write_back(picture, corrected)

    report = _sync_depth_report(reading)
    _log_once(field, "sync_depth",
              "model stages: sync_depth - spacing %.3f IRE against the "
              "specified %.1f on %d lines (scatter %.3f), gain %.4f "
              "(%+.2f dB), drift %+.3f IRE a field, tip clipped %s at "
              "%.4f%% of samples",
              report["spacing_ire"], sync_depth.SPECIFIED_DEPTH_IRE,
              report["lines"], report["scatter_ire"], report["gain"],
              report["as_decibels"], report["slope_ire_per_field"],
              report["clipped"], 100.0 * report["fraction_at_floor"])
    spectrum = report["spectrum"]
    if spectrum is not None:
        _log_once(field, "sync_depth_spectrum",
                  "model stages: sync_depth frequency axis - DC gain %.4f "
                  "against the spacing's %.4f, tilt %+.2f dB/MHz across the "
                  "band, %d pulses",
                  float(spectrum["dc_gain"]), report["gain"],
                  float(spectrum["tilt_db_per_mhz"]), int(spectrum["pulses"]))
    field.sync_depth = report
    return report


def _sync_depth_superseded(field):
    """Whether `--ire0_adjust hsync` already derives the output scale from
    the porch and the tip, in which case the spacing is 40 IRE by
    construction and a gain fitted to it would be fitted to its own
    output. Decided per decode and said once."""
    modes = getattr(field.rf.options, "ire0_adjust", ()) or ()
    if "hsync" in modes:
        _log_once(field, "sync_depth_superseded",
                  "model stages: sync_depth declined for this decode - "
                  "--ire0_adjust hsync already derives the output scale from "
                  "the porch and the tip, so the spacing is 40 IRE by "
                  "construction and this stage would fit its own output")
        return True
    return False


def _read_sync_depth(field, picture, measured):
    """THE MEASUREMENT HALF OF `correct_sync_depth`: the spacing, its
    trustworthiness, the per-row gain a straight line through it gives and
    the anchor it is applied about - measured, and not yet applied.

    Factored out so the picture transform can take the same reading on the
    field as it arrived; `correct_sync_depth` composes this with
    `_realise_sync_depth` and behaves exactly as it did. Every refusal below
    is logged once and returns None.
    """
    spans = measured["spans"]
    lines = np.asarray(picture, dtype=np.float64).reshape(spans["lines"],
                                                          spans["width"])
    rows = spans["rows"]
    white = spans["blanking"] + 100.0 * spans["units_per_ire"]
    tip_columns = range(*spans["tip"])
    porch_columns = range(*spans["porch"])

    clip = sync_depth.detect_clip(lines[rows][:, list(tip_columns)],
                                  spans["blanking"], white)
    result = sync_depth.spacing(lines[rows], spans["blanking"], white,
                                tip_columns, porch_columns)
    # THE CENSORED FIT HAS A BOUNDARY, AND IT IS THE MEDIAN. Censoring exists
    # to recover a mean whose TAIL the container has cut off; while the level
    # itself stands above the floor that is an interpolation of observed data,
    # and `sync_depth` measures its worth there - on a planted tip with 6.7
    # per cent of its samples below the floor it recovers the truth to 0.08
    # IRE where masking is out by 27.8. Once the fitted mean falls BELOW the
    # floor more than half the distribution is gone and the estimator is
    # extrapolating a level nothing ever showed. That is the boundary, and it
    # is the estimator's own rather than a fraction anyone picked.
    #
    # IT FIRES ON REAL MATERIAL. On the countdown decode 78.9 per cent of the
    # pulse's flat interior sits on the container floor and its 99th
    # percentile is up at +60 IRE where picture has replaced the pulse; the
    # fit returned a tip 18 250 units beneath the floor with a fitted sigma of
    # 19 739 - 55 IRE of scatter on a level the standard fixes - which came
    # out as a spacing of 107.7 IRE and a gain of 2.69. That passed the
    # module's own plausibility guard and would have been applied.
    censored = result.get("censored")
    floor = result.get("censor_floor")
    if censored and censored.get("fitted") and floor is not None:
        below = float(censored["mean"])
        if np.isfinite(below) and below < float(floor):
            _log_once(field, "sync_depth_clipped",
                      "model stages: sync_depth declined - %.1f%% of the "
                      "pulse's flat interior sits on the container floor and "
                      "the censored fit puts the tip %.0f units BENEATH it "
                      "with a fitted sigma of %.0f, so more than half the "
                      "level is gone and what it reports was never observed; "
                      "the sync tip is not a level on this material",
                      100.0 * float(censored.get("censored_fraction", 0.0)),
                      float(floor) - below, float(censored["sigma"]))
            return None

    # AND THE LINES HAVE TO AGREE ABOUT A CONSTANT. Both levels are
    # specified constants, so the per-line spacing is one number measured
    # many times; a scatter comparable to the number itself means the lines
    # are not all measuring it. On the countdown decode, where picture
    # replaces the pulse on part of the population, a field came back with a
    # spacing of 14.4 IRE and a scatter of 37.3 - a gain of 0.36 that would
    # have been applied to the whole picture. The bound is the STANDARD'S
    # OWN tolerance on this level (`standard_levels.IRE_TOLERANCE`, 1 IRE on
    # the sync tip and the setup), read robustly so a handful of bad lines
    # cannot set it; the two good decodes sit at 0.26 and 0.36 IRE against
    # it. It is a stricter reading than the standard intends, taken
    # deliberately: refusing costs a correction of a fifth of an IRE, and
    # applying a gain fitted to a population that is not all sync costs the
    # picture.
    spread = np.asarray(result["per_line"], dtype=np.float64)
    spread = spread[np.isfinite(spread)]
    disagreement = (1.4826 * float(np.median(np.abs(spread - np.median(spread))))
                    if spread.size else float("inf"))
    if not disagreement < standard_levels.IRE_TOLERANCE:
        _log_once(field, "sync_depth_scatter",
                  "model stages: sync_depth declined - the %d lines disagree "
                  "about the spacing by %.2f IRE against the %.1f IRE the "
                  "standard allows on this level, so they are not all "
                  "measuring one constant and a gain fitted to them would "
                  "not be the channel's",
                  int(spread.size), disagreement,
                  standard_levels.IRE_TOLERANCE)
        return None

    try:
        scale = sync_depth.correction(result)
    except ValueError as error:
        _log_once(field, "sync_depth_refused",
                  "model stages: sync_depth declined - %s", error)
        return None

    # THE TIME AXIS, FITTED ON THE SAME LINES. Both levels are specified
    # constants, so a drift down the field is an error; the straight line
    # through the per-line spacing is removed with its level, which makes
    # the correction a per-line gain rather than a per-field one.
    per_line = np.asarray(result["per_line"], dtype=np.float64)
    good = np.isfinite(per_line)
    index = np.asarray(rows, dtype=np.float64)
    if good.sum() > 8 and np.ptp(index[good]) > 0:
        slope, intercept = np.polyfit(index[good], per_line[good], 1)
        fitted = slope * index + intercept
    else:
        slope = 0.0
        fitted = np.full(index.shape, float(result["spacing_ire"]))
    fitted = np.where(np.isfinite(fitted), fitted, result["spacing_ire"])
    gain = fitted / sync_depth.SPECIFIED_DEPTH_IRE
    if not np.isfinite(gain).all() or gain.min() <= 0.1 or gain.max() >= 3.0:
        _log_once(field, "sync_depth_refused",
                  "model stages: sync_depth declined - the per-line gain "
                  "spans %.3f to %.3f, outside anything a chain plausibly "
                  "does", float(np.nanmin(gain)), float(np.nanmax(gain)))
        return None

    # THE FREQUENCY AXIS, measured and reported beside the correction.
    try:
        spectrum = sync_depth.depth_spectrum(
            measured["pulses"][:, spans["front_porch"]:],
            spans["sample_rate_hz"])
    except (ValueError, np.linalg.LinAlgError):
        spectrum = None

    # A GAIN ABOUT THE MEASURED BLANKING, on every row and not only the ones
    # the fit was taken on: the scale is the CHANNEL's and applies to the
    # whole field. The straight line is evaluated at every row index, which
    # is an interpolation inside the fitted span and an extrapolation of at
    # most the vertical interval's own length beyond it.
    anchor = spans["blanking"] + float(result["blanking_ire"]) \
        * spans["units_per_ire"]
    per_row = _gain_per_row(slope, fitted[0] - slope * index[0],
                            spans["lines"])
    return {
        "spans": spans,
        "result": result,
        "clip": clip,
        "scale": scale,
        "slope": float(slope),
        "anchor": float(anchor),
        "per_row": per_row,
        "spectrum": spectrum,
    }


def _gain_per_row(slope, intercept, lines):
    """The per-row gain a straight line through the spacing gives, bounded
    to what a chain plausibly does and taken over the specified depth."""
    every = np.arange(int(lines), dtype=np.float64)
    return np.clip(slope * every + intercept,
                   0.1 * sync_depth.SPECIFIED_DEPTH_IRE,
                   3.0 * sync_depth.SPECIFIED_DEPTH_IRE) \
        / sync_depth.SPECIFIED_DEPTH_IRE


def _realise_sync_depth(lines, reading):
    """THE REALISATION HALF: a gain about the measured blanking on a float
    field, in place, and nothing else moved.

    Written as three in-place steps in the same order the expression
    `anchor + (lines - anchor) / per_row` evaluates, so the two paths agree
    to the bit.
    """
    anchor = float(reading["anchor"])
    per_row = np.asarray(reading["per_row"], dtype=np.float64)
    lines -= anchor
    lines /= per_row[:, None]
    lines += anchor


def _sync_depth_report(reading):
    """The reading `correct_sync_depth` has always attached to the field."""
    result = reading["result"]
    scale = reading["scale"]
    clip = reading["clip"]
    return {
        "spacing_ire": float(result["spacing_ire"]),
        "blanking_ire": float(result["blanking_ire"]),
        "tip_ire": float(result["tip_ire"]),
        "scatter_ire": float(result["scatter_ire"]),
        "gain": float(scale["gain"]),
        "as_decibels": float(scale["as_decibels"]),
        "deficit_ire": float(result["deficit_ire"]),
        "slope_ire_per_field": float(reading["slope"])
        * reading["spans"]["lines"],
        "clipped": bool(clip["clipped"]),
        "fraction_at_floor": float(clip["fraction_at_floor"]),
        "lines": int(result["lines"]),
        "spectrum": reading["spectrum"],
    }


def measure_burst_sync_lock(field, uphet):
    """THE BURST LOCKED ABSOLUTELY TO THE SYNC PULSE, WITHIN THE FIELD.

    Node `burst_sync_lock`. Applies nothing.

    Ethan: *"it is directly tied to frequency, i.e. burst is a constant
    phase, and time is known across the period of the burst, it should be
    locked absolutely to the sync pulse with in the field by comparing the
    position relative to the position in the burst."*

    THE TWO REFERENCES FAIL IN OPPOSITE WAYS AND COMPOSE. The sync edge
    happens once a line, so there is no question which one is being looked
    at, but it is a band-limited edge in noise. The burst's phase is fixed by
    specification, so a degree of departure is 0.776 nanoseconds, but phase
    repeats every subcarrier cycle. The sync chooses the cycle; the burst
    places the line within it.

    THE COARSE SIDE IS THE SYNC-ONLY LINE POSITIONS, `linelocs2`, and that
    matters more than it looks. Reading the coarse side off the CORRECTED
    line positions would read back the decoder's own burst refinement and
    return a lock that is true by construction - the null space this
    module's own docstring warns about. `linelocs2` is the hsync refinement
    and nothing else, so the two sides share no arithmetic and the vernier's
    displacement can be compared with the one the decoder actually applied.

    THE BURST IS READ FROM THE CHROMA, NEVER THE LUMA. There is no burst in
    the VHS luma - it is carried down the colour-under path - so the phase
    is taken on the up-converted chroma over the burst window the system
    parameters name, as a COMPLEX phasor whose magnitude is that line's own
    evidence.

    MEASURED BY THIS FUNCTION, 2026-09-06, on the 75 per cent bar SP decode
    over 245 lines a field:

        the sync alone localises to     31.5 ns
        one degree of burst phase is     0.776 ns
        the lock's spread is             0.79 ns
        its drift across the field is    1.24 ns
        `absolute_within_field`          False

    so the vernier is forty times finer than the reference it refines, and
    the claim it exists to test FAILS on this tape by a small margin: the
    systematic drift down the field is 1.6 times the random spread, which
    is 1.6 degrees of subcarrier from the top of the picture to the bottom.
    That is a real reading and not a failure of the measurement - `resolved`
    is True with a 128 ns margin, so the right subcarrier cycle was chosen
    everywhere.

    AND THE DECODER'S OWN REFINEMENT AGREES AT r = -0.958. The sign is the
    convention and not a disagreement: a line the burst says is LATE is one
    the decoder shifts EARLIER, so the two are anticorrelated by
    construction and the magnitude is the agreement. Neither is derived from
    the other - one is `linelocs - linelocs2` and the other is a phase read
    off the chroma - so 0.958 over 245 lines is an independent confirmation
    of the decoder's burst refinement rather than a restatement of it.
    """
    if getattr(field.rf, "color_system", None) != "NTSC":
        _log_once(field, "burst_lock_system",
                  "model stages: burst_sync_lock declined - the 227.5 cycle "
                  "line and the specified burst phase it removes are NTSC's, "
                  "and this decode is %s",
                  getattr(field.rf, "color_system", "unknown"))
        return None
    spans = sync_spans(field)
    if spans is None or uphet is None:
        return None
    coarse = getattr(field, "linelocs2", None)
    applied = getattr(field, "linelocs", None)
    offset = int(getattr(field, "lineoffset", 0)) + 1
    if coarse is None or applied is None:
        _log_once(field, "burst_lock_linelocs",
                  "model stages: burst_sync_lock declined - this field has "
                  "no sync-only line positions to use as the coarse "
                  "reference, and the corrected ones would read back the "
                  "decoder's own burst refinement")
        return None

    rows = spans["rows"]
    # Output row r is resampled from `linelocs[r + lineoffset + 1]`
    # (`Field.computewow_scaled`), so the two are indexed apart and reading
    # them one to one would compare a line with its neighbour.
    located = rows + offset
    rows = rows[located < min(len(coarse), len(applied))]
    located = located[located < min(len(coarse), len(applied))]
    if rows.size < 8:
        return None
    chroma = np.asarray(uphet, dtype=np.float64).reshape(spans["lines"],
                                                         spans["width"])
    burst = _burst_phasor(field, chroma, rows, spans)
    if burst is None:
        return None

    rate = float(field.rf.freq_hz)
    positions = np.asarray(coarse, dtype=np.float64)[located] / rate
    # The sync's OWN error, from its own residual about the uniform grid the
    # line rate defines - not a constant anyone chose. It is what decides
    # whether the vernier picks the right subcarrier cycle, and `resolve`
    # reports the margin rather than assuming it.
    index = np.asarray(rows, dtype=np.float64)
    trend = np.polyfit(index, positions, 1)
    uncertainty = float(np.std(positions - np.polyval(trend, index)))

    # The coarse side is the line's own departure from the uniform grid the
    # line rate defines, which is what the burst can refine; the grid itself
    # carries no information about one line against another.
    residual = positions - np.polyval(trend, index)
    lock = burst_sync_lock.lock_from_lines(
        residual, np.angle(burst["phasor"]), uncertainty, line_indices=index)

    # THE INDEPENDENT COMPARISON. The decoder's own burst refinement moved
    # each line by `linelocs - linelocs2`; the vernier says it should have
    # moved by `displacement_s`. Neither is derived from the other.
    decoder_shift = (np.asarray(applied, dtype=np.float64)[located]
                     - np.asarray(coarse, dtype=np.float64)[located]) / rate
    decoder_shift = decoder_shift - float(np.median(decoder_shift))
    vernier = np.asarray(lock["per_line_s"], dtype=np.float64)
    vernier = vernier - float(np.median(vernier))
    agreement = float(np.corrcoef(decoder_shift, vernier)[0, 1]) \
        if np.std(decoder_shift) > 0 and np.std(vernier) > 0 else float("nan")

    reading = {
        "phasor": burst["phasor"],
        "burst_magnitude": float(np.mean(np.abs(burst["phasor"]))),
        "displacement_mean_s": float(lock["displacement_mean_s"]),
        "spread_s": float(lock["spread_s"]),
        "drift_s_per_line": float(lock["drift_s_per_line"]),
        "drift_across_field_s": float(lock["drift_across_field_s"]),
        "resolved": bool(lock["resolved"]),
        "margin_s": float(lock["margin_s"]),
        "absolute_within_field": bool(lock["absolute_within_field"]),
        "sync_uncertainty_s": uncertainty,
        "decoder_agreement": agreement,
        "seconds_per_degree": burst_sync_lock.seconds_per_degree("NTSC"),
        "lines": int(rows.size),
    }
    if not lock["resolved"]:
        _log_once(field, "burst_lock_unresolved",
                  "model stages: burst_sync_lock declined - the sync's own "
                  "error is %.1f ns against the %.1f ns half period the "
                  "vernier needs, so it would pick the wrong subcarrier "
                  "cycle and be wrong by exactly one period",
                  uncertainty * 1e9,
                  0.5 * burst_sync_lock.ambiguity_interval_s("NTSC") * 1e9)
    else:
        _log_once(field, "burst_lock",
                  "model stages: burst_sync_lock - %d lines, sync alone "
                  "localises to %.1f ns, the burst to %.3f ns a degree; "
                  "lock spread %.2f ns, drift %.2f ns across the field, "
                  "absolute within the field %s; the decoder's own burst "
                  "refinement agrees at r=%+.3f",
                  reading["lines"], uncertainty * 1e9,
                  reading["seconds_per_degree"] * 1e9,
                  reading["spread_s"] * 1e9,
                  reading["drift_across_field_s"] * 1e9,
                  reading["absolute_within_field"], agreement)
    field.burst_sync_lock = reading
    return reading


def _burst_phasor(field, chroma, rows, spans):
    """The burst as a COMPLEX phasor per line, from the reserved window.

    Read at the subcarrier by projection onto the specified carrier rather
    than by a magnitude, because the whole measurement is a phase. The
    window is `SysParams["colorBurstUS"]`, which is the reserved interval
    the standard sets aside for exactly this.
    """
    sys_params = field.rf.SysParams
    start = int(math.ceil(field.usectooutpx(sys_params["colorBurstUS"][0])))
    stop = int(math.floor(field.usectooutpx(sys_params["colorBurstUS"][1])))
    if stop - start < 8 or stop > spans["width"]:
        return None
    columns = np.arange(start, stop, dtype=np.float64)
    carrier = np.exp(-2j * np.pi * burst_sync_lock.SUBCARRIER_HZ
                     / spans["sample_rate_hz"] * columns)
    window = chroma[rows, start:stop]
    # `chroma_to_u16` has not run at this point, so the up-converted chroma
    # is still signed about zero - the same statement `measure` makes above.
    phasor = (window @ carrier) / float(columns.size)
    if not np.isfinite(phasor).all() or not np.abs(phasor).any():
        return None
    return {"phasor": phasor, "window": (start, stop)}


def measure_vertical_interval(field, picture):
    """THE INSERTION TEST SIGNAL, WHOSE SYNTHETIC SIDE IS SPECIFIED.

    Node `vertical_interval`. Applies nothing.

    WHY IT IS THE MOST VALUABLE INPUT THE METHOD CAN BE GIVEN, in the
    module's own words: every other component's synthetic side is a fitted
    curve, and a curve fitted to data describes that data by construction.
    An insertion test signal has no such problem - its bar amplitude, its
    pulse half-amplitude durations, its multiburst frequencies and its
    subcarrier phases are printed in a standard to a tenth of an IRE.

    MEASURED, 2026-09-06, ON THREE DECODES: IT DECLINES ON TWO AND FINDS ONE.

        tape        verdict     best line        r        floor
        75 bars SP  declined    ntc7_composite  +0.157    0.225
        home        declined    ntc7_combin.    +0.092    0.225
        countdown   FOUND       colour_bars     +0.674    0.225

    On the two that decline the reserved lines are simply blank - 0.7 to 1.5
    IRE peak to peak on the bar tape's rows 9 to 19 and 0.7 to 1.1 on home's
    - and there is nothing there to measure a channel with. On countdown
    rows 16 to 18 carry 72 to 104 IRE peak to peak above a quiet row 19, and
    row 18 matches the specified 75 per cent colour bars. Three independent
    things point the same way there: the correlation is three times the null
    floor, the line repeats between fields of the same parity at r = 0.999
    where that tape's picture rows repeat at 0.64 to 0.91, and it sits as an
    ISLAND in an otherwise blank interval rather than at the head of a run
    that continues into the picture. So that tape carries an inserted
    colour-bar line, which is an ordinary thing for a countdown leader to
    have, and the channel it yields carries 13.6 effective dimensions.

    AND THE FIRST VERSION OF THIS STAGE WAS FOOLED BY THE OBVIOUS THING. On
    the bar tape it matched the specified 75 per cent colour bars on row 20
    at r=+0.487 against a floor of 0.225 - a correct match to the wrong
    object, because that source begins its picture one line inside the
    interval the standard reserves and a static test pattern repeats between
    fields exactly as an inserted line would. Neither correlation nor
    repetition separates those two. POSITION does, and that is what
    `_not_spilled_picture` reads.

    THE FLOOR IS DERIVED, NOT CHOSEN. A band-limited line of duration T
    holds 2 B T independent real samples - 318 of them for a 53 microsecond
    active window inside the VHS luma's 3 MHz - so the correlation between
    two unrelated lines has standard deviation 1/sqrt(318) = 0.056, and the
    detection threshold is `chroma_head_switch.DETECTION_SIGMA` of that. One
    significance governs this stage, the chroma detector and the colour
    lock's gate.
    """
    spans = sync_spans(field)
    if spans is None or picture is None:
        return None
    if spans["system"] != "NTSC":
        _log_once(field, "vits_system",
                  "model stages: vertical_interval declined - the lines it "
                  "carries are the 525-line standards' and this decode is %s",
                  getattr(field.rf, "color_system", "unknown"))
        return None
    if spans["width"] != vertical_interval.SAMPLES_PER_LINE:
        _log_once(field, "vits_grid",
                  "model stages: vertical_interval declined - the specified "
                  "lines are rendered on the %d sample four-times-subcarrier "
                  "grid and this decode's line is %d samples",
                  vertical_interval.SAMPLES_PER_LINE, spans["width"])
        return None

    lines = np.asarray(picture, dtype=np.float64).reshape(spans["lines"],
                                                          spans["width"])
    ire = (lines - spans["blanking"]) / spans["units_per_ire"]
    sys_params = field.rf.SysParams
    # THE ACTIVE WINDOW IS IN THE SPECIFICATION'S FRAME, NOT THE ROW'S.
    # `activeVideoUS` is quoted from the row's own origin, which is the sync
    # edge; the specified lines put that edge one front porch into the line
    # (`vertical_interval._LAYOUT_ORIGIN_S`), and `_align_to_specified` moves
    # the row onto that frame. So the window has to move with it, or the two
    # sides are compared a front porch apart - which reads the back porch and
    # the burst against the picture and quietly weakens every correlation.
    active = (min(spans["width"] - 1,
                  spans["front_porch"] + int(math.ceil(field.usectooutpx(
                      sys_params["activeVideoUS"][0])))),
              min(spans["width"],
                  spans["front_porch"] + int(math.floor(field.usectooutpx(
                      sys_params["activeVideoUS"][1])))))
    if active[1] - active[0] < 32:
        return None
    candidates = _reserved_rows(field, spans)
    if candidates.size == 0:
        return None
    candidates, spilled = _not_spilled_picture(ire, candidates, active,
                                               spans)
    if candidates.size == 0:
        _log_once(field, "vits_spill",
                  "model stages: vertical_interval declined - every one of "
                  "the %d reserved lines is part of the run of active lines "
                  "that continues into the picture, so this source starts "
                  "its picture inside the interval the standard reserves and "
                  "there is no reserved line left to read",
                  int(spilled.size))
        field.vertical_interval = {"present": False, "spilled": True}
        return field.vertical_interval

    band = spans["band_limit_hz"] or sync_shape.VHS_LUMA_BAND_HZ
    duration = (active[1] - active[0]) / spans["sample_rate_hz"]
    dimensions = _DIMENSIONS_PER_SECOND * band * duration
    sigma = float(chroma_head_switch.DETECTION_SIGMA)
    floor = float(sigma / math.sqrt(max(dimensions, 1.0)))

    specified = _specified_lines(spans["sample_rate_hz"], spans["width"])
    best = {"correlation": 0.0, "row": -1, "signal": None}
    for row in candidates:
        aligned = _align_to_specified(ire[row], spans)
        if aligned is None:
            continue
        for name, line in specified.items():
            reference = np.asarray(line["composite_ire"],
                                   dtype=np.float64)[active[0]:active[1]]
            got = aligned[active[0]:active[1]]
            if np.std(reference) <= 0 or np.std(got) <= 0:
                continue
            correlation = float(np.corrcoef(reference, got)[0, 1])
            if abs(correlation) > abs(best["correlation"]):
                best = {"correlation": correlation, "row": int(row),
                        "signal": name, "line": line, "measured": aligned}

    # AND IT MUST REPEAT, WHICH IS WHAT AN INSERTION SIGNAL IS. A source
    # inserts a test line on the SAME line of every field; picture content
    # changes from field to field. Without this the stage matched the actual
    # 75 per cent colour bars at r=+0.495 on row 20 of the bars tape - a
    # correct match to the wrong thing, because that source starts its
    # picture a line inside the interval the standard reserves, and fitting
    # a channel there would be fitting it to the picture. The floor is the
    # same one, so no second threshold is introduced.
    repeat = _reserved_repeat(field, ire, best.get("row", -1), active)
    if best["signal"] is not None and repeat is None:
        # NO VERDICT YET, WHICH IS NOT THE SAME AS A FINDING. The repeat
        # cannot be tested until a second field of this parity has been
        # seen, and admitting the first one on its correlation alone is how
        # the countdown decode - a moving picture - accepted its own leader
        # as a specified colour bar line at r=+0.672. A stage that cannot
        # measure its subject yet declines.
        ldd.logger.debug(
            "vertical_interval: row %d matches %s at %+.3f, held until a "
            "second field of this parity can say whether it repeats",
            int(best["row"]), best["signal"], float(best["correlation"]))
        field.vertical_interval = {"present": False, "undecided": True,
                                   "correlation": float(best["correlation"]),
                                   "floor": floor}
        return field.vertical_interval
    if best["signal"] is not None and repeat is not None \
            and abs(repeat) < floor:
        _log_once(field, "vits_moves",
                  "model stages: vertical_interval declined - row %d matches "
                  "the specified %s at r=%+.3f but does not repeat between "
                  "fields of the same parity (r=%+.3f against %.3f), so it "
                  "is picture that has run up into the reserved interval "
                  "rather than an inserted test line",
                  int(best["row"]), best["signal"],
                  float(best["correlation"]), float(repeat), floor)
        field.vertical_interval = {"present": False, "moves": True,
                                   "correlation": float(best["correlation"]),
                                   "repeat": float(repeat), "floor": floor}
        return field.vertical_interval

    if best["signal"] is None or abs(best["correlation"]) < floor:
        _log_once(field, "vits_absent",
                  "model stages: vertical_interval declined - no insertion "
                  "test signal on rows %d to %d. The best of %d specified "
                  "lines against %d reserved lines correlates at %+.3f "
                  "against the %.3f a pair of unrelated band-limited lines "
                  "reaches at %.1f sigma (%.0f independent samples), so "
                  "there is nothing here to measure a channel with",
                  int(candidates[0]), int(candidates[-1]), len(specified),
                  int(candidates.size), float(best["correlation"]), floor,
                  sigma, dimensions)
        field.vertical_interval = {"present": False,
                                   "correlation": float(best["correlation"]),
                                   "floor": floor}
        return field.vertical_interval

    channel = vertical_interval.line_dimension(best["line"], best["measured"])
    _log_once(field, "vits",
              "model stages: vertical_interval - %s found on row %d at "
              "r=%+.3f against a floor of %.3f; the channel against the "
              "standard carries %.2f effective dimensions",
              best["signal"], best["row"], best["correlation"], floor,
              float(channel["effective_length"]))
    field.vertical_interval = {"present": True, "signal": best["signal"],
                               "row": best["row"],
                               "correlation": float(best["correlation"]),
                               "floor": floor, "channel": channel}
    return field.vertical_interval


def _not_spilled_picture(ire, candidates, active, spans):
    """The reserved lines the picture has NOT run into.

    AN INSERTION TEST SIGNAL IS AN ISLAND; SPILLED PICTURE IS A RUN. A test
    line is written into an interval that is otherwise blank, so it has
    blank reserved lines beneath it. A source that simply starts its picture
    early fills every line from some point down and keeps going into the
    picture proper.

    Measured 2026-09-06, and this is why the test exists: on the 75 per cent
    bar tape the stage matched the SPECIFIED colour bars on row 20 at
    r=+0.487 - a correct match to the wrong object, because that source
    begins its picture one line inside the reserved interval and a static
    test pattern repeats between fields exactly as an inserted line would.
    Neither correlation nor repetition can separate those two; position can.

    The activity test is a robust outlier test on the reserved population
    itself - a line is active when its peak-to-peak stands more than the
    detector's own significance in median absolute deviations above the
    median of the reserved lines - so no level is chosen. On that tape the
    blank reserved lines span 0.7 to 1.5 IRE peak to peak and the spilled
    one spans 95.
    """
    span = np.ptp(ire[:, active[0]:active[1]], axis=1)
    reserved = span[candidates]
    middle = float(np.median(reserved))
    deviation = float(np.median(np.abs(reserved - middle)))
    # 1.4826 makes the median absolute deviation an estimate of sigma for a
    # normal; where the reserved lines are identically blank it is zero, and
    # the output's own least significant bit is then the smallest difference
    # that can mean anything.
    scale = max(1.4826 * deviation, 1.0 / spans["units_per_ire"])
    threshold = middle + float(chroma_head_switch.DETECTION_SIGMA) * scale
    last = int(candidates[-1])
    following = last + 1
    spilled = []
    if following < ire.shape[0] and span[following] > threshold:
        row = last
        while row >= int(candidates[0]) and span[row] > threshold:
            spilled.append(row)
            row -= 1
    spilled = np.asarray(sorted(spilled), dtype=np.intp)
    keep = np.asarray([r for r in candidates if r not in set(spilled.tolist())],
                      dtype=np.intp)
    return keep, spilled


def _reserved_repeat(field, ire, row, active):
    """How well a reserved line repeats against the previous field of its own
    parity. None until such a field has been seen - the first field of each
    parity cannot answer the question and is admitted on the correlation
    alone rather than refused for want of evidence."""
    if row < 0:
        return None
    store = field.rf.__dict__.setdefault("_vits_rows", {})
    parity = bool(field.isFirstField)
    current = np.asarray(ire[row][active[0]:active[1]], dtype=np.float64)
    previous = store.get((parity, int(row)))
    if not field.__dict__.get("_vits_row_stored"):
        field.__dict__["_vits_row_stored"] = True
        store[(parity, int(row))] = current.copy()
    if previous is None or previous.size != current.size:
        return None
    if np.std(previous) <= 0 or np.std(current) <= 0:
        return None
    return float(np.corrcoef(previous, current)[0, 1])


def _reserved_rows(field, spans):
    """The reserved lines between the end of vertical sync and the first
    picture line, from the specification's own numbering - the vertical
    interval's length for the first, `sync_geometry.VBI_LAST_LINE` for the
    last. Never waveform-derived."""
    from vhsdecode.models import sync_geometry

    first = int(np.ceil(float(sync_geometry.interval_geometry(
        spans["system"])["total_lines"])))
    last = int(sync_geometry.VBI_LAST_LINE)
    last = min(last, spans["lines"] - 1)
    if last <= first:
        return np.empty(0, dtype=np.intp)
    return np.arange(first, last, dtype=np.intp)


def _specified_lines(sample_rate_hz, samples):
    """The standards' own lines on the decode's grid, built once a process.

    Rendering three lines costs a few milliseconds and they do not depend on
    the field, so they are memoised on the module rather than rebuilt for
    every reserved line of every field.
    """
    key = (float(sample_rate_hz), int(samples))
    cached = _SPECIFIED_LINES.get(key)
    if cached is not None:
        return cached
    built = {
        "ntc7_composite": vertical_interval.ntc7_composite(samples,
                                                           sample_rate_hz),
        "ntc7_combination": vertical_interval.ntc7_combination(samples,
                                                              sample_rate_hz),
        "colour_bars": vertical_interval.colour_bars(samples, sample_rate_hz),
    }
    _SPECIFIED_LINES[key] = built
    return built


_SPECIFIED_LINES = {}


def _align_to_specified(row, spans):
    """A decoded line moved onto the specification's own time axis.

    The specified lines put the sync leading edge one front porch into the
    line (`vertical_interval._LAYOUT_ORIGIN_S`), while a time base corrected
    row begins at that edge, so the row is shifted by the difference - which
    is measured on the row's own sync edge rather than assumed, because the
    decoder's own alignment leaves a fraction of a microsecond in it.
    """
    values = np.asarray(row, dtype=np.float64)
    tip = values[spans["tip"][0]:spans["tip"][1]]
    porch = values[spans["porch"][0]:spans["porch"][1]]
    if tip.size < 2 or porch.size < 2:
        return None
    half = 0.5 * (float(np.median(tip)) + float(np.median(porch)))
    crossed = np.nonzero(values[:int(spans["sync_samples"])] <= half)[0]
    if not crossed.size:
        return None
    edge = int(crossed[0])
    target = int(round(spans["front_porch"]))
    return np.roll(values, target - edge)


def measure_tape_speed(field):
    """THE TAPE'S SPEED AGAINST ITS EXPECTED, AS A COMPONENT.

    Node `tape_speed`. Applies nothing.

    Ethan: *"Let's use the tape speed difference from its expected as
    another component, this is what ties in the tapes mechanical model we
    built earlier."*

    READ BEFORE THE TIME BASE CORRECTION, WHICH IS THE ONLY PLACE IT
    SURVIVES. The corrector locks sync to sync, so the line period it
    outputs is the nominal one by construction and a speed error read there
    is exactly zero. `linelocs` holds the positions the corrector had to
    resample FROM, on the demodulator's own sample grid, so their successive
    differences are the line period as the tape delivered it.

    THREE AXES, AND THE MODULE'S CENTRAL FINDING IS ABOUT WHICH ONE CAN
    WITNESS IT:

      TIME        the line period against the format's own, which is the
                  measurement. Nothing here is fitted.
      FREQUENCY   `loss_departure`, what that speed error does to the log
                  response - because speed enters the magnetics only through
                  the wavelength, so an error of eps rescales the frequency
                  axis every loss lives on. Derived, and REPORTED as the
                  size of the effect rather than used to identify it: over
                  0.5 to 6 MHz a speed error is degenerate with a common
                  scaling of every magnetic length at coherence 1.000000, an
                  identity rather than an approximation.
      AMPLITUDE   the same departure in nepers, which is what that frequency
                  axis actually moves, and the drum-rate line in the envelope
                  where one has been measured - an amplitude modulation, so
                  the time base cannot null it.

    AND IT SAYS WHICH TRANSPORT ELEMENT COULD HAVE DONE IT. `attribute`
    separates a capstan error, which moves the track geometry and barely
    moves a wavelength, from a drum error, which does the opposite.

    MEASURED, 2026-09-06, on three decodes, 245 line periods a field read
    off `linelocs2`:

        tape          epsilon    spread     wavelength   response
        75 bars SP   +0.00015   0.00010     x 1.00015    0.002 dB
        home         -0.00006   0.00021     x 0.99994    0.001 dB
        countdown    +0.00276   0.00247     x 1.00276    0.028 dB

    The two studio-sourced tapes sit at one and a half parts in ten thousand
    or less, and countdown is eighteen times either - 0.28 per cent fast,
    with a per-line spread almost as large as the mean, which is the same
    tape whose sync tips `sync_depth` refuses. THE FREQUENCY-AXIS COLUMN IS
    WHY THE TIME AXIS HAD TO MEASURE THIS: even countdown's error moves the
    magnetic response by twenty-eight thousandths of a decibel over five and
    a half megahertz, which is far below anything the frequency axis could
    separate from a common scaling of every magnetic length - and that
    separation is exactly zero, the two being the same operation written
    twice.
    """
    spans = sync_spans(field)
    if spans is None:
        return None
    # SYNC ONLY. `linelocs2` is the hsync refinement and nothing else; the
    # burst refinement that follows it nudges each line by a fraction of a
    # sample, which is a perturbation of exactly the quantity being read.
    locations = getattr(field, "linelocs2", None)
    if locations is None:
        locations = getattr(field, "linelocs", None)
    if locations is None or len(locations) < 16:
        _log_once(field, "tape_speed_declined",
                  "model stages: tape_speed declined - this field has no "
                  "pre-correction line positions, and the corrected ones "
                  "are nulled by construction")
        return None

    rate = float(field.rf.freq_hz)
    offset = int(getattr(field, "lineoffset", 0)) + 1
    located = spans["rows"] + offset
    located = located[located + 1 < len(locations)]
    if located.size < 8:
        return None
    values = np.asarray(locations, dtype=np.float64)
    periods = (values[located + 1] - values[located]) / rate
    expected = spans["line_samples"] / spans["sample_rate_hz"]
    speed = tape_speed.from_line_period(periods, expected)
    if not np.isfinite(speed["epsilon"]):
        return None

    epsilon = float(speed["epsilon"])
    where = tape_speed.attribute(epsilon)
    scaling = tape_speed.wavelength_scaling(epsilon)
    # The frequency axis: what this speed error does to the log response
    # across the band the luma occupies.
    frequencies = np.linspace(0.5e6, 6.0e6, 23)
    departure = tape_speed.loss_departure(frequencies, epsilon)
    reading = {
        "epsilon": epsilon,
        "epsilon_rms": float(speed["epsilon_rms"]),
        "spread": float(speed["spread"]),
        "lines": int(speed["lines"]),
        "wavelength_scaling": float(scaling),
        "frequency_hz": frequencies,
        "loss_departure_nepers": departure,
        "peak_departure_db": float(np.max(np.abs(departure)) * 8.685889638),
        "attribution": where,
        "expected_period_s": expected,
    }
    _log_once(field, "tape_speed",
              "model stages: tape_speed - line period over %d lines gives "
              "eps %+.5f (spread %.5f), so every wavelength scales by "
              "%.5f and the log response moves by up to %.3f dB over 0.5 to "
              "6 MHz - which the frequency axis cannot witness, being "
              "degenerate with a common scaling of every magnetic length",
              reading["lines"], epsilon, reading["spread"], scaling,
              reading["peak_departure_db"])
    field.tape_speed = reading
    return reading


# --------------------------------------------------------------------------
# iq_imbalance - the chroma's quadrature imbalance, from the burst itself
# --------------------------------------------------------------------------

def measure_iq_imbalance(field, uphet, amount):
    """THE QUADRATURE IMBALANCE, measured on the colour-under burst.

    Ethan, 2026-09-06, looking at a composite vectorscope: *"The vectorscope
    looks like an IQ imbalance on the chroma channel"*, and then the method:
    *"I think we can see the IQ imbalance by using the entire complex shape
    of the burst itself in all dimensions."*

    An imbalanced quadrature pair maps a phasor to `alpha b + beta conj(b)`,
    and the conjugate term counter-rotates: a circle becomes an ellipse and
    the whole colour coordinate system shears. `|beta/alpha|` is the image
    rejection and it is invisible to a magnitude, because a phasor and its
    conjugate have the same modulus - which is why every quantity handed
    across this stage's boundaries is complex.

    THE COLOUR-UNDER BURST AND NOT THE UP-CONVERTED ONE, which the module
    proves rather than asserts. At the composite burst's 180 degrees a line
    `b` and `conj(b)` both change sign every line, so they are the same
    sequence - coherence 1.0000, condition 1.3e14 - and no fit can separate
    them. At the colour-under's 90 degrees a line they counter-rotate,
    coherence 0.0000 and condition 1.0. The record-side rotation of SMPTE
    32M clause 3.9.2.1.5 is the only reason this is measurable at all.

    THE RUNS ARE KEPT INSIDE ONE FIELD, and that is this wiring's own
    contribution to the same trap. The standard's rotation ADVANCES on one
    track and RETARDS on the other, so a run spanning a head switch carries
    two rotations and fits neither; `from_burst_series` already takes each
    run's rotation from that run, and keeping every run inside a field is
    what makes the assumption behind that true here. Each field contributes
    exactly two runs of half its picture lines, trimmed so the runs align.

    IT REPORTS AND IT DOES NOT CORRECT, and the reason is a measured margin
    rather than caution. The correction is one complex division - the
    inverse of `alpha b + beta conj(b)` once both are known - and the
    measurement supplies both. What it does not yet supply is confidence:
    `iq_imbalance` records the image standing at only 1.77 to 1.97 times
    its own control on the bars, and only the chroma-noise record tap
    reaching 5.77. The arc's correction gain law says an estimate is worth
    applying at half its believed optimum when the model error is known;
    here the model error is the same size as the estimate, and a correction
    at that margin would be as likely to add an image as remove one. What
    would change the ruling is a margin over the control on the order of
    the ten the leakage statistic reaches, which is a matter of pooling
    more runs rather than of a better estimator.
    """
    if not amount:
        return None
    state = _measurement_state(field, "iq_imbalance")
    burst = _colour_under_burst(field)
    if burst is None:
        _log_once(field, "iq_imbalance_declined",
                  "model stages: iq_imbalance declined, the colour-under "
                  "burst is not available - this stage must run before "
                  "luma_beat, which releases the down-converted chroma")
        return None

    phasors = np.asarray(burst["phasors"], dtype=np.complex128)
    # TWO RUNS A FIELD, BOTH INSIDE IT, so no run spans the head switch and
    # each one carries a single track's rotation. The run length is half the
    # field's own picture lines rather than a written number, so it follows
    # the geometry into any system.
    run = phasors.size // 2
    if run < 8:
        return None
    series = phasors[:2 * run]
    store = state.setdefault("series", [])
    if not field.__dict__.get("_iq_imbalance_folded"):
        field.__dict__["_iq_imbalance_folded"] = True
        store.append(series)
        state["fields"] += 1

    try:
        found = iq_imbalance.from_burst_series(series, run_lines=run)
    except ValueError:
        return None
    if not found.get("usable"):
        _log_once(field, "iq_imbalance_unusable",
                  "model stages: iq_imbalance declined - %s",
                  found.get("why", "the burst carried no measurable level"))
        return None

    quarter = bool(found["rotation_is_a_quarter_turn"])
    ldd.logger.debug(
        "iq_imbalance: %d runs, rotation %+.2f deg/line (%s), image ratio "
        "%.5f (%.2f dB), control %.5f, over control %.2fx, gain imbalance "
        "%.2f%% or %.2f deg of quadrature error",
        found["runs"], found["rotation_deg_per_line"],
        "a quarter turn" if quarter else "NOT a quarter turn, refused",
        found["image_ratio"], found["image_rejection_db"],
        found["control_ratio"], found["over_control"],
        100.0 * found["gain_imbalance_fraction"],
        found["quadrature_error_deg"])

    if not quarter:
        # A RUN WHOSE ROTATION IS NOT A QUARTER TURN HAS A FAILED DETECTOR,
        # and its row must not be used - the module records exactly that
        # case, a chroma-noise playback capture returning -18.75 degrees a
        # line. Refused rather than averaged in, and said in the log.
        _log_once(field, "iq_imbalance_rotation",
                  "model stages: iq_imbalance declined, the runs carry "
                  "%+.2f degrees a line where SMPTE 32M clause 3.9.2.1.5 "
                  "puts a quarter turn - the run detector has failed on "
                  "this material and its reading is not usable",
                  found["rotation_deg_per_line"])
        return None

    if len(store) >= _COLOUR_FRAME_FIELDS and not state["reported"]:
        state["reported"] = True
        # POOLED OVER EVERY FIELD'S RUNS AT ONCE, because the statistic is a
        # median over runs and a median of medians is not one. The fields
        # are concatenated and the same run length is used, which is exactly
        # aligned by construction: every field contributed two runs of that
        # length and nothing else.
        pooled = iq_imbalance.from_burst_series(np.concatenate(store),
                                                run_lines=run)
        state["pooled"] = pooled
        if pooled.get("usable"):
            _log_once(
                field, "iq_imbalance_pooled",
                "model stages: iq_imbalance over %d runs of %d lines - "
                "rotation %+.2f deg/line, image ratio %.5f (%.2f dB), "
                "%.2f times its control; that is a gain imbalance of %.2f "
                "per cent or a quadrature error of %.2f degrees, and it is "
                "REPORTED and not applied - the margin over the control is "
                "%s",
                pooled["runs"], run, pooled["rotation_deg_per_line"],
                pooled["image_ratio"], pooled["image_rejection_db"],
                pooled["over_control"],
                100.0 * pooled["gain_imbalance_fraction"],
                pooled["quadrature_error_deg"],
                "too slim to correct on" if pooled["over_control"] < 3.0
                else "wide enough that a correction is now a judgement "
                     "for Ethan rather than a refusal")

    field.iq_imbalance = found
    return found


# ==========================================================================
# THE LEVELS, GAIN AND SOURCE-SIDE STAGES
#
# Ethan, 2026-09-06: *"Every stage that we have designed must be called and
# used. Exhaustively go through the stages and make sure they are being
# used. All of them need to be used and all of them need to model all
# dimensions."*
#
# Seven more modules were designed and could not be reached from a decode.
# They are wired below in the shape the sync group above already has - a
# node in `pipeline/stages.toml`, a default of ON carried by the
# declaration (`declared_default = true`), `--stages -<name>` as the only
# control, and no flag of their own.
#
# THE SUBJECT IS THE LEVELS, AND THE LEVELS ARE IN THE RESERVED INTERVALS.
# Blanking is read on the back porch and the synchronizing level on the
# pulse's flat interior, on the specification's own line population; the
# chrominance readings are taken on the colour burst. Nothing in this group
# derives a correction from active picture. Where the active area appears at
# all it is as a REPORT - `standard_levels.compliance` says whether the
# corrected signal sits inside the standard's excursion limits, which is a
# statement about the picture and not an input to the correction.
#
# WHAT EACH ONE IS:
#
#   `standard_levels`      CORRECTS. The back porch is the specified zero,
#                          and the level read there is not the porch's mean:
#                          the colour-under carrier and the subcarrier leak
#                          onto it at frequencies the specification fixes,
#                          so they are projected out and the BASELINE that
#                          remains is driven to 0 IRE. The gain is
#                          `sync_depth`'s and is not touched here.
#
#   `level_from_frequency` MEASURES. The same levels read as CARRIER
#                          FREQUENCIES on the radio-frequency input, before
#                          the demodulator and before every stage the
#                          picture-domain reading inherits. The two readings
#                          share nothing, so their agreement bounds both.
#
#   `source_agc`           MEASURES, chain position 7. The gain control
#                          ahead of the recorder, derived by asserting that
#                          both levels are constants and attributing what
#                          MOVES to the only element whose purpose is to
#                          move.
#
#   `vcr_agc`              MEASURES, chain position 8. The recorder's own
#                          loop, bounded from the straightness of the same
#                          two level series across the field.
#
#   `composite_channel`    MEASURES. The channel the specifications define,
#                          confronted with what this decode carries - the
#                          luma's specified flatness, the chrominance chain
#                          SMPTE 32M states with numbers, and the burst
#                          doubler that is a factor of two in every level
#                          taken from a burst.
#
#   `picture_stage`        MEASURES. The references and alignments that act
#                          after the radio-frequency matrix: the burst's
#                          lock to the horizontal reference, what the
#                          decode's own band can and cannot separate among
#                          them, and how precisely the sync edge would have
#                          to be located to make two of them two dimensions.
#
#   `multipath`            MEASURES. A tapped-delay description of the sync
#                          channel's departure, judged on fields it never
#                          saw. THE MECHANISM CLAIM IS WITHDRAWN and stays
#                          withdrawn: this reports what the taps do, not
#                          what they are.
#
# THE UNITS TRAP THAT GOVERNS THIS WHOLE GROUP. `--level_adjust` defaults to
# a tenth and SPREADS the black and white levels in the written JSON while
# leaving blanking alone, so an offline reader taking
# `(white16bIre - black16bIre) / 100` as the units per IRE gets 400.768
# where the decode used 358.4 and every absolute IRE it reports is out by
# 11.82 per cent. Nothing in this group reads the written file: every
# reading below takes its scale from `sync_spans`, which evaluates
# `hz_to_output` at 0 and 100 IRE on the running decode.
# `precursor.output_scale` is the same inversion for a reader that has only
# the file, and the two agree by construction.
# ==========================================================================

from vhsdecode.models import composite_channel, level_from_frequency
from vhsdecode.models import multipath, picture_stage
from vhsdecode.models import source_agc, vcr_agc


def _level_series(field, picture):
    """THE TWO SPECIFIED LEVELS, PER LINE, WITH THEIR OWN TRUSTWORTHINESS.

    Blanking on the back porch and the synchronizing level on the pulse's
    flat interior, both in IRE on the decode's own scale, for the lines the
    specification's population names. One measurement shared by the three
    stages that read levels, taken once and cached on the FIELD, because a
    stale series would be attributed to the wrong picture.

    THE TRUSTWORTHINESS TRAVELS WITH THE SERIES, and that is the point of
    returning it here rather than leaving each caller to decide. `sync_depth`
    already established what makes this population unreadable and measured
    both failures on real material: a censored fit whose mean falls BENEATH
    the container floor is extrapolating a level nothing ever showed - on the
    countdown decode 78.9 per cent of the pulse's interior sits on the floor
    and the fit returned a spacing of 107.7 IRE - and lines that disagree
    about a specified constant by more than the standard's own tolerance are
    not all measuring it. Three stages reading the same series must reach the
    same verdict about it, so the verdict is computed once.
    """
    cached = field.__dict__.get("_level_series")
    if cached is not None:
        return cached or None
    field.__dict__["_level_series"] = False

    # `sync_spans` and not `sync_pulses`, deliberately. Only the geometry and
    # the scale are wanted here, and `sync_pulses` caches a POOLED PULSE
    # ARRAY taken the first time any stage asked for one - which on a decode
    # running `precursor` or `sync_depth` is the picture as it stood BEFORE
    # those corrections. The spans are geometry and a scale and do not move;
    # the levels are read from the picture handed in, so they are always this
    # stage's own view of it.
    spans = sync_spans(field)
    if spans is None or picture is None:
        return None
    if len(picture) != spans["lines"] * spans["width"]:
        return None
    rows = spans["rows"]
    lines = np.asarray(picture, dtype=np.float64).reshape(spans["lines"],
                                                          spans["width"])
    white = spans["blanking"] + 100.0 * spans["units_per_ire"]
    tip_columns = range(*spans["tip"])
    porch_columns = range(*spans["porch"])
    try:
        result = sync_depth.spacing(lines[rows], spans["blanking"], white,
                                    tip_columns, porch_columns)
        clip = sync_depth.detect_clip(lines[rows][:, list(tip_columns)],
                                      spans["blanking"], white)
    except (ValueError, IndexError, np.linalg.LinAlgError):
        return None

    porch_block = ((lines[rows][:, spans["porch"][0]:spans["porch"][1]]
                    - spans["blanking"]) / spans["units_per_ire"])
    spacing_ire = np.asarray(result["per_line"], dtype=np.float64)
    blanking_ire = porch_block.mean(axis=1)

    why_not = None
    censored = result.get("censored")
    floor = result.get("censor_floor")
    if censored and censored.get("fitted") and floor is not None:
        below = float(censored["mean"])
        if np.isfinite(below) and below < float(floor):
            why_not = (
                "%.1f%% of the pulse's flat interior sits on the container "
                "floor and the censored fit puts the tip %.0f units beneath "
                "it, so more than half the level is gone and what it reports "
                "was never observed"
                % (100.0 * float(censored.get("censored_fraction", 0.0)),
                   float(floor) - below))
    if why_not is None:
        finite = spacing_ire[np.isfinite(spacing_ire)]
        scatter = (1.4826 * float(np.median(np.abs(finite - np.median(finite))))
                   if finite.size else float("inf"))
        if not scatter < standard_levels.IRE_TOLERANCE:
            why_not = ("the %d lines disagree about the spacing by %.2f IRE "
                       "against the %.1f the standard allows on this level, "
                       "so they are not all measuring one constant"
                       % (int(finite.size), scatter,
                          standard_levels.IRE_TOLERANCE))

    series = {
        "spans": spans,
        "rows": rows,
        "porch": porch_block,
        "blanking_ire": blanking_ire,
        "spacing_ire": spacing_ire,
        "result": result,
        "clip": clip,
        "trustworthy": why_not is None,
        "why_not": why_not,
        "line_rate_hz": 1e6 / float(field.rf.SysParams["line_period"]),
    }
    field.__dict__["_level_series"] = series
    return series


def correct_standard_levels(field, amount):
    """THE BACK PORCH DRIVEN TO THE SPECIFIED ZERO, WITH THE COLOUR TAKEN OUT
    OF IT FIRST.

    Node `standard_levels`. Ethan: *"after RF correction, I do want to
    normalize the signal so it sits within the video standard"*, and *"the
    important thing is that the back porch is centered at 0 ire ... The
    scaling of the video is done in our scaling step in the picture stage."*

    THE DIVISION OF LABOUR IS EXPLICIT AND IS NOT NEW HERE. `sync_depth` owns
    the GAIN - the blanking-to-tip spacing over the specified forty - and
    applies it about the measured blanking, which leaves the zero exactly
    where it found it; its own docstring says a shift of the zero *"is a
    different defect and belongs to `standard_levels`"*. This stage is that
    defect and touches nothing else.

    WHAT IT ADDS OVER `--ire0_adjust backporch`, WHICH ALREADY MOVES THE
    ZERO. That mode takes ONE window level for the whole field and calls it
    ire0. Two things are wrong with a window level as a zero, and this stage
    repairs both:

      THE POINTED ONE. The colour-under carrier sits at 40 f_H and the
      subcarrier at 455/2 f_H, both EXACT multiples of the line rate, so
      both leak onto the back porch at frequencies known before a sample is
      read. Averaging the window returns the baseline plus whatever that
      leakage happens to average to, which is zero only if the window spans a
      whole number of cycles of every leaking carrier - it does not. The
      carriers are therefore projected out, not averaged down, and the
      BASELINE that remains is the level.

      THE TIME AXIS. One number a field cannot follow a zero that moves down
      the field, and `sync_depth.field_slope` measures blanking drifting by
      0.34 to 0.60 IRE across a field on all three of this arc's tapes -
      seven to thirteen times the ten-bit output floor of 0.0461 IRE. The
      correction here is a per-line trend, taken with
      `standard_levels.dominant_trend` so that a contiguous run of bad fields
      cannot drag it, which is Ethan's *"we can base the levels on the
      dominant trend that should be constant given our other corrections"*.

    THREE AXES, AND EACH IS MEASURED:

      AMPLITUDE   the projected baseline in IRE, driven to the specified
                  zero. `porch_dimensions` returns it beside the raw window
                  mean, so the size of the leakage error is reported rather
                  than assumed.
      FREQUENCY   the two leaking carriers, each as ONE COMPLEX amplitude at
                  a specified frequency. Complex because an amplitude
                  without its phase cannot say whether two lines' leakage
                  adds or cancels.
      TIME        the baseline's own trend down the field, rejected on the
                  median absolute deviation and applied per line.

    THE EXCURSION IS A REPORT AND NOT AN INPUT. `standard_levels.compliance`
    reads the whole corrected field to say whether it still sits inside the
    140 IRE the standard allows without chrominance (SMPTE 170M clause 12.1),
    which is a statement ABOUT the picture; the correction itself is derived
    from the back porch alone.

    MEASURED BY THIS STAGE, 2026-09-06, `--stages +standard_levels`:

        decode                     baseline   projection   trend    lines
                                   IRE        corrects     swing    rejected
        75 bars SP, --ire0_adjust   -0.007      +0.004      0.487     0.4 %
        multiburst y-only SP        -0.041      +0.003      0.907     0.0 %
        home, --ire0_adjust         -0.097      -0.326      3.200     5.3 %
        75 bars SP, no ire0_adjust  +4.417      +0.001      0.435     0.4 %
        countdown                      -           -          -      DECLINED

    THE LAST TWO ROWS ARE THE WHOLE OF THE STAGE'S CASE. With the decoder's
    own back-porch anchoring on, the field-mean baseline is already at zero by
    construction and what is left is the trend and the projection; with it
    OFF the same field's porch stands 4.417 IRE high and this stage is the
    only thing that moves it. And on countdown it DECLINES - 79.0 per cent of
    the pulse's flat interior sits on the container floor - which is visible
    in the output as well as in the log: that decode's luma is byte-identical
    with the stage on.

    THE GAUGE IT IS NOT FITTED WITH IS THE PICTURE. Regressing the eight
    plateaus of the 75 per cent bar pattern on SMPTE 170M's specified
    luminances, on the decode with no `--ire0_adjust`:

        intercept, before          +3.253 IRE      gain 0.98830
        intercept, after           -0.983 IRE      gain 0.98836
        the decoder's own anchor   -0.927 IRE      gain 0.98813

    So 4.24 IRE of zero error is removed from the ACTIVE PICTURE by a
    correction that never reads it, landing 0.056 IRE from where the
    decoder's own back-porch anchoring puts the same picture - and THE GAIN
    IS UNTOUCHED to six parts in a hundred thousand, which is what an offset
    correction must do and is the thing a mistake here would break.

    THE SECOND GAUGE IS THE FRONT PORCH, a reserved interval on the other
    side of the pulse that this stage never reads. Its line-to-line scatter
    about its own trend falls from 0.236 to 0.168 IRE on the bars tape and
    from 0.577 to 0.317 on home - 29 and 45 per cent - so the per-line
    correction is flattening a zero that really is moving. ITS FIELD-LONG
    SLOPE DOES NOT AGREE, going -0.077 to -0.302 IRE a field on bars and
    -0.063 to +0.637 on home, which says the back porch's own trend is not
    entirely shared with the front porch. The front porch is the interval
    this arc has already measured to be content-dependent at 41 to 66 sigma,
    so it is the weaker of the two witnesses - but the disagreement is
    reported rather than left out.
    """
    if not amount:
        return None
    picture = getattr(field, "dspicture", None)
    if picture is None:
        return None
    series = _level_series(field, picture)
    if series is None:
        _log_once(field, "standard_levels_declined",
                  "model stages: standard_levels declined - no sync "
                  "population on an IRE scale in this field")
        return None
    if not series["trustworthy"]:
        _log_once(field, "standard_levels_untrusted",
                  "model stages: standard_levels declined - %s; the back "
                  "porch cannot be anchored on a field whose reserved "
                  "intervals are not readable", series["why_not"])
        return None
    reading = _read_standard_levels(field, series)
    if reading is None:
        return None

    spans = series["spans"]
    corrected = np.array(picture, dtype=np.float64).reshape(spans["lines"],
                                                            spans["width"])
    _realise_standard_levels(corrected, reading)

    # THE GAUGE THIS STAGE IS NOT FITTED WITH. Two anchors determine the
    # affine map and therefore prove nothing about themselves; the excursion
    # is over-determined once they are fixed, so it is what the compliance
    # check is for. Read on the CORRECTED field, whose blanking now sits at
    # the container value the scale calls zero and whose synchronizing level
    # sits one measured spacing beneath it.
    spacing = float(np.nanmean(series["spacing_ire"]))
    verdict = standard_levels.compliance(
        corrected.reshape(-1),
        spans["blanking"] - spacing * spans["units_per_ire"],
        spans["blanking"])
    _write_back(picture, corrected)
    # THE CACHED SERIES IS NOW STALE, AND A STALE ONE IS THE FAILURE THIS
    # WHOLE GROUP IS ORDERED TO AVOID. Every level below has just moved, so a
    # stage added after this point that asked `_level_series` for the field's
    # levels would be handed the picture as it stood BEFORE the correction and
    # would find a departure this stage has already removed. Dropped rather
    # than left for the next reader to discover.
    field.__dict__.pop("_level_series", None)

    report = _standard_levels_report(series, reading, verdict)
    # WHAT IS LEFT FOR THIS STAGE DEPENDS ON WHETHER THE DECODER ALREADY
    # MOVED THE ZERO, and saying which is the difference between a report
    # that means something and a number that looks like nothing happened.
    # `--ire0_adjust backporch` anchors ire0 on ONE window level a field, so
    # on such a decode the field-mean baseline is already near zero by
    # construction and the whole of this stage's amplitude contribution is
    # the projection's own correction to that level - while the TIME axis,
    # the trend down the field, is untouched by it and is the larger term.
    anchored = "backporch" in (getattr(field.rf.options, "ire0_adjust", ())
                               or ())
    _log_once(field, "standard_levels_anchor",
              "model stages: standard_levels - %s, so %s",
              "--ire0_adjust backporch has already set this decode's zero "
              "from one window level a field" if anchored else
              "nothing upstream has moved this decode's zero",
              "the level below is what the PROJECTION corrects in that "
              "anchor and the trend is what a per-field number could not "
              "follow" if anchored else
              "the level below is the whole departure from the specified "
              "zero")
    _log_once(field, "standard_levels",
              "model stages: standard_levels - the back porch baseline reads "
              "%+.3f IRE against the specified zero, %+.3f IRE away from the "
              "window's own mean once the %s carriers are projected out "
              "(%s withheld, the window holds %.2f us); removed as a trend "
              "swinging %.3f IRE down the field over %d lines, %.1f%% "
              "rejected; the corrected field spans %.1f IRE, %s the %.0f the "
              "standard allows without chrominance",
              report["baseline_ire"],
              report["baseline_ire"] - report["raw_mean_ire"],
              ", ".join("%g f_H" % m for m in reading["offered"]) or "no",
              ", ".join("%g f_H" % m for m in reading["withheld"]) or "none",
              1e6 * reading["window_s"], report["trend_swing_ire"],
              report["lines"], 100.0 * report["rejected_fraction"],
              report["excursion_ire"],
              "inside" if report["within_luma_limit"] else "OUTSIDE",
              140.0)
    if report["leakage_ire"]:
        _log_once(field, "standard_levels_leakage",
                  "model stages: standard_levels frequency axis - the porch "
                  "carries %s, each one complex amplitude at a specified "
                  "multiple of the line rate",
                  ", ".join("%s at %.4f IRE, %+.1f deg"
                            % (name, report["leakage_ire"][name],
                               report["leakage_phase_deg"][name])
                            for name in sorted(report["leakage_ire"])))
    field.standard_levels = report
    return report


def _read_standard_levels(field, series):
    """THE MEASUREMENT HALF OF `correct_standard_levels`: the porch's
    projected baseline per line, the carriers taken out of it, and the
    dominant trend that becomes the per-row zero - measured, not applied.

    Factored out so the picture transform can take the same reading on the
    field as it arrived; `correct_standard_levels` composes this with
    `_realise_standard_levels` and behaves exactly as it did. Refusals are
    logged once and return None.
    """
    spans = series["spans"]
    rate = spans["sample_rate_hz"]
    line_rate = series["line_rate_hz"]
    porch = series["porch"]

    # WHICH CARRIERS THE WINDOW CAN ACTUALLY SEPARATE, decided by the window
    # and not by a preference. A carrier the porch does not complete one
    # cycle of has no period inside the window to be projected on, and
    # offering it anyway makes the design matrix nearly singular and lets it
    # absorb the baseline it exists to protect. So each specified multiple is
    # admitted only where the window is at least its own period long, and the
    # ones that are refused are named.
    window_s = porch.shape[1] / rate
    offered, withheld = [], []
    for multiple in (standard_levels.COLOUR_UNDER_MULTIPLE,
                     standard_levels.SUBCARRIER_MULTIPLE):
        if window_s * multiple * line_rate >= 1.0:
            offered.append(multiple)
        else:
            withheld.append(multiple)

    baselines = np.empty(porch.shape[0], dtype=np.float64)
    leakage = {}
    conditioning = []
    for index in range(porch.shape[0]):
        decomposed = standard_levels.porch_dimensions(
            porch[index], rate, line_rate,
            multiples=offered if offered else ())
        baselines[index] = float(decomposed["baseline"])
        conditioning.append(float(decomposed.get("conditioning", 1.0)))
        for name, value in (decomposed.get("leakage") or {}).items():
            leakage.setdefault(name, []).append(value)

    if not np.isfinite(baselines).all():
        _log_once(field, "standard_levels_refused",
                  "model stages: standard_levels declined - the porch "
                  "projection did not resolve on every line")
        return None

    # A ZERO ERROR WIDER THAN THE SIGNAL'S OWN SCALE IS NOT A ZERO ERROR. The
    # bound is the one absolute distance the standard fixes, blanking to the
    # synchronizing level; a baseline further from zero than that is a
    # measurement that has lost the porch rather than a level to be removed.
    trend = standard_levels.dominant_trend(baselines)
    level = float(trend["level"])
    if not abs(level) < sync_depth.SPECIFIED_DEPTH_IRE:
        _log_once(field, "standard_levels_refused",
                  "model stages: standard_levels declined - the porch "
                  "baseline reads %+.2f IRE, further from the specified zero "
                  "than the whole %.0f IRE the standard puts between blanking "
                  "and the synchronizing level",
                  level, sync_depth.SPECIFIED_DEPTH_IRE)
        return None

    return {
        "spans": spans,
        "rows": np.asarray(series["rows"], dtype=np.float64),
        "baselines": baselines,
        "leakage": leakage,
        "conditioning": conditioning,
        "offered": offered,
        "withheld": withheld,
        "window_s": window_s,
        "trend": trend,
        "level": level,
        "per_row": _zero_per_row(trend, np.asarray(series["rows"],
                                                   dtype=np.float64),
                                 spans["lines"]),
    }


def _zero_per_row(trend, rows, lines):
    """THE TREND EVALUATED AT EVERY ROW, not only at the ones it was
    measured on: the zero belongs to the FIELD and the vertical interval
    has the same one. Interpolation inside the measured span and a held
    value beyond it, which is an extrapolation of at most the interval's
    own length."""
    shape = np.asarray(trend.get("trend"), dtype=np.float64) \
        if trend.get("trend") is not None \
        else np.full(rows.size, float(trend["level"]))
    every = np.arange(int(lines), dtype=np.float64)
    return np.interp(every, rows, shape)


def _realise_standard_levels(lines, reading):
    """THE REALISATION HALF: the per-row zero taken off a float field, in
    place. One subtraction per row and nothing scaled."""
    spans = reading["spans"]
    lines -= (np.asarray(reading["per_row"], dtype=np.float64)
              * spans["units_per_ire"])[:, None]


def _standard_levels_report(series, reading, verdict):
    """The reading `correct_standard_levels` has always attached."""
    trend = reading["trend"]
    leakage = reading["leakage"]
    return {
        "baseline_ire": float(reading["level"]),
        "raw_mean_ire": float(np.nanmean(series["blanking_ire"])),
        "leakage_ire": {name: float(np.abs(np.mean(values)))
                        for name, values in leakage.items()},
        "leakage_phase_deg": {name: float(np.degrees(np.angle(np.mean(values))))
                              for name, values in leakage.items()},
        "carriers_withheld": [float(m) for m in reading["withheld"]],
        "trend_swing_ire": float(trend["trend_swing"]),
        "flat": bool(trend["flat"]),
        "rejected_fraction": float(trend["rejected_fraction"]),
        "conditioning": float(np.median(reading["conditioning"])),
        "excursion_ire": float(verdict["excursion_ire"]),
        "within_luma_limit": bool(verdict["within_luma_limit"]),
        "within_composite_limit": bool(verdict["within_composite_limit"]),
        "lines": int(reading["baselines"].size),
    }


# How much of the radio-frequency input one reading of the carrier is taken
# on, stated as a count of LINES rather than as a duration: the pulses are
# what the reading is made of, so the window is sized by how many of them it
# must contain. `find_sync` refuses below eight and `instantaneous` guards a
# fraction off each end against the transform's wrap-around, so the window
# has to hold the eight the estimator needs plus the ones the guard removes.
# Sixty-four leaves about forty-eight after the guard, which is a fifth of a
# field: long enough that the median is not one pulse's noise and short
# enough that a whole field's worth of transform is not run per field.
_CARRIER_WINDOW_LINES = 64


def measure_level_from_frequency(field, amount):
    """THE LEVELS READ AS CARRIER FREQUENCIES, BEFORE THE DEMODULATOR.

    Node `level_from_frequency`. Applies nothing.

    Ethan, 2026-09-06: *"the frequency will have been refined to the point
    where this will perfectly resolve the drift and shape of the luma channel
    to remove the agc and other level inconsistencies ... Let's build that
    inversion."*

    WHY IT IS AN INVERSION AND NOT A SECOND OPINION. SMPTE 32M-2004 clause
    3.9.1.1.4 puts the synchronizing tip at 3.4 MHz and peak white at 4.4, so
    the 140 IRE excursion IS the 1.0 MHz deviation and a level and a
    frequency are one quantity in two units. The picture-domain reading
    inherits the demodulator, the de-emphasis, the time base and every
    correction after them; this one inherits none of them, because it is
    taken on `field.data["input"]` - the radio-frequency samples the field
    was decoded from.

    ALL THREE AXES, AND ALL THREE MEASURED:

      FREQUENCY   the tip's and blanking's own carriers in hertz, which is
                  the axis this stage exists on. Band-limited to the luma
                  first, because the instantaneous frequency of the SUM of
                  the luma's frequency-modulated carrier and the
                  chrominance's amplitude-modulated one is neither of theirs
                  - unlimited, an earlier reading put blanking 200 kHz low
                  and returned a gain of 0.22.
      AMPLITUDE   the same gap in IRE at the specification's own exchange
                  rate of 7.14 kHz an IRE, which is the level.
      TIME        the per-pulse series across the window, and the window's
                  own position in the decode. The scatter of that series is
                  what separates a measurement from a dropout: the module
                  records a window that returned 61.8 IRE with a tip scatter
                  of 74 kHz against 7 to 10 kHz in the good ones.

    THE GATE IS `sync_depth`'S, TRANSPLANTED. Both levels are specified
    constants, so the per-pulse gap is one number measured many times and
    the lines must agree about it to within the standard's own tolerance -
    read robustly, so a handful of bad pulses cannot set it. In this domain
    that tolerance is `IRE_TOLERANCE` times the exchange rate, which is the
    same statement in hertz.

    THE PAIR IS THE POINT. `agreement` puts this reading beside the picture's
    own, and the two share no stage between them, so their disagreement
    bounds both. The module's own finding is that the tape holds the
    specified forty IRE and a little over while the decoded picture delivers
    thirty-five to thirty-seven, which places the deficit AFTER the
    demodulator rather than on the tape. This stage is that finding taken by
    the runtime, on the runtime's own fields.

    MEASURED BY THIS STAGE, 2026-09-06, `--stages +level_from_frequency`,
    191 or 192 pulses over four fields of 64 lines each:

        decode              tip MHz  blanking  gap IRE  carrier  picture
        75 bars SP           3.4280   3.7234    41.46   1.0366   0.9738
        multiburst y-only    3.4322   3.7250    40.97   1.0242   0.9745
        home                 3.4644   3.7654    42.52   1.0631   0.9466
        countdown            3.3822   3.7291    48.76   1.2191   DECLINED

    THE TAPE HOLDS ITS LEVELS AND THE DECODE DOES NOT, on every capture: the
    carrier stands 5.1, 6.4 and 12.3 per cent above the picture on the three
    where both can be read.

    AND THE FOURTH ROW IS THE STAGE EARNING ITS PLACE. Countdown's sync tip
    is unreadable in the picture - 79 per cent of it on the container floor,
    which makes `sync_depth`, `source_agc`, `vcr_agc` and `standard_levels`
    all decline - and the carrier reading is unaffected, because a clip
    imposed after the demodulator does not exist before it. So the pair is
    withheld there and the frequency-domain reading still stands.

    THE FIGURE IS 1.17 IN THE MODULE AND 1.065 HERE, AND THE DIFFERENCE IS
    THE IRE TRAP. The module's picture-side numbers were read from the
    written JSON as `(white16bIre - black16bIre) / 100`, which
    `--level_adjust` inflates to 400.768 units an IRE where the decode used
    358.4 - every absolute IRE taken that way is short by 11.82 per cent.
    This stage reads its scale from `hz_to_output` on the running decode, so
    the bar tape's picture gain comes out 0.9738 rather than 0.873, and
    0.873 x 1.1182 is 0.976. The deficit is real and it is 6.5 per cent, not
    seventeen.

    THE GAUGE IT IS NOT FITTED WITH is the picture's own bar plateaus,
    regressed through the origin on SMPTE 170M's specified luminances:
    0.9734 against the sync pulse's 0.9738, four parts in ten thousand
    apart, while the carrier stands 6.5 per cent above both. Two
    independent readings of the picture agree with each other and disagree
    with the tape, which is exactly the shape a defect downstream of the
    demodulator makes.
    """
    if not amount:
        return None
    state = _measurement_state(field, "level_from_frequency")
    if state["reported"]:
        return None

    data = getattr(field, "data", None)
    samples = data.get("input") if isinstance(data, dict) else None
    if samples is None or not len(samples):
        _log_once(field, "level_from_frequency_declined",
                  "model stages: level_from_frequency declined - this field "
                  "carries no radio-frequency input to read a carrier on")
        state["reported"] = True
        return None

    rate = float(getattr(field.rf, "freq_hz", 0.0))
    if not rate > 0:
        _log_once(field, "level_from_frequency_declined",
                  "model stages: level_from_frequency declined - the input "
                  "sample rate is not stated")
        state["reported"] = True
        return None

    line_s = float(field.rf.SysParams["line_period"]) * 1e-6
    want = int(round(_CARRIER_WINDOW_LINES * line_s * rate))
    block = np.asarray(samples, dtype=np.float64).ravel()
    if block.size < want:
        want = block.size
    start = max((block.size - want) // 2, 0)
    window = block[start:start + want]

    try:
        got = level_from_frequency.levels(window, rate)
    except (ValueError, np.linalg.LinAlgError) as error:
        _log_once(field, "level_from_frequency_declined",
                  "model stages: level_from_frequency declined - %s", error)
        state["reported"] = True
        return None

    gaps = np.asarray(got["per_pulse_gap_hz"], dtype=np.float64)
    gaps = gaps[np.isfinite(gaps)]
    per_ire = level_from_frequency.hz_per_ire()
    scatter = (1.4826 * float(np.median(np.abs(gaps - np.median(gaps))))
               if gaps.size else float("inf"))
    # WHAT THE STANDARD BOUNDS IS THE LEVEL, NOT ONE PULSE'S READING OF IT,
    # AND THAT DISTINCTION IS THE WHOLE GATE. `sync_depth` applies the 1 IRE
    # tolerance to a PER-LINE spacing because each of its lines averages the
    # tip over a wide flat interior and scatters 0.26 to 0.36 IRE; here every
    # pulse is one median of an instantaneous frequency, which is a far
    # noisier reading of the same constant, so the same test applied to the
    # per-pulse spread measures the ESTIMATOR and not the signal. Applied
    # that way it refused the bars tape at 1.64 IRE - a window whose spread
    # sits squarely in the 7 to 10 kHz range the module records for its GOOD
    # windows once the two levels' noise is added - which is a gate refusing
    # a measurement because it was asking the wrong question.
    #
    # SO THE TEST IS ON THE ESTIMATE'S OWN ERROR. The gap is a difference of
    # two levels each allowed 1 IRE, so it may carry root-two of that, and
    # the quantity that must land inside it is the standard error of the
    # pooled reading - the spread over the root of the pulse count. On the
    # module's recorded numbers this separates the two regimes it names: a
    # good window's 7 to 10 kHz over forty-eight pulses is 0.14 to 0.20 IRE,
    # and the dropout's 74 kHz is 1.50 IRE, outside. The spread itself is
    # reported beside it so the regime is visible rather than implied.
    allowed = math.sqrt(2.0) * standard_levels.IRE_TOLERANCE * per_ire
    error = scatter / math.sqrt(max(gaps.size, 1))
    if not error < allowed:
        _log_once(field, "level_from_frequency_scatter",
                  "model stages: level_from_frequency declined - %d pulses "
                  "spread %.1f kHz about the blanking-to-tip gap, so the "
                  "pooled reading carries %.2f IRE of error against the %.2f "
                  "a difference of two levels each allowed %.1f IRE may "
                  "carry; this window is a dropout or an unlocked passage "
                  "rather than a measurement",
                  int(gaps.size), scatter / 1e3, error / per_ire,
                  allowed / per_ire, standard_levels.IRE_TOLERANCE)
        state["reported"] = True
        return None
    got["gap_error_ire"] = error / per_ire
    got["gap_spread_ire"] = scatter / per_ire

    state["fields"] += 1
    state.setdefault("readings", []).append(got)
    if state["fields"] < _SOURCE_DECISION_FIELDS:
        return got

    readings = state["readings"]
    pooled = {
        "tip_hz": float(np.median([r["tip_hz"] for r in readings])),
        "blanking_hz": float(np.median([r["blanking_hz"] for r in readings])),
        "gap_hz": float(np.median([r["gap_hz"] for r in readings])),
        "spacing_ire": float(np.median([r["spacing_ire"] for r in readings])),
        "gain": float(np.median([r["gain"] for r in readings])),
        "tip_scatter_hz": float(np.median([r["tip_scatter_hz"]
                                           for r in readings])),
        "gap_spread_ire": float(np.median([r["gap_spread_ire"]
                                           for r in readings])),
        "gap_error_ire": float(np.median([r["gap_error_ire"]
                                          for r in readings])),
        "pulses": int(np.sum([r["pulses"] for r in readings])),
        "fields": len(readings),
        "window_lines": _CARRIER_WINDOW_LINES,
    }

    # THE PAIR. The picture's own gain, read on the same field's reserved
    # intervals through every stage this reading skipped.
    series = _level_series(field, getattr(field, "dspicture", None))
    if series is not None and series["trustworthy"]:
        level_gain = (float(np.nanmean(series["spacing_ire"]))
                      / sync_depth.SPECIFIED_DEPTH_IRE)
        pooled["against_the_picture"] = level_from_frequency.agreement(
            pooled["gain"], level_gain)
    state["pooled"] = pooled
    state["reported"] = True

    _log_once(field, "level_from_frequency",
              "model stages: level_from_frequency - on the radio frequency "
              "itself the synchronizing tip sits at %.4f MHz and blanking at "
              "%.4f, a gap of %.1f kHz which is %.2f IRE at the "
              "specification's %.2f kHz an IRE, so a gain of %.4f; %d pulses "
              "over %d fields of %d lines each, tip scatter %.1f kHz, the "
              "gap spreading %.2f IRE pulse to pulse for %.3f IRE of error "
              "on the pooled reading",
              pooled["tip_hz"] / 1e6, pooled["blanking_hz"] / 1e6,
              pooled["gap_hz"] / 1e3, pooled["spacing_ire"], per_ire / 1e3,
              pooled["gain"], pooled["pulses"], pooled["fields"],
              _CARRIER_WINDOW_LINES, pooled["tip_scatter_hz"] / 1e3,
              pooled["gap_spread_ire"], pooled["gap_error_ire"])
    if "against_the_picture" in pooled:
        pair = pooled["against_the_picture"]
        _log_once(field, "level_from_frequency_pair",
                  "model stages: level_from_frequency against the picture - "
                  "the carrier says %.4f and the decoded levels say %.4f, a "
                  "difference of %+.4f (%+.1f per cent). The two share no "
                  "stage, so %s",
                  pair["frequency_gain"], pair["level_gain"],
                  pair["difference"], 100.0 * pair["relative"],
                  "the deficit is downstream of the demodulator, not on the "
                  "tape" if pair["difference"] > 0 else
                  "the picture carries MORE than the carrier does, which is "
                  "a gain added after the demodulator")
    field.level_from_frequency = pooled
    return pooled


def measure_source_agc(field, amount):
    """THE SOURCE'S GAIN CONTROL, DERIVED BY ASSERTING THE LEVELS.

    Node `source_agc`. Chain position 7, between the propagation terms and
    the recorder's own loop at 8. Applies nothing.

    Ethan, 2026-09-06: *"The levels changing related to the luma level is
    likely AGC drift and this has a attack and decay component to it. We
    should model that in all dimensions ... AGC can be derived by asserting
    the levels should be constant, and back porch should be ire 0."*

    NOT FITTED TO A RESIDUAL - ASSERTED. Blanking is the reference zero and
    the synchronizing level is 40 IRE beneath it (ITU-R BT.1700), so on a
    signal that has passed through nothing the back porch reads zero on every
    line and the spacing reads forty. What is measured instead is the chain,
    and the part of it that MOVES belongs to the only element whose purpose
    is to move. The two observables separate cleanly and by construction: a
    gain cannot move a zero and an offset cannot change a spacing.

    THE THREE AXES, AND ONE OF THEM IS DERIVED RATHER THAN MEASURED:

      AMPLITUDE   MEASURED. The offset from `assert_levels`, and the gain
                  from the same call - the two departures from the two
                  specified constants, taken on the same lines by the same
                  arithmetic.
      TIME        MEASURED, AND ORDINAL ONLY. A gain control is asymmetric by
                  design - fast down to avoid clipping, slow up to avoid
                  pumping - and the asymmetry is a property of the
                  TRAJECTORY, so it survives the drive being unobserved. The
                  ratio of a high percentile of the downward increments to
                  the same percentile of the upward ones is what `fit`
                  returns, and the module is explicit that the number ORDERS
                  rather than measures: against planted controls a ratio of
                  10 reads 1.0 and one of 100 reads 4.3.
      FREQUENCY   DERIVED, NOT MEASURED, and it cannot be otherwise here.
                  `levels_as_frequencies` turns the measured gain into what
                  it is once the signal is on tape - the levels ARE the
                  carrier's frequencies, so a gain of g puts the deviation at
                  g megahertz and moves every channel above it. That is the
                  specification applied to a measurement, not a second
                  measurement. The band the control can act over at all,
                  `signatures`, needs the two time constants, and those need
                  the drive.

    WHAT IS NOT IDENTIFIABLE, SAID PLAINLY. The driving level is never
    observed - only the control's output is - so the two time constants
    cannot both be recovered. `fit` reports the ratio and says so; the
    module's `solve_with_drive` recovers 2.015 ms and 222.8 ms against a
    planted 2 and 200 when the drive IS known, and nothing in a decode
    supplies it. If the returned asymmetry sits at unity this stage is inert
    on the effect it exists for, and it says that rather than dressing it up.

    MEASURED BY THIS STAGE, 2026-09-06, `--stages +source_agc`, over four
    fields of 245 lines each:

        decode              offset   moves   gain     moves    asymmetry
        75 bars SP          -0.013   0.259   0.9752   0.0077     0.995
        multiburst y-only   -0.019   0.381   0.9803   0.0128     1.058
        home                +0.204   0.876   0.9480   0.0107     1.084
        countdown              -       -       -        -      DECLINED

    IT IS INERT ON THE EFFECT IT EXISTS FOR, AND THAT IS THE FINDING RATHER
    THAN A FAILURE TO REPORT. The asymmetry runs 0.995 to 1.084 on all three
    readable decodes, where a gain control falling ten times faster than it
    recovers reads 1.0 on the module's own planted control and one falling a
    hundred times faster reads 4.3. Nothing here is evidence of a control
    acting: the levels move, but they move symmetrically, which is what a
    level shift looks like and not what a gain control looks like. The two
    LEVELS are measured and are not inert - home's back porch stands +0.204
    IRE from the specified zero and moves 0.876 IRE within a field - and the
    gain is 0.948 to 0.980 on all three. It is only the ASYMMETRY, the axis
    that would identify a control, that is silent.

    THE GAUGE IT IS NOT FITTED WITH is the radio-frequency carrier, which
    `level_from_frequency` reads on the same fields and which shares no stage
    with this one: 1.0242, 1.0366 and 1.0631 against this stage's asserted
    0.9803, 0.9752 and 0.9480. The gain this stage measures is the PICTURE's
    and the tape's is above it on every capture, so whatever moves the gain
    is not ahead of the record head - which is precisely the conclusion a
    stage at chain position 7 is placed to be able to draw.
    """
    if not amount:
        return None
    picture = getattr(field, "dspicture", None)
    if picture is None:
        return None
    series = _level_series(field, picture)
    if series is None:
        _log_once(field, "source_agc_declined",
                  "model stages: source_agc declined - no sync population on "
                  "an IRE scale in this field")
        return None
    if not series["trustworthy"]:
        _log_once(field, "source_agc_untrusted",
                  "model stages: source_agc declined - %s; a gain control "
                  "cannot be derived by asserting levels that cannot be read",
                  series["why_not"])
        return None

    asserted = source_agc.assert_levels(series["blanking_ire"],
                                        series["spacing_ire"])
    # THE SAMPLE PERIOD IS THE LINE, because that is the rate the two levels
    # are observed at: one blanking and one spacing per line of the
    # specification's own population.
    line_s = float(field.rf.SysParams["line_period"]) * 1e-6
    try:
        asymmetry = source_agc.fit(asserted["gain"], line_s)
    except ValueError as error:
        _log_once(field, "source_agc_declined",
                  "model stages: source_agc declined - %s", error)
        return None

    carriers = source_agc.levels_as_frequencies(asserted["gain_mean"],
                                                asserted["offset_mean"])
    state = _measurement_state(field, "source_agc")
    state["fields"] += 1
    store = state.setdefault("readings", [])
    store.append((asserted["offset_mean"], asserted["gain_mean"],
                  asymmetry.get("asymmetry", float("nan")),
                  asserted["offset_moves"], asserted["gain_moves"]))

    reading = {
        "offset_ire": float(asserted["offset_mean"]),
        "offset_moves_ire": float(asserted["offset_moves"]),
        "gain": float(asserted["gain_mean"]),
        "gain_moves": float(asserted["gain_moves"]),
        "asymmetry": float(asymmetry.get("asymmetry", float("nan"))),
        "falls_faster": bool(asymmetry.get("falls_faster", False)),
        "deviation_hz": float(carriers["deviation_hz"]),
        "deviation_error_hz": float(carriers["deviation_error_hz"]),
        "peak_white_error_hz": float(carriers["peak_white_error_hz"]),
        "chain_position": source_agc.CHAIN_POSITION,
        "drive_observed": False,
    }
    field.source_agc = reading

    if len(store) >= _SOURCE_DECISION_FIELDS and not state["reported"]:
        state["reported"] = True
        columns = [np.asarray([row[i] for row in store], dtype=np.float64)
                   for i in range(5)]
        offsets, gains, ratios, offset_moves, gain_moves = columns
        ratios = ratios[np.isfinite(ratios)]
        pooled_ratio = float(np.median(ratios)) if ratios.size else float("nan")
        _log_once(field, "source_agc",
                  "model stages: source_agc (chain position %d) over %d "
                  "fields - the back porch sits %+.3f IRE from the specified "
                  "zero and moves %.3f IRE within a field, the spacing gives "
                  "a gain of %.4f moving %.4f; asserted from the two "
                  "constants and never fitted to a residual",
                  source_agc.CHAIN_POSITION, len(store),
                  float(np.median(offsets)), float(np.median(offset_moves)),
                  float(np.median(gains)), float(np.median(gain_moves)))
        _log_once(field, "source_agc_time",
                  "model stages: source_agc time axis - the trajectory's "
                  "asymmetry reads %.3f, %s. THE DRIVE IS NEVER OBSERVED, so "
                  "the two time constants are not identifiable and only this "
                  "ratio is; the module's own controls read 1.0 for a planted "
                  "ratio of 10 and 4.3 for a planted 100, so this ORDERS "
                  "controls and is not the ratio. %s",
                  pooled_ratio,
                  "the gain falls faster than it recovers"
                  if pooled_ratio > 1.0 else
                  "the gain recovers at least as fast as it falls, which is "
                  "not what a gain control does",
                  "At this value the stage is INERT on the effect it exists "
                  "for: the level series carries no asymmetry, so nothing "
                  "here is evidence of a control acting."
                  if abs(pooled_ratio - 1.0) < 0.1 else
                  "The asymmetry is resolved, so something in the chain is "
                  "moving the gain in one direction faster than the other.")
        _log_once(field, "source_agc_frequency",
                  "model stages: source_agc frequency axis, DERIVED from the "
                  "measurement and not measured beside it - a gain of %.4f "
                  "puts the recorded deviation at %.1f kHz against the 1000.0 "
                  "SMPTE 32M clause 3.9.1.1.4 specifies, so peak white lands "
                  "%+.1f kHz from 4.4 MHz and every channel above the luma "
                  "moves with it",
                  float(np.median(gains)),
                  float(np.median(gains)) * 1e3,
                  float(np.median(gains) - 1.0) * 1e3)
    return reading


def measure_vcr_agc(field, amount):
    """THE RECORDER'S OWN GAIN CONTROL, BOUNDED BY THE STRAIGHTNESS OF THE
    LEVEL SERIES.

    Node `vcr_agc`. Chain position 8, after the transmission path and BEFORE
    the recorder's input clip at 9 - and that order is the whole reason the
    two cannot be folded together, because THE AGC SETS THE LEVEL AT WHICH
    THE CLIP IS REACHED. Applies nothing.

    Ethan: *"Remember the clipping constraint, and AGC may also be a concern
    here, since this shifts the frequency response of the composite channel
    as it relates to the television standard."* Both halves are the same
    mechanism seen twice, and both are reported here.

    THREE AXES:

      TIME        MEASURED, AND IT IS A BOUND RATHER THAN A VALUE.
                  `time_constant_from_drift` reads the two level series
                  across the field and asks whether either BENDS. A loop
                  relaxing inside the window curves its trajectory by a known
                  function of the window over the time constant; the series
                  are straight, so every loop that would have shown a
                  resolvable bend is excluded and the time constant is
                  bounded BELOW. On this arc's three tapes that bound cleared
                  the 1 ms row and left only the 50 ms candidate of the five
                  the module had assumed - the first measurement it has ever
                  had of its own central quantity.
      AMPLITUDE   MEASURED. `reference_error` is what a clipped
                  synchronizing tip costs a loop referenced to it: the loop
                  drives the measured tip to nominal and SUCCEEDS, so the tip
                  comes out looking right while everything above blanking is
                  stretched by `1/(1-d)`. The clip depth comes from
                  `sync_depth.detect_clip`, which finds a clip by the tip's
                  PILE-UP and not by its depth - because the depth is exactly
                  what the loop has already normalised away.
      FREQUENCY   DERIVED from the measured bound. A gain that varies in time
                  is not a filter, but its effect on the LEVEL is one: the
                  loop suppresses the slow amplitude changes it can track and
                  passes the ones it cannot, which is a first-order high-pass
                  at `1/(2 pi tau)`. That is Ethan's *"shift in the frequency
                  response of the composite channel"*, and it is the loop's
                  own closed-loop transfer rather than a metaphor. Evaluated
                  at the bound, so the corner quoted is the HIGHEST the
                  measurement permits.

    THE PROBE IS THE FIELD, AND THAT IS THE EXTENSION THIS MODULE BROUGHT.
    `sync_geometry.probes` stops at the vertical interval, 572 microseconds,
    because that is the longest contiguous run of sync in the signal. But
    both levels are readable on EVERY line, so a level series across a field
    is a probe of the field's own length - twenty-six times the interval's
    reach - and it is still sync-only, because both levels are read in the
    reserved intervals and never in the picture.

    MEASURED BY THIS STAGE, 2026-09-06, `--stages +vcr_agc`. The probe runs
    15571 microseconds over 245 lines and resolves 64.2 Hz on every decode:

        decode             spacing  sigma  blanking  sigma   tau at    corner
                           drift           drift             least
        75 bars SP         +0.1379   4.9   +0.2571   1.0    1.27 ms   125.5 Hz
        multiburst y-only  +0.0275   0.3   +0.2068   3.3    0.52 ms   304.1 Hz
        home               -0.2663   4.8   -0.9226   5.0    1.95 ms    81.8 Hz
        countdown             -       -       -       -        -      DECLINED

    THE PAIR DOES DISAGREE, AND DIFFERENTLY ON EACH TAPE, which is what a
    pair exists to produce and what one level read twice never could. On the
    bars tape only the GAIN is resolved and blanking is not; on the
    multiburst pattern the reverse, blanking at 3.3 sigma and the spacing not
    resolved at all; on home both move together. A single level would have
    reported one number on all three.

    THE TIME CONSTANT IS BOUNDED AND THE BOUND IS REAL. None of the three
    series BENDS, so every relaxation fast enough to have shown a resolvable
    curvature is excluded, and of the five candidates spanning five decades
    only 50 ms survives on two of the three decodes and only 1 ms and 50 ms
    on the third. That is a five-decade assumption narrowed to one, taken by
    the runtime on its own fields.

    THE GAUGE IT IS NOT FITTED WITH is the same blanking level ACROSS fields,
    which this stage never looks at - it fits a straight line WITHIN a field
    and asks only whether it bends. Field to field the level scatters 0.0169
    IRE over 26 fields on the bars tape and 0.0314 over 28 on home, both
    below the 0.0461 IRE ten-bit output floor, which is consistent with a
    corner above the 59.94 Hz field rate suppressing field-rate level content
    - and on countdown, where this stage declines, the same statistic is
    2.911 IRE with a 12.93 IRE swing, which is what an unreadable level looks
    like from the outside.
    """
    if not amount:
        return None
    picture = getattr(field, "dspicture", None)
    if picture is None:
        return None
    series = _level_series(field, picture)
    if series is None:
        _log_once(field, "vcr_agc_declined",
                  "model stages: vcr_agc declined - no sync population on an "
                  "IRE scale in this field")
        return None
    if not series["trustworthy"]:
        _log_once(field, "vcr_agc_untrusted",
                  "model stages: vcr_agc declined - %s; a loop cannot be "
                  "bounded by the straightness of a series that is not the "
                  "level", series["why_not"])
        return None

    spacing = np.asarray(series["spacing_ire"], dtype=np.float64)
    blanking = np.asarray(series["blanking_ire"], dtype=np.float64)
    good = np.isfinite(spacing) & np.isfinite(blanking)
    if good.sum() < 8:
        _log_once(field, "vcr_agc_declined",
                  "model stages: vcr_agc declined - %d usable lines is not a "
                  "series", int(good.sum()))
        return None
    try:
        drift = vcr_agc.time_constant_from_drift(spacing[good],
                                                 blanking[good],
                                                 series["spans"]["system"])
        probe = vcr_agc.field_probe(int(good.sum()),
                                    series["spans"]["system"])
    except (ValueError, KeyError, np.linalg.LinAlgError) as error:
        _log_once(field, "vcr_agc_declined",
                  "model stages: vcr_agc declined - %s", error)
        return None

    # A BOUND THAT DOES NOT EXIST IS NOT AN INFINITE ONE. `x is None` means
    # the series could not have resolved even the fastest bend the window can
    # watch, so NOTHING is excluded and there is no lower bound to quote;
    # carried as "not a number" so the pooling drops it rather than reading
    # it as an exclusion of everything.
    bound = drift.get("time_constant_us_at_least")
    bound_us = float(bound) if bound is not None else float("nan")
    state = _measurement_state(field, "vcr_agc")
    state["fields"] += 1
    store = state.setdefault("readings", [])
    store.append((float(drift["differential"]["drift_ire"]),
                  float(drift["common_mode"]["drift_ire"]), bound_us,
                  float(drift["differential"]["sigma"]),
                  float(drift["common_mode"]["sigma"])))

    # THE CLIP, ON THE AMPLITUDE AXIS. The depth is the fraction of the
    # tip-to-white span the container took, read from where the censored fit
    # puts the true level against the floor it was censored at - not from the
    # tip's depth against the standard, which the loop has already removed.
    clip = series["clip"]
    result = series["result"]
    censored = result.get("censored")
    floor_ire = float(clip["container_floor_ire"])
    depth = 0.0
    if censored and censored.get("fitted"):
        true_tip = (float(censored["mean"]) - series["spans"]["blanking"]) \
            / series["spans"]["units_per_ire"]
        span = standard_levels.REFERENCE_WHITE_IRE - true_tip
        if span > 0:
            depth = max(0.0, (floor_ire - true_tip) / span)
    cost = vcr_agc.reference_error(min(depth, 0.99))

    reading = {
        "chain_position": vcr_agc.CHAIN_POSITION,
        "spacing_drift_ire": float(drift["differential"]["drift_ire"]),
        "spacing_sigma": float(drift["differential"]["sigma"]),
        "blanking_drift_ire": float(drift["common_mode"]["drift_ire"]),
        "blanking_sigma": float(drift["common_mode"]["sigma"]),
        "which_moves": str(drift["which_moves"]),
        "the_pair_disagrees": bool(drift["the_pair_disagrees"]),
        "surviving_constants_us": [float(v) for v in
                                   drift["assumed_constants_surviving_us"]],
        "time_constant_us_at_least": bound_us,
        "probe_resolution_hz": float(probe["resolution_hz"]),
        "probe_duration_us": float(probe["duration_us"]),
        "clipped": bool(clip["clipped"]),
        "clip_depth_in_span": float(depth),
        "gain_error_from_clip": float(cost["gain_error"]),
        "corner_hz_at_the_bound": (vcr_agc.corner_hz(bound_us)
                                   if np.isfinite(bound_us) and bound_us > 0
                                   else 0.0),
    }
    field.vcr_agc = reading

    if len(store) >= _SOURCE_DECISION_FIELDS and not state["reported"]:
        state["reported"] = True
        bounds = np.asarray([row[2] for row in store], dtype=np.float64)
        bounds = bounds[np.isfinite(bounds)]
        pooled = float(np.median(bounds)) if bounds.size else float("nan")
        _log_once(field, "vcr_agc",
                  "model stages: vcr_agc (chain position %d) over %d fields - "
                  "the spacing drifts %+.4f IRE across the %d lines of the "
                  "window at %.1f sigma and blanking %+.4f at %.1f sigma; the "
                  "two are different mechanisms, a gain drift and a level "
                  "shift, and only the first belongs to this loop - on "
                  "this field %s moves and the pair %s",
                  vcr_agc.CHAIN_POSITION, len(store),
                  float(np.median([row[0] for row in store])),
                  int(good.sum()),
                  float(np.median([row[3] for row in store])),
                  float(np.median([row[1] for row in store])),
                  float(np.median([row[4] for row in store])),
                  reading["which_moves"],
                  "disagrees" if reading["the_pair_disagrees"]
                  else "agrees")
        if np.isfinite(pooled):
            _log_once(field, "vcr_agc_time",
                      "model stages: vcr_agc time axis - the field-long "
                      "probe runs %.0f us and resolves %.1f Hz, twenty-six "
                      "times the vertical interval's reach; the series does "
                      "not BEND, so every loop relaxing faster than %.2f ms "
                      "is excluded and the time constant is bounded BELOW "
                      "rather than measured. Of the five constants the "
                      "module assumed: %s",
                      reading["probe_duration_us"],
                      reading["probe_resolution_hz"], pooled / 1e3,
                      ", ".join("%.0f us %s" % (
                          candidate,
                          "survives" if candidate >= pooled else "excluded")
                          for candidate in vcr_agc.ASSUMED_TIME_CONSTANTS_US))
        else:
            _log_once(field, "vcr_agc_time",
                      "model stages: vcr_agc time axis declined to bound - "
                      "the field-long probe runs %.0f us and resolves %.1f "
                      "Hz, but the drift it shows is too small against its "
                      "own scatter for even the fastest resolvable bend to "
                      "have stood out, so NOTHING is excluded and no lower "
                      "bound on the time constant exists on this material",
                      reading["probe_duration_us"],
                      reading["probe_resolution_hz"])
        _log_once(field, "vcr_agc_amplitude",
                  "model stages: vcr_agc amplitude axis - the synchronizing "
                  "tip %s, %.3f%% of its samples at the container floor and "
                  "a fitted depth of %.4f of the tip-to-white span; a loop "
                  "referenced to that tip would set a gain of %.4f and the "
                  "error would be exactly ABSENT from the one level it is "
                  "read at",
                  "IS clipped" if reading["clipped"] else "is not clipped",
                  100.0 * float(clip["fraction_at_floor"]),
                  reading["clip_depth_in_span"],
                  reading["gain_error_from_clip"])
        if np.isfinite(pooled) and pooled > 0:
            _log_once(field, "vcr_agc_frequency",
                      "model stages: vcr_agc frequency axis, DERIVED from "
                      "that bound - the loop is a first-order high-pass on "
                      "the LEVEL, so at the fastest time constant the "
                      "measurement still permits its corner sits at %.2f Hz "
                      "and everything slower than that in the picture's own "
                      "level is removed with the fading it was built to "
                      "track", vcr_agc.corner_hz(pooled))
    return reading


# Two complete colour frames, because the reading is a HELD-OUT one: the
# fit is formed on the first half of the fields and judged on the second,
# and each half has to be a whole colour frame on both heads or the two
# halves are not the same kind of evidence.
_MULTIPATH_DECISION_FIELDS = 2 * _SOURCE_DECISION_FIELDS

_MULTIPATH_STATE = "_multipath"


def _multipath_verdict(reading):
    """What the held-out shares say, READ SEPARATELY.

    The module is explicit that the three dimensions can fail separately and
    that a tap matching the magnitude and not the phase is not a delayed copy
    of anything, so a single pooled share is the wrong summary: it can be
    healthy while the half of the evidence that identifies a delay has gone
    negative. Both are therefore read, and a negative phase share - the fit
    making the held-out phase WORSE than doing nothing - is called out as the
    failure it is rather than averaged into a passing total.
    """
    holds = reading["held_out_removed"] > 0.5 * reading["in_sample_removed"]
    if not holds:
        return ("The fit does NOT hold out, so what it removed in sample was "
                "this half's own noise.")
    if reading["phase_removed"] <= 0.0:
        return ("The total holds out, but ONLY THE MAGNITUDE DOES: the fit "
                "makes the held-out PHASE worse than doing nothing, and a "
                "tap that matches a comb's depth without its phase is not a "
                "delayed copy of anything.")
    return ("The fit holds out on both halves of the evidence, so the "
            "description is stable on fields it did not build.")


def measure_multipath(field, amount):
    """A TAPPED-DELAY DESCRIPTION OF THE SYNC CHANNEL, JUDGED ON FIELDS IT
    NEVER SAW.

    Node `multipath`. Applies nothing.

    THE MECHANISM CLAIM IS WITHDRAWN AND STAYS WITHDRAWN. A general
    propagation model will not account for this arc's all-pass residual: an
    echo WEAKER than the direct path is minimum phase and contributes none of
    it, one STRONGER contributes about seven radians, and the measurement
    sits between the two regimes at 0.87 to 2.07 radians - matching neither.
    Two of the three tapes never passed a transmitter and carry 0.87 to 0.93
    of it. So this stage reports what a tapped-delay channel DESCRIBES, and
    nothing here may be read as saying the description is the air.

    WHAT IS STILL WORTH MEASURING, AND WHY IT IS WORTH MEASURING IN THE
    RUNTIME. The taps come out the SAME ON BOTH HEADS to three decimals -
    home's at 0.152, 0.341, 0.152 and 0.569 microseconds on head A and at the
    same delays with the same amplitudes and phases on head B - which places
    whatever they describe DOWNSTREAM of the head switch, on the two tapes
    where it holds out. And the one tape where propagation multipath is
    possible at all is the one that collapses: countdown holds 56 per cent in
    sample and 17 out of it, with its PHASE going negative, meaning the fit
    makes the held-out phase worse than doing nothing.

    ALL THREE AXES, ALL THREE MEASURED, AND THEY CAN FAIL SEPARATELY:

      FREQUENCY   a delayed copy puts a comb of period `1/tau` on the
                  response, and the fit is a complex least squares on the
                  frequency axis the sync pulse supports.
      AMPLITUDE   the comb's DEPTH fixes each tap's amplitude, reported as
                  its own held-out share, because a tap that matches the
                  magnitude and not the phase is not a delayed copy of
                  anything.
      TIME        the delays themselves, bounded by the probe rather than
                  chosen: `sync_geometry.ghost_reach` says a 4.7 microsecond
                  pulse cannot see a ghost beyond about 2.35 microseconds and
                  cannot resolve one closer than its own edge, so the search
                  runs over that window and the bound is reported with the
                  result.

    THE COMPLEX FIT IS NOT OPTIONAL. A delayed copy moves magnitude and phase
    together - it is one complex tap `a exp(-j 2 pi f tau)` - so the real and
    imaginary parts are stacked and the projection cannot be blind to a
    quarter turn. Fitting the phase alone throws away half the evidence and
    lets a tap match a phase it has no magnitude for, which is the standing
    rule this arc has caught five defects with.

    MEASURED BY THIS STAGE, 2026-09-06, `--stages +multipath`, eight fields
    each - four fitted, four judged - over 14 bins of the sync segment's own
    spectrum, with the delays bounded by the probe at 3.14 microseconds:

        decode              in sample  held out  magnitude   phase
        75 bars SP            46.0 %    45.8 %    +59.1 %   -23.8 %
        multiburst y-only     49.2      46.9      +58.9     +19.2
        home                  63.9      64.0      +59.8     +68.1
        countdown             61.3      29.9      +16.1     +65.2

    COUNTDOWN COLLAPSES AND THE OTHERS DO NOT, which reproduces the module's
    own offline finding on the runtime's own fields: the one capture where
    propagation multipath is possible at all loses half its explanatory
    power on fields the fit never saw - 61.3 per cent to 29.9 - while the two
    non-broadcast tapes hold to within 0.3 of a percentage point. That is
    backwards from the hypothesis, and it is the measured reason the
    mechanism claim stays withdrawn.

    THE THREE AXES FAIL SEPARATELY AND THE STAGE SAYS SO. The MAGNITUDE holds
    out at 58.9 to 59.8 per cent on the three non-broadcast captures and
    collapses to 16.1 on countdown; the PHASE holds at +19.2 to +68.1 on
    three and goes NEGATIVE on the bars tape, meaning the fit makes the
    held-out phase worse than doing nothing there. A pooled share alone would
    have called the bars tape a success at 45.8 per cent; a tap that matches a
    comb's depth without its phase is not a delayed copy of anything, and the
    log names that case rather than averaging it away.

    THE GAUGE IT IS NOT FITTED WITH is a SECOND CAPTURE FROM THE SAME DECK.
    The 75-bar and multiburst y-only patterns were played back on one machine
    in one session, and the fit never sees more than one of them: they return
    held-out shares of 45.8 and 46.9 per cent with magnitudes of 59.1 and
    58.9, and they share two of their four delays exactly - 0.185 and 3.097
    microseconds. So the description is a property of the machine and the
    tape rather than of the pattern, on the magnitude axis. On the PHASE axis
    the same pair disagrees in sign, which is the sharpest form of the
    limitation this stage exists to report.
    """
    if not amount:
        return None
    picture = getattr(field, "dspicture", None)
    if picture is None:
        return None
    store = field.rf.__dict__.setdefault(
        _MULTIPATH_STATE, {"profiles": [], "reported": False})
    if store["reported"]:
        return None

    try:
        geom = geometry(field)
        segment = sync_segment_geometry(field)
    except Exception as error:                                  # noqa: BLE001
        _log_once(field, "multipath_declined",
                  "model stages: multipath declined - the sync segment "
                  "geometry could not be derived (%s)", error)
        store["reported"] = True
        return None

    # ONE FIELD FOLDS IN ONCE, for the reason `source_correction` records:
    # `downscale` can be re-run on the same field object and a profile
    # counted twice would weight that field twice.
    if not field.__dict__.get("_multipath_folded"):
        profile = _sync_profile(field, geom, segment)
        if profile is not None:
            field.__dict__["_multipath_folded"] = True
            store["profiles"].append(profile)

    if len(store["profiles"]) < _MULTIPATH_DECISION_FIELDS:
        return None
    store["reported"] = True

    # THE SPLIT IS IN TIME AND NOT BY HEAD, and that is what makes it a
    # held-out test rather than a head comparison: the taps are a property
    # of a PATH, so they must hold on fields the fit never saw, and fields
    # that arrived later are exactly those.
    profiles_seen = store["profiles"]
    half = len(profiles_seen) // 2
    early = np.mean(np.asarray(profiles_seen[:half], dtype=np.float64), axis=0)
    late = np.mean(np.asarray(profiles_seen[half:], dtype=np.float64), axis=0)

    rate = geom["sample_rate_hz"]
    system = _sync_system(field)
    if system is None:
        _log_once(field, "multipath_declined",
                  "model stages: multipath declined - this tree states no "
                  "sync geometry for %r, so the probe's reach cannot be "
                  "derived", getattr(field.rf, "color_system", None))
        return None
    # THE PROBE IS THE SEGMENT THE CHANNEL WAS MEASURED ON, in microseconds,
    # taken from the decoder's own line period rather than from a written
    # duration. `line_period` is already in microseconds, which is the unit
    # `ghost_reach` reads.
    probe_us = (float(field.rf.SysParams["line_period"])
                * (segment["tail"] + segment["stop"]) / geom["width"])
    try:
        from vhsdecode.models import sync_geometry as geometry_module

        first = source_correction.channel(early, rate)
        second = source_correction.channel(late, rate)
        reach = geometry_module.ghost_reach(probe_us, system)
    except Exception as error:                                  # noqa: BLE001
        _log_once(field, "multipath_declined",
                  "model stages: multipath declined - the sync segment does "
                  "not support a channel measurement (%s)", error)
        return None

    carried = np.asarray(first["carried"]) & np.asarray(second["carried"])
    if int(carried.sum()) < 8:
        _log_once(field, "multipath_declined",
                  "model stages: multipath declined - only %d bins of the "
                  "sync pulse's own spectrum stand above the noise on both "
                  "halves, which is fewer than the four complex taps the fit "
                  "would place", int(carried.sum()))
        return None

    # THE DEPARTURE IS THE COMPLEX LOG, which is the domain a cascade of
    # channels adds in and the only one a tap is linear in. A bin whose
    # response has reached zero has no logarithm, so the carried set is
    # narrowed to the bins where both halves actually have one rather than
    # letting an infinity into the least squares.
    with np.errstate(divide="ignore", invalid="ignore"):
        fit_departure = np.log(np.asarray(first["response"]))
        judge_departure = np.log(np.asarray(second["response"]))
    carried = carried & np.isfinite(fit_departure) & np.isfinite(judge_departure)
    if int(carried.sum()) < 8:
        _log_once(field, "multipath_declined",
                  "model stages: multipath declined - only %d bins carry a "
                  "finite complex logarithm on both halves", int(carried.sum()))
        return None
    hertz = np.asarray(first["frequency_hz"])[carried]
    fit_departure = fit_departure[carried]
    judge_departure = judge_departure[carried]

    try:
        held = multipath.held_out(hertz, fit_departure, judge_departure,
                                  reach_s=float(reach["longest_us"]) * 1e-6)
    except (ValueError, np.linalg.LinAlgError) as error:
        _log_once(field, "multipath_declined",
                  "model stages: multipath declined - %s", error)
        return None

    reading = {
        "taps": [{"delay_us": 1e6 * float(tap["delay_s"]),
                  "magnitude": float(tap["magnitude"]),
                  "phase_deg": float(tap["phase_deg"])}
                 for tap in held["taps"]],
        "in_sample_removed": float(held["in_sample_removed"]),
        "held_out_removed": float(held["held_out_removed"]),
        "magnitude_removed": float(held["magnitude_removed"]),
        "phase_removed": float(held["phase_removed"]),
        "bins": int(carried.sum()),
        "reach_us": float(reach["longest_us"]),
        "shortest_us": float(reach["shortest_us"]),
        "fields": len(profiles_seen),
        "mechanism": "withdrawn",
    }
    _log_once(field, "multipath",
              "model stages: multipath over %d fields, fitted on the first "
              "half and judged on the second - %.1f%% of the sync channel's "
              "complex departure removed in sample and %.1f%% held out "
              "(magnitude %+.1f%%, phase %+.1f%%) over %d bins; the taps sit "
              "at %s microseconds, inside the %.2f us this probe reaches and "
              "outside the %.3f us it can resolve. %s",
              reading["fields"], 100.0 * reading["in_sample_removed"],
              100.0 * reading["held_out_removed"],
              100.0 * reading["magnitude_removed"],
              100.0 * reading["phase_removed"], reading["bins"],
              ", ".join("%.3f" % tap["delay_us"] for tap in reading["taps"]),
              reading["reach_us"], reading["shortest_us"],
              _multipath_verdict(reading))
    _log_once(field, "multipath_withdrawn",
              "model stages: multipath - THE MECHANISM CLAIM IS WITHDRAWN "
              "and this reading does not reinstate it. An echo weaker than "
              "the direct path is minimum phase and adds no all-pass, one "
              "stronger adds about seven radians, and the arc's measured "
              "0.87 to 2.07 matches neither - while two tapes that never "
              "passed a transmitter carry 0.87 to 0.93 of it. What is "
              "reported above is what a tapped-delay channel DESCRIBES, not "
              "what it is")
    field.multipath = reading
    return reading


def _capture_source(field):
    """Which source chain this capture came through, or the neutral one.

    `profiles.py` owns the question. A capture with no recorded history is
    read as a plain composite line input rather than as a broadcast, because
    that is the assumption that ADDS nothing: attaching a television
    channel's limit to a tape that never saw one applies a mechanism where
    it does not exist, while withholding it from a tape that did only leaves
    a limit unmodelled.
    """
    name = _capture_name(field)
    if name:
        try:
            return str(profiles.for_capture(name)["source"]["name"]), True
        except (ValueError, KeyError):
            pass
    return "composite", False


def measure_composite_channel(field, amount):
    """THE CHANNEL THE SPECIFICATIONS DEFINE, CONFRONTED WITH THIS DECODE.

    Node `composite_channel`. Applies nothing.

    Ethan, 2026-09-06: *"Look at the video spec for the frequency response of
    the luma signal, it should be scaled according to that"*, *"The modeling
    needs to consider the RF channel response of the composite signal, just
    as we do with the other stages with the VCR"*, and *"Same with the chroma
    and color under carrier response."*

    THE FIRST ANSWER IS A NEGATIVE ONE AND IT IS THE IMPORTANT ONE. SMPTE
    170M-2004 clause 7.1: *"This standard does not impose a bandwidth
    restriction on the luminance part of the NTSC signal."* There is no
    specified luma response to scale by - the specification IS flatness, and
    the familiar 4.2 MHz belongs to the transmission CHANNEL, which the note
    to clause 7 says outright. So this stage reports the departure from flat
    and applies the broadcast limit only where the capture's own source
    profile says the signal passed through a broadcast channel.

    WHY IT STANDS BESIDE `source_correction` RATHER THAN DUPLICATING IT.
    That stage measures the head-COMMON luma departure and CORRECTS it,
    pooled over a colour frame and referred to the same specified flat. What
    it cannot supply is the reference's own status: whether the departure it
    is removing is a defect or a channel the standard permits, and what the
    standard's own tolerance on such a measurement is. That is this stage,
    and it applies nothing.

    THREE AXES:

      FREQUENCY   MEASURED. The sync pulse's own channel against the
                  specified flat, on the bins the pulse supports at its 213
                  kHz resolution, with the broadcast limit taken from the
                  capture's source profile rather than assumed. Reported
                  with how much of the decode's own band lies above that
                  limit - on VHS the answer is none, because the tape returns
                  about 3 MHz, so the limit is NOT OBSERVABLE here and saying
                  so is the finding.
      AMPLITUDE   MEASURED, WITH THE STANDARD'S OWN ERROR BAR. The departure
                  in decibels root-mean-square, quoted against the tolerance
                  the multiburst measurement carries - 0.5 IRE on 50, which
                  is 0.086 dB (ITU-R J.63 Table III, EBU Tech 3209 section
                  7.2.9). A departure is only a departure against a stated
                  tolerance, and this is the one the standard states.
      TIME        MEASURED. The two exactness relations the specifications
                  fix: the colour-under carrier at EXACTLY 40 f_H (SMPTE 32M
                  clause 3.9.2.1.4, so one line is exactly 40 cycles) and the
                  subcarrier at exactly 455/2 f_H. Both are checked against
                  the decode's OWN line rate, and both are time-domain
                  statements - an exact multiple of the line rate is a
                  carrier that repeats line to line, which is what makes the
                  90 degrees a line of clause 3.9.2.1.5 well defined.

    THE BURST DOUBLER IS THE TRAP AND IS REPORTED WHETHER OR NOT ANYTHING
    ASKS. SMPTE 32M clause 3.9.2.1.3 raises the burst 6.0 +/- 0.5 dB BEFORE
    recording, so the recorded burst is twice the recorded chroma and any
    level taken from a played-back burst and applied to the picture's colour
    is out by a factor of two unless the doubler is undone. It is a
    record-side step: present on every tape and absent from every composite
    reference.

    MEASURED BY THIS STAGE, 2026-09-06, `--stages +composite_channel`, over
    four fields, 14 bins at 213 kHz resolution:

        decode              source       departure   over the    bins above
                                         dB rms      tolerance   4.2 MHz
        75 bars SP          composite       8.98      103.9 x        0
        home                home vcr        9.13      105.7 x        0
        countdown           television      8.51       98.5 x        0
        multiburst y-only   composite      12.95      149.8 x        0

    THE BROADCAST LIMIT IS NOT OBSERVABLE ON ANY OF THEM, and the profile
    system is what makes that a statement rather than an omission: countdown
    is read as a `television` source and so the 4.2 MHz limit IS attached to
    its specified response - and not one of its 14 bins reaches it, because
    the sync pulse carries evidence only to 2.86 MHz. No decode here can
    separate a television source from a line input on this axis, which is a
    limit of the material and not of the model.

    THE GAUGE IT IS NOT FITTED WITH IS THE MULTIBURST, which is the
    measurement the specification actually names for this quantity and which
    lives in the active picture where this stage never reads. On the
    multiburst y-only capture the six packets, specified EQUAL to within 0.5
    IRE on 50, come back at

        0.5 MHz   1.0      2.0      3.0      3.58     4.2
         0.00   -0.17    -2.01    -7.57   -25.42   -28.99   dB

    which is 16.06 dB rms from equality against the 12.95 dB rms this stage
    read from the sync pulse alone on the same capture. Two instruments in
    two different parts of the line agree that the departure is of order 13
    to 16 dB and both put it a hundred times outside the standard's own
    tolerance.

    AND THE SAME GAUGE CONFIRMS THE HEADLINE FINDING. The channel is 25.4 dB
    down at the subcarrier and 29.0 dB down at 4.2 MHz, so the tape has
    already removed the band a broadcast channel's limit would have removed -
    which is why that limit cannot be seen here, measured rather than
    argued.
    """
    if not amount:
        return None
    picture = getattr(field, "dspicture", None)
    if picture is None:
        return None
    state = _measurement_state(field, "composite_channel")
    if state["reported"]:
        return None

    try:
        geom = geometry(field)
        segment = sync_segment_geometry(field)
        profile = _sync_profile(field, geom, segment)
    except Exception as error:                                  # noqa: BLE001
        _log_once(field, "composite_channel_declined",
                  "model stages: composite_channel declined - the sync "
                  "segment geometry could not be derived (%s)", error)
        state["reported"] = True
        return None
    if profile is None:
        return None

    state["fields"] += 1
    store = state.setdefault("profiles", [])
    if not field.__dict__.get("_composite_channel_folded"):
        field.__dict__["_composite_channel_folded"] = True
        store.append(profile)
    if len(store) < _SOURCE_DECISION_FIELDS:
        return None
    state["reported"] = True

    rate = geom["sample_rate_hz"]
    try:
        measured = source_correction.channel(
            np.mean(np.asarray(store, dtype=np.float64), axis=0), rate)
    except Exception as error:                                  # noqa: BLE001
        _log_once(field, "composite_channel_declined",
                  "model stages: composite_channel declined - the sync "
                  "segment does not support a channel measurement (%s)",
                  error)
        return None

    source, recorded = _capture_source(field)
    hertz = np.asarray(measured["frequency_hz"], dtype=np.float64)
    carried = np.asarray(measured["carried"], dtype=bool)
    specified = composite_channel.luma_response(hertz, source)
    above_limit = int(np.count_nonzero(
        carried & (hertz > composite_channel.BROADCAST_LIMIT_HZ)))

    reference = composite_channel.multiburst_reference()
    tolerance_db = float(reference["tolerance_db"])
    departure_db = float(measured["departure_db_rms"])

    # THE TIME AXIS: the two exactness relations, against the decode's own
    # clock rather than against a written figure.
    line_hz = 1e6 / float(field.rf.SysParams["line_period"])
    subcarrier_hz = composite_subcarrier_hz(field)
    colour_under_hz = float(field.rf.DecoderParams["color_under_carrier"])
    cycles_per_line = subcarrier_hz / line_hz
    colour_under_cycles = colour_under_hz / line_hz
    separator = composite_channel.separator_response(
        np.asarray([colour_under_hz, subcarrier_hz]))

    reading = {
        "source": source,
        "source_recorded": bool(recorded),
        "specified_flat": bool(specified["specified_flat"]),
        "limit_applies": bool(specified["limit_applies"]),
        "limit_hz": composite_channel.BROADCAST_LIMIT_HZ,
        "bins_above_the_limit": above_limit,
        "departure_db_rms": departure_db,
        "tolerance_db": tolerance_db,
        "over_tolerance": departure_db / tolerance_db if tolerance_db else
        float("inf"),
        "bins": int(measured["bins"]),
        "resolution_hz": float(measured["resolution_hz"]),
        "highest_carried_hz": (float(hertz[carried].max()) if carried.any()
                               else float("nan")),
        "line_hz": line_hz,
        "subcarrier_hz": subcarrier_hz,
        "subcarrier_cycles_per_line": cycles_per_line,
        "specified_cycles_per_line": composite_channel.SUBCARRIER_CYCLES_PER_LINE,
        "colour_under_hz": colour_under_hz,
        "colour_under_cycles_per_line": colour_under_cycles,
        "specified_colour_under_multiple": 40.0,
        "separator_at_colour_under": float(separator["response"][0]),
        "separator_at_subcarrier": float(separator["response"][1]),
        "burst_doubler_db": composite_channel.BURST_DOUBLER_DB,
        "burst_doubler_linear": 10.0 ** (composite_channel.BURST_DOUBLER_DB
                                         / 20.0),
        "fields": len(store),
    }

    _log_once(field, "composite_channel",
              "model stages: composite_channel over %d fields - this capture "
              "reads as a %r source (%s), so the specified luma response is "
              "%s and the measured sync channel departs from it by %.2f dB "
              "rms over %d bins at %.0f kHz resolution, which is %.1f times "
              "the %.3f dB the multiburst tolerance allows",
              reading["fields"], source,
              "from the recorded history" if recorded else
              "assumed, no history recorded for this capture",
              "flat with the broadcast channel's %.1f MHz limit"
              % (composite_channel.BROADCAST_LIMIT_HZ / 1e6)
              if reading["limit_applies"] else "FLAT, with no bandwidth "
              "restriction at all (SMPTE 170M clause 7.1)",
              departure_db, reading["bins"], reading["resolution_hz"] / 1e3,
              reading["over_tolerance"], tolerance_db)
    _log_once(field, "composite_channel_limit",
              "model stages: composite_channel frequency axis - the pulse "
              "carries evidence to %.2f MHz and %d of its bins lie above the "
              "%.1f MHz transmission limit, so the broadcast channel's own "
              "restriction is %s on this material%s",
              reading["highest_carried_hz"] / 1e6, above_limit,
              composite_channel.BROADCAST_LIMIT_HZ / 1e6,
              "observable" if above_limit else "NOT OBSERVABLE",
              "" if above_limit else " - the tape returns less band than the "
              "channel would have removed, so no decode here can separate a "
              "television source from a line input on this axis")
    _log_once(field, "composite_channel_time",
              "model stages: composite_channel time axis - the decode's "
              "subcarrier sits at %.6f MHz, which is %.4f cycles a line "
              "against the specified %.2f (%+.4f), and its colour-under at "
              "%.6f MHz, %.4f cycles against the exactly 40 SMPTE 32M clause "
              "3.9.2.1.4 requires (%+.4f). An exact multiple of the line rate "
              "is a carrier that repeats line to line, which is what makes "
              "the specified 90 degrees a line well defined",
              subcarrier_hz / 1e6, cycles_per_line,
              composite_channel.SUBCARRIER_CYCLES_PER_LINE,
              cycles_per_line - composite_channel.SUBCARRIER_CYCLES_PER_LINE,
              colour_under_hz / 1e6, colour_under_cycles,
              colour_under_cycles - 40.0)
    _log_once(field, "composite_channel_chroma",
              "model stages: composite_channel chroma chain, every figure "
              "cited and none fitted - the separator band-pass (clause "
              "3.9.2.1.1, 3.58 MHz centre, -3 dB at 3.08 and 4.08) passes "
              "%.4f of the subcarrier and %.5f of the colour-under carrier; "
              "the record level sits %g to %g dB below saturation (clause "
              "3.9.2.1.2); and THE BURST IS RAISED %.1f +/- %.1f dB BEFORE "
              "RECORDING (clause 3.9.2.1.3), so the recorded burst is a "
              "factor of %.3f above the recorded chroma and any level taken "
              "from a played-back burst is out by that factor until the "
              "doubler is undone",
              reading["separator_at_subcarrier"],
              reading["separator_at_colour_under"],
              composite_channel.RECORD_LEVEL_BELOW_SATURATION_DB[0],
              composite_channel.RECORD_LEVEL_BELOW_SATURATION_DB[1],
              composite_channel.BURST_DOUBLER_DB,
              composite_channel.BURST_DOUBLER_TOLERANCE_DB,
              reading["burst_doubler_linear"])
    field.composite_channel = reading
    return reading


# How many places the picture stage's own modifications are evaluated on.
# The question they answer - how many directions this band tells apart - is
# a property of the SHAPES and not of the grid, and the module measured it
# stable from 512 to 16384 places; a power of two keeps the transform grid
# and this one commensurable.
_PICTURE_STAGE_PLACES = 4096


def measure_picture_stage(field, uphet, amount):
    """THE REFERENCES AND ALIGNMENTS THAT ACT AFTER THE RADIO-FREQUENCY
    MATRIX.

    Node `picture_stage`. Applies nothing.

    Ethan named four: *"sync pulse amplitude and phase; genlocking, which
    aligns the colour burst to hsync; burst locking, which aligns the hsyncs
    to the colour burst; and the time-base residual, which is the existing
    residual subtracted from a constant."*

    WHY THEY ARE A STAGE OF THEIR OWN AND NOT MORE ENTRIES IN THE
    RADIO-FREQUENCY KEY. Every mechanism in the magnetic and interference
    models acts on the carrier ahead of a NON-LINEAR demodulator, which is
    why the standing law forbids inverting them with a video-domain filter.
    These act on the demodulated line, on quantities the radio-frequency
    stage does not carry at all - the horizontal reference, the subcarrier's
    phase against it, and the line-to-line timing - and their abscissa is the
    video baseband rather than the carrier.

    HERE, ON THE CHROMA PATH, BECAUSE THIS IS THE ONLY PLACE THE MEASUREMENT
    CAN BE TAKEN. The burst's lock to the horizontal reference needs the
    up-heterodyned chroma and the sync-referred line origin at once, and
    those coexist only inside `decode_chroma` - the same argument
    `chroma_leakage` stands on. It runs BEFORE `cti`, which accelerates the
    sweep between colour states at exactly the transitions the burst window
    abuts.

    THREE AXES:

      AMPLITUDE   MEASURED. The burst window's complex lock and its LINE
                  COHERENCE - the length of the mean phasor against the mean
                  length - which says how much of the window is actually
                  locked rather than merely present.
      TIME        MEASURED. The burst's phase drift across the field, read as
                  a frequency, which is the reading that does NOT depend on
                  the picture: on this arc's three tapes the up-heterodyne
                  holds to a tenth of a hertz within a field. Beside it, the
                  precision the sync edge would have to be located to for
                  genlock and burst lock to be two dimensions rather than one
                  - a ninetieth of a 4fsc sample at one degree of burst
                  phase, which is 0.776 nanoseconds.
      FREQUENCY   DERIVED, and it is a property of the MODEL rather than of
                  the tape. `distinguishable` reports how many of the four
                  named modifications this decode's own baseband tells apart,
                  by the participation ratio of the singular values - the
                  same measure the tape magnetics' 1.58 of 6 was taken with.
                  It says what the decode COULD separate, not what it found.

    THE ACTIVE WINDOW IS READ AND IS REPORTED AS PICTURE, WHICH IS THE
    MODULE'S OWN FINDING RATHER THAN A LAPSE. `colour_lock` takes the same
    lock in the burst window and in the active area and the two disagree: on
    two recordings made by one machine within minutes of each other the
    burst-to-active difference is -42 degrees on colour bars and +78 on a
    chroma noise pattern, and a lock cannot depend on what was filmed. The
    active area's mean phasor is the vector sum of the hues in the window.
    Nothing here is corrected from it; it is reported so that the
    disagreement is visible on any decode rather than on the three it was
    measured on.

    MEASURED BY THIS STAGE, 2026-09-06, `--stages +picture_stage`, four
    fields each:

        decode              burst   active   difference  drift Hz   across
                            coh.    coh.     deg                    the line
        75 bars SP          0.9999  0.9646    -33.3      +0.273     -18724
        home                0.9998  0.9562    +46.9      +0.232       -416
        countdown           0.9987  0.8015    +73.6      -0.830      -7789
        multiburst y-only   0.9748  0.1082    +89.4      +0.499     +14653

    THE BURST WINDOW IS LOCKED ON EVERY CAPTURE AND THE ACTIVE ONE IS THE
    PICTURE, DEMONSTRATED FOUR WAYS. The burst holds 0.975 to 0.9999 while
    the burst-to-active difference takes four different values on four
    captures from two decks, and the y-only multiburst - which carries no
    chrominance in its picture at all - drops the active coherence to 0.108
    while its burst still locks at 0.975. A lock cannot depend on what was
    filmed, and this is the negative control for that statement rather than
    an argument for it.

    THE HETERODYNE HOLDS. The burst's phase drifts +0.23 to -0.83 Hz across a
    field, so the fixed-frequency assumption costs nothing these captures can
    see; the -18724 Hz the bars tape reads ACROSS the line is the hue rotation
    between the two halves of a bar pattern and reproduces the module's own
    offline -18749.7 +/- 11.0 Hz.

    THE GAUGE IT IS NOT FITTED WITH is where the sync edge actually lands.
    This stage states the precision that would make genlock and burst lock
    two dimensions - 0.776 nanoseconds, a ninetieth of a 4fsc sample - and
    says nothing about whether the decoder reaches it. Measured on the
    written time base corrected output, as the robust scatter of the
    interpolated half-amplitude crossing over 1680 to 1960 lines:

        75 bars SP           5.34 ns      6.9 times the requirement
        home                11.46 ns     14.8 times
        countdown           20.93 ns     27.0 times
        multiburst y-only   88.54 ns    114.1 times

    So on this decoder the two ARE one dimension, on every capture, by a
    factor of seven at best. The frequency axis says they cannot be told
    apart by a band and the sync's position would have to separate them; the
    gauge says the sync's position does not.
    """
    if not amount:
        return None
    state = _measurement_state(field, "picture_stage")
    if state["reported"]:
        return None

    geom = geometry(field)
    window = _burst_window(field)
    rows = _picture_rows(field, geom)
    expected = geom["lines"] * geom["width"]
    if window is None or rows is None or uphet is None \
            or len(uphet) != expected:
        _log_once(field, "picture_stage_declined",
                  "model stages: picture_stage declined - the field has no "
                  "burst window or no up-heterodyned chroma to read the lock "
                  "on")
        state["reported"] = True
        return None

    block = np.asarray(uphet, dtype=np.float64).reshape(geom["lines"],
                                                        geom["width"])
    active = _output_active_span(field, geom)
    if active is None:
        state["reported"] = True
        return None
    try:
        lock = picture_stage.colour_lock(
            block[rows[0]:rows[1]], geom["sample_rate_hz"], window, active,
            first_line=rows[0], subcarrier_hz=geom["subcarrier_hz"])
    except (ValueError, np.linalg.LinAlgError) as error:
        _log_once(field, "picture_stage_declined",
                  "model stages: picture_stage declined - %s", error)
        state["reported"] = True
        return None

    state["fields"] += 1
    store = state.setdefault("locks", [])
    store.append(lock)
    if len(store) < _COLOUR_FRAME_FIELDS:
        return lock
    state["reported"] = True

    # THE MODEL'S OWN AXIS, on the band this decode actually delivers: from
    # zero to the output's Nyquist, which is where the picture stage's
    # modifications live and where a caller would enter them.
    hertz = np.linspace(0.0, 0.5 * geom["sample_rate_hz"],
                        _PICTURE_STAGE_PLACES)
    try:
        separable = picture_stage.distinguishable(
            hertz, subcarrier_hz=geom["subcarrier_hz"])
        tie = picture_stage.alignment_tie(hertz,
                                          subcarrier_hz=geom["subcarrier_hz"],
                                          lines=rows[1] - rows[0])
    except (ValueError, np.linalg.LinAlgError, ImportError) as error:
        _log_once(field, "picture_stage_model",
                  "model stages: picture_stage could not evaluate its own "
                  "separability on this band (%s); the measured lock below "
                  "stands regardless", error)
        separable = tie = None

    burst_coherence = float(np.median([l["burst"]["line_coherence"]
                                       for l in store]))
    active_coherence = float(np.median([l["active"]["line_coherence"]
                                        for l in store]))
    difference = float(np.median([l["difference_deg"] for l in store]))
    drift = np.asarray([l["burst_drift_hz"] for l in store],
                       dtype=np.float64)
    drift = drift[np.isfinite(drift)]
    within = np.asarray([l["within_line_frequency_hz"] for l in store],
                        dtype=np.float64)
    within = within[np.isfinite(within)]

    reading = {
        "fields": len(store),
        "burst_line_coherence": burst_coherence,
        "active_line_coherence": active_coherence,
        "burst_to_active_deg": difference,
        "burst_drift_hz": float(np.median(drift)) if drift.size else
        float("nan"),
        "burst_drift_scatter_hz": float(drift.std()) if drift.size else
        float("nan"),
        "within_line_frequency_hz": (float(np.median(within)) if within.size
                                     else float("nan")),
        "separating_precision_s": float(
            picture_stage.separating_precision_s(
                1.0, geom["subcarrier_hz"])["required_sync_precision_s"]),
        "separating_precision_in_4fsc_samples": float(
            picture_stage.separating_precision_s(
                1.0, geom["subcarrier_hz"])["in_4fsc_samples"]),
    }
    if separable is not None:
        reading["effective_directions"] = float(separable["effective"])
        reading["entries"] = int(separable["count"])
        reading["condition"] = float(separable["condition"])
    if tie is not None and tie.get("full_band") is not None:
        reading["genlock_against_burst_lock"] = float(
            tie["full_band"]["effective"])
        reading["one_dimension_on_frequency"] = bool(
            tie["one_dimension_on_the_frequency_axis"])

    _log_once(field, "picture_stage",
              "model stages: picture_stage over %d fields - the burst window "
              "is locked to %.4f line coherence and the active window to "
              "%.4f, and the two disagree by %+.1f degrees. THE ACTIVE "
              "READING IS THE PICTURE'S, not the lock's: its mean phasor is "
              "the vector sum of the hues in the window, and nothing here is "
              "corrected from it",
              reading["fields"], burst_coherence, active_coherence,
              difference)
    _log_once(field, "picture_stage_time",
              "model stages: picture_stage time axis - the burst's phase "
              "drifts %+.3f +/- %.3f Hz across the field, so the "
              "up-heterodyne %s within a field; the active window reads "
              "%+.1f Hz across the line, which is the hue rotation and not a "
              "heterodyne error. Genlock and burst lock become two "
              "dimensions only if the sync edge is located to %.3f ns, a "
              "%.4f of a 4fsc sample",
              reading["burst_drift_hz"], reading["burst_drift_scatter_hz"],
              "holds to a fraction of a hertz"
              if abs(reading["burst_drift_hz"]) < 1.0 else "is drifting",
              reading["within_line_frequency_hz"],
              1e9 * reading["separating_precision_s"],
              reading["separating_precision_in_4fsc_samples"])
    if separable is not None:
        _log_once(field, "picture_stage_frequency",
                  "model stages: picture_stage frequency axis, DERIVED from "
                  "the specified shapes rather than measured on the tape - "
                  "this decode's baseband to %.2f MHz tells %.3f of %d "
                  "directions apart among the named modifications at "
                  "condition %.2f, "
                  "against the tape magnetics' 1.58 of 6 at 1.06e4. They are "
                  "a different KIND of family and not more members of a "
                  "collinear one. Genlock against burst lock is %.3f of 2 on "
                  "the same band, so on the frequency axis they are %s",
                  0.5 * geom["sample_rate_hz"] / 1e6,
                  reading["effective_directions"], reading["entries"],
                  reading["condition"],
                  reading.get("genlock_against_burst_lock", float("nan")),
                  "ONE direction and only the sync's own position separates "
                  "them" if reading.get("one_dimension_on_frequency")
                  else "separable")
    field.picture_stage = reading
    return reading


# ==========================================================================
# THE RADIO-FREQUENCY, CAPTURE AND TRANSPORT GROUP
#
# Eight stages that read the RADIO FREQUENCY - the signal as the capture card
# delivered it - rather than the demodulated picture. Ethan: *"Every stage
# that we have designed must be called and used ... All of them need to be
# used and all of them need to model all dimensions."* Each was designed,
# tested offline and never reachable from a decode; each is a NODE in
# `vhsdecode/pipeline/stages.toml`, defaults to ON through
# `declared_default = true`, and has no flag of its own - `--stages
# -band_delay` turns one off, and so on.
#
# NOT ONE OF THEM WRITES A SAMPLE, and that is a statement about the arc
# rather than about the wiring. Every one of these modules identifies
# something - a converter's word length, a filter's order, the two machines'
# separability, a mechanical rate, a differential delay, a record-side mark -
# and identification is what the whole group is for. Their findings go to the
# log and to attributes on the field; a decode with any of them enabled is
# byte identical to one without, which is what makes them safe to leave on.
#
# THE RAW RADIO FREQUENCY IS THE SUBJECT, and `field.rawdata` is where the
# decode holds it: the capture's own samples for this field's read span, on
# the demodulator's sample grid, before the notch, the band-pass, the
# equalizer or the analytic signal. Reading it here rather than in
# `demodblock` is deliberate on two counts - the field thread is serial, so a
# stage may accumulate across fields without a lock, and the field carries the
# line locations, so a reading may be confined to the reserved intervals.
#
# WHERE THE READING IS TAKEN, PER STAGE, because the sync-only law governs it:
#
#   `interference`      THE RESERVED PULSES ALONE. Its signatures live across
#                       the RF band, which is where the picture's own
#                       sidebands are, so the spectrum is taken inside the
#                       field-sync broad pulses where they exist and the
#                       line-sync pulses otherwise - during a pulse the
#                       modulator holds one frequency, so the carrier is a
#                       line to be excised and the floor beside it is the
#                       channel.
#   `capture_filter`    THE WHOLE FIELD, ABOVE THE SIGNAL BAND. Its subject is
#                       the noise floor above everything the format puts on
#                       the tape, and it is identified by the ABSENCE of
#                       signal there: the fitted band starts above the video
#                       band-pass's own upper corner, so no picture enters it
#                       by construction.
#   `capture_profile`   THE CODE LATTICE, which is a property of the converter
#                       and not of any interval.
#   `band_delay`        THE SYNC RISE AND THE BURST, both reserved.
#   `head_switch_pair`  THE BURST, once a line, and the vertical interval,
#                       over two spliced fields.
#   `filter_model`      THE POOLED SYNC PULSE, shared with the sync group.
#   `transport_model`   THE ENVELOPE AND THE LINE PERIODS, per line.
#   `rf_stages`         NOTHING MEASURED FROM THE SIGNAL - it is the two
#                       machines' identifiability on this decode's own band,
#                       and it says so.
#
# COMPLEX THROUGHOUT. `band_delay` carries the chroma band as its complex
# envelope and the burst's phase beside its centroid; `head_switch_pair`
# carries the per-line rotation as a complex number and projects it only at
# the last step; `interference` and `rf_stages` build complex signatures whose
# real part is what a mechanism does to amplitude and whose imaginary part is
# what it does to phase; `filter_model` reads the pulse's complex spectrum.
# The two places a magnitude IS taken are named where they happen and both are
# the subject's own definition rather than a reduction: a noise power spectrum
# has no phase to keep, and a code lattice is a set of levels.
#
# ONE NAME PER IMPORT LINE, for the same reason the sync group states: the
# stage inventory reads imports as text and credits only the names sharing a
# line with `models import`, so a wrapped tuple would leave most of this group
# reported as designed and never called - which is the exact gap it exists to
# close.
# ==========================================================================

from vhsdecode.models import band_delay
from vhsdecode.models import capture_filter
from vhsdecode.models import capture_profile
from vhsdecode.models import filter_model
from vhsdecode.models import head_switch_pair
from vhsdecode.models import interference
from vhsdecode.models import rf_stages
from vhsdecode.models import sync_geometry
from vhsdecode.models import transport_model

# The recorded track width per tape speed, in micrometres, exactly as
# `format_defs/vhs.py` writes it and `rf_stages.mechanics_at_speed` reads it
# back. The decoder stores the WIDTH rather than the speed it was given, so
# this is how the speed is recovered from a running decode - the alternative
# would be a second copy of the command line, and the two would drift.
#
# PER SYSTEM, AND PAL'S SLOW SPEEDS ARE DELIBERATELY ABSENT. NTSC's three
# widths are distinct, so the map is one to one. PAL records 49.0 at SP and
# 24.5 at BOTH of its slower speeds, so the decoder's own record cannot say
# which of the two a decode is running at - and a stage that needs the speed
# declines on a PAL long-play tape rather than guessing between them.
_TRACK_WIDTH_TO_SPEED = {
    "NTSC": {58.0: "SP", 29.0: "LP", 19.3: "EP"},
    "PAL": {49.0: "SP"},
}

# How many fields the group pools before it reports. One complete NTSC colour
# frame, the same span the chroma stages use, so a figure quoted by this group
# and one quoted by that one rest on the same number of fields.
_RF_POOL_FIELDS = _LOCK_DECISION_FIELDS


def _rf_state(field, key):
    """The per-decode store for one stage of this group.

    On `field.rf` because every one of these pools across fields, and keyed by
    name so eight stages do not share one dictionary. Separate from the chroma
    group's `_chroma_measurements` for the same reason.
    """
    return field.rf.__dict__.setdefault("_rf_measurements", {}).setdefault(
        key, {"fields": 0, "reported": False})


def rf_geometry(field):
    """Everything this group needs about the capture, derived from the decode.

    Nothing here is written down: the sample rate is the demodulator's own,
    the band is the decoder's own video band-pass, the carriers are the
    format's own sync-tip and peak-white frequencies as `DecoderParams` holds
    them, and the tape speed is read back from the recorded track width the
    format definition set from it.

    Returns None where the decode cannot supply one, which is the case for a
    format whose definition carries no video band-pass corners.
    """
    cached = field.rf.__dict__.get("_rf_geometry")
    if cached is not None:
        return cached or None
    field.rf.__dict__["_rf_geometry"] = False

    params = field.rf.DecoderParams
    rate = float(getattr(field.rf, "freq_hz", 0.0))
    low = params.get("video_bpf_low")
    high = params.get("video_bpf_high")
    if not rate > 0 or low is None or high is None:
        return None
    system = getattr(field.rf, "color_system", None)
    system = "NTSC" if system in ("NTSC", "PAL_M") else system
    width = params.get("video_track_width")
    speed = (_TRACK_WIDTH_TO_SPEED.get(system, {}).get(float(width))
             if width else None)
    geometry = {
        "sample_rate_hz": rate,
        "nyquist_hz": 0.5 * rate,
        "band_hz": (float(low), float(high)),
        "sync_tip_hz": float(field.rf.iretohz(params["vsync_ire"])),
        "blanking_hz": float(field.rf.iretohz(0)),
        "white_hz": float(field.rf.iretohz(100)),
        "carrier_hz": 0.5 * (float(field.rf.iretohz(params["vsync_ire"]))
                             + float(field.rf.iretohz(100))),
        "colour_under_hz": float(params["color_under_carrier"]),
        "hz_ire": float(params["hz_ire"]),
        "system": system,
        "tape_speed": speed,
        "track_width_um": float(width) if width else None,
        "block_len": int(getattr(field.rf, "blocklen", 32768)),
        "line_rate_hz": (1e6 / float(field.rf.SysParams["line_period"])
                         if field.rf.SysParams.get("line_period") else None),
        "field_rate_hz": (2e6 / (float(field.rf.SysParams["line_period"])
                                 * float(field.rf.SysParams["frame_lines"]))
                          if field.rf.SysParams.get("line_period") else None),
    }
    field.rf.__dict__["_rf_geometry"] = geometry
    return geometry


def _shared_rf_analytics(field, raw, sample_rate_hz, system="NTSC"):
    """The two full-field transforms, computed once a field and shared.

    Ethan, 2026-09-07: *"I believe the hypercomplex stages and the folding
    can be consolidated through some mathmetical simplification."*

    THIS IS THE CONSOLIDATION, and it is a sharing rather than an
    approximation. `band_delay` and `head_switch_pair` read the SAME
    samples of the SAME field and each computed the luma band's
    instantaneous frequency and the chroma band's complex envelope
    independently. Measured on one field of 667334 samples at 40 MSps, the
    band split costs 35.3 ms and the analytic signal 108.1 ms, so the
    second caller was paying 143.4 ms a field for arrays the first already
    held.

    Cached on the FIELD and keyed by the raw array's size and its own
    identity, so a field the decoder redoes recomputes rather than
    silently reusing another field's transforms - which is the failure the
    key exists to prevent, and it is worth the two comparisons.
    """
    if raw is None:
        return None
    store = field.__dict__.setdefault("_shared_rf_analytics", {})
    key = (int(raw.size), float(sample_rate_hz), system, id(raw))
    if store.get("key") != key:
        from vhsdecode.models import band_delay as _band_delay
        store["key"] = key
        store["value"] = _band_delay.analytics(raw, sample_rate_hz, system)
    return store["value"]


def _raw_field(field):
    """This field's raw radio frequency, as float, or None.

    A converted copy: the stages below filter and transform it and must not
    write into the block cache the decoder is still reading. Double precision
    because two of them run zero-phase filters and a transform over most of a
    million samples. The one stage that needs the INTEGER lattice -
    `capture_profile` - reads `field.rawdata` itself, because that lattice is
    exactly what a conversion to float would erase.
    """
    raw = getattr(field, "rawdata", None)
    if raw is None:
        return None
    values = np.asarray(raw)
    if values.ndim != 1 or values.size < 1024:
        return None
    return values.astype(np.float64)


def _mechanics(geometry):
    """The head-to-tape mechanics at this decode's own tape speed, or None
    where the format definition gives no width this group can read a speed
    from - a decode of a format `rf_stages` has no mechanics for declines
    rather than borrowing VHS's."""
    speed = geometry.get("tape_speed")
    if speed is None or geometry.get("system") not in ("NTSC", "PAL"):
        return None
    try:
        return rf_stages.mechanics_at_speed(speed, geometry["system"])
    except (ValueError, KeyError):
        return None


# The fraction of a reserved pulse that is used as a window. The pulse's own
# transitions are excluded at both ends, and `sync_depth.TIP_INTERIOR` is the
# fraction the depth stage already reads its level on, so the two stages read
# the same part of the same pulse rather than two nearby parts.
_PULSE_INTERIOR = sync_depth.TIP_INTERIOR


def _reserved_carrier_windows(field, geometry):
    """The raw radio frequency inside the longest reserved pulses available.

    THE PART OF THE RF BAND THAT CARRIES NO PICTURE. During a synchronizing
    pulse the carrier sits at the frequency the standard specifies for the
    synchronizing level and nothing modulates it, so what is in the band
    beside that one line is the channel's own noise shaped by the channel -
    the same premise `capture_filter` uses above the video band, applied
    inside it. The sync-only law is met by construction rather than by
    argument.

    THE LENGTH OF THE PULSE IS THE RESOLUTION, WHICH IS THE 1/T PROBE LAW.
    A line-sync pulse is 4.70 microseconds and resolves 213 kHz; a
    FIELD-sync broad pulse is 27.1 (`sync_geometry.FIELD_SYNC_PULSE_US`) and
    resolves 37, six times finer, and there are six of them in every field's
    vertical interval. So the broad pulses are preferred and the line pulses
    are the fallback, and which was used is reported with the reading -
    a spectrum's resolution is not a detail when the thing being measured is
    a shape across frequency.

    Returns the windows, the name of the pulse they came from, and None where
    the decode can supply neither.
    """
    raw = _raw_field(field)
    if raw is None:
        return None
    system = "525" if geometry.get("system") == "NTSC" else "625"
    try:
        sync_us = sync_geometry.LINE_SYNC_US[system]
        broad_us = sync_geometry.FIELD_SYNC_PULSE_US[system]
    except (KeyError, AttributeError):
        return None
    rate = geometry["sample_rate_hz"]
    low, high = _PULSE_INTERIOR

    def interior(starts, pulse_us):
        span = pulse_us * 1e-6 * rate
        first = int(round(low * span))
        length = int(round(high * span)) - first
        if length < 64:
            return None
        begin = np.rint(np.asarray(starts, dtype=np.float64)).astype(np.intp)
        begin = begin + first
        keep = (begin >= 0) & (begin + length < raw.size)
        begin = begin[keep]
        if begin.size < 3:
            return None
        return raw[begin[:, None] + np.arange(length)[None, :]]

    # THE FIELD-SYNC BROAD PULSES, found in the decoder's own demodulated
    # channel rather than assumed at a line number. A run below the midpoint
    # carrier that is more than twice a line pulse long is field sync and
    # nothing else, which is the same test `head_switch_pair.vertical_sync`
    # makes - taken here against the DECODE'S own two levels rather than the
    # specification's carrier law, so a decode whose ire0 has moved still
    # finds them.
    video = getattr(field, "data", {}).get("video") if hasattr(field, "data") \
        else None
    if video is not None and "demod" in video:
        demod = np.asarray(video["demod"], dtype=np.float64)
        threshold = 0.5 * (geometry["sync_tip_hz"] + geometry["blanking_hz"])
        below = demod < threshold
        change = np.diff(below.astype(np.int8))
        starts = np.flatnonzero(change == 1) + 1
        stops = np.flatnonzero(change == -1) + 1
        if stops.size and starts.size and stops[0] < starts[0]:
            stops = stops[1:]
        count = min(starts.size, stops.size)
        starts, stops = starts[:count], stops[:count]
        if count:
            width_us = (stops - starts) / rate * 1e6
            broad = starts[width_us > 2.0 * sync_us]
            windows = interior(broad, broad_us) if broad.size else None
            if windows is not None:
                return windows, "field-sync broad pulse", broad_us

    locations = getattr(field, "linelocs", None)
    if locations is None:
        return None
    values = np.asarray(locations, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 16:
        return None
    windows = interior(values[:-1], sync_us)
    if windows is None:
        return None
    return windows, "line-sync pulse", sync_us


def _welch_db(values, rate_hz, segment):
    """A pooled power spectrum in decibels, and the frequencies it sits on.

    A MAGNITUDE IS TAKEN HERE AND IT IS THE SUBJECT'S OWN DEFINITION. A noise
    power spectral density has no phase - the phase of noise is what makes it
    noise - so this is not the reduction at a boundary the arc forbids. Every
    stage below that needs a phase reads the waveform, not this.
    """
    from scipy import signal as _signal

    data = np.asarray(values, dtype=np.float64)
    nperseg = min(int(segment), data.size)
    if nperseg < 64:
        return None, None
    frequency, power = _signal.welch(data, fs=float(rate_hz), nperseg=nperseg,
                                     noverlap=nperseg // 2)
    return frequency, 10.0 * np.log10(np.maximum(power, 1e-300))


# --------------------------------------------------------------------------
# capture_profile - the converter's own limit, read from the capture
# --------------------------------------------------------------------------

def measure_capture_profile(field):
    """THE FOURTH LINK IN THE CHAIN, AND THE ONE THAT TERMINATES THE REST.

    Node `capture_profile`. Applies nothing.

    Ethan named the chain as four links and the component model carried
    three: *"Capture chain: VCR -> capture card; capture card profile ->
    complete limits of the sample rate i.e. 8bits, and 40mhz"*. The module
    reads that profile from the samples rather than asserting it, and this is
    it running on the capture the decode is actually reading - which is the
    whole point, because the three constants it replaced (eight bits, forty
    megahertz, a 13.3 MHz roll-off) were wrong on every tape in this arc, the
    test-pattern set being fifty.

    THE LATTICE, NOT THE CONTAINER, and the integer samples are what carry
    it. An eight-bit capture left-shifted into a sixteen-bit container is
    still an eight-bit capture: the greatest common divisor of the differences
    between the distinct codes present is the quantiser's step whatever the
    container, and reading the container's word length instead reports twice
    the depth. This is why the stage reads `field.rawdata` directly and not
    the converted copy every other stage in this group takes.

    ALL THREE AXES, AND THE THIRD IS A DRIFT RATHER THAN A CONSTANT.
    AMPLITUDE is the code lattice - the step, the occupancy of the converter's
    full scale, and the quantisation floor that follows from them, which no
    residual anywhere in this arc may claim to sit below. FREQUENCY is the
    Carson band the format occupies and the Shannon capacity across it, which
    is what decides whether the converter or the tape is the binding limit.
    TIME is the per-field series of the same two amplitude readings: the
    converter's operating point is not guaranteed constant down a tape, and a
    drift in the occupancy is a drift in every level derived from it. Pooled
    over the decode and reported as a spread, so a moving front end is
    visible rather than averaged away.

    MEASURED by this stage, 2026-09-06, on three decodes taken here. Read
    through `--no_resample` at the capture's own 50 MSps the bars SP tape is
    8 bits on a step of 256, 45.8 per cent of full scale at 24.5 codes rms;
    the home tape, natively 40 MSps, is 8 bits on the same step at 81.6 per
    cent; and the same bars capture with `-f 50` and the loader resampling it
    to 40 reads 16 bits on a step of 1 with 26825 distinct codes. The third
    is the reason for the refusal below. The full figures, and the binding
    limit each of them gives, are in this stage's node in
    `pipeline/stages.toml`.
    """
    geometry = rf_geometry(field)
    raw = getattr(field, "rawdata", None)
    if geometry is None or raw is None:
        return None
    samples = np.asarray(raw)
    if samples.ndim != 1 or samples.size < 4096:
        return None
    state = _rf_state(field, "capture_profile")

    profile = capture_profile.profile_of(samples, geometry["sample_rate_hz"])
    if not profile.get("bits", 0) > 0:
        _log_once(field, "capture_profile_declined",
                  "model stages: capture_profile declined - this field's "
                  "samples land on no readable lattice, so the converter's "
                  "word length cannot be measured and is not guessed")
        return None

    # THE TAPE'S OWN CARRIER-TO-NOISE, from the envelope's spread within the
    # field. An FM carrier is nominally constant in amplitude, so what the
    # envelope does between samples is the channel's noise and nothing else -
    # the module's own definition, fed here from the decoder's envelope
    # rather than from a second one taken here.
    envelope = None
    video = getattr(field, "data", {}).get("video") if hasattr(field, "data") \
        else None
    if video is not None and "envelope" in video:
        envelope = np.asarray(video["envelope"], dtype=np.float64)
    spread_db = float("nan")
    if envelope is not None and envelope.size > 16:
        centre = float(np.median(envelope))
        if centre > 0:
            spread_db = float(20.0 * np.log10(
                1.0 + np.std(envelope) / centre))

    binding = None
    if np.isfinite(spread_db) and spread_db > 0:
        deviation = 0.5 * (geometry["white_hz"] - geometry["sync_tip_hz"])
        baseband = float(field.rf.DecoderParams.get("video_lpf_freq",
                                                    3.0e6))
        binding = capture_profile.binding_limit(
            spread_db, profile["signal_rms_codes"], profile,
            deviation_hz=deviation, baseband_hz=baseband)

    history = state.setdefault("history", {"occupancy": [], "rms_codes": [],
                                           "bits": []})
    history["occupancy"].append(float(profile["occupancy"]))
    history["rms_codes"].append(float(profile["signal_rms_codes"]))
    history["bits"].append(float(profile["bits"]))
    state["fields"] += 1

    reading = {"profile": profile, "binding": binding,
               "envelope_spread_db": spread_db}
    ldd.logger.debug(
        "capture_profile: %.0f bits, step %.0f, %.1f%% of full scale, "
        "quantisation floor %.2f codes rms, envelope spread %.3f dB",
        profile["bits"], profile["step"], 100.0 * profile["occupancy"],
        profile["rms"] / max(profile["step"], 1e-12), spread_db)

    if state["fields"] >= _RF_POOL_FIELDS and not state["reported"]:
        state["reported"] = True
        occupancy = np.asarray(history["occupancy"], dtype=np.float64)
        codes = np.asarray(history["rms_codes"], dtype=np.float64)
        depths = np.asarray(history["bits"], dtype=np.float64)
        _log_once(
            field, "capture_profile",
            "model stages: capture_profile over %d fields - %.0f bits on a "
            "step of %.0f in a %.0f-code full scale at %.3f MSps, the signal "
            "occupying %.1f%% of it at %.1f codes rms; quantisation floor "
            "%.3f codes rms (%.2f dB signal to noise), and the capture's "
            "budget is %.1f Mbit/s. The word length is the same on all %d "
            "fields (spread %.3f bits) and the occupancy drifts by %.2f "
            "per cent of full scale across them, which is the time axis a "
            "single reading of this profile cannot show",
            state["fields"], profile["bits"], profile["step"],
            profile["full_scale"], geometry["sample_rate_hz"] / 1e6,
            100.0 * float(occupancy.mean()), float(codes.mean()),
            profile["rms"] / max(profile["step"], 1e-12),
            profile["signal_to_noise_db"],
            profile["budget_bits_per_s"] / 1e6, state["fields"],
            float(depths.max() - depths.min()),
            100.0 * float(occupancy.max() - occupancy.min()))
        # WHOSE LATTICE IS THIS? The decoder resamples whenever the capture's
        # own rate is not its working rate - `-f 50` on a fifty-megasample
        # capture is resampled to forty before a single sample reaches a
        # field - and a resampled stream lands on the CONTAINER's lattice
        # rather than on the converter's. Measured on the same 8-bit capture:
        # read through `--no_resample` this stage finds 8 bits on a step of
        # 256 with 112 codes present, and read with the resampler in the path
        # it finds 16 bits on a step of 1 with tens of thousands. The lattice
        # cannot tell a genuinely wider converter from a resampled narrow one,
        # so the stage says which of the two it is looking at by the one thing
        # that does distinguish them - a converter cannot produce more codes
        # than its own word length allows, and a resampler produces as many as
        # the container has.
        present = float(profile["codes"])
        span = float(profile["full_scale"]) / max(profile["step"], 1e-12)
        if profile["step"] <= 1.0 and present > 0.05 * span:
            _log_once(
                field, "capture_profile_lattice",
                "model stages: capture_profile DECLINES to call this the "
                "capture card's word length. The samples land on a step of "
                "1 with %.0f distinct codes of the container's %.0f, which "
                "is what a stream RESAMPLED between the card and the decoder "
                "looks like and is also what a genuinely %.0f-bit converter "
                "looks like; the lattice cannot separate the two. Either "
                "way the floor it implies sits %.1f dB below the channel's "
                "own carrier-to-noise, so it is not the limit on this "
                "decode. Re-run with --no_resample to read the card's own "
                "lattice",
                present, span, profile["bits"],
                (binding["capture_snr_db"] - binding["tape_snr_db"])
                if binding is not None else float("nan"))
        if binding is not None:
            _log_once(
                field, "capture_profile_binding",
                "model stages: capture_profile - over the %.3f MHz Carson "
                "band the tape carries %.1f dB and the converter %.1f dB, "
                "so the capacities are %.2f and %.2f Mbit/s and the %s is "
                "the binding limit by a factor of %.2f",
                binding["band_hz"] / 1e6, binding["tape_snr_db"],
                binding["capture_snr_db"],
                binding["tape_capacity_bits_per_s"] / 1e6,
                binding["capture_capacity_bits_per_s"] / 1e6,
                binding["binds"], binding["ratio"])
        else:
            _log_once(
                field, "capture_profile_binding_declined",
                "model stages: capture_profile declines the binding-limit "
                "comparison - the envelope this field carries gives no "
                "usable carrier-to-noise, so which link binds is not stated "
                "rather than assumed")
    field.capture_profile = reading
    return reading


# --------------------------------------------------------------------------
# capture_filter - the analogue low-pass between the RF tap and the converter
# --------------------------------------------------------------------------

# The spur rejection: a running median over this many spectral bins. Spurs are
# NARROW by definition - `capture_chain_noise` measures the capture chain's own
# as single lines at the crystal's sub-harmonics and at the computer's 6.000
# and 12.000 MHz - and a filter's roll-off is broad, so a median over a window
# many bins wide keeps the roll-off and drops the lines. The width is stated as
# a span in hertz rather than in bins so it follows the transform length.
_SPUR_MEDIAN_HZ = 200e3

# The fit's grid, decimated to this many points. The identification searches a
# fixed grid of orders, corners and converter floors, so its cost is set by
# that search and not by the number of points; decimating costs nothing in the
# fit and keeps the stage from spending twenty seconds on five thousand
# strongly correlated bins. Measured on the bars SP capture: 5735 points take
# 18.7 seconds and 250 take 7.1, and the two fits agree to 0.1 dB rms.
_FILTER_FIT_POINTS = 250


def measure_capture_filter(field):
    """THE ANALOGUE LOW-PASS AFTER THE RF TAP, FITTED TO THE NOISE FLOOR.

    Node `capture_filter`. Applies nothing.

    Ethan: *"I also have a low pass filter just after the RF tap on my VCR.
    Model that based on the noise profile of the capture, where it rolls off
    consistent with an analog low pass filter. There may be many orders in
    the filter. The comparison component is the nyquist roll off."*

    THE NOISE IS THE PROBE, so this stage needs no stimulus and takes no
    picture. Above the video band-pass's own upper corner the capture carries
    nothing the format put there, and the noise entering the filter is
    broadband, so the measured spectrum up there IS the filter's squared
    magnitude scaled by a constant. The fitted band therefore starts above
    that corner and the sync-only law is met by construction: there is no
    picture in the region the fit reads.

    POOLED ACROSS FIELDS AND FITTED ONCE. A single field's periodogram is one
    realisation of a random process, and the module fits four parameters to
    its shape; pooling a colour frame's worth of fields before fitting is what
    makes the shape the channel's rather than the draw's. The fit is then run
    once and latched.

    TWO ADDITIVE TERMS AND NOT ONE, which is the module's central finding and
    the reason its `identify` fits the converter's own floor as a free
    parameter: a flat floor beneath a falling one is noise ADDED after the
    filter, so it is the frequency at which the filter stops being observable
    and fixing it would smuggle in the answer.

    THE AXES, AND ONE OF THEM IS DERIVED. AMPLITUDE is the two levels in
    decibels - what the filter passes and what the converter adds beneath it.
    FREQUENCY is the corner and the order, read off a slope that a pole
    cascade fixes at six decibels an octave per pole. TIME IS DERIVED AND NOT
    MEASURED, and the reason is the subject: a noise power spectrum has no
    phase, so nothing in this measurement can witness a delay. What the stage
    carries on the time axis instead is the sampler's own APERTURE - a
    sample-and-hold of one sample period, whose response is `sinc(f/fs)` with
    no parameters at all - which is Ethan's comparison component and is a
    time-domain quantity known exactly rather than fitted. The fitted
    cascade's group delay follows from its magnitude by Bode's relation and
    is not an independent reading; it is reported as such and never as a
    measurement.

    MEASURED by this stage, 2026-09-06, on three decodes. At its own 50 MSps
    the bars SP capture fits 4.0 poles at a 14.783 MHz corner leaving 3.31 dB
    rms of a 8.61 dB fall; the home tape at its native 40 MSps fits 13.0 poles
    at 14.019 MHz leaving 0.96 dB of 22.06; and the same bars capture
    resampled to 40 MSps fits 7.5 poles at 15.481 MHz - which is the
    resampler and not the VCR, and is why the refusal below exists. Full
    figures in this stage's node in `pipeline/stages.toml`.
    """
    geometry = rf_geometry(field)
    raw = _raw_field(field)
    if geometry is None or raw is None:
        return None
    state = _rf_state(field, "capture_filter")
    if state.get("latched"):
        return state.get("result")

    rate = geometry["sample_rate_hz"]
    segment = min(16384, 1 << int(math.floor(math.log2(max(raw.size, 2)))))
    frequency, decibels = _welch_db(raw, rate, segment)
    if frequency is None:
        return None
    store = state.setdefault("power", None)
    linear = 10.0 ** (decibels / 10.0)
    state["power"] = linear if store is None else store + linear
    state["fields"] += 1
    state["frequency_hz"] = frequency
    if state["fields"] < _RF_POOL_FIELDS:
        return None

    pooled = 10.0 * np.log10(np.maximum(state["power"] / state["fields"],
                                        1e-300))
    pooled = pooled - float(pooled.max())
    # THE SPURS ARE REJECTED AND NOT FITTED. `capture_chain_noise` measures
    # this capture chain's own lines - the converter clock's sub-harmonics,
    # which move when the crystal is changed, and two at exactly 6.000000 and
    # 12.000000 MHz which do not, being the computer - and a four-parameter
    # roll-off fitted through them reports the lines rather than the filter.
    from scipy.ndimage import median_filter

    step_hz = float(frequency[1] - frequency[0]) if frequency.size > 1 else 0.0
    window = max(3, int(round(_SPUR_MEDIAN_HZ / max(step_hz, 1e-9))) | 1)
    cleaned = median_filter(pooled, size=window, mode="nearest")

    low = float(geometry["band_hz"][1]) * 1.05
    high = 0.98 * geometry["nyquist_hz"]
    if not high > low * 1.2:
        _log_once(field, "capture_filter_declined",
                  "model stages: capture_filter declined - this capture's "
                  "%.3f MHz Nyquist leaves no span above the %.3f MHz video "
                  "band for the filter to be seen in, so the roll-off is "
                  "not identifiable here and is not fitted",
                  geometry["nyquist_hz"] / 1e6, geometry["band_hz"][1] / 1e6)
        state["latched"] = True
        return None
    use = (frequency >= low) & (frequency <= high)
    grid, level = frequency[use], cleaned[use]
    if grid.size > _FILTER_FIT_POINTS:
        stride = int(math.ceil(grid.size / _FILTER_FIT_POINTS))
        grid, level = grid[::stride], level[::stride]
    if grid.size < 16:
        state["latched"] = True
        return None

    fit = capture_filter.identify(grid, level, rate, band=(low, high))
    state["latched"] = True
    if not fit.get("fitted"):
        _log_once(field, "capture_filter_unfitted",
                  "model stages: capture_filter declined - %s",
                  fit.get("why", "no candidate fitted"))
        return None

    fold = capture_filter.fold_suppression(rate, fit["corner_hz"],
                                           fit["order"], fit["converter_db"],
                                           fit["filtered_db"])
    # WHAT THE FIT HAS TO EXPLAIN, so its residual can be read against
    # something rather than quoted alone: the total fall across the fitted
    # band. A residual that is a large fraction of it means the two-term
    # model does not describe this capture, which is a finding and not a
    # failure to report.
    fall_db = float(level[:8].mean() - level[-8:].mean())
    credible = abs(fit["rms_db"]) < 0.5 * abs(fall_db)
    result = {"fit": fit, "fold": fold, "fall_db": fall_db,
              "credible": credible, "fields": state["fields"],
              "frequency_hz": grid, "measured_db": level}
    state["result"] = result

    _log_once(
        field, "capture_filter",
        "model stages: capture_filter over %d fields, %.3f to %.3f MHz - "
        "%.1f poles at a %.3f MHz corner, %.1f dB an octave, passing noise "
        "at %.1f dB over a converter floor at %.1f dB, observable to %.3f "
        "MHz; the fit's residual is %.2f dB rms against a %.2f dB fall, so "
        "the two-term model %s this capture",
        state["fields"], low / 1e6, high / 1e6, fit["poles"],
        fit["corner_hz"] / 1e6, fit["db_per_octave"], fit["filtered_db"],
        fit["converter_db"], fit["observable_to_hz"] / 1e6, fit["rms_db"],
        fall_db, "describes" if credible else "does NOT describe")
    _log_once(
        field, "capture_filter_aperture",
        "model stages: capture_filter - the comparison component is the "
        "sampler's own aperture, one sample period wide and therefore "
        "%.2f dB at Nyquist with no parameters at all, against the fitted "
        "filter's %.2f dB there; the aperture is a TIME-domain quantity "
        "known exactly and the fitted cascade's own delay follows from its "
        "magnitude by Bode's relation rather than being measured, because "
        "a noise power spectrum carries no phase. Aliasing: the fold this "
        "filter passes sits %.1f dB %s the converter's floor",
        fit["aperture_db_at_nyquist"], fit["filter_db_at_nyquist"],
        abs(float(fold["margin_db"])),
        "below" if fold["below_the_floor"] else "ABOVE")
    # WHOSE ROLL-OFF IS THIS? The same question `capture_profile` asks of the
    # lattice, and the lattice is what answers it here too. The decoder
    # resamples whenever the capture's rate is not its working rate, and a
    # resampler is itself a low-pass at the new Nyquist - so a fit taken on a
    # resampled stream reads the resampler's corner and not the VCR's. A
    # resampled stream also leaves the converter's lattice, which is a test
    # this stage can make on the same samples it just fitted.
    #
    # Measured on the bars SP capture, both ways: resampled from 50 to 40
    # MSps it reads 7.5 poles at a 15.481 MHz corner with a 1.42 dB residual,
    # and read at its own 50 MSps through `--no_resample` it reads 4.0 poles
    # at 14.783 with 3.31 dB. Two different measurements, and only the second
    # is the VCR's. On the home capture, which is 40 MSps natively and is
    # therefore not resampled at all, the lattice stays on its 256-code step
    # and this caveat does not fire.
    lattice = capture_profile.measure_bit_depth(np.asarray(field.rawdata))
    dense = (lattice.get("step", 0.0) <= 1.0
             and lattice.get("codes", 0.0) > 0.05 * 2.0 ** lattice["bits"])
    if dense:
        _log_once(
            field, "capture_filter_resample",
            "model stages: capture_filter - these samples are NOT on a "
            "converter's own lattice (step %.0f, %.0f distinct codes), which "
            "is what a stream resampled between the card and the decoder "
            "looks like. A resampler is a low-pass at the new Nyquist, so "
            "what is fitted above may be the resampler's corner rather than "
            "the analogue filter's; `--no_resample` is what makes this "
            "reading the VCR's",
            lattice["step"], lattice["codes"])
    if not credible:
        _log_once(
            field, "capture_filter_credibility",
            "model stages: capture_filter DECLINES to call this corner "
            "identified. A single pole cascade over a flat converter floor "
            "leaves %.2f dB rms unexplained of a %.2f dB fall, so the "
            "reported order and corner are what that model comes to on this "
            "capture and not a measurement of the analogue filter",
            fit["rms_db"], fall_db)
    field.capture_filter = result
    return result


# --------------------------------------------------------------------------
# rf_stages - the two machines, and whether they can be told apart at all
# --------------------------------------------------------------------------

# How many points the identifiability grids carry. `rf_stages.BASELINE_POINTS`
# is the module's own figure and is used rather than a second one, so a count
# quoted by a decode and one quoted by the module's tests are the same count.
_RF_STAGE_POINTS = rf_stages.BASELINE_POINTS


def measure_rf_stages(field):
    """THE RECORDING MACHINE AND THE PLAYING MACHINE, AS SEPARATE COMPONENTS -
    AND WHETHER THIS DECODE'S BAND CAN TELL THEM APART.

    Node `rf_stages`. Applies nothing.

    Ethan named the chain as four links, two of them radio frequency: *"VCR
    chain: VCR -> Tape; Tape -> VCR"*. They are two DIFFERENT MACHINES - the
    recorder has its own emphasis, amplifier, head and record current, the
    player its own head, preamplifier, equalisation and de-emphasis - and a
    single measured response is their product. The module builds them apart
    and then measures whether taking them apart was possible, and the answer
    is no.

    WHY THIS RUNS ON A DECODE AT ALL, GIVEN THAT ANSWER. Because the answer
    is a property of the BAND and of the tape speed, and both of those are
    properties of the decode rather than of the module. The grid this runs on
    is the decoder's own video band-pass; the mechanics are those of the tape
    speed the decode was given, read back from the recorded track width the
    format definition set from it. Every decode gets its own figure, and a
    capture with a wider band would get a different one - the band limit at
    either end of the chain is the only one of the four collinearities that is
    a limit rather than an identity, and it is the only one a wider band
    would break.

    THE CONTROL RUNS FIRST AND THE STAGE DECLINES IF IT FAILS. A construction
    that manufactures two directions where the physics has one would report
    the two machines as separable BECAUSE the construction split them, and an
    earlier component in this arc reported 4.65 of 5 for exactly that reason.
    `cascade_control` asserts two things known independently: one mechanism
    entered on both sides spans one direction, and the specified emphasis
    round trip is unity to machine precision. If either fails the count is not
    reported, because it would not mean anything.

    ALL THREE AXES, AND THE TIME AXIS IS THE MODULE'S OWN FINDING RATHER THAN
    AN OMISSION. AMPLITUDE is each mechanism's log magnitude. FREQUENCY is the
    abscissa itself, and there are TWO of them - the RF one the magnetics and
    the amplifiers live on, and the baseband one the emphases live on - which
    are counted separately and never pooled, because an inner product across
    two abscissae is memory layout rather than coherence. TIME is carried as
    phase, complex, through `minimum_phase_of`, and the finding is that it
    buys one direction for the PAIR - the total delay - and none for either
    machine, because a record delay and a playback delay are only ever
    observed as their sum. That is measured, not assumed: stacking the phase
    beneath the log magnitude leaves the participation ratio unchanged.

    MEASURED by this stage, 2026-09-06, on the bars SP and home decodes -
    which share a band and a tape speed and therefore give the same figures
    to the digit. Over the decoder's own 0.500 to 6.500 MHz at SP the
    magnetics alone span 1.566 of 6 directions, the record stage takes that
    to 2.008 of 9, the playback stage to 2.104 of 9, and both together to
    2.356 of 12 at a condition of 2.13e5. The control passes at 1.0000
    directions and 4.44e-16 nepers. Full figures, and the four
    collinearities, in this stage's node in `pipeline/stages.toml`.
    """
    geometry = rf_geometry(field)
    if geometry is None:
        return None
    state = _rf_state(field, "rf_stages")
    if state.get("latched"):
        return state.get("result")
    state["latched"] = True

    mechanics = _mechanics(geometry)
    if mechanics is None:
        _log_once(field, "rf_stages_declined",
                  "model stages: rf_stages declined - this tree states no "
                  "head-to-tape mechanics for %r at a %s track width, so "
                  "the two machines' signatures cannot be built and are not "
                  "borrowed from another format",
                  geometry.get("system"), geometry.get("track_width_um"))
        return None

    low, high = geometry["band_hz"]
    rf_hz = np.linspace(low, high, _RF_STAGE_POINTS)
    baseband = np.linspace(50e3,
                           float(field.rf.DecoderParams.get("video_lpf_freq",
                                                            3.0e6)),
                           _RF_STAGE_POINTS)

    control = rf_stages.cascade_control(rf_hz, baseband)
    if not control["passes"]:
        _log_once(
            field, "rf_stages_control",
            "model stages: rf_stages DECLINES - its own control fails on "
            "this band. One mechanism entered on both sides spans %.4f "
            "directions where it must span exactly 1, and the specified "
            "emphasis round trip comes to %.3g nepers where it must be "
            "zero. A separability count taken through a construction that "
            "fails this would be reporting the construction",
            control["effective"], control["round_trip_nepers"])
        return None

    counts = rf_stages.distinguishable(rf_hz, mechanics)
    baseband_counts = rf_stages.baseband_distinguishable(
        baseband, mechanics["tape_speed"])
    null = rf_stages.null_space(rf_hz, baseband)
    result = {"control": control, "rf": counts, "baseband": baseband_counts,
              "null_space": null, "mechanics": mechanics,
              "band_hz": (low, high)}
    state["result"] = result

    _log_once(
        field, "rf_stages",
        "model stages: rf_stages on this decode's own %.3f to %.3f MHz at "
        "%s (writing speed %.4f m/s, track width %.1f um) - the magnetics "
        "alone span %.3f of %d directions; the record stage takes that to "
        "%.3f of %d, the playback stage to %.3f of %d, and both together to "
        "%.3f of %d at a condition of %.3g. Six more members buy %.3f of a "
        "direction, so the two machines are the same collinear family seen "
        "twice",
        low / 1e6, high / 1e6, mechanics["tape_speed"],
        mechanics["writing_speed_m_s"],
        1e6 * mechanics["track_width_m"],
        counts["baseline"]["effective"], counts["baseline"]["count"],
        counts["with_record"]["effective"], counts["with_record"]["count"],
        counts["with_playback"]["effective"], counts["with_playback"]["count"],
        counts["with_both"]["effective"], counts["with_both"]["count"],
        counts["with_both"]["condition"], counts["gain_over_the_baseline"])
    _log_once(
        field, "rf_stages_null",
        "model stages: rf_stages - what NEITHER machine owns, measured on "
        "this band: the emphasis pair is one direction (coherence %.6f, "
        "residual %.3g nepers), the two wavelength losses compose in the "
        "exponent so only their sum survives (coherence %.9f), a flat gain "
        "splits with residual %.3g and a pure delay with %.3g radians, and "
        "the band limit at either end reads %.6f coherent - the only one of "
        "the four that is a limit rather than an identity, and the only one "
        "a wider capture would break. On the baseband abscissa the emphases "
        "span %.3f of %d at a condition of %.3g",
        null["emphasis_pair_coherence"], null["emphasis_pair_residual_nepers"],
        null["wavelength_loss_coherence"],
        null["flat_gain_split_residual_nepers"],
        null["pure_delay_split_residual_radians"],
        null["band_limit_coherence"],
        baseband_counts["effective"], baseband_counts["count"],
        baseband_counts["condition"])
    field.rf_stages = result
    return result


# --------------------------------------------------------------------------
# filter_model - the VCR's own de-emphasis, in the decoder's parametrization
# --------------------------------------------------------------------------

def measure_filter_model(field, picture):
    """THE DE-EMPHASIS SHELF AS PARAMETERS, NOT AS A CURVE.

    Node `filter_model`. Applies nothing.

    The head model gives the magnetics; this is its counterpart for the
    electronics. The recording and playback machines apply a shelf whose FORM
    the format fixes and whose PARAMETERS drift with component tolerance, age
    and temperature, and fitting a measured response to those parameters
    rather than to a free curve is what makes the result travel: *"this
    machine's shelf sits at 268 kHz, not the assumed 274"* is a statement
    about a machine, and a spline coefficient is not.

    THE PARAMETRIZATION IS THE DECODER'S OWN, which is the module's first
    rule and the reason this can be wired at all. The shelf is built by
    calling `compute_video_filters.gen_video_main_deemp_fft`, the same
    function the runtime uses, at the decoder's own sample rate and block
    length - so what is reported is the DEPARTURE from what this decode
    already assumed, which is the only quantity a decode could act on.

    THE MEASUREMENT IS THE SYNC PULSE AND NOTHING ELSE. The pooled pulse the
    sync group already takes is read against the standard's own 4.7
    microsecond pulse - `pair_dimension.spec_sync`, the same reference
    `sync_depth`'s frequency axis uses - and the log ratio of the two
    spectra over the format's luma band is what the assumed shelf failed to
    account for. No active picture enters.

    AND IT DECLINES, LOUDLY, WHEN THE FIT DOES NOT MEAN WHAT IT LOOKS LIKE.
    This arc has already recorded that the shelf is not determinable from
    sync pulses - the fit lands on the difference between the measured pulse
    and the analytic ideal rather than on the machine - so the stage reports
    the fitted departure WITH the residual it leaves, and refuses to call the
    parameters identified whenever that residual is not small against the
    departure it purports to explain. A fit reported without that test is the
    trap, not the fit.

    ALL THREE AXES, ALL MEASURED. AMPLITUDE is the shelf's gain in decibels
    and the departure's own size in nepers. FREQUENCY is the shelf's mid
    frequency and Q, fitted across the band the pulse supports. TIME is the
    pulse's WIDTH in microseconds - one number, which is exactly why the
    module states that at most one degree of freedom can ever be recovered
    from it - together with the per-parameter sensitivity that says which
    parameters could account for a width deficit at all and which could not,
    however far they were mis-set.

    MEASURED by this stage, 2026-09-06, on two decodes that fail in two
    different ways. The bars SP fit runs to its bounds - 900000 Hz of 900000
    and 24.000 dB of 24.0 - and leaves 85 per cent of the 0.8268 neper
    departure; the home tape's converges tidily at 423111 +/- 5119 Hz and
    10.678 +/- 0.114 dB and leaves 99 per cent of 0.6556. Neither explains
    what it was fitted to. The pulses are 4.6126 and 4.6495 microseconds wide
    against the standard's 4.700, and no shelf parameter moves the width by
    more than 0.0019 microseconds per one per cent. Full figures in this
    stage's node in `pipeline/stages.toml`.
    """
    from vhsdecode.models import pair_dimension

    geometry = rf_geometry(field)
    measured = sync_pulses(field, picture)
    if geometry is None or measured is None:
        _log_once(field, "filter_model_declined",
                  "model stages: filter_model declined - this field carries "
                  "no pooled sync pulse, and the shelf is not fitted to "
                  "anything else")
        return None
    state = _rf_state(field, "filter_model")

    spans = measured["spans"]
    rate = spans["sample_rate_hz"]
    profile = np.asarray(measured["profile"], dtype=np.float64)
    band = spans["band_limit_hz"]
    if band is None:
        return None

    parameters = dict(field.rf.DecoderParams)
    parameters["deviation"] = filter_model.nominal_deviation(
        geometry["hz_ire"], float(field.rf.DecoderParams["vsync_ire"]))

    # THE SPECIFIED PULSE, at the same length and rate, at the depth ITU-R
    # BT.1700 fixes. The measured profile is in IRE with blanking at zero, so
    # the two are on one scale without a fitted gain between them.
    depth = -abs(sync_depth.SPECIFIED_DEPTH_IRE)
    ideal = pair_dimension.spec_sync(rate, profile.size, depth_ire=depth)
    ideal = np.asarray(ideal, dtype=np.float64)

    frequencies = np.fft.rfftfreq(profile.size, 1.0 / rate)
    got = np.fft.rfft(profile - profile.mean())
    want = np.fft.rfft(ideal - ideal.mean())
    inside = (frequencies > 0) & (frequencies <= band) & (np.abs(want) > 0)
    if inside.sum() < 6:
        return None
    departure = np.log(np.abs(got[inside]) / np.abs(want[inside]))
    departure = departure - departure.mean()

    fitted = filter_model.fit_deemphasis(
        frequencies[inside], departure, field.rf.DecoderParams,
        geometry["sample_rate_hz"], geometry["block_len"])
    if not fitted:
        _log_once(field, "filter_model_unfitted",
                  "model stages: filter_model declined - the shelf fit did "
                  "not converge on this field's pulse")
        return None

    # THE TIME AXIS. The measured width against the standard's, and which
    # parameters could move it. `width_sensitivity` runs on the SPECIFIED
    # pulse in hertz - the non-linear stage divides by a deviation and is
    # meaningless fed IRE - so it reports how far each parameter moves a
    # known pulse rather than being fitted to this one.
    scale = geometry["hz_ire"]
    sensitivity = filter_model.width_sensitivity(ideal * scale, parameters,
                                                 rate)
    measured_us = filter_model.pulse_width(profile, rate)
    specified_us = sync_geometry.LINE_SYNC_US[
        "525" if spans["system"] == "NTSC" else "625"]
    deficit_us = (measured_us - specified_us
                  if np.isfinite(measured_us) else float("nan"))

    reading = {"fitted": fitted, "sensitivity": sensitivity,
               "measured_width_us": measured_us,
               "specified_width_us": specified_us,
               "width_deficit_us": deficit_us,
               "departure_rms_nepers": float(np.sqrt((departure ** 2).mean())),
               "bins": int(inside.sum())}
    history = state.setdefault("history", [])
    history.append(reading)
    state["fields"] += 1
    ldd.logger.debug(
        "filter_model: shelf mid %.0f Hz (assumed %.0f), gain %.3f dB "
        "(assumed %.3f), q %.4f (assumed %.4f), residual %.4f nepers over "
        "%d bins; pulse %.4f us against a specified %.3f",
        fitted["deemph_mid"], fitted["assumed_deemph_mid"],
        fitted["deemph_gain"], fitted["assumed_deemph_gain"],
        fitted["deemph_q"], fitted["assumed_deemph_q"],
        fitted["residual_rms_nepers"], reading["bins"], measured_us,
        specified_us)

    if state["fields"] >= _RF_POOL_FIELDS and not state["reported"]:
        state["reported"] = True
        mids = np.asarray([r["fitted"]["deemph_mid"] for r in history])
        gains = np.asarray([r["fitted"]["deemph_gain"] for r in history])
        qs = np.asarray([r["fitted"]["deemph_q"] for r in history])
        residuals = np.asarray([r["fitted"]["residual_rms_nepers"]
                                for r in history])
        departures = np.asarray([r["departure_rms_nepers"] for r in history])
        widths = np.asarray([r["measured_width_us"] for r in history])
        widths = widths[np.isfinite(widths)]
        explained = float(np.median(residuals)
                          / max(float(np.median(departures)), 1e-30))
        movers = sorted(((name, float(value)) for name, value
                         in sensitivity.items()
                         if name != "baseline_us" and np.isfinite(value)),
                        key=lambda item: -abs(item[1]))
        _log_once(
            field, "filter_model",
            "model stages: filter_model over %d fields, on the sync pulse "
            "alone and in the decoder's own parametrization - the shelf's "
            "mid frequency fits %.0f +/- %.0f Hz against the assumed %.0f, "
            "its gain %.3f +/- %.3f dB against %.3f, its Q %.4f +/- %.4f "
            "against %.4f - inside bounds of %.0f to %.0f Hz, %.1f to %.1f "
            "dB and %.2f to %.2f. The departure it is fitted to is %.4f "
            "nepers rms and the fit leaves %.4f, which is %.0f per cent of "
            "it",
            state["fields"], float(mids.mean()), float(mids.std()),
            fitted["assumed_deemph_mid"], float(gains.mean()),
            float(gains.std()), fitted["assumed_deemph_gain"],
            float(qs.mean()), float(qs.std()), fitted["assumed_deemph_q"],
            filter_model.DEEMPHASIS_BOUNDS["deemph_mid"][0],
            filter_model.DEEMPHASIS_BOUNDS["deemph_mid"][1],
            filter_model.DEEMPHASIS_BOUNDS["deemph_gain"][0],
            filter_model.DEEMPHASIS_BOUNDS["deemph_gain"][1],
            filter_model.DEEMPHASIS_BOUNDS["deemph_q"][0],
            filter_model.DEEMPHASIS_BOUNDS["deemph_q"][1],
            float(np.median(departures)), float(np.median(residuals)),
            100.0 * explained)
        if fitted.get("pinned"):
            _log_once(
                field, "filter_model_pinned",
                "model stages: filter_model - %s sat AT a bound, so those "
                "were not determined by the data at all. A parameter driven "
                "to its bound is the fit saying the shape it is being asked "
                "to reproduce is not a shelf's, and it is reported here "
                "rather than quoted as a value",
                fitted["pinned"].replace(",", ", "))
        _log_once(
            field, "filter_model_width",
            "model stages: filter_model - the TIME axis. This decode's "
            "pulses are %.4f +/- %.4f us wide at half amplitude against the "
            "standard's %.3f, a deficit of %+.0f ns. The width is ONE "
            "number, so at most one degree of freedom is recoverable from "
            "it, and the parameters that move it at all are %s (microseconds "
            "per one per cent)",
            float(widths.mean()) if widths.size else float("nan"),
            float(widths.std()) if widths.size else float("nan"),
            specified_us,
            1e3 * (float(widths.mean()) - specified_us) if widths.size
            else float("nan"),
            ", ".join("%s %+.4f" % pair for pair in movers[:4])
            or "none measurably")
        if explained > 0.5:
            _log_once(
                field, "filter_model_credibility",
                "model stages: filter_model DECLINES to hand these "
                "parameters back as a decode setting. The fit leaves %.0f "
                "per cent of the departure it is fitted to, which is the "
                "recorded finding rather than a surprise: a sync pulse is "
                "213 kHz wide as a probe and the shelf's corner sits below "
                "that, so what the fit lands on is the difference between "
                "the measured pulse and the analytic ideal and not the "
                "machine. The flat-field identification is the measurement "
                "that settles this one",
                100.0 * explained)
    field.filter_model = reading
    return reading


# --------------------------------------------------------------------------
# transport_model - the mechanical parts, at the rates their diameters fix
# --------------------------------------------------------------------------

# The largest number of line samples the transport search will hold. Two
# series a field at the line rate is about 260 samples a field, so this is
# nearly two hundred fields - far more than any decode in this arc - and it
# exists so a long decode cannot grow the store without bound.
_TRANSPORT_MAXIMUM_LINES = 49152


def measure_transport(field):
    """THE VCR'S MECHANICAL TRANSPORT, AS PREDICTED FREQUENCIES.

    Node `transport_model`. Applies nothing.

    Every rotating part turns at a rate its own diameter fixes, and any
    eccentricity, bearing wear or stiff bushing in it disturbs the tape once
    per revolution. Those disturbances are not diffuse noise: each lands at a
    KNOWN frequency, so the specification predicts where to look before
    anything is measured and a line found at a predicted rate names the part
    that made it. The prediction coming first is what separates this from
    trawling a spectrum for whatever peak is tallest.

    TWO SERIES, BECAUSE THE TWO FAULTS ARE DIFFERENT. A speed disturbance
    shows in the TIME BASE - the tape momentarily running fast or slow - and a
    tension or contact disturbance shows in the RF ENVELOPE, because
    head-to-tape pressure sets the spacing and spacing sets the level by
    Wallace's law. An eccentric capstan moves the first and barely touches the
    second; a sticky guide does the opposite. Both are read here, per line, so
    a part can be attributed rather than merely found.

    IT DECLINES BEFORE IT SEARCHES, AND THAT IS THE POINT OF `observable`.
    Nothing slower than one cycle across the record can be told from a
    constant, so on a short decode most of the transport is absent from the
    MEASUREMENT rather than from the tape - and saying so first is what stops
    a null result being read as a clean transport. The stage reports which
    parts its own duration can resolve and searches only for those.

    AND IT NAMES THE CONFOUND IT CANNOT REMOVE. The drum turns once per two
    fields, so its rate IS the head alternation's. Sampled once per FIELD the
    two are bit-identical and no search could attribute a line there; sampled
    once per LINE, which is what this does, the same correlation falls to
    0.0296 and they separate. The head-to-head amplitude difference still
    sits at that frequency in the ENVELOPE, so a drum line found there is
    reported with that caveat attached rather than without it.

    ALL THREE AXES, ALL MEASURED. AMPLITUDE is the envelope series and the
    height of each line over its own local background. FREQUENCY is the
    predicted rate of every part and the frequency the line is actually found
    at. TIME is the line-period series - the second of the two, and the one
    that responds to a different set of faults - together with the per-field
    decomposition of the sweep into the three shapes a path error can take:
    a constant is a height offset, a linear tilt is a guide standing wrong,
    and a bow is the tape curving across its width.

    MEASURED by this stage, 2026-09-06, on fourteen-frame decodes - 0.4322
    seconds and 6800 lines, against the 0.4171 the drum's own search band
    requires. The DRUM is found on both series of both tapes at 30.0802 Hz
    against a predicted 29.9700, and the capstan in neither: 723.2 times its
    local background in the envelope and 13.4 in the time base on the bars
    tape, 720.7 and 13.9 on the same capture at its own 50 MSps, and 1975.9
    and 71.2 on the home tape. Full figures in this stage's node in
    `pipeline/stages.toml`.
    """
    geometry = rf_geometry(field)
    if geometry is None or geometry.get("line_rate_hz") is None:
        return None
    video = getattr(field, "data", {}).get("video") if hasattr(field, "data") \
        else None
    locations = getattr(field, "linelocs", None)
    if video is None or "envelope" not in video or locations is None:
        return None
    values = np.asarray(locations, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 32:
        return None
    envelope = np.asarray(video["envelope"], dtype=np.float64)
    state = _rf_state(field, "transport_model")

    # THE TWO SERIES, ONE SAMPLE A LINE. The envelope is averaged over each
    # whole line rather than sampled at a point, so a per-line reading is the
    # line's mean level and not one instant of the carrier's own ripple.
    starts = np.rint(values[:-1]).astype(np.intp)
    stops = np.rint(values[1:]).astype(np.intp)
    keep = (starts >= 0) & (stops <= envelope.size) & (stops > starts)
    starts, stops = starts[keep], stops[keep]
    if starts.size < 32:
        return None
    edges = np.concatenate(([0], np.cumsum(envelope)))
    level = (edges[stops] - edges[starts]) / (stops - starts)
    period = (values[1:] - values[:-1])[keep] / geometry["sample_rate_hz"]

    sweep = transport_model.fit_sweep(level)
    if sweep.get("usable"):
        state.setdefault("sweeps", []).append(
            {name: float(sweep[name]) for name in
             ("share_height_offset_m", "share_guide_tilt_m",
              "share_parabolic_bow_m", "explained")})
    store = state.setdefault("series", {"level": [], "period": []})
    if sum(len(part) for part in store["level"]) < _TRANSPORT_MAXIMUM_LINES:
        store["level"].append(level.astype(np.float32))
        store["period"].append(period.astype(np.float64))
    state["fields"] += 1

    reading = {"sweep": sweep, "lines": int(level.size)}
    if state["fields"] < _RF_POOL_FIELDS:
        field.transport_model = reading
        return reading

    envelope_series = np.concatenate(store["level"]).astype(np.float64)
    period_series = np.concatenate(store["period"])
    line_rate = float(geometry["line_rate_hz"])
    duration = period_series.sum()
    mechanics = _mechanics(geometry)
    speed = (mechanics["linear_tape_speed_m_s"] if mechanics
             else None)
    if speed is None:
        _log_once(field, "transport_model_declined",
                  "model stages: transport_model declined - no linear tape "
                  "speed is stated for this format and speed, so no part's "
                  "rotation rate can be predicted and none is guessed")
        return reading

    rates = transport_model.rotation_rates(speed,
                                           float(geometry["field_rate_hz"]))
    seen = transport_model.observable(rates, duration, line_rate)
    resolvable = [entry for entry in seen if entry["resolvable"]]
    found_level = transport_model.find_lines(envelope_series, line_rate,
                                             resolvable)
    found_period = transport_model.find_lines(period_series, line_rate,
                                              resolvable)
    reading.update({"rates": rates, "observable": seen,
                    "envelope_lines": found_level,
                    "time_base_lines": found_period,
                    "duration_s": duration})

    # WHEN THE REPORT IS TAKEN, AND WHY NOT AT THE FOURTH FIELD LIKE THE
    # REST OF THIS GROUP. The whole subject here is DURATION: `find_lines`
    # searches a fractional band around each predicted rate, so the series
    # has to run long enough for a transform bin to land inside that band
    # before any search means anything. For the fastest predicted part -
    # the drum, whose rate the format fixes - that is one over the
    # tolerance times the rate, which on NTSC is 0.417 seconds, or about
    # twenty-five fields. Reported at four fields the search would say "no
    # predicted rate found" on every decode ever run, which is a statement
    # about the report's timing and not about the transport. So the
    # requirement is announced early and the verdict waits for it.
    slowest = min((float(entry["rate_hz"]) for entry in resolvable),
                  default=float("nan"))
    fastest = max((float(entry["rate_hz"]) for entry in resolvable),
                  default=float("nan"))
    required = (1.0 / (transport_model.SEARCH_TOLERANCE * fastest)
                if np.isfinite(fastest) and fastest > 0 else float("inf"))
    reading["required_duration_s"] = required
    ldd.logger.debug(
        "transport_model: %.4f s of %.4f needed, %d lines, %d resolvable, "
        "%d envelope lines, %d time-base lines",
        duration, required, envelope_series.size, len(resolvable),
        len(found_level), len(found_period))
    _log_once(
        field, "transport_model_requirement",
        "model stages: transport_model - `find_lines` searches a %.0f per "
        "cent band around each predicted rate, so a bin only lands inside "
        "the fastest one's band (%s at %.4f Hz) once the series runs %.4f "
        "seconds, and the slowest resolvable part here is %s at %.4f Hz. "
        "The search below is therefore withheld until the decode has that "
        "much tape rather than reported early and read as a clean transport",
        100.0 * transport_model.SEARCH_TOLERANCE,
        next((entry["name"] for entry in resolvable
              if float(entry["rate_hz"]) == fastest), "nothing"), fastest,
        required,
        next((entry["name"] for entry in resolvable
              if float(entry["rate_hz"]) == slowest), "nothing"), slowest)

    if duration >= required and not state["reported"]:
        state["reported"] = True
        _log_once(
            field, "transport_model_observable",
            "model stages: transport_model - %.4f seconds of tape at a "
            "%.1f Hz line rate resolves %d of the %d modelled parts (%s); "
            "the rest are absent from the MEASUREMENT rather than from the "
            "tape - %s",
            duration, line_rate, len(resolvable), len(seen),
            ", ".join("%s %.4f Hz" % (entry["name"], entry["rate_hz"])
                      for entry in resolvable) or "none",
            "; ".join("%s %s" % (entry["name"], entry["why_not"])
                      for entry in seen if not entry["resolvable"]))
        for label, found in (("envelope", found_level),
                             ("time base", found_period)):
            if not found:
                _log_once(
                    field, "transport_model_%s" % label.replace(" ", "_"),
                    "model stages: transport_model found no predicted rate "
                    "in the %s over %d lines", label, envelope_series.size)
                continue
            _log_once(
                field, "transport_model_%s" % label.replace(" ", "_"),
                "model stages: transport_model in the %s over %d lines - %s",
                label, envelope_series.size,
                "; ".join(
                    "%s predicted %.4f Hz found at %.4f, %.1fx its own local "
                    "background%s" % (
                        entry["name"], entry["rate_hz"], entry["at_hz"],
                        entry["height_over_background"],
                        (" - and at this rate the ENVELOPE cannot tell the "
                         "drum from the head-to-head amplitude difference, "
                         "which sits at the same frequency"
                         if entry["name"] == "head drum"
                         and label == "envelope" else ""))
                    for entry in found))
        shares = state.get("sweeps") or []
        if shares:
            pooled = {name: float(np.median([row[name] for row in shares]))
                      for name in ("share_height_offset_m",
                                   "share_guide_tilt_m",
                                   "share_parabolic_bow_m", "explained")}
            dominant = max(("height_offset_m", "guide_tilt_m",
                            "parabolic_bow_m"),
                           key=lambda n: pooled["share_" + n])
            _log_once(
                field, "transport_model_sweep",
                "model stages: transport_model - the head's sweep down a "
                "field, pooled over %d of them, decomposes into %.1f per "
                "cent height offset, %.1f per cent guide tilt and %.1f per "
                "cent parabolic bow, explaining %.1f per cent of the "
                "per-line variation; the dominant shape is %r, which is %s",
                len(shares), 100.0 * pooled["share_height_offset_m"],
                100.0 * pooled["share_guide_tilt_m"],
                100.0 * pooled["share_parabolic_bow_m"],
                100.0 * pooled["explained"], dominant,
                transport_model.PATH_ERRORS[dominant]["cause"])
    field.transport_model = reading
    return reading


# --------------------------------------------------------------------------
# band_delay - the luma and chroma bands as one clock, read twice
# --------------------------------------------------------------------------

def measure_band_delay(field):
    """THE LUMA AND CHROMA BANDS AS ONE CLOCK, READ TWICE.

    Node `band_delay`. Applies nothing.

    Ethan, 2026-09-06: *"The difference in time between the luma and chroma
    bands are the measurement we can use to observe the delay on each video
    head. This applies to playback and recording. Use this to relate on the
    time axis for the head measurements."*

    WHY THIS IS A CLEAN MEASUREMENT. One head wrote both bands in one pass
    and one head reads both back in one pass, so everything that moves the
    signal in time as a whole - the drum's phase, the capstan, the tape's
    stretch, the capture clock - moves the two together and cancels in their
    difference. What survives is a delay that depends on frequency, over a
    span of about six to one, and that is the head's own dispersion.

    BOTH CLOCKS ARE IN RESERVED INTERVALS. The luma band carries the sync
    pulse, whose rise out of the tip is a specified event; the chroma band
    carries the burst, whose start is specified at 5.3 microseconds after the
    leading edge of sync. The interval between them is therefore a specified
    constant and any departure from it is the two bands' differential delay.
    No picture is read.

    THE HEADS ARE LABELLED BY THE DECODE AND NOT BY THE MODULE, and that is
    this wiring's own contribution. The module labels heads from a gap in the
    line sequence at the vertical interval, which comes once per drum
    revolution - two fields - so a single field cannot supply it, and it
    correctly returns nothing. The decoder already knows which head read this
    field: it is the field parity, which the format fixes and which
    `docs/` records as a KNOWN CONSTANT rather than a component. So each
    field's reading is filed under its own parity and the per-head figure is
    pooled across the decode.

    THE REFUSALS ARE PRESERVED, and they are the measurement's own. The
    record tap is the instrument's null - the same drive reaches whichever
    head is switched in, so the two heads MUST agree there, and they do at
    1.3 and 0.8 standard errors. The per-head figure at the playback tap is
    BOUNDED at 4.6 nanoseconds and not measured, because two captures that
    should agree disagree in sign. And the tap difference of -78.27 ns is NOT
    a spacing loss whatever its size: a positive separation lengthens this
    interval, so a measurement that shortens it would need -0.39 micron. This
    stage reports its own reading against those and does not overwrite them.

    ALL THREE AXES, ALL MEASURED. AMPLITUDE is the burst's own level, which is
    what tells a burst from its absence and gates every reading - the y-only
    captures are refused by exactly that. FREQUENCY is the two bands
    themselves and their ratio, which is what buys the measurement. TIME is
    the interval in nanoseconds, which is the measurement.

    MEASURED by this stage, 2026-09-06, on two decodes of four fields each.
    The interval reads 2060.2 ns on the bars SP playback tap and 1774.7 on
    the home tape, against the 1857.1 the standard specifies - departures of
    +203.0 and -82.5 nanoseconds, of OPPOSITE SIGN, so the second is not a
    spacing loss at any size and would need -0.412 micron. Between the two
    field parities the bars decode gives +0.20 +/- 0.90 ns and the home tape
    -24.97 +/- 33.04, both consistent with zero and neither improving on the
    offline 4.6 nanosecond bound. Full figures in this stage's node in
    `pipeline/stages.toml`.
    """
    geometry = rf_geometry(field)
    raw = _raw_field(field)
    if geometry is None or raw is None:
        return None
    if geometry.get("system") != "NTSC":
        _log_once(field, "band_delay_declined",
                  "model stages: band_delay declined - the burst's position "
                  "is cited from BTS-3 for the 525-line system and no 625 "
                  "evidence was taken, so this reading is not extended to "
                  "%r", geometry.get("system"))
        return None
    state = _rf_state(field, "band_delay")

    shared = _shared_rf_analytics(field, raw, geometry["sample_rate_hz"])
    measured = band_delay.measure_capture(raw, geometry["sample_rate_hz"],
                                          "NTSC", shared=shared)
    pooled = measured["pooled"]
    if not pooled["lines"] or not np.isfinite(pooled["median_us"]):
        _log_once(field, "band_delay_no_burst",
                  "model stages: band_delay declined on this field - no "
                  "line carried a burst inside the specified window, which "
                  "is what a chrominance-free recording looks like to this "
                  "instrument and is reported rather than fitted")
        return None

    parity = int(bool(getattr(field, "isFirstField", False)))
    history = state.setdefault("history", {0: [], 1: []})
    history[parity].append({
        "median_us": float(pooled["median_us"]),
        "sd_us": float(pooled["sd_us"]),
        "lines": int(pooled["lines"]),
        "amplitude": float(measured["burst_amplitude"]),
    })
    state["fields"] += 1
    reading = {"pooled": pooled, "bands": measured["bands"], "parity": parity,
               "burst_amplitude": float(measured["burst_amplitude"])}
    ldd.logger.debug(
        "band_delay: parity %d, %.5f us over %d lines (sd %.5f), specified "
        "%.5f, departure %+.1f ns, burst %.1f",
        parity, pooled["median_us"], pooled["lines"], pooled["sd_us"],
        pooled["specified_us"], pooled["departure_ns"],
        measured["burst_amplitude"])

    if state["fields"] >= _RF_POOL_FIELDS and not state["reported"]:
        per_head = {}
        for which, rows in history.items():
            if not rows:
                continue
            centres = np.asarray([row["median_us"] for row in rows])
            per_head[which] = {
                "median_us": float(np.median(centres)),
                "standard_error_us": float(np.std(centres)
                                           / math.sqrt(max(centres.size, 1))),
                "lines": int(sum(row["lines"] for row in rows)),
                "fields": int(centres.size),
            }
        if len(per_head) == 2:
            state["reported"] = True
            difference = band_delay.head_difference({"per_head": per_head})
            band = measured["bands"]
            spacing = band_delay.wallace_comparison(pooled["departure_ns"])
            reading["per_head"] = per_head
            reading["head_difference"] = difference
            _log_once(
                field, "band_delay",
                "model stages: band_delay over %d fields - the sync rise to "
                "burst interval is %.1f ns against the %.1f ns BTS-3 clause "
                "2.4.9 specifies, a departure of %+.1f ns, read across a "
                "band ratio of %.3f (luma centre %.3f MHz against a "
                "colour-under at %.4f MHz)",
                state["fields"], 1e3 * pooled["median_us"],
                1e3 * pooled["specified_us"], pooled["departure_ns"],
                band["ratio"], band["luma_centre_hz"] / 1e6,
                band["chroma_hz"] / 1e6)
            _log_once(
                field, "band_delay_heads",
                "model stages: band_delay per head, labelled by the field "
                "parity the format fixes - head 0 %.1f +/- %.1f ns over %d "
                "fields, head 1 %.1f +/- %.1f ns over %d, a difference of "
                "%+.2f +/- %.2f ns at %.1f standard errors. The offline "
                "measurement BOUNDS this at 4.6 ns rather than measuring it, "
                "because two captures that should agree disagree in sign, "
                "and this reading is reported against that bound and does "
                "not replace it",
                1e3 * per_head[0]["median_us"],
                1e3 * per_head[0]["standard_error_us"], per_head[0]["fields"],
                1e3 * per_head[1]["median_us"],
                1e3 * per_head[1]["standard_error_us"], per_head[1]["fields"],
                difference["difference_ns"], difference["error_ns"],
                difference["sigma"])
            _log_once(
                field, "band_delay_wallace",
                "model stages: band_delay - the departure is %s a spacing "
                "loss. %s, so this one would need %+.3f micron, and the "
                "equivalent separation is quoted WITH its sign for exactly "
                "that reason",
                "consistent with" if spacing["separation_is_physical"]
                else "NOT",
                spacing["sign_note"], spacing["equivalent_separation_um"])
    field.band_delay = reading
    return reading


# --------------------------------------------------------------------------
# head_switch_pair - the record head's mark and the playback head's, together
# --------------------------------------------------------------------------

def measure_head_switch_pair(field):
    """THE RECORD HEAD SWITCH, READ OFF THE TAPE AS A PERMANENT MARK.

    Node `head_switch_pair`. Applies nothing.

    Ethan, 2026-09-06: *"Record head switch is the point where the phase
    rotates over time, as the heads rotate around the tape at record, the
    head switches when the phase of the color rotates ... The point where the
    phase rotates at record time in the chroma is the point where the heads
    switch at record time."* And: *"The color phase rotation is introduced on
    the record head, the luma head switching is already identified."*

    WHY THIS IS THE ONE MARK THAT CARRIES THE RECORDING MACHINE. SMPTE
    32M-2004 clause 3.9.2.1.5 makes the colour-under carrier's phase advance
    ninety degrees a line on one track and retard ninety on the other. Ninety
    degrees a line is not a step but a RATE - a quarter turn per line period,
    f_H/4 - so the recorded phase rotates steadily while the head sweeps and
    its direction says which head is writing. Where the rotation reverses is
    where the record head switched, written into the oxide and replayed
    identically on any deck for ever afterwards. The luminance switch belongs
    to the machine doing the playing; this one belongs to the machine that
    did the recording, and it is the only thing in the signal that does.

    IT MUST BE READ IN THE RAW RADIO FREQUENCY, which is why this stage sits
    here and not in the chroma path. In a decoded chroma file the rotation is
    a hundred and eighty degrees a line and not ninety: the decoder has
    already up-converted and undone the record side's own quarter turn. The
    colour-under envelope this reads is taken from the capture's own samples,
    before any of that.

    IT NEEDS TWO FIELDS AND SPLICES THEM, and the reason is the module's own
    gate rather than a preference. A reversal is credited only when a run of
    at least sixty lines in each state brackets it, and a single field's read
    span puts the mark within a handful of lines of one end - measured, one
    field of the bars capture finds a clean rotation axis at +90.14 degrees
    and NO mark, while two spliced fields find one mark and its luminance
    partner. The splice is exact: the previous field's samples are cut at the
    absolute offset between the two read positions, so the stream handed to
    the module is contiguous and carries no seam.

    ALL THREE AXES, ALL MEASURED. AMPLITUDE is the burst level that gates
    every line and the blanking-level contrast across the luminance switch -
    the mark's own size. FREQUENCY is the rotation as a RATE, f_H/4 =
    3933.566 Hz, which is what reverses and what the specification fixes.
    TIME is the mark's position: where it sits ahead of vertical sync in
    lines, its bracket in fractions of a line, and the line-timing
    displacement at the luminance switch in nanoseconds.

    MEASURED by this stage, 2026-09-06, on two decodes of four spliced pairs
    each. The bars SP tape - recorded and played on one Sony SLV-778HF - reads
    +90.24 +/- 0.01 degrees a line with 0.9952 of lines clean and the two
    marks -0.406 +/- 0.000 lines apart; the home tape, recorded on a machine
    we have never seen, reads +90.05 +/- 0.07 with 0.9720 clean and the marks
    +8.421 +/- 0.436 lines apart. Twenty standard errors between two decks, on
    a quantity neither instrument is fitted to. Full figures in this stage's
    node in `pipeline/stages.toml`.
    """
    geometry = rf_geometry(field)
    raw = _raw_field(field)
    if geometry is None or raw is None:
        return None
    if geometry.get("system") not in ("NTSC", "PAL"):
        return None
    state = _rf_state(field, "head_switch_pair")

    # THE SPLICE. `readloc` is the absolute sample this field's raw data
    # begins at, so the previous field's samples up to that point are exactly
    # what is missing from the front of this one. Where the two do not abut -
    # a seek, a dropped field, the first field of a decode - the stage says
    # so and reads nothing rather than splicing a seam.
    previous = state.get("tail")
    state["tail"] = (float(getattr(field, "readloc", 0.0)), raw)
    if previous is None:
        return None
    start, earlier = previous
    offset = int(round(float(getattr(field, "readloc", 0.0)) - start))
    if not 0 < offset <= earlier.size:
        _log_once(field, "head_switch_pair_seam",
                  "model stages: head_switch_pair declined a field - the "
                  "previous read ended %d samples from this one's start, so "
                  "the two do not abut and no seam is spliced",
                  offset)
        return None
    stream = np.concatenate([earlier[:offset], raw])
    shared = (_shared_rf_analytics(field, raw, geometry["sample_rate_hz"],
                                   geometry["system"])
              if stream.size == raw.size else None)

    # SHARE ONLY WHEN THE ARRAY IS THE SAME ONE. This stage prepends the
    # previous field's tail so its record starts where the last one ended,
    # so `stream` is a different and longer array than `raw` whenever that
    # tail is non-empty, and handing it another array's transforms would
    # be a real error rather than a saving. It shares only when the
    # concatenation was a no-op.
    measured = head_switch_pair.measure_capture(
        stream, geometry["sample_rate_hz"], geometry["system"],
        shared=(shared if stream.size == raw.size else None))
    rotation = measured["rotation"]
    if not measured["record"]:
        _log_once(
            field, "head_switch_pair_no_mark",
            "model stages: head_switch_pair found no record-side reversal "
            "in a spliced pair - the per-line rotation clusters on %+.2f "
            "degrees with a clean fraction of %.4f, and where that fraction "
            "collapses the recording carries no chrominance and the "
            "instrument reports nothing rather than reporting noise",
            rotation["axis_deg"], rotation["clean_fraction"])
        return None

    store = state.setdefault("marks", [])
    store.append(measured)
    state["fields"] += 1
    reading = {"rotation": rotation, "rate": measured["rate"],
               "record": measured["record"], "luma": measured["luma"],
               "separation": measured["separation"],
               "against_vertical_sync": measured["against_vertical_sync"]}
    ldd.logger.debug(
        "head_switch_pair: axis %+.2f deg, clean %.4f, %d record marks, "
        "%d luma marks",
        rotation["axis_deg"], rotation["clean_fraction"],
        len(measured["record"]), len(measured["luma"]))

    if state["fields"] >= _RF_POOL_FIELDS and not state["reported"]:
        state["reported"] = True
        axes = np.asarray([item["rotation"]["axis_deg"] for item in store])
        clean = np.asarray([item["rotation"]["clean_fraction"]
                            for item in store])
        separations = np.asarray(
            [item["separation"]["pooled_timing_h"] for item in store],
            dtype=np.float64)
        separations = separations[np.isfinite(separations)]
        specified = rotation["specified"]
        _log_once(
            field, "head_switch_pair",
            "model stages: head_switch_pair over %d spliced field pairs - "
            "the recorded colour-under rotation clusters on %+.2f +/- %.2f "
            "degrees a line against the %+.0f SMPTE 32M clause 3.9.2.1.5 "
            "specifies, with %.4f of lines projecting more than nine tenths "
            "onto that axis; that is a RATE of %.1f Hz, a quarter of the "
            "line rate, and it is what reverses at the record head switch",
            state["fields"], float(axes.mean()), float(axes.std()),
            specified["track_1_deg"], float(clean.mean()),
            measured["rate"]["specified_hz"])
        if separations.size:
            _log_once(
                field, "head_switch_pair_separation",
                "model stages: head_switch_pair - the record-side mark and "
                "the playback-side luminance switch sit %+.3f +/- %.3f "
                "lines apart over %d pairs. They coincide on a tape played "
                "on the deck that recorded it and separate by five to ten "
                "lines on a foreign deck, so this number is a statement "
                "about WHICH MACHINE made this recording and not about the "
                "signal",
                float(np.median(separations)), float(separations.std()),
                separations.size)
        else:
            _log_once(
                field, "head_switch_pair_separation_declined",
                "model stages: head_switch_pair declines the separation - "
                "the record mark was found but no luminance switch was, so "
                "the pair cannot be placed on one time axis and no distance "
                "between them is reported")
    field.head_switch_pair = reading
    return reading


# --------------------------------------------------------------------------
# interference - the disturbance types, as complex components
# --------------------------------------------------------------------------

def measure_interference(field):
    """THE INTERFERENCE TYPES, AS COMPONENTS THE TRANSFORM CAN DIFFERENTIATE
    OVER - AND HOW MUCH OF THIS CAPTURE'S OWN CHANNEL THEY SPAN.

    Node `interference`. Applies nothing.

    Ethan's reading, and it is the right one: *"randomness is a residual here
    that we can effectively differentiate over since we can model different
    types of radio interference, maybe that's all the RF stage boils down to
    using all of our measured components."*

    THE FINDING THAT MOTIVATES THE MODULE. Given a random phase per field so
    that none of them could correlate trivially, four of five disturbances
    are not random to the transform at all: a beat has a frequency even when
    its phase is random and a dropout has a shape even when its position is
    not. Only particle noise sits at the sphere floor, and it sits there to
    within a tenth of a sigma. So what a single measurement lumps together as
    noise is mostly modellable interference, and the irreducible remainder is
    one specific mechanism - the finite number of particles in the volume the
    head reads.

    WHAT THIS ADDS BY RUNNING ON A DECODE. Two things the module cannot get
    on its own. First, the band and the mechanics are the DECODE'S - the
    video band-pass the format definition gives, the writing speed and track
    width of the tape speed the decode was told - so the count of
    distinguishable types is this capture's rather than a nominal one.
    Second, and this is the measurement, the key is asked to span something
    real: the channel this capture actually has, taken over the RF band
    against the tape's own ideal response, and the share that lands OUTSIDE
    the key is the honest upper bound on what any amount of further fitting
    could recover.

    THE RESERVED PULSES AND NOTHING ELSE. The signatures live across the RF
    band, which is exactly where the picture's own sidebands are, so the
    spectrum is taken inside synchronizing pulses and nowhere else. During a
    pulse the modulator holds one frequency, so the spectrum is a LINE at the
    sync-tip carrier standing on the channel's own noise - the line is the
    modulator and is excised, and the floor beside it is the channel. This is
    the same premise `capture_filter` uses above the video band, applied
    inside it.

    THE PULSE'S LENGTH IS THE RESOLUTION, WHICH IS THE 1/T PROBE LAW AND IS
    WHY THE FIELD-SYNC PULSES ARE PREFERRED. A line-sync pulse is 4.70
    microseconds and resolves 213 kHz; a field-sync broad pulse is 27.1 and
    resolves 37, six times finer, and every field's vertical interval holds
    six of them. The first wiring of this stage read the line pulse and got
    12 bins for an 11-vector basis, where the completeness is 1.0000 by
    arithmetic and says nothing; the broad pulse is what makes the question
    answerable, and the stage declines when whichever pulse it found does
    not give three times the key's own size in bins.

    THE SPAN IS JUDGED AGAINST ITS OWN NULL, because a basis of `k` vectors
    captures `k/L` of anything by chance. What is reported is the excess over
    that, which is what says the key covers the channel rather than merely
    being large.

    ALL THREE AXES, AND THE SIGNATURES ARE COMPLEX BY CONSTRUCTION. Each
    returns a complex number per frequency whose REAL part is what the
    mechanism does to amplitude and whose IMAGINARY part is what it does to
    phase - which is the time axis, and is why the echo family is a Fourier
    basis over delay and separates where the tape's monotone decays do not.
    The count is taken with `real_parameters=True`, so a gain and a delay of
    the same shape are different mechanisms, which the Hermitian inner
    product cannot see: measured, that distinction moves the effective count
    from 2.20 to 3.79 on the propagation set.

    MEASURED by this stage, 2026-09-06. Of the 14 modelled types, 13 are
    independent enough to be counted and they span 7.355 effective directions
    at a condition of 10.7 on the bars decode and 7.332 at 10.6 on the home
    tape. Against each capture's own channel - 48 field-sync broad pulses at
    92 kHz resolution with the carrier line excised - the key spans 0.9655,
    0.9663 and 0.9747 of the residual against a chance null of 0.2034, so
    between 2.5 and 3.5 per cent lies outside it. Full figures in this
    stage's node in `pipeline/stages.toml`.
    """
    geometry = rf_geometry(field)
    if geometry is None:
        return None
    state = _rf_state(field, "interference")
    probe = _reserved_carrier_windows(field, geometry)
    if probe is None:
        _log_once(field, "interference_declined",
                  "model stages: interference declined - this field gives no "
                  "usable reserved-pulse windows, and the RF band is not "
                  "read anywhere else because everywhere else is picture")
        return None
    windows, pulse, pulse_us = probe
    state["pulse"] = pulse
    state["pulse_us"] = pulse_us

    rate = geometry["sample_rate_hz"]
    length = windows.shape[1]
    taper = np.hanning(length)
    spectrum = np.fft.rfft(windows * taper[None, :], axis=1)
    frequency = np.fft.rfftfreq(length, 1.0 / rate)
    power = (np.abs(spectrum) ** 2).mean(axis=0)
    store = state.setdefault("power", None)
    state["power"] = power if store is None else store + power
    state["frequency_hz"] = frequency
    state["resolution_hz"] = float(rate / length)
    state["windows"] = state.get("windows", 0) + int(windows.shape[0])
    state["fields"] += 1
    if state["fields"] < _RF_POOL_FIELDS:
        return None
    if state.get("reported"):
        return state.get("result")

    mechanics = _mechanics(geometry)
    if mechanics is None:
        _log_once(field, "interference_mechanics",
                  "model stages: interference declined - no mechanics are "
                  "stated for this format and tape speed, so the particle "
                  "floor the other types are judged against cannot be built")
        state["reported"] = True
        return None

    low, high = geometry["band_hz"]
    # THE CARRIER'S OWN LINE IS EXCISED AND THE REST IS THE CHANNEL. During a
    # synchronizing pulse the modulator holds one frequency, so the spectrum
    # is a LINE at the sync-tip carrier standing on the channel's noise. The
    # line is the modulator and not the channel, and leaving it in would let
    # every signature score against one spike; what carries the channel's
    # shape is the floor beside it. The excision is three resolution widths
    # each side, which is the transform's own leakage width for the taper
    # used and not a chosen margin.
    resolution = float(state["resolution_hz"])
    guard = 3.0 * resolution
    inside = ((frequency >= low) & (frequency <= high)
              & (np.abs(frequency - geometry["sync_tip_hz"]) > guard))
    speed = mechanics["writing_speed_m_s"]
    width = mechanics["track_width_m"]
    key = {"mechanics": {"writing_speed_m_s": speed, "track_width_m": width},
           "carrier_hz": geometry["carrier_hz"]}
    grid = frequency[inside]
    # How many entries the key has on this band, counted on a grid dense
    # enough to build them whatever the measurement's own resolution is.
    entries = len(interference.signatures(np.linspace(low, high, 64), **key))

    # THE MEASUREMENT MUST HAVE MORE DIMENSIONS THAN THE KEY HAS ENTRIES, and
    # by a margin. `span_completeness` is charitable to the key by
    # construction - a basis of k vectors captures k/L of ANY residual by
    # chance - so at L close to k the answer is one whatever the channel does.
    # Measured on a first wiring that read the 4.7 microsecond line pulse: 12
    # bins, 11 basis vectors, a null of 0.9167 and an "inside" of 1.0000,
    # which said nothing at all. Three times the key's size is the floor, and
    # below it the stage declines and says which pulse it was reading.
    if inside.sum() < 3 * entries:
        _log_once(field, "interference_resolution",
                  "model stages: interference declined - the %s is %.2f us "
                  "long, so by the 1/T probe law it resolves %.0f kHz and "
                  "puts only %d bins inside the %.3f to %.3f MHz band with "
                  "the carrier excised. A %d-entry key spans %d numbers by "
                  "chance alone, so the completeness is not measured here",
                  state["pulse"], state["pulse_us"], resolution / 1e3,
                  int(inside.sum()), low / 1e6, high / 1e6, entries,
                  int(inside.sum()))
        state["reported"] = True
        return None

    channel = 0.5 * np.log(np.maximum(state["power"][inside]
                                      / state["fields"], 1e-300))
    ideal = interference.ideal_tape_response(grid, speed, width)
    reference = np.asarray(ideal["log_response"], dtype=np.float64)
    residual = (channel - channel.mean()) - (reference - reference.mean())

    counts = interference.distinguishable(grid, **key)
    cover = interference.span_completeness({"channel": residual}, grid, **key)
    shapes = interference.signatures(grid, **key)
    subtractable = [name for name, value in shapes.items()
                    if interference.subtractable(value)]
    result = {"distinguishable": counts, "span": cover,
              "subtractable": subtractable, "residual": residual,
              "frequency_hz": grid, "resolution_hz": resolution,
              "pulse": state["pulse"]}
    state["result"] = result
    state["reported"] = True

    _log_once(
        field, "interference",
        "model stages: interference on this decode's own %.3f to %.3f MHz "
        "at a writing speed of %.4f m/s - %d of %d modelled types are "
        "linearly independent enough to be counted and they span %.3f "
        "effective directions at a condition of %.3g, with %d of %d "
        "subtractable at all. Only particle noise is meant to sit at the "
        "floor; the rest are structure a single measurement would have "
        "called noise",
        low / 1e6, high / 1e6, speed, counts["count"], len(shapes),
        counts["effective"], counts["condition"], len(subtractable),
        len(shapes))
    _log_once(
        field, "interference_span",
        "model stages: interference against this capture's OWN channel - "
        "the spectrum inside %d %ss over %d fields, %.0f kHz resolution "
        "with the sync-tip carrier line excised, against the tape's ideal "
        "response - the key spans %.4f of the residual where a basis of %d "
        "vectors would capture %.4f of %d bins by chance, an excess of "
        "%+.4f. What is outside the span is %.4f, and that is the upper "
        "bound on what any further fitting could recover, because nothing "
        "in the model represents it",
        state["windows"], state["pulse"], state["fields"], resolution / 1e3,
        cover["inside"], cover["basis"], cover["null"], cover["length"],
        cover["excess"], cover["outside"])
    field.interference = result
    return result


# ==========================================================================
# THE HEAD, THE MEDIUM AND THE PATH
# ==========================================================================
#
# Five stages, and between them they carry the physical chain from the coil
# to the oxide to the clearance the two are separated by. Not one of them
# writes a sample, so a decode is byte-identical whether they are on or off
# and the finding goes to the log.
#
#   `head_model`        MEASURES. The physical head's own parameters, fitted
#                       to the two heads' DIFFERENCE on the RF envelope.
#                       A difference and never one head: the two levels are
#                       read by two instruments with two arbitrary scales,
#                       and only a difference between heads at one frequency
#                       is free of both.
#
#   `head_differential` MEASURES. The same head pair as Ethan's set of three
#                       - an amplitude, a phase and a frequency - each on its
#                       own axis and each with its own closed form.
#
#   `magnetic`          MEASURES AND DECLARES. The record level as an axis
#                       rather than a shape: the self-demagnetisation cap
#                       binds at a different frequency for every level, and
#                       that movement is the whole of the information.
#
#   `magnetic_circuit`  MEASURES. The same separation read TWICE, once from
#                       the two bands' levels and once from their timing, so
#                       that the two routes can disagree - which is the only
#                       thing a pair is for.
#
#   `tape_path`         DECLARES, AND SAYS SO. Seven mechanisms that set the
#                       clearance, three controls that can fail, and NO
#                       measurement pair: its `joint_axis` against its
#                       `time_axis` is an arithmetic identity by the module's
#                       own docstring and is reported as a construction check
#                       rather than as a second view.
#
# THE SYNC-ONLY LAW IS MET BY CONSTRUCTION. Every reading below is taken
# inside the synchronizing pulse's flat interior, the back porch after the
# burst, or the colour burst itself. No sample of active picture enters a
# level, a difference, a fit or a verdict.

# ONE IMPORT A LINE, for the reason the chroma group's block records: the
# inventory reads these files as text and its pattern for an import does not
# cross a newline.
from vhsdecode.models import head_differential
from vhsdecode.models import head_model
from vhsdecode.models import magnetic
from vhsdecode.models import magnetic_circuit
from vhsdecode.models import tape_path

# The per-decode store of reserved-interval band levels, shared by all five
# stages so that enabling two of them costs one measurement rather than two.
_BAND_LEVELS = "_reserved_band_levels"

# A sentinel, so that a field whose measurement legitimately came back empty
# is not measured again on every stage that asks.
_UNREAD = object()


def _reserved_windows_us(field):
    """The three reserved intervals this group reads, in microseconds from
    the sync pulse's LEADING EDGE - which is where the decoder's own line
    locations sit.

    THREE WINDOWS AND NOT ONE, because a head's response is a shape across
    frequency and one window holds one carrier. During the SYNC TIP the
    modulator holds the synchronizing level's frequency and nothing modulates
    it; during the BACK PORCH it holds blanking's; and the COLOUR BURST is
    the same head's writing at forty times the line rate, six times further
    down the band than either. No picture is inside any of them.

    EVERY BOUNDARY IS A SYSTEM PARAMETER. The tip's interior is
    `sync_depth.TIP_INTERIOR`, the same fraction `sync_depth` reads its own
    level on, so the two stages read the same part of the same pulse. The
    porch is bounded by the burst's end and the start of active video, and
    guarded at both ends by the format's own specified sync transition time
    rather than by a margin chosen here.

    THE FRONT PORCH IS DELIBERATELY NOT AMONG THEM. It is at the same
    carrier as the back porch and buys no frequency, and this arc has
    already measured that it is not a level: its content dependence runs to
    41-66 standard errors where the back porch after the burst has none.
    """
    sys_params = field.rf.SysParams
    sync_us = float(sys_params.get("hsyncPulseUS", 0.0) or 0.0)
    burst = sys_params.get("colorBurstUS")
    active = sys_params.get("activeVideoUS")
    guard = float(sys_params.get("syncTransitionUS", 0.0) or 0.0)
    if not sync_us > 0 or burst is None or active is None or not guard > 0:
        return None
    low, high = _PULSE_INTERIOR
    porch = (float(burst[1]) + guard, float(active[0]) - guard)
    if porch[1] - porch[0] <= guard:
        return None
    return {
        "tip": (low * sync_us, high * sync_us),
        "burst": (float(burst[0]), float(burst[1])),
        "porch": porch,
        "guard_us": guard,
        "sync_us": sync_us,
    }


def _window_bounds(rate_hz, span_us):
    """One window's offset and length in samples, for a span in microseconds."""
    first = int(round(float(span_us[0]) * 1e-6 * rate_hz))
    length = int(round(float(span_us[1]) * 1e-6 * rate_hz)) - first
    return first, length


def _common_lines(rate_hz, locations, spans, limit):
    """The lines every one of these windows fits inside, and nothing else.

    ONE LINE SET FOR EVERY WINDOW, which is not tidiness. The three levels
    and the two clocks are differenced and paired against each other, so a
    window that quietly kept a different subset of lines from its neighbour
    would be differencing two populations - the defect that makes a level
    difference read as a channel effect when it is a bookkeeping one.

    Returns None where fewer lines survive than carry a statistic, which is
    a refusal rather than a truncation.
    """
    values = np.rint(np.asarray(locations, dtype=np.float64)).astype(np.intp)
    keep = np.ones(values.shape, dtype=bool)
    for span in spans:
        first, length = _window_bounds(rate_hz, span)
        if length < 8:
            return None
        keep &= (values + first >= 0) & (values + first + length <= int(limit))
    values = values[keep]
    if values.size < 16:
        return None
    return values


def _line_windows(rate_hz, lines, span_us):
    """The index matrix for one window over an already-validated line set."""
    first, length = _window_bounds(rate_hz, span_us)
    return lines[:, None] + first + np.arange(length)[None, :]


def _measure_band_levels(field, geometry):
    """This field's two on-tape bands, read in the reserved intervals alone.

    WHAT IS MEASURED, AND FROM WHICH CHANNEL. The luma band's LEVEL is the
    decoder's own analytic envelope - `np.abs` of the very analytic signal
    the demodulator runs on, so no second transform is taken and no second
    band-pass is chosen here - and its FREQUENCY is the decoder's own
    demodulated channel, averaged over the same window. So the carrier each
    level belongs to is measured rather than assumed, on both of the luma
    windows. The colour-under band's level is the magnitude of a COMPLEX
    envelope at the specified carrier, `band_delay.chroma_envelope`, which is
    narrow enough to exclude the luma FM and which keeps the phase.

    THE COHERENCE IS MEASURED AND IT IS LOAD-BEARING. `|sum s| / sum |s|`
    over each burst window is one for a pure tone and falls toward the
    reciprocal of the square root of the window length for noise, so it is
    what tells a burst from its absence. A luminance-only recording has no
    colour-under carrier at all, and the estimators this group's separation
    rests on must refuse such a capture rather than fit it:
    `magnetic_circuit.two_band_delay_pair`'s own docstring records a luma-only
    capture reporting a head delay difference of +2229.70 +/- 104.29 ns -
    twenty-one standard deviations of nonsense that an error bar alone would
    have accepted.

    Returns one record a field, or None where the decode cannot supply one.
    """
    windows = _reserved_windows_us(field)
    if windows is None:
        return None
    video = getattr(field, "data", {}).get("video") if hasattr(field, "data") \
        else None
    if not video or "envelope" not in video or "demod" not in video:
        return None
    envelope = np.asarray(video["envelope"], dtype=np.float64)
    demod = np.asarray(video["demod"], dtype=np.float64)
    locations = getattr(field, "linelocs", None)
    if locations is None or envelope.size < 4096:
        return None
    values = np.asarray(locations, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 32:
        return None
    # the last location opens a line the field's buffer does not finish
    values = values[:-1]
    raw = _raw_field(field)
    if raw is None:
        return None
    rate = float(geometry["sample_rate_hz"])
    limit = min(envelope.size, demod.size, raw.size)

    # THE RISE SEARCH SPAN is a window like the others and is validated with
    # them, so every reading below comes off the same lines.
    # A span bracketing the pulse's TRAILING edge, half a pulse width either
    # side of where the format puts it. Wide enough that a pulse arriving
    # early or late is still inside it and narrow enough to hold one crossing.
    rise_span = (0.5 * windows["sync_us"], 1.5 * windows["sync_us"])
    lines = _common_lines(rate, values,
                          (windows["tip"], windows["burst"], windows["porch"],
                           rise_span), limit)
    if lines is None:
        return None

    record = {
        "parity": int(bool(getattr(field, "isFirstField", False))),
        "level_nepers": {},
        "frequency_hz": {},
        "scatter_nepers": {},
        "lines": {},
    }
    for name in ("tip", "porch"):
        index = _line_windows(rate, lines, windows[name])
        level = envelope[index].mean(axis=1)
        carrier = demod[index].mean(axis=1)
        good = np.isfinite(level) & (level > 0) & np.isfinite(carrier)
        if int(good.sum()) < 16:
            return None
        logged = np.log(level[good])
        record["level_nepers"][name] = float(np.median(logged))
        record["scatter_nepers"][name] = float(
            np.std(logged) / math.sqrt(max(int(good.sum()), 1)))
        record["frequency_hz"][name] = float(np.median(carrier[good]))
        record["lines"][name] = int(good.sum())

    system = "NTSC" if geometry.get("system") in ("NTSC", "PAL_M") \
        else geometry.get("system")
    try:
        chroma = band_delay.chroma_envelope(raw, rate, system)
    except Exception:                                        # noqa: BLE001
        return None
    if chroma.size < limit:
        return None
    index = _line_windows(rate, lines, windows["burst"])
    segment = chroma[index]
    magnitude = np.abs(segment)
    total = magnitude.sum(axis=1)
    amplitude = magnitude.mean(axis=1)
    resultant = np.abs(segment.sum(axis=1))
    good = np.isfinite(amplitude) & (amplitude > 0) & (total > 0)
    if int(good.sum()) < 16:
        return None
    logged = np.log(amplitude[good])
    record["level_nepers"]["burst"] = float(np.median(logged))
    record["scatter_nepers"]["burst"] = float(
        np.std(logged) / math.sqrt(max(int(good.sum()), 1)))
    record["frequency_hz"]["burst"] = float(colour_under.carrier_hz(system))
    record["lines"]["burst"] = int(good.sum())

    # THE COHERENCE, WITH ITS OWN CONTROL - the same band read where NO
    # BURST CAN BE. The colour-under carrier is present in the burst window
    # and absent during the synchronizing pulse, so the band's power in the
    # tip is this capture's own noise floor at that frequency, measured on
    # the same lines through the same filter. The signal's share of the
    # total power is the coherence `PHASE_COHERENCE_FLOOR` is stated
    # against, and it is what tells a burst from its absence.
    #
    # THE WITHIN-WINDOW RESULTANT IS NOT THAT, AND MEASURING BOTH IS HOW
    # THAT WAS FOUND. `|sum s| / sum |s|` over one burst window reads 0.999
    # on the bars decode and 0.854 on `multiburst-y-only`, which carries no
    # chrominance at all - because a 750 kHz-wide filter cannot turn its
    # phase inside a 2.5 microsecond window whatever is fed to it, so that
    # statistic measures the FILTER and would have passed a capture with no
    # subject. It is kept beside the coherence as a diagnostic and named for
    # what it is.
    tip_band = np.abs(chroma[_line_windows(rate, lines, windows["tip"])])
    burst_power = float(np.median(np.mean(magnitude[good] ** 2, axis=1)))
    floor_power = float(np.median(np.mean(tip_band[good] ** 2, axis=1)))
    record["coherence"] = float(np.clip(
        1.0 - floor_power / max(burst_power, 1e-30), 0.0, 1.0))
    record["window_resultant"] = float(np.median(resultant[good]
                                                 / total[good]))
    record["burst_over_floor_db"] = float(
        10.0 * np.log10(max(burst_power, 1e-30) / max(floor_power, 1e-30)))

    # THE BAND-TO-BAND INTERVAL, from the same two channels and the same
    # lines. The sync pulse's rise out of the tip is the luma band's clock
    # and the burst envelope's centroid is the colour-under band's, and the
    # interval between them is a specified constant - so its departure is the
    # two bands' differential delay. Both clocks are in reserved intervals.
    record.update(_band_interval(field, geometry, lines, demod, magnitude,
                                 index, rise_span, system))
    return record


def _band_interval(field, geometry, lines, demod, magnitude, burst_index,
                   rise_span, system):
    """The sync rise to burst centroid interval, in seconds, for this field.

    THE RISE IS FOUND AND NOT ASSUMED. The threshold is the carrier at the
    midpoint IN IRE between the synchronizing level and blanking, which no
    picture content reaches, and the crossing is interpolated between the two
    samples that bracket it. The centroid is weighted by the burst envelope
    above half its own peak, which is `band_delay.burst_readings`' own
    definition, so this stage and that one measure the same quantity the same
    way rather than two nearby ones.

    EVERYTHING THAT MOVES THE FIELD AS A WHOLE CANCELS. One head wrote both
    bands in one pass and one head reads both back in one pass, so the drum's
    phase, the capstan, the tape's stretch and the capture clock move the two
    clocks together and vanish in their difference. What survives depends on
    frequency across the ratio between the bands, and that is dispersion.
    """
    rate = float(geometry["sample_rate_hz"])
    threshold = 0.5 * (float(geometry["sync_tip_hz"])
                       + float(geometry["blanking_hz"]))
    search = _line_windows(rate, lines, rise_span)
    trace = demod[search]
    below = trace < threshold
    crossing = below[:, :-1] & (~below[:, 1:])
    found = crossing.any(axis=1)
    column = np.argmax(crossing, axis=1)
    start = search[np.arange(search.shape[0]), column]
    before = demod[start]
    after = demod[np.minimum(start + 1, demod.size - 1)]
    step = np.where(after != before, (threshold - before) / (after - before),
                    0.0)
    rise = start.astype(np.float64) + step

    peak = magnitude.max(axis=1)
    weight = np.where(magnitude > 0.5 * peak[:, None], magnitude, 0.0)
    total = weight.sum(axis=1)
    position = np.arange(magnitude.shape[1], dtype=np.float64)
    centroid = burst_index[:, 0].astype(np.float64) + np.where(
        total > 0, (weight * position).sum(axis=1) / np.maximum(total, 1e-30),
        np.nan)

    interval = (centroid - rise) / rate
    good = found & np.isfinite(interval) & (total > 0)
    if int(good.sum()) < 16:
        return {"interval_s": float("nan"), "interval_lines": 0}
    specified = band_delay.specified_interval(system)
    return {
        "interval_s": float(np.median(interval[good])),
        "interval_error_s": float(np.std(interval[good])
                                  / math.sqrt(max(int(good.sum()), 1))),
        "interval_lines": int(good.sum()),
        "specified_interval_s": float(specified["centre_us"]) * 1e-6,
    }


def _reserved_band_levels(field, geometry):
    """This field's record, measured once however many stages ask for it."""
    cached = field.__dict__.get(_BAND_LEVELS, _UNREAD)
    if cached is not _UNREAD:
        return cached
    field.__dict__[_BAND_LEVELS] = None
    reading = _measure_band_levels(field, geometry)
    field.__dict__[_BAND_LEVELS] = reading
    if reading is not None:
        field.rf.__dict__.setdefault(
            _BAND_LEVELS, {"fields": []})["fields"].append(reading)
    return reading


def _band_level_history(field):
    return field.rf.__dict__.get(_BAND_LEVELS, {}).get("fields", [])


_BAND_NAMES = ("burst", "tip", "porch")


def _head_band_pairs(field):
    """Consecutive opposite-parity fields, paired - one drum revolution each.

    FIELD PARITY IS THE ONLY HEAD LABEL A DECODE HAS, and which parity is
    head A is not knowable from a time base corrected field, so the labels
    are a convention that holds across one decode. That is enough: every
    quantity below is a difference between the two, and a convention that is
    consistent within a decode fixes its sign and nothing else.

    THE PAIRS DO NOT OVERLAP. Pairing every field with its successor would
    use each field twice and halve the error bars without adding a single
    measurement, which is the arithmetic that makes a scatter look better
    than the evidence supports.
    """
    fields = _band_level_history(field)
    grid, difference, pairs, delays = [], [], 0, []
    index = 0
    while index + 1 < len(fields):
        first, second = fields[index], fields[index + 1]
        if first["parity"] == second["parity"]:
            index += 1
            continue
        head0 = first if first["parity"] == 0 else second
        head1 = second if first["parity"] == 0 else first
        for name in _BAND_NAMES:
            grid.append(head0["frequency_hz"][name])
            difference.append(head0["level_nepers"][name]
                              - head1["level_nepers"][name])
        if np.isfinite(head0.get("interval_s", np.nan)) and \
                np.isfinite(head1.get("interval_s", np.nan)):
            delays.append(head0["interval_s"] - head1["interval_s"])
        pairs += 1
        index += 2
    return {
        "frequency_hz": np.asarray(grid, dtype=np.float64),
        "difference_nepers": np.asarray(difference, dtype=np.float64),
        "delay_difference_s": np.asarray(delays, dtype=np.float64),
        "pairs": pairs,
    }


def _head_series(field, window):
    """Each head's own level series down the tape, for the time axis."""
    out = {0: [], 1: []}
    for record in _band_level_history(field):
        out[record["parity"]].append(record["level_nepers"][window])
    return {which: np.asarray(values, dtype=np.float64)
            for which, values in out.items()}


def _head_mechanics(field, geometry):
    """The format's own head-to-tape mechanics for this decode, or None.

    The decoder's own recorded track width WINS over the table's, which is
    `head_model.mechanics_for`'s own rule: a second copy of a figure the repo
    already holds is a figure that can drift away from it silently.
    """
    speed = geometry.get("tape_speed")
    system = geometry.get("system")
    if speed is None or system not in ("NTSC", "PAL"):
        return None
    return head_model.mechanics_for(
        str(getattr(field.rf.options, "tape_format", "VHS")),
        system, speed, field.rf.DecoderParams)


def _luma_centre_hz(pairs):
    """The geometric centre of the two luma carriers this decode measured.

    GEOMETRIC AND NOT ARITHMETIC, because the quantity it centres is a group
    delay, which is linear in the LOGARITHM of frequency;
    `magnetic_circuit.spec_band_centres` states the distinction and keeps two
    centres for exactly this reason.
    """
    grid = np.asarray(pairs["frequency_hz"], dtype=np.float64)
    if grid.size < 3:
        return float("nan")
    luma = grid.reshape(-1, len(_BAND_NAMES))[:, 1:]
    return float(np.exp(np.mean(np.log(np.maximum(luma, 1.0)))))


def _colour_under_measured_hz(pairs):
    grid = np.asarray(pairs["frequency_hz"], dtype=np.float64)
    if grid.size < 3:
        return float("nan")
    return float(np.median(grid.reshape(-1, len(_BAND_NAMES))[:, 0]))


def _band_group(field, geometry, name):
    """The shared entry point for the five: the mechanics, this field's
    record, and the pooled pairs - or a stated refusal.

    Refuses in one place rather than five, so that a decode of a format this
    tree states no mechanics for gets the same sentence whichever of the
    stages is named.
    """
    mechanics = _head_mechanics(field, geometry)
    if mechanics is None:
        _log_once(field, name + "_mechanics",
                  "model stages: %s declined - this tree states no "
                  "head-to-tape mechanics for %r at a %s track width, so "
                  "no wavelength can be formed and none is borrowed from "
                  "another format", name, geometry.get("system"),
                  geometry.get("track_width_um"))
        return None
    reading = _reserved_band_levels(field, geometry)
    if reading is None:
        _log_once(field, name + "_windows",
                  "model stages: %s declined on this field - the reserved "
                  "intervals could not be read, so no level was taken; the "
                  "sync-only law leaves this stage nothing else to read",
                  name)
        return None
    return {"mechanics": mechanics, "reading": reading,
            "speed_m_s": float(head_model.writing_speed(mechanics)),
            "track_width_m": float(mechanics["track_width_m"])}


# --------------------------------------------------------------------------
# head_model - the physical head, fitted to the two heads' difference
# --------------------------------------------------------------------------

def measure_head_model(field):
    """THE PHYSICAL HEAD, FITTED TO WHAT THE TWO HEADS DIFFER BY.

    Node `head_model`. Applies nothing.

    A magnetic recording head has a known transfer function built from a
    small number of mechanisms, each with a physical parameter established
    long before this format existed - Wallace's separation loss, the gap's
    sinc, the coating's thickness loss, the azimuth's sinc across the track.
    Fitting a measurement to THAT rather than to a free curve is what makes
    the fitted numbers mean something: a separation in nanometres and a gain
    in decibels, rather than spline coefficients.

    IT IS FITTED TO A DIFFERENCE AND NEVER TO ONE HEAD, and that is a
    refusal this stage makes rather than an approximation it accepts. The
    three levels below are read by TWO instruments - the decoder's own
    analytic envelope in the luma band and a narrow complex envelope at the
    colour-under carrier - and each carries its own arbitrary scale. A
    single head's three levels therefore contain an instrument constant that
    no fit can tell from the head's own flat gain. The same two instruments
    read BOTH heads, so the scale cancels exactly in the head difference at
    every frequency, and what is left belongs to the heads. Everything else
    the two share cancels with it: the tape, the record-side path, the
    de-emphasis and the decoder's own filters.

    AND THE FITTED SEPARATION IS TURNED INTO A DIFFERENCE BEFORE IT IS
    REPORTED. `fit_difference_response` builds the model as the fitted head
    against the module's TYPICAL head, so the number it returns is an
    absolute parameter measured from a 50 nanometre reference, and reading it
    as a head-to-head difference would report that reference as though the
    measurement had found it. Subtracting it is what makes this stage and
    `head_differential` agree to the digit; before it they differed by
    exactly 50 nanometres, which is the reference and not a discrepancy.

    TWO PARAMETERS ARE FREE AND THE CHOICE IS THE MODULE'S OWN. `gain_db` is
    a constant and Wallace's spacing loss is linear in frequency, and
    `head_model.CARRIER_LAW_NULL_SPACE` names both as invisible to any
    video-domain measurement, because the carrier-normalization law
    annihilates a constant and a linear term exactly. They are measurable on
    the RF ENVELOPE, where a level is a level and a slope is a slope, which
    is what this stage reads - so this is the one place in the decoder where
    those two can be fitted at all.

    THE TIME AXIS IS THE PER-FIELD SERIES, AND ITS LIMIT IS STATED WITH IT.
    Each head's own level is tracked down the tape and described by
    `head_model.drift`, which reports the slope against an EFFECTIVE sample
    size and runs a unit-root test beside it, because a random walk fits a
    straight line about half the time and a short capture is exactly where
    that fools an observer. `head_model.drift_resolution` states what a
    capture of this length can see at all before anything is looked for.

    AND THE PER-FIELD LOOP CANNOT SEE THE DRUM, WHICH IS MEASURED HERE
    RATHER THAN ASSERTED. The drum turns once every two fields, so sampled
    once a field its phase is exactly `(-1)^k` - the same vector as the head
    A/B alternation. The stage computes both regressors and reports the
    largest difference between them; it is zero, so a once-per-revolution
    effect and a head difference are one direction and no per-field fit
    separates them. Seeing the drum needs a per-LINE loop, which is 525
    times faster and is a missing loop rather than a missing capture.

    THE HEAD'S DELAY IS NOT TAKEN FROM `head_model.group_delay_s`, and the
    reason is measured live below rather than quoted. That function Hilbert
    transforms the log magnitude on whatever grid it is handed, and a
    separation loss is a straight line, so the transform's own periodicity
    dominates the answer: the same head on five grids gives delays that
    differ by an order of magnitude and change sign. The delay reported here
    is `magnetic_circuit.band_delay_difference`, the closed form
    `-(2 d / (pi v)) ln(f2 / f1)`, which has no grid at all.
    """
    geometry = rf_geometry(field)
    if geometry is None:
        return None
    state = _rf_state(field, "head_model")
    common = _band_group(field, geometry, "head_model")
    if common is None:
        return None
    state["fields"] += 1
    reading = common["reading"]
    ldd.logger.debug(
        "head_model: parity %d, burst %.4f, tip %.4f, porch %.4f nepers at "
        "%.4f, %.4f, %.4f MHz over %d, %d, %d lines",
        reading["parity"], reading["level_nepers"]["burst"],
        reading["level_nepers"]["tip"], reading["level_nepers"]["porch"],
        reading["frequency_hz"]["burst"] / 1e6,
        reading["frequency_hz"]["tip"] / 1e6,
        reading["frequency_hz"]["porch"] / 1e6,
        reading["lines"]["burst"], reading["lines"]["tip"],
        reading["lines"]["porch"])

    if state["fields"] < _RF_POOL_FIELDS or state["reported"]:
        field.head_model = state.get("result")
        return state.get("result")

    pairs = _head_band_pairs(field)
    if pairs["pairs"] < 2:
        _log_once(field, "head_model_pairs",
                  "model stages: head_model declined - %d drum revolutions "
                  "carry both parities, and a two-parameter fit on three "
                  "carriers needs at least two so that a residual exists to "
                  "judge it by", pairs["pairs"])
        state["reported"] = True
        return None

    speed = common["speed_m_s"]
    width = common["track_width_m"]
    fitted = head_model.fit_difference_response(
        pairs["frequency_hz"], pairs["difference_nepers"],
        common["mechanics"], free=("gain_db", "spacing_m"))
    if not fitted:
        _log_once(field, "head_model_fit",
                  "model stages: head_model declined - the difference over "
                  "%d places did not admit a two-parameter fit",
                  int(pairs["frequency_hz"].size))
        state["reported"] = True
        return None

    # THE TIME AXIS, and the resolution it is read against.
    limits = head_model.drift_resolution(
        state["fields"], float(geometry.get("field_rate_hz") or 59.94))
    series = _head_series(field, "tip")
    drifts = {which: head_model.drift(values)
              for which, values in series.items() if values.size >= 4}

    # THE DRUM AT THE PER-FIELD NYQUIST, computed rather than claimed.
    count = int(state["fields"])
    index = np.arange(count, dtype=np.float64)
    field_rate = float(geometry.get("field_rate_hz") or 59.94)
    drum_rate = 0.5 * field_rate
    parity_regressor = (-1.0) ** index
    drum_regressor = np.cos(2.0 * np.pi * drum_rate * index / field_rate)
    aliasing = float(np.max(np.abs(parity_regressor - drum_regressor)))

    # THE DELAY, in closed form - and the grid dependence of the other route
    # measured on this decode's own fitted separation.
    #
    # THE FITTED SPACING IS NOT THE DIFFERENCE AND MUST BE TURNED INTO ONE.
    # `fit_difference_response` builds `head_difference_log(fitted, TYPICAL)`,
    # so the value it returns is the first head's own parameter against the
    # module's typical head, and the quantity that means anything between two
    # heads is what it exceeds that reference by. Reading the raw number as a
    # separation difference reports the typical head's 50 nm as though the
    # measurement had found it, and the two stages of this group would then
    # disagree by exactly that constant.
    reference_spacing = float(head_model.TYPICAL["spacing_m"])
    difference_m = float(fitted.get("spacing_m", 0.0)) - reference_spacing
    fitted["spacing_difference_m"] = difference_m
    fitted["reference_spacing_m"] = reference_spacing
    spacing = abs(difference_m)
    low_hz = _colour_under_measured_hz(pairs)
    high_hz = _luma_centre_hz(pairs)
    closed_form_s = magnetic_circuit.band_delay_difference(
        spacing, low_hz=low_hz, high_hz=high_hz, writing_speed_m_s=speed)
    # THREE OF THE FIVE GRIDS `magnetic_circuit.band_delay_difference` names
    # in its own docstring, so this decode's answer and the module's recorded
    # table are taken over the same windows rather than two similar ones.
    grids = ((0.2e6, 7e6, 1024), (0.5e6, 7e6, 1024), (0.05e6, 20e6, 4096))
    grid_delays = []
    for start, stop, places in grids:
        axis = np.linspace(start, stop, places)
        delay = head_model.group_delay_s(axis, speed, width,
                                         spacing_m=max(spacing, 1e-12))
        grid_delays.append(float(np.interp(high_hz, axis, delay)
                                 - np.interp(low_hz, axis, delay)))

    result = {
        "fitted": fitted,
        "pairs": pairs["pairs"],
        "places": int(pairs["frequency_hz"].size),
        "drift": drifts,
        "resolution": limits,
        "aliasing": aliasing,
        "closed_form_delay_s": closed_form_s,
        "grid_delays_s": grid_delays,
        "band_hz": (low_hz, high_hz),
        "mechanics": common["mechanics"],
    }
    state["result"] = result
    state["reported"] = True

    _log_once(
        field, "head_model",
        "model stages: head_model over %d fields - the two heads' RF "
        "envelope difference at %.4f, %.4f and %.4f MHz, %d places from %d "
        "drum revolutions, fits a flat gain of %+.4f dB and a separation "
        "difference of %+.2f nm against the module's %.1f nm typical head, "
        "with a residual of %.5f nepers, explaining "
        "%.1f per cent of the variance and %.1f per cent about zero. Both "
        "parameters are in the carrier law's null space and are measurable "
        "ONLY on the envelope; a difference and never one head, because the "
        "two instruments' own scales cancel only between heads",
        state["fields"], low_hz / 1e6,
        float(pairs["frequency_hz"].reshape(-1, 3)[0, 1]) / 1e6,
        float(pairs["frequency_hz"].reshape(-1, 3)[0, 2]) / 1e6,
        result["places"], pairs["pairs"], fitted.get("gain_db", float("nan")),
        1e9 * difference_m, 1e9 * reference_spacing,
        fitted.get("residual_rms_nepers", float("nan")),
        100.0 * fitted.get("explained", float("nan")),
        100.0 * fitted.get("explained_with_constant", float("nan")))
    for which in sorted(drifts):
        described = drifts[which]
        _log_once(
            field, "head_model_drift_%d" % which,
            "model stages: head_model head %d down the tape - %.3e nepers a "
            "field at %.2f standard errors on %.1f effective samples of %d, "
            "unit-root statistic %.3f against a critical %.3f, so a random "
            "walk is %s. This capture is %.3f s long and resolves drifts "
            "between %.3f and %.2f Hz; nothing slower is distinguishable "
            "from a constant",
            which, described.get("slope_per_field", float("nan")),
            described.get("t", float("nan")),
            described.get("effective_samples", float("nan")),
            int(described.get("count", 0)),
            described.get("unit_root_t", float("nan")),
            described.get("unit_root_critical", float("nan")),
            "not ruled out" if described.get("random_walk_risk", 1.0)
            else "ruled out",
            limits["duration_seconds"], limits["slowest_resolvable_hz"],
            limits["fastest_resolvable_hz"])
    _log_once(
        field, "head_model_nyquist",
        "model stages: head_model - the per-field loop CANNOT see the drum, "
        "measured on this decode. The drum turns at %.3f Hz against a field "
        "rate of %.3f, which is exactly the per-field Nyquist, so its "
        "sampled phase and the head A/B alternation are the same vector to "
        "%.3g over %d fields. A once-per-revolution effect and a head "
        "difference are one direction here; separating them needs a "
        "per-LINE loop, %d times faster, which is a missing loop and not a "
        "missing capture",
        drum_rate, field_rate, aliasing, count,
        int(round(float(field.rf.SysParams["frame_lines"]) / 2.0)))
    _log_once(
        field, "head_model_delay",
        "model stages: head_model - the head's own band-to-band delay is "
        "%+.2f ns by the closed form -(2 d / pi v) ln(f2/f1) at %.4f m/s, "
        "and it is NOT taken from group_delay_s. That route Hilbert "
        "transforms a straight line on the caller's grid, and on this "
        "decode's own fitted separation it returns %+.1f, %+.1f and %+.1f ns "
        "on three grids where the closed form returns %+.2f - an order of "
        "magnitude and a change of sign, decided by a window nobody knows "
        "to choose",
        1e9 * closed_form_s, speed, 1e9 * grid_delays[0],
        1e9 * grid_delays[1], 1e9 * grid_delays[2], 1e9 * closed_form_s)
    field.head_model = result
    return result


# --------------------------------------------------------------------------
# head_differential - the head pair's amplitude, phase and frequency
# --------------------------------------------------------------------------

_SYNC_PROFILES = "_head_sync_profiles"


def _pool_sync_profile(field):
    """This field's mean sync profile, filed under the head that read it.

    The profile is the pulse as it arrived, in IRE, averaged over the
    picture lines of the field - and the averaging is a measurement of the
    channel rather than of the content, because the pulse is the same
    specified shape on every one of them. Kept per field rather than
    accumulated, so the fields can be split into disjoint halves afterwards:
    `fixed_response_verdict` needs each head's own within-head variation,
    and a running sum cannot be taken apart again.
    """
    store = field.rf.__dict__.setdefault(_SYNC_PROFILES, {0: [], 1: []})
    if field.__dict__.get(_SYNC_PROFILES):
        return store
    field.__dict__[_SYNC_PROFILES] = True
    try:
        profile = _sync_profile(field, geometry(field),
                               sync_segment_geometry(field))
    except Exception:                                        # noqa: BLE001
        return store
    if profile is None or not np.all(np.isfinite(profile)):
        return store
    store[int(bool(getattr(field, "isFirstField", False)))].append(
        np.asarray(profile, dtype=np.float64))
    return store


def _profile_phase(field, store, upper_hz):
    """The two heads' pooled profiles as a COMPLEX ratio, and its phase.

    A PHASE NEEDS A REFERENCE AND THE SYNC PULSE IS THE ONE THE
    SPECIFICATION PRINTS. The two heads' pooled profiles are transformed and
    divided, so everything the two share - the pulse itself, the decoder's
    filters, the de-emphasis - divides out and what is left is the transfer
    between the two heads. Complex throughout: no magnitude is taken at any
    boundary, which is what lets the delay be separated from the excess.

    The band runs from the segment's own resolution, `1/T`, to the format's
    luma limit, because outside that the pooled pulse carries nothing and a
    ratio of two small numbers is not a measurement.
    """
    lengths = {which: {len(row) for row in rows}
               for which, rows in store.items()}
    if not lengths[0] or not lengths[1] or len(lengths[0]) != 1 \
            or lengths[0] != lengths[1]:
        return None
    length = lengths[0].pop()
    first = np.mean(np.asarray(store[0], dtype=np.float64), axis=0)
    second = np.mean(np.asarray(store[1], dtype=np.float64), axis=0)
    rate = float(field.rf.SysParams["outfreq"]) * 1e6
    grid = np.fft.rfftfreq(length, d=1.0 / rate)
    spectrum_a = np.fft.rfft(first)
    spectrum_b = np.fft.rfft(second)
    inside = (grid > rate / float(length)) & (grid <= float(upper_hz))
    if int(inside.sum()) < 8:
        return None
    ratio = spectrum_a[inside] / np.where(spectrum_b[inside] == 0,
                                          np.nan, spectrum_b[inside])
    good = np.isfinite(ratio)
    if int(good.sum()) < 8:
        return None
    return {
        "frequency_hz": grid[inside][good],
        "ratio": ratio[good],
        "phase_rad": np.unwrap(np.angle(ratio[good])),
        "log_magnitude": np.log(np.abs(ratio[good])),
        "bins": int(good.sum()),
        "sample_rate_hz": rate,
        "length": length,
    }


def measure_head_differential(field):
    """THE HEAD PAIR AS AN AMPLITUDE, A PHASE AND A FREQUENCY.

    Node `head_differential`. Applies nothing.

    Ethan: *"It will be the amplitude, phase, and frequency of this
    differential pair. We have three sets of three differential pairs, which
    form a matrix to calculate over the residuals."* This is the head pair's
    set of three, taken where the decoder can take it.

    A drum carries two video heads and they are not identical. Everything the
    two share cancels in their difference - the record-side path, the tape,
    the de-emphasis, the decoder's own filters - so the difference is the
    honest quantity, and exactly three physical parameters can survive it,
    each with its own closed form.

    AMPLITUDE, on the two bands at once. A separation difference contributes
    `-2 pi dd f / v` and a gain difference contributes a constant, and over
    ONE band those two are nearly the same shape. Over the two bands the
    format itself provides they are not, and the module measures the
    improvement: condition 27.0 and a clearance error of 186 nm over the luma
    band alone against condition 3.1 and 23 nm over both. The stage solves
    them jointly on the three reserved carriers and reports which mechanism
    the departure actually is - a pure clearance predicts a band-to-band
    ratio equal to the frequency ratio and a pure gain predicts one, and
    nothing is fitted to decide between them.

    PHASE, on the sync pulse, because a phase needs a reference and the sync
    pulse is the one the specification prints. The two heads' pooled profiles
    are transformed and divided and `residual_phase_bound` removes the
    best-fitting pure delay; what remains is EXCESS phase, and that is the
    discriminating quantity rather than the delay itself. Wallace's law is a
    real attenuation and carries no phase at all, so a separation leaves no
    excess, while a gap or an azimuth difference would turn the phase through
    its sinc null. The offline instrument puts the excess within plus or
    minus 1.7 degrees on four captures; this stage reports its own figure
    against that bound rather than replacing it.

    FREQUENCY, as a BOUND, and that is the module's own verdict rather than a
    limitation of the wiring. `frequency_verdict` will not return a
    coefficient without an out-of-sample result that clears, and none is
    offered here. The stage reports the bound the data gives and the bound
    geometry gives - a head standing proud by `dr` sweeps at `(R + dr)/R`,
    and `dr` is the same protrusion the amplitude quantity measures as a
    separation, so the amplitude measurement PREDICTS the frequency
    quantity's size - and says which of the two binds.

    THE CONTROLS RUN BEFORE THE READING. `azimuth_cancels_control` asserts
    that a common drum misalignment cancels exactly between two heads written
    at opposite azimuth, which is why azimuth is not among the three;
    `wallace_round_trip_control` exercises the SIZE of the law rather than
    its shape; `collinear_control` asserts that a family of five separations
    is one direction, which is the separability law. A count taken through a
    construction that fails one of them would be reporting the construction,
    so the stage declines instead.
    """
    geometry = rf_geometry(field)
    if geometry is None:
        return None
    state = _rf_state(field, "head_differential")
    common = _band_group(field, geometry, "head_differential")
    if common is None:
        return None
    store = _pool_sync_profile(field)
    state["fields"] += 1
    if state["fields"] < _RF_POOL_FIELDS or state["reported"]:
        field.head_differential = state.get("result")
        return state.get("result")

    pairs = _head_band_pairs(field)
    if pairs["pairs"] < 2:
        _log_once(field, "head_differential_pairs",
                  "model stages: head_differential declined - %d drum "
                  "revolutions carry both parities, and the joint gain and "
                  "clearance solve needs at least four places",
                  pairs["pairs"])
        state["reported"] = True
        return None

    speed = common["speed_m_s"]
    width = common["track_width_m"]
    grid = pairs["frequency_hz"].reshape(-1, len(_BAND_NAMES))
    departure = pairs["difference_nepers"].reshape(-1, len(_BAND_NAMES))

    # THE CONTROLS, first, on this decode's own band.
    control_grid = np.linspace(geometry["band_hz"][0], geometry["band_hz"][1],
                               head_differential.EVIDENCE_PLACES)
    azimuth = head_differential.azimuth_cancels_control(control_grid)
    wallace = head_differential.wallace_round_trip_control()
    collinear = head_differential.collinear_control(
        control_grid, geometry["system"], geometry["tape_speed"],
        rf_params=field.rf.DecoderParams)
    failed = [name for name, result in (("azimuth", azimuth),
                                        ("wallace", wallace),
                                        ("collinear", collinear))
              if not result.get("passes")]
    if failed:
        _log_once(
            field, "head_differential_control",
            "model stages: head_differential DECLINES - its own control "
            "fails on this band: %s. A common drum misalignment leaves "
            "%.3g nepers rms where it must leave zero, the Wallace round "
            "trip errs by %.3g dB where it must be exact, and a family of "
            "five separations spans %.4f directions where it must span "
            "exactly one. A reading taken through a construction that fails "
            "this would be reporting the construction",
            ", ".join(failed), azimuth["common_misalignment_rms_nepers"],
            wallace["round_trip_error_db"], collinear["effective"])
        state["reported"] = True
        return None

    # AMPLITUDE. The joint gain and clearance over the two bands.
    amplitude = head_differential.two_band_difference(
        grid[:, 1:].ravel(), departure[:, 1:].ravel(),
        grid[:, 0], departure[:, 0], speed)
    luma_hz = _luma_centre_hz(pairs)
    colour_hz = _colour_under_measured_hz(pairs)
    discriminator = head_differential.band_ratio_discriminator(luma_hz,
                                                               colour_hz)
    luma_departure = float(np.mean(departure[:, 1:]))
    colour_departure = float(np.mean(departure[:, 0]))
    measured_ratio = (luma_departure / colour_departure
                      if abs(colour_departure) > 0 else float("nan"))
    level_db = luma_departure * head_model.DB_PER_NEPER
    shape_db = float(np.std(departure[:, 1:]) * head_model.DB_PER_NEPER)
    separation = head_differential.spacing_from_level(level_db, luma_hz,
                                                      speed)
    consistency = head_differential.wallace_consistency(level_db, shape_db,
                                                        luma_hz)

    # PHASE. The two heads' pooled sync profiles as a complex ratio.
    # THE BAND IS THE FORMAT'S OWN LUMA BAND AND NOT THE DECODER'S LOW PASS.
    # `video_lpf_freq` is 6.6 MHz on this format - the decoder's filter, not
    # the channel - and above the recorded luma band the pooled pulse carries
    # no energy, so a ratio there is a ratio of two small numbers. Read to
    # 6.5 MHz the excess phase came to 6.618 degrees rms against an offline
    # bound of 1.7; read to the format's stated band it is the same
    # measurement without the bins that hold nothing.
    upper = _LUMA_BAND_HZ.get(str(field.rf.options.tape_format).upper())
    if upper is None:
        _log_once(field, "head_differential_band",
                  "model stages: head_differential PHASE declined - this "
                  "tree states no luma band for %r, and a band borrowed "
                  "from a neighbouring format would be a constant invented "
                  "at a call site",
                  str(field.rf.options.tape_format))
    transfer = _profile_phase(field, store, upper) if upper else None
    phase = None
    if transfer is not None:
        phase = head_differential.residual_phase_bound(
            transfer["phase_rad"], transfer["frequency_hz"])

    # FREQUENCY. A bound, and which of the two bounds binds.
    verdict = head_differential.frequency_verdict(
        pairs["frequency_hz"], pairs["difference_nepers"], speed, width,
        drum_diameter_m=float(common["mechanics"]["drum_diameter_m"]),
        spacing_difference_m=separation)

    # IS EACH HEAD'S RESPONSE FIXED? The between-head difference against each
    # head's own within-head variation, on disjoint halves of its own fields.
    fixed = None
    if len(store[0]) >= 2 and len(store[1]) >= 2:
        halves = {}
        for which, rows in store.items():
            stack = np.asarray(rows, dtype=np.float64)
            middle = stack.shape[0] // 2
            halves[which] = (stack[:middle].mean(axis=0),
                             stack[middle:].mean(axis=0))
        span = state["fields"] / float(geometry.get("field_rate_hz") or 59.94)
        fixed = head_differential.fixed_response_verdict(
            halves[0], halves[1], span_s=span)

    result = {"amplitude": amplitude, "phase": phase, "frequency": verdict,
              "fixed": fixed, "separation_m": separation,
              "consistency": consistency, "discriminator": discriminator,
              "measured_band_ratio": measured_ratio,
              "controls": {"azimuth": azimuth, "wallace": wallace,
                           "collinear": collinear},
              "pairs": pairs["pairs"]}
    state["result"] = result
    state["reported"] = True

    _log_once(
        field, "head_differential_amplitude",
        "model stages: head_differential AMPLITUDE over %d fields and %d "
        "drum revolutions - solved jointly on the two bands, the head pair "
        "differs by %+.4f dB of flat gain (%s) and %+.2f nm of clearance "
        "(%s), the gain taking %.1f per cent of the departure and the "
        "clearance %.1f - the two shares exceed a hundred because the terms "
        "partly CANCEL across the band, which is what a joint solve on two "
        "bands looks like when one carries most of the departure - with a "
        "residual of %.5f nepers. The band ratio decides which mechanism it "
        "is with nothing fitted: a pure clearance predicts %.3f and a pure "
        "gain 1.000, and this decode measures %.3f",
        state["fields"], pairs["pairs"],
        amplitude.get("gain_db", float("nan")),
        "significant" if amplitude.get("gain_significant")
        else "not significant",
        1e9 * amplitude.get("clearance_m", float("nan")),
        "significant" if amplitude.get("clearance_significant")
        else "not significant",
        100.0 * amplitude.get("gain_share", float("nan")),
        100.0 * amplitude.get("clearance_share", float("nan")),
        amplitude.get("residual_rms", float("nan")),
        discriminator["clearance_predicts"], measured_ratio)
    _log_once(
        field, "head_differential_wallace",
        "model stages: head_differential - the luma-band level split of "
        "%+.4f dB inverts through Wallace's law to %.2f nm at %.4f MHz, and "
        "the consistency test the measurement could have failed: a pure "
        "separation is a straight line through the origin, so its shape over "
        "its level FIXES the band it was measured on, and %.4f dB rms of "
        "shape implies a half width of %.4f MHz against the %.4f MHz this "
        "stage actually read",
        level_db, 1e9 * separation, luma_hz / 1e6, shape_db,
        consistency["implied_half_width_hz"] / 1e6,
        0.5 * abs(float(np.mean(grid[:, 2] - grid[:, 1]))) / 1e6)
    if phase is not None:
        _log_once(
            field, "head_differential_phase",
            "model stages: head_differential PHASE over %d bins of the two "
            "heads' pooled sync profiles - %d profiles for head 0 and %d for "
            "head 1, which is the group's own pooling span and is thin for a "
            "phase - %.4f to %.4f MHz: a pure delay of "
            "%+.2f ns and an EXCESS of %.3f degrees rms, %.3f peak, against "
            "the %.1f degrees the offline instrument bounds it at on four "
            "captures. The excess is the discriminating quantity: Wallace's "
            "law is a real attenuation and carries no phase, so a separation "
            "leaves none, while a gap or an azimuth difference would turn "
            "the phase through its sinc null",
            transfer["bins"], len(store[0]), len(store[1]),
            transfer["frequency_hz"][0] / 1e6,
            transfer["frequency_hz"][-1] / 1e6, 1e9 * phase["delay_s"],
            phase["excess_rms_degrees"], phase["excess_peak_degrees"],
            head_differential.MEASURED_PHASE_BOUND_DEGREES)
    else:
        _log_once(field, "head_differential_phase",
                  "model stages: head_differential PHASE declined - the two "
                  "heads' pooled sync profiles are not the same length on "
                  "this decode, so no complex ratio between them exists and "
                  "none is manufactured by resampling one onto the other")
    _log_once(
        field, "head_differential_frequency",
        "model stages: head_differential FREQUENCY is reported as %s, which "
        "is the module's own verdict. The data bounds the fractional scaling "
        "at %.3g and geometry at %.3g, so %s binds - a head standing proud "
        "by dr sweeps at (R+dr)/R and dr is the protrusion the amplitude "
        "quantity already measured, so the amplitude measurement predicts "
        "this one's size. It is not merely unmeasured here; at this evidence "
        "level it is unmeasurable, and more care with the estimator does not "
        "change that",
        verdict.get("report_as", "a bound"),
        verdict.get("bound_from_the_data", float("nan")),
        verdict.get("bound_from_geometry", float("nan")),
        verdict.get("binds", "the data"))
    if fixed is not None:
        _log_once(
            field, "head_differential_fixed",
            "model stages: head_differential - is each head's response "
            "FIXED? Between the heads the level differs by %+.4f IRE and the "
            "shape by %.4f rms, against a within-head %.4f and %.4f taken on "
            "disjoint halves of each head's own %d and %d fields: ratios of "
            "%.2f and %.2f, so the level is %s and the shape is %s. The span "
            "is "
            "%.3f s, which tests reproducibility rather than stability - the "
            "transport's drift lives from a quarter second upward, so a "
            "verdict from this span speaks for the CIRCUIT alone, which is "
            "exactly the part that should be fixed",
            fixed["between_level"], fixed["between_shape_rms"],
            fixed["within_head"]["A"]["level"],
            fixed["within_head"]["A"]["shape"], len(store[0]), len(store[1]),
            fixed["level_ratio"], fixed["shape_ratio"],
            "resolved" if fixed["level_resolved"] else "not resolved",
            "resolved" if fixed["shape_resolved"] else "not resolved",
            fixed["span_s"])
    _log_once(
        field, "head_differential_controls",
        "model stages: head_differential controls all pass on this band - a "
        "common drum misalignment of %.2f degrees cancels to %.3g nepers rms "
        "between two heads written at opposite azimuth while independent "
        "construction errors leave %.3g, which is why azimuth is not one of "
        "the three; the Wallace round trip errs by %.3g dB; and a family of "
        "five separations spans %.6f of 5 directions, which is the "
        "separability law - a family differing only in a RATE is one "
        "direction",
        azimuth["misalignment_degrees"],
        azimuth["common_misalignment_rms_nepers"],
        azimuth["independent_errors_rms_nepers"],
        wallace["round_trip_error_db"], collinear["effective"])
    field.head_differential = result
    return result


# --------------------------------------------------------------------------
# magnetic - the record level as an axis, and where the cap takes over
# --------------------------------------------------------------------------

def measure_magnetic(field):
    """THE RECORD LEVEL AS AN AXIS, AND THE CAP THAT MAKES IT ONE.

    Node `magnetic`. Applies nothing.

    Ethan: *"I think the physical models we used are extra dimensions into
    the magnetic information, such as record level for example. There should
    be a shape that defines that against the curve the tape will exhibit."*

    RECORD LEVEL IS NOT A NEW FREQUENCY SHAPE, and the module has already
    measured that: it reaches the response through the recording depth and
    the transition length, at coherences of 0.9995 and 1.0000 with losses
    already in the family, and adding both LOWERS the effective count from
    1.56 to 1.40. A second copy of a direction only worsens the conditioning.

    WHAT IS NEW IS THE LEVEL DEPENDENCE, AND THE SELF-DEMAGNETISATION CAP IS
    THE WHOLE REASON. `d <= lambda / (2 pi)` binds at a DIFFERENT FREQUENCY
    FOR EVERY LEVEL, so the crossover between level-limited and cap-limited
    recording moves along the band as the level changes, and that movement is
    the shape. Differentiating with respect to level gives 2.90 of 5
    directions with the cap and 1.01 without it.

    THE CONTROL RUNS FIRST AND THE STAGE DECLINES IF IT FAILS. Without the
    cap a depth linear in level must give exactly ONE direction however many
    levels are used, because the response is then a fixed function of a
    single scaled variable. A first estimate of this axis read 4.65 of 5 and
    was wrong - the depth had been written as a fraction of the cap, which
    makes the dimensionless group independent of frequency - and what exposed
    it was this control reading 4.97 where it had to read 1.00.

    THE PHASE OF THE DECLARED SIGNATURE IS EXACTLY ZERO AND THE STAGE SAYS
    SO. `magnetic.level_signature` returns a ratio of two real thickness
    losses cast to complex, so every phase in it is zero to machine
    precision - which is what a cast does rather than what the physics says.
    A thickness loss is a causal minimum-phase response and its phase is
    fixed by its magnitude. The stage MEASURES the declared phase, reports it
    as zero, and reports `minimum_phase_level_signature` beside it. THAT
    PHASE IS DERIVED FROM THE MAGNITUDE AND IS NOT AN INDEPENDENT
    MEASUREMENT; nothing in a decode observes it, and both this log line and
    that function's docstring say so.

    THE TIME AXIS IS DERIVED TOO, AND ITS LIMIT IS THE HONEST PART OF IT. The
    record level was set once, when the recording was made, and no decode
    moves it; there is no per-field record level to measure. What a decode
    does supply is the playback envelope's own level down the tape, which is
    turned into a drive ratio against the decode's own median and carried
    through `crossover_hz`. So the series is a DERIVED level on a measured
    axis, and it is reported as derived.

    THE NOMINAL DEPTH IS AN ASSUMPTION AND ITS WEIGHT IS REPORTED WITH IT.
    `magnetic.NOMINAL_DEPTH_M` is a scale rather than a constant of the
    format - its own comment says the caller supplies the tape's own where it
    is known, and no measurement in this project supplies one. The crossover
    is inversely proportional to it, so the stage states the assumption, its
    source, and what the crossover would become if the depth were half or
    twice what is assumed.
    """
    geometry = rf_geometry(field)
    if geometry is None:
        return None
    state = _rf_state(field, "magnetic")
    common = _band_group(field, geometry, "magnetic")
    if common is None:
        return None
    state["fields"] += 1
    if state["fields"] < _RF_POOL_FIELDS or state["reported"]:
        field.magnetic = state.get("result")
        return state.get("result")

    speed = common["speed_m_s"]
    low, high = geometry["band_hz"]
    grid = np.linspace(low, high, head_differential.EVIDENCE_PLACES)

    control = magnetic.linear_control(grid, speed)
    if not control["passes"]:
        _log_once(
            field, "magnetic_control",
            "model stages: magnetic DECLINES - its own control fails on this "
            "band. With the cap removed a depth linear in level must span "
            "exactly ONE direction and it spans %.4f. A level axis taken "
            "through a construction that fails this would be reporting the "
            "construction, which is exactly how an earlier estimate of this "
            "quantity read 4.65 of 5", control["effective"])
        state["reported"] = True
        return None

    # THE LEVELS ARE THE MODULE'S OWN and are not restated here: a second
    # copy of the set could drift away from the one the module's tests quote.
    axis = magnetic.level_axis(grid, speed)
    levels = list(axis["levels"])
    declared = magnetic.level_signature(grid, 1.0, speed)
    derived = magnetic.minimum_phase_level_signature(grid, 1.0, speed)
    declared_phase = float(np.degrees(
        np.sqrt(np.mean(np.angle(declared) ** 2))))
    derived_phase = float(np.degrees(
        np.sqrt(np.mean(np.angle(derived) ** 2))))

    # THE DERIVED TIME AXIS: the playback envelope's own level, per field,
    # as a drive ratio against this decode's own median.
    series = _head_series(field, "tip")
    pooled = np.concatenate([values for values in series.values()
                             if values.size])
    reference = float(np.median(pooled))
    drives = np.exp(pooled - reference)
    crossovers = np.asarray(
        [magnetic.crossover_hz(float(drive), speed) for drive in drives])
    drift = head_model.drift(drives)

    nominal = magnetic.NOMINAL_DEPTH_M
    result = {
        "control": control,
        "axis": axis,
        "declared_phase_degrees": declared_phase,
        "derived_phase_degrees": derived_phase,
        "crossovers_hz": crossovers,
        "drives": drives,
        "drift": drift,
        "nominal_depth_m": nominal,
        "band_hz": (low, high),
    }
    state["result"] = result
    state["reported"] = True

    _log_once(
        field, "magnetic",
        "model stages: magnetic on this decode's own %.3f to %.3f MHz at "
        "%.4f m/s - the record level spans %.3f of %d directions at a "
        "condition of %.3g, and its whole information content is the "
        "self-demagnetisation cap binding at a different frequency for every "
        "level: the crossover runs from %.3f MHz at a level of %.1f to %.3f "
        "MHz at %.1f. The control passes at %.4f of an expected 1.00",
        low / 1e6, high / 1e6, speed, axis["effective"], axis["count"],
        axis["condition"], axis["crossovers_hz"][0] / 1e6, levels[0],
        axis["crossovers_hz"][-1] / 1e6, levels[-1],
        control["effective"])
    _log_once(
        field, "magnetic_phase",
        "model stages: magnetic - the declared level signature's phase is "
        "%.4f degrees rms, which is EXACTLY ZERO and is a cast rather than a "
        "measurement. A thickness loss is causal and minimum phase, so its "
        "phase is fixed by its magnitude, and reconstructing it gives %.4f "
        "degrees rms on the same magnitude. THAT PHASE IS DERIVED AND NOT "
        "MEASURED: nothing in a decode observes the phase of a record-level "
        "dependence, and it is offered as the phase the magnitude implies "
        "rather than as a second measurement",
        declared_phase, derived_phase)
    _log_once(
        field, "magnetic_time",
        "model stages: magnetic time axis over %d fields, and it is DERIVED. "
        "The record level was set once when the recording was made and no "
        "decode moves it, so there is no per-field record level to measure. "
        "What IS measured is the playback envelope, %.4f nepers of spread "
        "about its own median; read AS IF it were a drive ratio it runs "
        "%.4f to %.4f and moves the crossover between %.3f and %.3f MHz. "
        "The envelope also carries every playback-side loss, so that reading "
        "is an upper bound on what the record level could have contributed "
        "and not an attribution to it. Its slope is %.3e a field at %.2f "
        "standard errors, and a random walk is %s",
        state["fields"], float(np.std(pooled)), float(drives.min()),
        float(drives.max()), float(crossovers.min()) / 1e6,
        float(crossovers.max()) / 1e6,
        drift.get("slope_per_field", float("nan")),
        drift.get("t", float("nan")),
        "not ruled out" if drift.get("random_walk_risk", 1.0)
        else "ruled out")
    _log_once(
        field, "magnetic_assumption",
        "model stages: magnetic - the nominal recording depth is an "
        "ASSUMPTION of %.3f um and the crossover is inversely proportional "
        "to it. Its source is the module's own scale, whose comment states "
        "that the caller supplies the tape's own where it is known and no "
        "measurement in this project supplies one. At half that depth the "
        "level-1.0 crossover would be %.3f MHz and at twice it %.3f, against "
        "the %.3f MHz assumed here and a band that ends at %.3f",
        1e6 * nominal, magnetic.crossover_hz(1.0, speed, 0.5 * nominal) / 1e6,
        magnetic.crossover_hz(1.0, speed, 2.0 * nominal) / 1e6,
        magnetic.crossover_hz(1.0, speed, nominal) / 1e6, high / 1e6)
    field.magnetic = result
    return result


# --------------------------------------------------------------------------
# magnetic_circuit - one separation, read in amplitude and in time
# --------------------------------------------------------------------------

def measure_magnetic_circuit(field):
    """THE SAME SEPARATION READ TWICE, SO THAT THE TWO ROUTES CAN DISAGREE.

    Node `magnetic_circuit`. Applies nothing.

    Ethan: *"The difference in time between the luma and chroma bands are the
    measurement we can use to observe the delay on each video head."* Wallace's
    separation loss is minimum phase, so the same `d` appears in the two
    bands' LEVELS and in their TIMING, and the two readings share that
    parameter and nothing else. One is a difference of levels and the other a
    difference of slopes of phase; one responds to every loss in the path and
    the other only to the part that is minimum phase; and their frequency
    dependence differs - `f2 - f1` against `ln(f2 / f1)`. A discrepancy
    between them says the path is not the pure minimum-phase separation loss
    the model assumes, which is a finding rather than a failure. That is what
    a pair is for.

    THE TIME ROUTE EXISTS BECAUSE THE CONSTANT CANCELS. Two captures with no
    common clock cannot measure an absolute delay: any instrument comparing
    them removes a linear phase, which removes a constant from the group
    delay at every frequency alike. A DIFFERENCE BETWEEN TWO BANDS is
    invariant under exactly that removal, so the one time quantity the
    apparatus can see is the one the physics puts information into.

    BOTH CLOCKS ARE IN RESERVED INTERVALS. The luma band's clock is the rise
    out of the sync tip, thresholded at the midpoint IN IRE between the
    synchronizing level and blanking, which no picture content reaches; the
    colour-under band's is the burst envelope's own centroid. Their interval
    is a specified constant, so its departure is the two bands' differential
    delay, and the HEAD difference of that departure is the head's own.

    THE COHERENCE GATE IS LOAD-BEARING AND THE ERROR BAR IS NOT ENOUGH. The
    offline measurement records a luminance-only capture, at a colour-under
    coherence of 0.127, reporting a head delay difference of
    +2229.70 +/- 104.29 ns - twenty-one standard deviations of nonsense that a
    reported error bar alone would have accepted. `PHASE_COHERENCE_FLOOR` is
    `1/sqrt(3)`, the coherence at which a single average's phase error reaches
    one radian, and this stage measures its own coherence to feed it: the
    burst's within-window resultant over its own magnitude, one for a pure
    tone and near zero for noise.

    THE COERCIVITY QUESTION IS DISPOSED OF RATHER THAN LEFT OPEN.
    `coercivity_constrains` gives the transition length and the loss it costs
    at the carrier, and this project's own arithmetic makes the record field
    the chrominance would need unreachable by a factor of 15.934. That is not
    a requirement that fails. SMPTE 32M 7.5.1.2.2 states that the chrominance
    is recorded with the luminance FM acting as bias, and `tape_bias` measures
    what that bias leaves behind - a luma-dependent chroma gain of
    +2.084 +/- 0.048 dB per MHz at 43 sigma. The factor is the quantitative
    reason the format had to bias the chroma, and the stage reports it as
    that rather than as a shortfall.

    THE RECORD SIDE'S PHASE IS REFUSED IN THE LOG. A record head reaches
    amplitude and not phase without a matched record tap; and even with one,
    on a single deck the head that wrote a track is the head that reads it, so
    the write and read shares are not two terms to be separated but one head
    measured twice. This decode has no tap, and the stage says so rather than
    attributing its reading to a side.
    """
    geometry = rf_geometry(field)
    if geometry is None:
        return None
    state = _rf_state(field, "magnetic_circuit")
    common = _band_group(field, geometry, "magnetic_circuit")
    if common is None:
        return None
    state["fields"] += 1
    if state["fields"] < _RF_POOL_FIELDS or state["reported"]:
        field.magnetic_circuit = state.get("result")
        return state.get("result")

    pairs = _head_band_pairs(field)
    if pairs["pairs"] < 2 or pairs["delay_difference_s"].size < 2:
        _log_once(field, "magnetic_circuit_pairs",
                  "model stages: magnetic_circuit declined - %d drum "
                  "revolutions carry both parities and %d carry both clocks, "
                  "and a pair whose two routes cannot be given error bars is "
                  "not a pair", pairs["pairs"],
                  int(pairs["delay_difference_s"].size))
        state["reported"] = True
        return None

    speed = common["speed_m_s"]
    bands = magnetic_circuit.spec_band_centres()
    grid = pairs["frequency_hz"].reshape(-1, len(_BAND_NAMES))
    departure = pairs["difference_nepers"].reshape(-1, len(_BAND_NAMES))
    colour_hz = float(np.median(grid[:, 0]))
    # the two centres are taken apart, which is the module's own rule: a
    # level averaged over a band lands on its ARITHMETIC centre and a delay
    # on its GEOMETRIC one, because one law is linear in frequency and the
    # other in its logarithm
    luma_linear_hz = float(np.mean(grid[:, 1:]))
    luma_geometric_hz = float(np.exp(np.mean(np.log(
        np.maximum(grid[:, 1:], 1.0)))))

    tilt = float(np.mean(departure[:, 1:]) - np.mean(departure[:, 0]))
    per_pair_tilt = np.mean(departure[:, 1:], axis=1) - departure[:, 0]
    tilt_error = float(np.std(per_pair_tilt)
                       / math.sqrt(max(per_pair_tilt.size, 1)))
    delays = pairs["delay_difference_s"]
    delay = float(np.mean(delays))
    delay_error = float(np.std(delays) / math.sqrt(max(delays.size, 1)))
    records = [record for record in _band_level_history(field)
               if "coherence" in record]
    coherence = float(np.median([record["coherence"] for record in records]))
    resultant = float(np.median([record["window_resultant"]
                                 for record in records]))
    over_floor = float(np.median([record["burst_over_floor_db"]
                                  for record in records]))

    verdict = magnetic_circuit.two_band_delay_pair(
        delay, tilt, delay_error, tilt_error, coherence=coherence,
        delay_low_hz=colour_hz, delay_high_hz=luma_geometric_hz,
        tilt_low_hz=colour_hz, tilt_high_hz=luma_linear_hz,
        writing_speed_m_s=speed)
    coercivity = magnetic_circuit.coercivity_constrains(
        writing_speed_m_s=speed, carrier_hz=luma_linear_hz)
    observability = magnetic_circuit.record_side_observability()

    result = {"pair": verdict, "bands": bands, "coercivity": coercivity,
              "observability": observability, "tilt_nepers": tilt,
              "delay_s": delay, "coherence": coherence,
              "centres_hz": (colour_hz, luma_linear_hz, luma_geometric_hz)}
    state["result"] = result
    state["reported"] = True

    _log_once(
        field, "magnetic_circuit",
        "model stages: magnetic_circuit over %d fields and %d drum "
        "revolutions - the head pair's separation read TWICE on this "
        "decode's own bands, %.4f MHz against %.4f MHz arithmetic and %.4f "
        "geometric. The AMPLITUDE route gives %+.2f +/- %.2f nm from a tilt "
        "of %+.5f +/- %.5f nepers at %.1f nm a neper; the TIME route gives "
        "%+.2f +/- %.2f nm from a delay of %+.2f +/- %.2f ns. They differ by "
        "%+.2f +/- %.2f nm, %.2f standard deviations, at a ratio of %.3f - "
        "and a discrepancy is the finding, because a pure minimum-phase "
        "separation would have to give both numbers alike",
        state["fields"], pairs["pairs"], colour_hz / 1e6,
        luma_linear_hz / 1e6, luma_geometric_hz / 1e6,
        1e9 * verdict["amplitude_spacing_m"],
        1e9 * verdict["amplitude_spacing_error_m"], tilt, tilt_error,
        1e9 * verdict["metres_per_neper"], 1e9 * verdict["time_spacing_m"],
        1e9 * verdict["time_spacing_error_m"], 1e9 * delay,
        1e9 * delay_error, 1e9 * verdict["difference_m"],
        1e9 * verdict["difference_error_m"], verdict["sigma"],
        verdict["ratio"])
    _log_once(
        field, "magnetic_circuit_coherence",
        "model stages: magnetic_circuit - the colour-under band's measured "
        "coherence is %.4f against a floor of %.4f, so the TIME route %s - "
        "the amplitude route is a difference of levels and does not depend "
        "on it. The "
        "gate is the band's own signal share: the burst window stands %.2f "
        "dB above the SAME BAND read inside the synchronizing pulse, where "
        "no colour-under carrier can be, so the tip is this capture's own "
        "noise floor at that frequency measured through the same filter on "
        "the same lines. The within-window resultant reads %.4f and is NOT "
        "the gate - a filter three quarters of a megahertz wide cannot turn "
        "its phase inside a 2.5 microsecond window whatever it is fed, so "
        "that statistic measures the filter and would pass a capture with "
        "no subject. The gate is load-bearing: a luminance-only capture at "
        "coherence 0.127 reported a head delay difference of +2229.70 +/- "
        "104.29 ns, twenty-one standard deviations of nonsense that the "
        "reported error bar alone would have accepted. %s",
        coherence, magnetic_circuit.PHASE_COHERENCE_FLOOR,
        "stands" if verdict.get("usable") else "is REFUSED and the pair with "
        "it, because a pair needs two routes",
        over_floor, resultant, verdict.get("why", ""))
    _log_once(
        field, "magnetic_circuit_coercivity",
        "model stages: magnetic_circuit - the coercivity's contribution is a "
        "LOSS and not a limit: a transition length of %.4f um costs %.2f dB "
        "at this decode's own %.4f MHz carrier, at every wavelength in the "
        "band rather than as a wall at the end of it, and the coating's "
        "thickness binds the bandwidth first (%.3f um against %.3f). The "
        "chrominance's own record field is unreachable by 15.934 times, and "
        "that is the quantitative reason the format had to bias it: SMPTE "
        "32M 7.5.1.2.2 records the chrominance with the luminance FM acting "
        "as bias, and `tape_bias` measures what the bias leaves at +2.084 "
        "+/- 0.048 dB per MHz at 43 sigma",
        1e6 * coercivity["transition_length_m"],
        coercivity["loss_at_carrier_db"], luma_linear_hz / 1e6,
        1e6 * coercivity["shortest_wavelength_m"],
        1e6 * coercivity["geometric_limit_m"])
    _log_once(
        field, "magnetic_circuit_record_side",
        "model stages: magnetic_circuit REFUSES to attribute this separation "
        "to a side, and the reason is not a shortage of care. A record-side "
        "spacing excursion wrote less magnetisation and perfect playback "
        "recovers exactly the magnetisation present, so the loss is an "
        "absence in the medium rather than a failure of the reading - which "
        "is why dropouts are baked in. The record head's PHASE needs a "
        "matched record tap, and even with one the write and read shares are "
        "not two terms to be separated but ONE HEAD MEASURED TWICE, because "
        "on a single deck the head that wrote a track is the head that reads "
        "it. This decode carries a playback tap alone, so what is reported "
        "above is the pair of heads and not the pair of machines")
    field.magnetic_circuit = result
    return result


# --------------------------------------------------------------------------
# tape_path - seven mechanisms, three controls, and no measurement pair
# --------------------------------------------------------------------------

# How many places the joint axis is built on. The joint grid is the OUTER
# PRODUCT of the frequency places with the fields, so a full evidence grid
# would be a matrix of a hundred thousand rows to answer a question whose
# answer is an identity. Sixteen is enough to show the identity holds.
_JOINT_PLACES = 16


def measure_tape_path(field):
    """SEVEN MECHANISMS THAT SET ONE CLEARANCE - AND NO MEASUREMENT PAIR.

    Node `tape_path`. Applies nothing.

    The head-to-tape clearance is not one number with an error bar; it is a
    budget, and the budget is what says which mechanism is worth chasing. At
    the middle of the specified tension range the clearance is an aerodynamic
    problem and a contact problem in roughly equal measure - air entrainment
    against tape tension at a ratio of about two to one - with the mechanical
    terms a tenth of it, and the wrap's contribution NEGATIVE, which is the
    sign the contact law gives rather than an error.

    THIS STAGE HAS NO PAIR AND SAYS SO, which is the point rather than an
    omission. Its three controls are CONTROLS - constructions whose answers
    are known independently and which can fail in both directions - and a
    control is not a second view of a measurement. And `joint_axis` against
    `time_axis` is an ARITHMETIC IDENTITY by the module's own docstring: each
    mechanism's joint signature is the outer product of its time shape with a
    frequency factor common to all seven, so the two Gram matrices are the
    same matrix and the two counts must agree to numerical precision. The
    stage reports that agreement as a CONSTRUCTION CHECK - if the joint count
    ever exceeded the time count, a direction would have been manufactured by
    the grid rather than found in the physics - and never as a pair.

    THE TIME AXIS IS THE DECODE-SPECIFIC PART. Every rate is stated by
    `transport_model` from the part's diameter and the tape speed before
    anything is measured, so what a capture decides is not the rates but
    whether it is long enough to hold them. The axis is built over THIS
    decode's own field count and field rate, and a capture too short for a
    reel rotation cannot separate the mechanisms that turn at one. Two
    mechanisms return an exactly zero series and that is the honest answer
    rather than a defect: the wrap and the penetration are set when the deck
    is threaded, and nothing that does not move can be told from anything
    else that does not move.

    THE ONE CONTACT WITH MEASURED DATA IS A STRAIN AGAINST A TIME BASE. The
    full width of the specified tension range strains the tape, and a
    strained tape plays back at the wrong rate; over one line that is a
    displacement in nanoseconds. The decode's own line periods, read off
    `linelocs` BEFORE the time base correction nulls them, are a measured
    scatter to put beside it - and the comparison is stated as what it is,
    a control quantity against a measurement rather than two measurements.

    THE MODULUS IS AN ASSUMPTION, STATED AS ONE. No specification in this
    project's sources gives the base film's Young's modulus, which is why it
    is an argument to `path_signature` rather than a constant inside it, and
    why `modulus_sensitivity` exists. The stage reports how far the strain
    delay moves across a sixteenfold range of modulus, so the weight the
    answer puts on the assumption is visible beside the answer.
    """
    geometry = rf_geometry(field)
    if geometry is None:
        return None
    state = _rf_state(field, "tape_path")
    common = _band_group(field, geometry, "tape_path")
    if common is None:
        return None
    state["fields"] += 1
    # THE LINE PERIODS, before the time base correction nulls them. Taken
    # every field, because the scatter this compares against is a scatter
    # across the decode and not within one field.
    history = state.setdefault("periods", [])
    locations = getattr(field, "linelocs", None)
    if locations is not None:
        values = np.asarray(locations, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size > 16:
            history.append(np.diff(values)
                           / float(geometry["sample_rate_hz"]))
    if state["fields"] < _RF_POOL_FIELDS or state["reported"]:
        field.tape_path = state.get("result")
        return state.get("result")

    speed = common["speed_m_s"]
    linear = float(common["mechanics"].get("linear_tape_speed_m_s", 0.0))
    field_rate = float(geometry.get("field_rate_hz") or 59.94)
    low, high = geometry["band_hz"]
    grid = np.linspace(low, high, head_differential.EVIDENCE_PLACES)

    controls = {
        "separability": tape_path.separability_control(),
        "entrainment": tape_path.entrainment_exponent_control(),
        "pressure": tape_path.pressure_control(),
    }
    failed = [name for name, result in controls.items()
              if not result.get("passes")]
    if failed:
        _log_once(
            field, "tape_path_control",
            "model stages: tape_path DECLINES - its own controls fail on "
            "this decode: %s. One family of Wallace separations must span "
            "exactly one direction and a Fourier basis of delays must span "
            "its full rank; the foil bearing's exponents must be two thirds "
            "and minus two thirds; and the contact pressure is an identity "
            "rather than an approximation. A budget taken through a "
            "construction that fails these would be reporting the "
            "construction", ", ".join(sorted(failed)))
        state["reported"] = True
        return None

    budget = tape_path.spacing_budget(writing_speed_m_s=speed)
    without_phase = tape_path.frequency_axis(grid, phase=False,
                                             writing_speed_m_s=speed)
    with_phase = tape_path.frequency_axis(grid, phase=True,
                                          writing_speed_m_s=speed)
    # The reference frequency the time axis carries its modulations onto is
    # this decode's own measured luma carrier where the two parities have
    # given one, and the band's own upper corner where they have not - never
    # a figure typed here.
    reference_hz = _luma_centre_hz(_head_band_pairs(field))
    if not np.isfinite(reference_hz):
        reference_hz = high
    # THE LINEAR TAPE SPEED IS PASSED ONLY WHERE THE FORMAT STATES ONE. It
    # sets the reel and capstan rates the time axis separates the mechanisms
    # by, and a figure typed here for a format whose mechanics this tree does
    # not carry would be a constant invented at a call site; where it is
    # absent the module's own default stands and the log says which was used.
    transport = {} if not linear > 0 else {"linear_speed_m_s": linear}
    time_axis = tape_path.time_axis(fields=int(state["fields"]),
                                    field_rate_hz=field_rate,
                                    reference_hz=reference_hz,
                                    writing_speed_m_s=speed, **transport)
    joint = tape_path.joint_axis(np.linspace(low, high, _JOINT_PLACES),
                                 fields=int(state["fields"]),
                                 field_rate_hz=field_rate,
                                 writing_speed_m_s=speed, **transport)
    identity = abs(float(joint["effective"]) - float(time_axis["effective"]))

    # THE STRAIN AGAINST THE MEASURED TIME BASE.
    line_period_s = 1.0 / float(
        geometry.get("line_rate_hz") or (1.0 / tape_path.LINE_PERIOD_S))
    tension_range = float(tape_path.TENSION_RANGE_N[1]
                          - tape_path.TENSION_RANGE_N[0])
    strain_s = tape_path.strain_delay_s(tension_range,
                                        interval_s=line_period_s)
    periods = np.concatenate(history) if history else np.zeros(0)
    periods = periods[np.isfinite(periods)]
    measured_scatter_s = (float(np.std(periods)) if periods.size > 16
                          else float("nan"))
    sensitivity = tape_path.modulus_sensitivity(grid)

    result = {"budget": budget, "controls": controls,
              "frequency_axis": {"amplitude_only": without_phase,
                                 "with_phase": with_phase},
              "time_axis": time_axis, "joint_axis": joint,
              "identity": identity, "strain_delay_s": strain_s,
              "line_period_scatter_s": measured_scatter_s,
              "sensitivity": sensitivity, "fields": int(state["fields"])}
    state["result"] = result
    state["reported"] = True

    _log_once(
        field, "tape_path",
        "model stages: tape_path at %.4f m/s and the specified %.3f N - the "
        "clearance budget is air entrainment %.4f um, tape tension %.4f, "
        "guide geometry and stiffness %.4f, wrap and penetration %+.4f, head "
        "protrusion %.4f, total %.4f um at %.1f Pa of contact pressure. The "
        "wrap's term is NEGATIVE because more tension across the wrap is "
        "more pressure and less separation, which is the sign the contact "
        "law gives",
        speed, budget["tension_n"], 1e6 * budget["air entrainment"],
        1e6 * budget["tape tension"],
        1e6 * budget["guide geometry and tape stiffness"],
        1e6 * budget["wrap angle and penetration"],
        1e6 * budget["head protrusion"], 1e6 * budget["total_m"],
        budget["pressure_pa"])
    _log_once(
        field, "tape_path_axes",
        "model stages: tape_path axes on this decode's own %.3f to %.3f MHz "
        "and its own %d fields - the frequency axis spans %.6f of %d "
        "directions on amplitude alone, which is one by algebra because every "
        "member's log magnitude is the same straight line through the origin "
        "differing only in slope, and %.3f with the phase, because a gain "
        "and a delay of the same shape are two mechanisms. The time axis "
        "spans %.3f of %d at a condition of %.3g, with %d members inert - "
        "%s, which do not move at all and are therefore separable from "
        "nothing",
        low / 1e6, high / 1e6, state["fields"], without_phase["effective"],
        without_phase["count"], with_phase["effective"],
        time_axis["effective"], time_axis["count"], time_axis["condition"],
        len(time_axis["inert"]), ", ".join(time_axis["inert"]))
    _log_once(
        field, "tape_path_identity",
        "model stages: tape_path - the joint axis reads %.6f against the "
        "time axis's %.6f, agreeing to %.3g. THAT IS AN ARITHMETIC IDENTITY "
        "AND NOT A PAIR, and it is reported as a construction check: every "
        "mechanism's joint signature is the outer product of its own time "
        "shape with a frequency factor common to all seven, so the two Gram "
        "matrices are the same matrix. A joint count EXCEEDING the time "
        "count would mean the grid had manufactured a direction. This stage "
        "has no measurement pair, and offering a control as one would be "
        "offering arithmetic as evidence",
        joint["effective"], time_axis["effective"], identity)
    _log_once(
        field, "tape_path_strain",
        "model stages: tape_path - the specified tension range of %.3f N "
        "strains the tape by %.3e and displaces one %.4f us line by %.3f ns, "
        "against a measured line-period scatter of %.3f ns over %d periods "
        "read off `linelocs` before the time base correction nulls them. The "
        "comparison is a CONTROL against a measurement and not two "
        "measurements: the strain is the widest difference the specification "
        "permits between the recorder and the player, not a fitted quantity",
        tension_range, budget["strain"], 1e6 * line_period_s, 1e9 * strain_s,
        1e9 * measured_scatter_s, int(periods.size))
    _log_once(
        field, "tape_path_assumption",
        "model stages: tape_path - the base film's Young's modulus is an "
        "ASSUMPTION of %.3g Pa, because no specification in this project's "
        "sources states one; that is why it is an argument rather than a "
        "constant. What rests on it, measured across a sixteenfold range of "
        "modulus: %s",
        tape_path.BASE_MODULUS_PA,
        sensitivity.get("why", "the module reports no sensitivity"))
    field.tape_path = result
    return result


# ==========================================================================
# THE ONE PICTURE TRANSFORM, AND THE SHIMS FOR THE RADIO-FREQUENCY ONE
#
# Ethan, 2026-09-07: *"I can make a singla picture stage that transforms
# luma chroma all up to the composite functions. There doesn't need to be
# sequencing, just all the dimensions execute at once in a single
# transform."* ... *"Essentially removing each indifvidual residual, instead
# of just substituting in null space, let's try both out."* ... *"This
# becomes a data transformation problem using our instruments."* And, on
# the amount: *"Let's see what happens if we take this to hyperspace as the
# null space model, the exact inverse within our possible area of measure."*
#
# WHAT STANDS HERE. `single_transform.Transform` is the mathematics: the
# vertices, the fold, the two treatments of the null space, the latch. This
# section is the glue between that and a field - the INSTRUMENTS, which are
# the absorbed nodes' own measurement halves factored out above, and the
# REALISATIONS, which are their application halves composed into one write.
# Nothing is sequenced between them: every measurement is taken on the field
# as it arrived, every vertex is filled from those measurements, and once
# the transform has latched the departures are realised together.
#
# THE VERTICES. Three live axes for a decode that writes chroma - colour,
# head, polarity - and two for one that does not, the colour axis then being
# a known constant of that decode exactly as the field and tap axes are of
# every decode. Colour 0 is the luma and 1 the chroma; head is field parity,
# the only head label a decode has; polarity 0 is the sync pulse's FALL,
# which lands on the tip, and 1 its RISE, which lands on blanking.
#
# THE CHANNELS, the same quantities at every vertex, each a DEPARTURE from
# what the specification prescribes so a perfect channel reads zero on all
# of them:
#
#   `response`  the complex log of the measured edge against the specified
#               edge, on the sync segment's own transform bins up to the
#               composite channel's limit. The luma's fall and rise are its
#               two polarity faces; the chroma's burst envelope, complex, is
#               placed on the same grid at the composite frequencies its
#               sidebands actually occupy about the subcarrier.
#   `levels`    the level each landing reaches against the level the
#               standard fixes for it, in IRE, one bin per specified sync
#               row: the luma's tip on the fall face and its blanking on
#               the rise face, so the polarity contrast is half the
#               spacing's deficit and the rise face is the zero; the burst's
#               amplitude against its specified forty on both of the
#               chroma's faces alike, a gain having no polarity state.
#   `timing`    the burst-to-sync vernier, the same on every face of a head.
#   `image`     the chroma's quadrature image, one complex number, zero on
#               the luma by construction; its colour contrast under REMOVE
#               is how far the image reproduces across the two banks.
#
# A quantity with no state on an axis is filled identically on both faces,
# and its differential on that axis is then exactly zero - refused by
# construction. What that does to the OTHER contrasts is recorded in
# `transform_picture`'s docstring, because it is not nothing.
# ==========================================================================

from vhsdecode.models import pair_dimension

# The absorbed nodes, by the option each one's declaration gates it on. A
# node's measurement is taken only where its own option is on, so
# `--stages -sync_shape` leaves the channel that node fills unfilled and the
# reading says so.
_PICTURE_NODES = (
    "sync_shape", "precursor", "sync_depth", "vertical_interval",
    "tape_speed_stage", "level_from_frequency", "source_agc", "vcr_agc",
    "composite_channel", "multipath", "standard_levels", "source_correction",
    "chroma_head_switch", "colour_free_luma", "burst_sync_lock", "colour_under",
    "iq_imbalance", "burst_instrument", "colour_framing", "vectorscope",
    "picture_stage", "chroma_leakage", "tape_bias",
)
_PICTURE_STAGE = "picture_transform"
_PICTURE_STATE = "_single_transform"
_PICTURE_CHANNELS = ("response", "levels", "timing", "image")

# THE WITNESS FOR A FIELD'S SYNC IS THE DECODER'S OWN GATE. `lddecode/core.py`
# takes a level from a field for its automatic gain only where
# `f.sync_confidence > 80`; beneath that the decoder does not believe the
# field's sync well enough to read a level from it, and neither does this
# transform. Not a new threshold: the decoder's, reused.
_SYNC_CONFIDENCE_FLOOR = 80

# THE TWO AMOUNTS. The correction-gain law: the optimum amount is
# a* = 1/(1 + rho) with rho the model error, which no resampling of the fit's
# own data can see, and applied at half the believed optimum a correction
# keeps three quarters of its benefit anywhere in the safe range
# (`source_correction.amount_law`, docs/RESIDUAL_LIMIT_DESIGN.md). The exact
# inverse is amount one and is Ethan's `exact_inverse` component: within the
# region the measurement reaches, everything; outside it, nothing.
_GAIN_LAW_HALF = 0.5
_EXACT_INVERSE = 1.0

# The timing channel's bins, from `measure_burst_sync_lock`'s reading.
_TIMING_BINS = ("displacement_mean_s", "drift_across_field_s")


def _transform_component(field, part, default=True):
    """Whether one declared component of the picture transform runs, from
    the selection the decoder resolved once at start-up - the same source
    `chroma._stage_enabled` and `ringing_tesseract.component_enabled`
    read."""
    from vhsdecode import pipeline_graph

    selection = getattr(field.rf.options, "stage_selection", None)
    return bool(pipeline_graph.component_enabled(selection, _PICTURE_STAGE,
                                                 part, default))


def _picture_components(field):
    """The components, resolved once a field. `remove_residuals` and
    `exact_inverse` default OFF: the substitute treatment and the gain law's
    half stand unless a decode names the other by
    `--stages +picture_transform.<part>`."""
    return {
        "latch": _transform_component(field, "latch", True),
        "causal_split": _transform_component(field, "causal_split", True),
        "report": _transform_component(field, "report", True),
        "remove_residuals": _transform_component(field, "remove_residuals",
                                                 False),
        "exact_inverse": _transform_component(field, "exact_inverse", False),
        # Ethan: "I believe there is still a wave underneath the convergence
        # that itself can be differentialed as random noise and subtracted
        # out" - folded and reported always, applied when named
        "wave": _transform_component(field, "wave", False),
    }


def _resolve_amount(amount, components):
    """The amount applied, and why: the caller's number where one is given,
    else the exact inverse where the component asks for it, else the
    correction-gain law's half."""
    if amount is not None:
        return float(amount), "caller"
    if components.get("exact_inverse"):
        return _EXACT_INVERSE, "exact_inverse component"
    return _GAIN_LAW_HALF, "correction-gain law"


def _shared_transform_store(field):
    """The dictionary BOTH transforms keep their state in, on the decoder
    and not the field: the transforms accumulate ACROSS fields, and a new
    decode is a new decoder object with an empty dictionary. The picture
    side owns three keys of it - `picture`, the transform itself;
    `picture_folded`, the fold-once set; `picture_state`, everything else -
    and the radio-frequency side owns its own. Nothing here ever clears the
    dictionary as a whole: a first form of this did, and on the first real
    decode it wiped the radio-frequency cube after the first field, counted
    a field twice and latched late."""
    return field.rf.__dict__.setdefault(_PICTURE_STATE, {})


def _picture_store(field):
    """The picture transform's OWN state, a sub-dictionary of the shared
    store; resetting it touches nothing of the radio-frequency side's."""
    return _shared_transform_store(field).setdefault("picture_state", {})


def _response_grid(spans):
    """THE RESPONSE CHANNEL'S GRID, fixed for the life of the decode.

    The sync segment's own transform bins - the grid every edge in this file
    is already measured on - from the first bin above zero to the composite
    channel's limit, `composite_channel.BROADCAST_LIMIT_HZ` (4.2 MHz, SMPTE
    170M, which specifies no luma bandwidth of its own; the limit is the
    channel's). That upper edge is chosen so that BOTH colours lie inside
    one grid: the format's luma band (`sync_shape.VHS_LUMA_BAND_HZ`, 3.0
    MHz for VHS) and the chroma's band about the subcarrier at 3.58 MHz,
    which the luma band alone would exclude entirely.

    Returns the grid, its indices on the segment's full rfft grid, and that
    full grid. The last two matter: the cepstrum inside
    `source_correction.corrector` assumes an rfft grid running from DC to
    Nyquist, and handing it the band alone stretches the band over the
    whole axis - the defect `tesseract.on_full_grid` records - so the
    corrector is always given the full grid with the band marked on it.
    """
    full = np.fft.rfftfreq(int(spans["segment"]), 1.0 / spans["sample_rate_hz"])
    index = np.flatnonzero((full > 0.0)
                           & (full <= composite_channel.BROADCAST_LIMIT_HZ))
    return full[index], index, full


def _picture_transform(field, spans, has_chroma, causal):
    """The decode's one picture transform, constructed once and keyed to
    the decode's geometry so a change of shape - which a decode does not
    have, but a test may - gets a fresh one rather than a mismatched fill.

    A decode that writes no chroma has the colour axis as a KNOWN CONSTANT
    with one state, exactly as every decode has the field axis and the tap:
    the transform then carries head and polarity alone, and `live_kept`
    drops the contrasts that name the missing axis.
    """
    shared = _shared_transform_store(field)
    store = _picture_store(field)
    rows = np.asarray(spans["rows"])
    key = (int(spans["lines"]), int(spans["width"]), int(rows.size),
           int(spans["segment"]), bool(has_chroma), bool(causal))
    if shared.get("picture") is not None and store.get("key") == key:
        return shared["picture"]

    grid, index, full = _response_grid(spans)
    axes = single_transform.LIVE_AXES["picture"]
    if not has_chroma:
        axes = tuple(axis for axis in axes if axis != "colour")
    channels = {"response": int(grid.size), "levels": int(rows.size),
                "timing": len(_TIMING_BINS), "image": 1}
    transform = single_transform.Transform(
        "picture", axes, channels, spans["sample_rate_hz"],
        bins_hz={"response": grid} if causal else None)
    # THE PICTURE'S OWN KEYS ARE RESET, AND ONLY THOSE: the sub-dictionary
    # and the two aliases the shared store carries for it.
    store.clear()
    store.update({
        "picture": transform,
        "key": key,
        "grid": grid,
        "grid_index": index,
        "full_grid": full,
        "folded": set(),
        "last": None,
        "snr": {},
        "kernels": {},
        "kernel_key": None,
        "fields": 0,
        "unreached": ({} if has_chroma else {
            "colour": "this decode writes no chroma, so the colour axis "
                      "has one state and is a known constant of the decode"}),
    })
    shared["picture"] = transform
    shared["picture_folded"] = store["folded"]
    return transform


def _vertex(transform, colour, head, polarity):
    """A vertex named by axis, for a transform that may lack the colour
    axis."""
    vertex = {"head": int(head), "polarity": int(polarity)}
    if "colour" in transform.axes:
        vertex["colour"] = int(colour)
    return vertex


# --------------------------------------------------------------------------
# The instruments, each turned into a channel's values
# --------------------------------------------------------------------------

def _interval_response(field, picture, spans, geom, grid):
    """THE WHOLE RESERVED INTERVAL AGAINST ITS SPECIFICATION, which is the
    band the pooled sync edge cannot reach at all.

    Ethan, 2026-09-07: *"Sync and eq pulses are one model"*, and *"3d
    analytical component of the entire RF with the expected 3d anaytical
    spec signal for each stage."* The specified field is
    `spec_signal.expected_field` - every picture row's sync pulse and the
    vertical interval's equalising and broad pulse train, from the format's
    timing table and nothing else - and the reading is taken on the mask's
    one long run, the interval itself, because that is where a ratio of
    spectra means anything. A 4.7 microsecond pulse says nothing below
    213 kHz; the interval's 572 microseconds reach 1.75 kHz.

    The expected field depends only on the decode's geometry, so it is
    built once and held on the decoder.
    """
    store = _picture_store(field)
    key = (int(spans["lines"]), int(spans["width"]),
           bool(getattr(field, "isFirstField", True)))
    expected = store.get("expected")
    if expected is None or store.get("expected_key") != key:
        expected = spec_signal.expected_field(
            int(spans["lines"]), int(spans["width"]),
            float(spans["sample_rate_hz"]),
            getattr(field.rf, "color_system", "NTSC"),
            first_field=bool(getattr(field, "isFirstField", True)),
            active=geom.get("active"))
        store["expected"] = expected
        store["expected_key"] = key
    lines = ((np.asarray(picture, dtype=np.float64).reshape(
        int(spans["lines"]), int(spans["width"])) - float(spans["blanking"]))
        / float(spans["units_per_ire"]))
    try:
        reading = spec_signal.analytic_component(
            lines, expected["waveform"], expected["mask"],
            float(spans["sample_rate_hz"]),
            band_hz=(float(expected["resolution_hz"]["lowest_hz"]),
                     float(spans["band_limit_hz"])))
    except (ValueError, np.linalg.LinAlgError) as reason:
        return {"declined": str(reason)}
    f = np.asarray(reading["frequency_hz"], dtype=np.float64)
    departure = np.asarray(reading["response"], dtype=np.complex128)
    inside = (grid >= f[0]) & (grid <= f[-1])
    value = np.zeros(grid.size, dtype=np.complex128)
    if inside.any():
        value[inside] = (np.interp(grid[inside], f, departure.real)
                         + 1j * np.interp(grid[inside], f, departure.imag))
    return {
        "value": value,
        "carried": inside,
        "frequency_hz": f,
        "response": departure,
        "amplitude": reading["amplitude"],
        "time": reading["time"],
        "run_us": reading["run_us"],
        "resolution_hz": reading["resolution_hz"],
        "bins": int(inside.sum()),
    }


def _join_interval_to_edges(faces, interval, grid, spans):
    """THE TWO PROBES, EACH USED WHERE ITS OWN DURATION LETS IT RESOLVE.

    Ethan: *"Use the eq pulses to get the longer value, and the hsync
    pulses to refine the higher frequency details."* Below the sync pulse's
    own resolution the pooled edge carries nothing, so the interval's
    reading takes those bins; above it the edge resolves detail the
    interval smears, and it keeps them. The crossover is not chosen: it is
    one over the pulse's specified width.

    THE INTERVAL CANNOT SEPARATE THE LANDINGS, and the join says so rather
    than pretending otherwise. Its window holds every pulse's fall and rise
    together, so what it measures is the polarity-COMMON response; both
    faces therefore receive the same values below the crossover and the
    polarity contrast is zero there by construction. What the fold reads on
    that axis below 213 kHz is a null, not a measurement, and the reading
    records the band each face owes to which probe.
    """
    if not faces or interval is None or "value" not in interval:
        return faces, None
    crossover = float(sync_geometry.crossover_hz(
        float(spans["sync_width_us"]) if "sync_width_us" in spans
        else sync_geometry.LINE_SYNC_US["525"],
        float(interval["run_us"]))["crossover_hz"])
    below = (grid < crossover) & np.asarray(interval["carried"], dtype=bool)
    if not below.any():
        return faces, {"crossover_hz": crossover, "bins_from_interval": 0}
    joined = {}
    for polarity, face in faces.items():
        if face is None:
            joined[polarity] = None
            continue
        value = np.array(face["value"], dtype=np.complex128)
        carried = np.array(face["carried"], dtype=bool)
        # match the levels across the join rather than fitting anything: a
        # step at the joint is two instruments' gains, not the channel's
        overlap = carried & np.asarray(interval["carried"], dtype=bool) \
            & (grid >= crossover)
        offset = 0.0
        if overlap.sum() >= 4:
            offset = float(np.median(value[overlap].real
                                     - interval["value"][overlap].real))
        value[below] = interval["value"][below] + offset
        carried[below] = True
        entry = dict(face)
        entry["value"] = value
        entry["carried"] = carried
        entry["from_interval"] = below
        entry["level_offset_nepers"] = offset
        joined[polarity] = entry
    return joined, {
        "crossover_hz": crossover,
        "bins_from_interval": int(below.sum()),
        "run_us": float(interval["run_us"]),
        "lowest_hz": float(interval["resolution_hz"]),
        "why": ("below the sync pulse's own resolution only the interval "
                "can speak, and it cannot separate the landings, so both "
                "polarity faces take its value and their differential is "
                "null there by construction"),
    }


def _luma_response_faces(measured, spans, grid):
    """THE LUMA'S RESPONSE ON ITS TWO POLARITY FACES: the pooled fall and the
    pooled rise, each against the specified edge, as a complex log
    departure on the fixed grid.

    The specified pulse is `pair_dimension.spec_sync` - SMPTE 170M table 2
    transcribed once in this tree - rolled to where `locate_transients`
    finds the measured pulse, exactly as `source_correction.channel` does
    it and for the reason recorded there: the framing is not a delay. The
    pulse is then cut at the midpoint between its transients, which is
    `sync_shape.edge_pair`'s split, and each half is read against the
    matching half of the reference by `transit_delay`: the cross spectrum
    over the reference's own power on the bins the format's band limit
    carries.

    Each face's magnitude is referred to its own lowest carried bin, so the
    level is not in the response - the level is the `levels` channel's -
    and its phase is carried whole. Placed on the fixed grid by
    interpolation inside the face's carried range and ZERO outside it,
    which is unity, the convention `source_correction.corrector` keeps for
    a bin nothing measured. The per-bin evidence, `sqrt(power / floor)`
    from the segment's own out-of-band noise, travels beside it for the
    corrector's cap.
    """
    profile = np.asarray(measured["profile"], dtype=np.float64)
    rate = float(spans["sample_rate_hz"])
    band = spans["band_limit_hz"]
    where = measured["transients"]
    length = profile.size
    reference = pair_dimension.spec_sync(rate, length)
    centre = (length - 1) / 2.0
    offset = 0.5 * (float(where["fall"]) + float(where["rise"]))
    reference = np.roll(reference, int(round(offset - centre)))
    cut = int(where["midpoint"])
    halves = ((0, profile[:cut], reference[:cut]),
              (1, profile[cut:], reference[cut:]))
    faces = {}
    for polarity, m, r in halves:
        try:
            fit = sync_shape.transit_delay(m, r, rate, float(band))
        except (ValueError, np.linalg.LinAlgError):
            faces[polarity] = None
            continue
        carried = np.asarray(fit["carried"], dtype=bool)
        f = np.asarray(fit["frequency_hz"])[carried]
        h = np.asarray(fit["response"])[carried]
        if f.size < 2:
            faces[polarity] = None
            continue
        level = max(float(np.abs(h[0])), 1e-12)
        log_magnitude = np.log(np.maximum(np.abs(h), 1e-12) / level)
        phase = np.unwrap(np.angle(h))
        floor = float(fit["noise_floor_power"])
        weight = np.asarray(fit["weight"])[carried]
        snr = np.sqrt(np.maximum(weight, 0.0) / floor) if floor > 0 \
            else np.zeros(f.size)
        inside = (grid >= f[0]) & (grid <= f[-1])
        value = np.zeros(grid.size, dtype=np.complex128)
        value[inside] = (np.interp(grid[inside], f, log_magnitude)
                         + 1j * np.interp(grid[inside], f, phase))
        evidence = np.zeros(grid.size, dtype=np.float64)
        evidence[inside] = np.interp(grid[inside], f, snr)
        faces[polarity] = {
            "value": value,
            "carried": inside,
            "snr": evidence,
            "bins": int(carried.sum()),
            "delay_s": float(fit["delay_s"]),
            "is_a_delay": bool(fit["is_a_delay"]),
        }
    return faces


def _chroma_response_face(field, uphet, spans, grid):
    """THE CHROMA'S RESPONSE ON THE LUMA'S GRID, complex, from the burst.

    `chroma.chroma_path_response` is the instrument this follows step for
    step - the specification's own gate envelope on the measured window,
    the flat top from the specified burst window, alignment before
    comparison, the decoder's own band-pass divided out, and a bin allowed
    to speak only where the expected envelope carries evidence - with one
    change that is this arc's standing finding: it reads the COMPLEX
    envelope (`chroma.burst_complex_envelope`), because a magnitude
    envelope made the burst instrument rank two of four and hid the tape's
    own colour-under roll-off. The complex envelope's spectrum is two-sided
    in the baseband offset, and that is what places it on the luma's grid:
    a composite frequency `f` carries the chroma's sideband at the signed
    offset `f - f_sc` from the subcarrier, the lower sideband beneath it
    and the upper above, and each grid bin takes the envelope's complex log
    response at its own offset. The magnitude form of the same relation is
    `picture_stage.baseband_from_rf`, which folds the two sidebands onto
    `|f - f_c|`; the complex envelope keeps them apart.

    The flat top's complex median is the chroma's gain and phase at the
    subcarrier; it is divided out here, so the response is the SHAPE about
    the carrier, and reported beside it as `gain_fsc`. The level itself is
    the `levels` channel's, from the same burst.
    """
    from vhsdecode import chroma as chroma_module

    envelope = chroma_module.burst_complex_envelope(field, uphet)
    if envelope is None:
        return None
    measured, first = envelope
    count = len(measured)
    reference = chroma_module._spec_burst_envelope(field, first, count)
    flat = chroma_module._spec_flat_span(field, first, count)
    if reference is None or flat is None:
        return None
    gain = complex(np.median(measured[flat].real)
                   + 1j * np.median(measured[flat].imag))
    if not abs(gain) > 0.0 or not np.isfinite(gain):
        return None
    normalised = measured / gain
    reference = reference / max(float(np.median(reference[flat])), 1e-12)
    index = np.arange(count, dtype=np.float64)
    shift = chroma_module._alignment_shift(np.abs(normalised), reference)
    aligned = (np.interp(index + shift, index, normalised.real)
               + 1j * np.interp(index + shift, index, normalised.imag))

    rate = float(spans["sample_rate_hz"])
    spectrum_m = np.fft.fft(aligned - aligned.mean())
    spectrum_r = np.fft.fft(reference - reference.mean())
    offsets = np.fft.fftfreq(count, d=1.0 / rate)
    expected = np.abs(spectrum_r)
    ours = chroma_module.decoder_envelope_response(field.rf, np.abs(offsets))
    if ours is not None:
        keep = np.isfinite(ours) & (ours > 1e-3)
        expected = np.where(keep, expected * ours, expected)
    usable = expected > chroma_module.TRANSFER_EVIDENCE_FRACTION * expected.max()
    usable[0] = False
    if usable.sum() < 2:
        return None
    denominator = spectrum_r[usable]
    if ours is not None:
        denominator = denominator * np.where(keep[usable], ours[usable], 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        response = np.log(spectrum_m[usable] / denominator)
    order = np.argsort(offsets[usable])
    f_m = offsets[usable][order]
    log_magnitude = response.real[order]
    phase = np.unwrap(response.imag[order])
    # the offset's origin: no departure at the carrier itself, the level
    # having been divided out
    below = int(np.sum(f_m < 0))
    f_m = np.concatenate([f_m[:below], [0.0], f_m[below:]])
    log_magnitude = np.concatenate([log_magnitude[:below], [0.0],
                                    log_magnitude[below:]])
    phase = np.concatenate([phase[:below], [0.0], phase[below:]])

    subcarrier = composite_subcarrier_hz(field)
    offset_of_bin = grid - subcarrier
    inside = (offset_of_bin >= f_m[0]) & (offset_of_bin <= f_m[-1])
    value = np.zeros(grid.size, dtype=np.complex128)
    value[inside] = (np.interp(offset_of_bin[inside], f_m, log_magnitude)
                     + 1j * np.interp(offset_of_bin[inside], f_m, phase))
    if not np.all(np.isfinite(value)):
        return None
    return {
        "value": value,
        "carried": inside,
        "gain_fsc": gain,
        "usable_bins": int(usable.sum()),
        "offset_range_hz": (float(f_m[0]), float(f_m[-1])),
    }


def _luma_level_faces(series, projected):
    """THE LUMA'S LEVELS ON ITS TWO POLARITY FACES, per specified row, as
    departures in IRE: the tip against the specified synchronizing level on
    the fall face, blanking against the specified zero on the rise face.

    `projected` is the porch baseline with the two leaking carriers taken
    out, the level `standard_levels` drives to zero, and is used where that
    node measured it; otherwise the porch's plain mean stands in, which
    differs from it by the leakage - 0.004 IRE on the bar tape, 0.33 on the
    home recording, by the figures `correct_standard_levels` records.
    """
    blanking = np.asarray(series["blanking_ire"], dtype=np.float64)
    spacing = np.asarray(series["spacing_ire"], dtype=np.float64)
    tip = blanking - spacing
    zero = (np.asarray(projected, dtype=np.float64) if projected is not None
            else blanking)
    return {
        "fall": tip - standard_levels.SYNC_TIP_IRE,
        "rise": zero - standard_levels.BLANKING_IRE,
        "finite": bool(np.isfinite(tip).all() and np.isfinite(zero).all()),
        "reading": "projected" if projected is not None else "porch_mean",
    }


def _chroma_units_per_ire(field):
    """The chroma's own scale, from the decoder's own reference: the chroma
    automatic gain drives the burst's fitted PEAK amplitude to
    `SysParams["burst_abs_ref"]` (`chroma.chroma_automatic_gain`, the
    amplitude being `2 sqrt(I^2 + Q^2) / N`, a peak), and the specification
    puts that peak at half of `composite_channel.BURST_IRE_PEAK_TO_PEAK`
    (40 IRE peak to peak, SMPTE 170M). Returns None where the format
    states no reference."""
    reference = field.rf.SysParams.get("burst_abs_ref")
    if reference is None:
        return None
    try:
        reference = float(reference)
    except (TypeError, ValueError):
        return None
    if not reference > 0.0:
        return None
    return reference / (composite_channel.BURST_IRE_PEAK_TO_PEAK / 2.0)


def _chroma_level_faces(field, phasors):
    """THE BURST'S AMPLITUDE AGAINST THE SPECIFIED FORTY IRE, per row, on
    BOTH of the chroma's polarity faces alike.

    The chroma has no polarity state: the burst's negative and positive
    peaks are one carrier's two half cycles and nothing physical makes one
    differ from the other, so a level departure read on the burst is a gain
    and is filled identically on the fall face and the rise face. Its
    polarity differential is then exactly zero and refused by construction,
    which is the face-duplication rule; the departure itself stands in the
    mean and the colour contrasts, beside the luma's.

    A first form of this put the two peaks on the two faces, each against
    plus or minus twenty IRE, which reads as a polarity asymmetry the burst
    does not have and, measured on a synthetic decode, moved the luma's own
    realisation under the substitute treatment by the sign of the burst's
    deficit: 0.79 IRE left on the luma tip against the 0.50 the faces
    alike leave. Withdrawn for the reason above.

    A phasor is the burst's projection onto the specified carrier over the
    burst window, so its modulus is half the peak amplitude and four times
    it is the peak-to-peak amplitude the standard states.
    """
    units = _chroma_units_per_ire(field)
    if units is None:
        return None
    peak_to_peak = 4.0 * np.abs(np.asarray(phasors, dtype=np.complex128)) / units
    departure = peak_to_peak - composite_channel.BURST_IRE_PEAK_TO_PEAK
    return {
        "fall": departure,
        "rise": departure,
        "finite": bool(np.isfinite(departure).all()),
    }


def _image_pair(burst):
    """THE CHROMA'S QUADRATURE IMAGE, COMPLEX, in the frame
    `iq_imbalance.correct` reads it in.

    `iq_imbalance.from_burst_series` returns the image's MODULUS, its
    control and its verdicts; the correction needs the complex coefficient,
    and this forms it by the same de-rotation: each run is turned back by
    the rotation it actually carries, its mean is the run's first line's
    imbalance-free burst, and the ratio of the period-two component to that
    mean is `beta` in the frame where that burst is real and positive - the
    frame `correct` documents.

    Every run's coefficient is referred to the FIELD's first picture row:
    in the burst's frame the coefficient alternates sign every line, so a
    run beginning an odd number of lines later reads it with the opposite
    sign, and pooling without that alignment would cancel it.
    """
    if burst is None:
        return None
    phasors = np.asarray(burst["phasors"], dtype=np.complex128).ravel()
    run = phasors.size // 2
    if run < 8:
        return None
    series = phasors[:2 * run]
    try:
        found = iq_imbalance.from_burst_series(series, run_lines=run)
    except ValueError:
        return None
    if not found.get("usable") or not found.get("rotation_is_a_quarter_turn"):
        return {"beta": None, "found": found,
                "why": ("the runs' rotation is not the quarter turn SMPTE "
                        "32M clause 3.9.2.1.5 puts on the tape, so the "
                        "run detector failed and the pair is not usable")}
    betas = []
    # every whole run, the last included: `from_burst_series` stops one run
    # short (its range ends at `size - span`, exclusive), which on a field
    # of two runs uses one
    for start in range(0, series.size - run + 1, run):
        segment = series[start:start + run]
        step = float(np.angle(complex((segment[1:] * np.conj(segment[:-1]))
                                      .sum())))
        index = np.arange(segment.size, dtype=np.float64)
        turned = segment * np.exp(-1j * step * index)
        mean = complex(turned.mean())
        if not abs(mean) > 0.0:
            continue
        period_two = complex((turned * (-1.0) ** index).mean())
        betas.append((period_two / mean) * ((-1.0) ** start))
    if not betas:
        return None
    return {
        "beta": complex(np.mean(betas)),
        "found": found,
        "first_row": int(np.asarray(burst["rows"])[0]),
        "over_control": float(found["over_control"]),
    }


# --------------------------------------------------------------------------
# The context: every instrument once, on the field as it arrived
# --------------------------------------------------------------------------

def picture_context(field, uphet):
    """ONE MEASUREMENT OF EVERYTHING THE PICTURE TRANSFORM FILLS FROM, taken
    on the field AS IT ARRIVED and before anything is written.

    One `sync_pulses`, one `_level_series`, one colour lock `measure`, one
    burst phasor per row, one chroma path response, and the colour-under
    burst while `chroma_under_tbc` still exists; the pure `measure_*`
    adapters are called where their nodes are on, because their readings
    and their pooled state are what the fillers draw on, and the six
    correctors' measurement halves are called in their place so that
    nothing here writes a sample. Each absorbed node is measured only where
    its own option is on, and a node that is off leaves its vertex to be
    declared unmeasured rather than read.

    The witnesses are decided here too: the decoder's own sync confidence
    beneath its automatic gain's floor, a broken parity alternation, and a
    colour lock refused on a decode whose lock is otherwise believed.
    """
    options = field.rf.options
    on = {name: bool(getattr(options, name, 0)) for name in _PICTURE_NODES}
    picture = field.dspicture
    geom = geometry(field)
    spans = sync_spans(field)
    context = {"on": on, "geometry": geom, "spans": spans, "declined": {},
               "measured": {}, "has_chroma": uphet is not None}
    if spans is None:
        context["declined"]["spans"] = (
            "this field cannot supply the specified sync population on an "
            "IRE scale (system %s, format %s)"
            % (getattr(field.rf, "color_system", "unknown"),
               getattr(options, "tape_format", "unknown")))
        return context
    grid = _picture_store(field).get("grid")
    if grid is None:
        grid = _response_grid(spans)[0]

    # ---- the luma's instruments -------------------------------------------
    wants_pulses = (on["sync_shape"] or on["precursor"] or on["sync_depth"]
                    or on["source_correction"])
    pulses = sync_pulses(field, picture) if wants_pulses else None
    context["pulses"] = pulses
    if wants_pulses and pulses is None:
        context["declined"]["sync_pulses"] = (
            "no sync population on an IRE scale in this field")
    if on["sync_shape"]:
        measure_sync_shape(field, picture)
        if pulses is not None:
            faces = _luma_response_faces(pulses, spans, grid)
            # THE WHOLE MASK, joined beneath the pulse's own resolution.
            interval = _interval_response(field, picture, spans, geom, grid)
            if "value" in interval:
                faces, join = _join_interval_to_edges(faces, interval, grid,
                                                      spans)
                context["interval"] = interval
                context["interval_join"] = join
            else:
                context["declined"]["interval"] = interval.get("declined", "")
            context["luma_response"] = faces
    if on["precursor"] and pulses is not None:
        context["precursor"] = _read_precursor(field, pulses)

    # `sync_depth`'s spacing is `_level_series`'s own `result`, taken once;
    # the gain is realised from the latched levels, so its measurement half
    # is not run a second time here.
    wants_levels = on["sync_depth"] or on["standard_levels"]
    series = _level_series(field, picture) if wants_levels else None
    context["levels"] = series
    if wants_levels and series is None:
        context["declined"]["level_series"] = (
            "no sync population on an IRE scale in this field")
    if series is not None:
        projected = None
        if on["standard_levels"] and series["trustworthy"]:
            context["standard_levels"] = _read_standard_levels(field, series)
            if context["standard_levels"] is not None:
                projected = context["standard_levels"]["baselines"]
        context["luma_levels"] = _luma_level_faces(series, projected)
        context["luma_levels"]["trustworthy"] = bool(series["trustworthy"])
        context["luma_levels"]["why_not"] = series["why_not"]

    if on["vertical_interval"]:
        measure_vertical_interval(field, picture)
    if on["tape_speed_stage"]:
        measure_tape_speed(field)
    if on["level_from_frequency"]:
        measure_level_from_frequency(field, options.level_from_frequency)
    if on["source_agc"]:
        measure_source_agc(field, options.source_agc)
    if on["vcr_agc"]:
        measure_vcr_agc(field, options.vcr_agc)
    if on["composite_channel"]:
        measure_composite_channel(field, options.composite_channel)
    if on["multipath"]:
        measure_multipath(field, options.multipath)
    if on["tape_bias"]:
        measure_tape_bias(field, picture, options.tape_bias)

    # ---- the chroma's instruments -----------------------------------------
    lock = None
    lock_refused = False
    if uphet is not None:
        # FIRST, because it reads the down-converted chroma this transform
        # releases at its end.
        if on["colour_under"]:
            measure_colour_under(field, uphet, options.colour_under)
        if on["iq_imbalance"]:
            measure_iq_imbalance(field, uphet, options.iq_imbalance)
            context["image"] = _image_pair(_colour_under_burst(field))
        if on["chroma_head_switch"] or on["colour_free_luma"]:
            lock = measure(field, uphet)
            lock_refused = (lock is None and _system_supported(field)
                            and field.rf.__dict__.get("_colour_lock_verdict")
                            is not False)
        if on["chroma_head_switch"] and lock is not None:
            context["chroma_head_switch"] = _read_chroma_head_switch(field,
                                                                     lock)
        if on["colour_free_luma"] and lock is not None:
            context["colour_free"] = _read_colour_free(
                lock, options.colour_free_luma)
        if on["burst_sync_lock"]:
            reading = measure_burst_sync_lock(field, uphet)
            context["lock_reading"] = reading
            if reading is not None:
                context["timing"] = np.asarray(
                    [float(reading[name]) for name in _TIMING_BINS])
        if on["burst_instrument"]:
            measure_burst_instrument(field, uphet, options.burst_instrument)
            context["chroma_response"] = _chroma_response_face(
                field, uphet, spans, grid)
            phasors = _rows_burst_phasors(field, uphet, spans,
                                          context.get("lock_reading"))
            if phasors is not None:
                context["chroma_levels"] = _chroma_level_faces(field, phasors)
        if on["colour_framing"]:
            measure_colour_framing(field, uphet, options.colour_framing)
        if on["vectorscope"]:
            measure_vectorscope(field, uphet, options.vectorscope)
        if on["picture_stage"]:
            measure_picture_stage(field, uphet, options.picture_stage)
        if on["chroma_leakage"]:
            measure_chroma_leakage(field, uphet, options.chroma_leakage)
    context["lock"] = lock
    context["lock_refused"] = lock_refused

    # ---- the witnesses ----------------------------------------------------
    confidence = getattr(field, "sync_confidence", None)
    context["witness"] = {
        "sync_confidence": (confidence is not None
                            and float(confidence) <= _SYNC_CONFIDENCE_FLOOR),
        "colour_lock": bool(lock_refused),
        "parity": False,      # decided against the store, by the caller
    }
    return context


def _rows_burst_phasors(field, uphet, spans, lock_reading):
    """The burst's complex phasor on every specified sync row, taken once:
    `measure_burst_sync_lock`'s own reading where it covers the whole
    population, else one projection here."""
    rows = np.asarray(spans["rows"])
    if lock_reading is not None:
        phasor = np.asarray(lock_reading.get("phasor"))
        if phasor.size == rows.size:
            return phasor
    chroma = np.asarray(uphet, dtype=np.float64).reshape(spans["lines"],
                                                         spans["width"])
    burst = _burst_phasor(field, chroma, rows, spans)
    return None if burst is None else burst["phasor"]


# --------------------------------------------------------------------------
# The fills
# --------------------------------------------------------------------------

def _fill_picture(transform, store, field, context, transient):
    """EVERY VERTEX THIS FIELD CAN FILL, filled, and what each fill was.

    A vertex whose instrument is OFF for the decode is filled with a
    declared zero - no departure, which is unity on a response and the
    specified level on a level - so that the channel's other vertices can
    still be judged; it is named as unmeasured in the outcome. A vertex
    whose instrument is on but declined on this field is simply not filled
    on this field.
    """
    spans = context["spans"]
    on = context["on"]
    head = int(bool(field.isFirstField))
    has_colour = "colour" in transform.axes
    rows = np.asarray(spans["rows"])
    outcome = {}
    values = {}

    # EVERY FILL AT WEIGHT ONE, with the lines pooled recorded beside it.
    # `Transform.fill` takes a weight in the caller's own currency, but its
    # Welford accumulator divides the second moment by W(W - 1) with W the
    # weight total, which treats every pooled line as an independent fill:
    # at 245 lines a fill and four fills a vertex the instrument floor came
    # out 0.0 to six places against a null pool of 2.9e-4, understated by
    # about the weight squared. One fill is one field's evidence, and the
    # floors are only right when it is counted as one.
    def put(colour, polarity, channel, x, flag, note=None):
        vertex = _vertex(transform, colour, head, polarity)
        accepted = transform.fill(vertex, channel, x, weight=1.0,
                                  transient=flag)
        key = (colour, head, polarity)
        outcome.setdefault(channel, {})[key] = {
            "accepted": bool(accepted), "note": note,
            "lines": int(rows.size)}
        if accepted:
            values.setdefault(channel, {})[key] = np.asarray(
                x, dtype=np.complex128)
        return accepted

    # -- response ------------------------------------------------------------
    faces = context.get("luma_response")
    if faces:
        for polarity in (0, 1):
            face = faces.get(polarity)
            if face is None:
                continue
            if put(0, polarity, "response", face["value"], transient):
                pooled = store["snr"].setdefault(head, {
                    "sum": np.zeros(store["grid"].size),
                    "count": np.zeros(store["grid"].size)})
                pooled["sum"] += face["snr"]
                pooled["count"] += face["carried"].astype(np.float64)
    if has_colour and faces:
        chroma = context.get("chroma_response")
        if chroma is not None:
            for polarity in (0, 1):
                put(1, polarity, "response", chroma["value"], transient)
        elif not on["burst_instrument"]:
            for polarity in (0, 1):
                put(1, polarity, "response", np.zeros(store["grid"].size),
                    transient, note="unmeasured: burst_instrument is off")

    # -- levels --------------------------------------------------------------
    levels = context.get("luma_levels")
    if levels:
        flag = transient or not levels["trustworthy"] or not levels["finite"]
        put(0, 0, "levels", levels["fall"], flag)
        put(0, 1, "levels", levels["rise"], flag)
    if has_colour and levels:
        chroma = context.get("chroma_levels")
        if chroma is not None:
            flag = transient or not chroma["finite"]
            put(1, 0, "levels", chroma["fall"], flag)
            put(1, 1, "levels", chroma["rise"], flag)
        elif not on["burst_instrument"]:
            for polarity in (0, 1):
                put(1, polarity, "levels", np.zeros(rows.size), transient,
                    note="unmeasured: burst_instrument is off")

    # -- timing, the same on every face of the head -------------------------
    timing = context.get("timing")
    if timing is not None:
        for colour in ((0, 1) if has_colour else (0,)):
            for polarity in (0, 1):
                put(colour, polarity, "timing", timing, transient)

    # -- image: the chroma's, and the luma's known zero ---------------------
    image = context.get("image")
    if has_colour and image is not None and image.get("beta") is not None:
        beta = np.asarray([image["beta"]], dtype=np.complex128)
        for polarity in (0, 1):
            put(1, polarity, "image", beta, transient)
            put(0, polarity, "image", np.zeros(1), transient,
                note="the luma is real and carries no quadrature image")
    return outcome, values


# --------------------------------------------------------------------------
# The realisations, from the departures
# --------------------------------------------------------------------------

def _departure(transform, colour, head, polarity, channel, treatment,
               wave=None):
    """One vertex's departure on one channel, under one treatment - plus
    this field's admitted wave at that vertex where the caller passes one
    (`wave` maps (colour, head, polarity) to the whole-vertex wave vector
    `Transform.wave_at` returns for the field being realised)."""
    sl = transform.channels[channel]
    value = transform.departure(_vertex(transform, colour, head, polarity),
                                treatment)[sl]
    if wave:
        extra = wave.get((colour, head, polarity))
        if extra is not None:
            value = value + np.asarray(extra, dtype=np.complex128)[sl]
    return value


def _response_kernels(transform, store, spans, geom, treatment, amount,
                      wave=None, wave_stamp=None):
    """ONE KERNEL PER HEAD from the luma's latched response, formed once per
    latched version and held.

    The head's response is the mean of its two polarity faces - the part a
    linear kernel can invert; the polarity contrast is a landing, not a
    filter, and stays a report. The corrector is
    `source_correction.corrector`, given the segment's FULL rfft grid with
    the band marked on it (see `_response_grid`), the pooled per-bin
    evidence for its cap, and a reference level of one because each face
    was referred to its own lowest carried bin when it was filled. Its own
    `amount` multiplies the gain law, so the law is neutralised - amount
    over `amount_law(0)`, at a model error of zero - and the transform's
    resolved amount is the fraction applied, exactly.
    """
    version = (None if transform.latched is None else transform.latched.version,
               treatment, float(amount),
               # with a wave applied the kernel is this field's own and is
               # formed again for the next; without one it is held
               None if not wave else wave_stamp)
    if store.get("kernel_key") == version:
        return store["kernels"]
    kernels = {}
    grid = store["grid"]
    index = store["grid_index"]
    full = store["full_grid"]
    band = float(spans["band_limit_hz"])
    for head in (0, 1):
        pooled = store["snr"].get(head)
        if pooled is None or pooled["count"].max() <= 0:
            kernels[head] = None
            continue
        count = pooled["count"]
        carried_grid = (count > 0) & (grid <= band)
        if carried_grid.sum() < 4:
            kernels[head] = None
            continue
        faces = [_departure(transform, 0, head, polarity, "response", treatment,
                            wave)
                 for polarity in (0, 1)]
        departure = 0.5 * (faces[0] + faces[1])
        magnitude = np.ones(full.size)
        magnitude[index] = np.exp(departure.real)
        snr = np.zeros(full.size)
        snr[index] = np.where(count > 0, pooled["sum"] / np.maximum(count, 1.0),
                              0.0)
        carried = np.zeros(full.size, dtype=bool)
        carried[index] = carried_grid
        measured = {"frequency_hz": full, "carried": carried,
                    "magnitude": magnitude, "signal_to_noise": snr,
                    "reference_level": 1.0}
        correction = source_correction.corrector(
            measured, amount=float(amount) / source_correction.amount_law(0.0),
            model_error=0.0)
        taps = source_correction.kernel(correction, geom["width"],
                                        spans["sample_rate_hz"])
        kernels[head] = {
            "taps": taps,
            "bins": int(carried_grid.sum()),
            "fraction": float(correction["fraction"]),
            "maximum_gain_db": float(correction["maximum_gain_db"]),
            "capped_bins": int(correction["capped_bins"]),
            "departure_db_rms": float(np.sqrt(np.mean(
                (20.0 / np.log(10.0) * departure.real[carried_grid]) ** 2))),
        }
    store["kernels"] = kernels
    store["kernel_key"] = version
    return kernels


def _level_map(transform, spans, head, treatment, amount, gain_on, zero_on,
               wave=None):
    """THE CLOSED-FORM LEVEL MAP: `sync_depth` then `standard_levels`, in the
    legacy order, derived analytically from the latched levels and applied
    in one write with no re-measurement between.

    THE LEGACY SEQUENCE, per field, was

        1.  y  = A + (x - A) / g_r            `correct_sync_depth`: a gain
                                              about the measured blanking A,
                                              g_r a straight line through
                                              the per-row spacing over the
                                              specified 40 IRE
        2.  round to the container
        3.  b'_r = the porch's projected baseline measured on y, in IRE
        4.  x' = y - t(b')_r . u              `correct_standard_levels`: the
                                              dominant trend t of that
                                              baseline, per row, u units an
                                              IRE

    Step 3 is where the sequence re-measured, and it need not: the porch
    projection is linear and every sample of a row maps affinely under
    step 1, so the projected baseline after the gain is exactly

        b'_r = A_ire + (b_r - A_ire) / g_r

    with b_r the baseline measured on the field as it arrived. The map is
    therefore

        x' = A + (x - A) / g_r  -  t(A_ire + (b - A_ire) / g)_r . u

    one write, and the only thing the sequence had that this has not is the
    rounding between its two writes, which is bounded by one least
    significant bit and is what `tests/unit/test_single_transform_stage.py`
    measures.

    WHERE THE NUMBERS COME FROM. Tip and blanking are the latched `levels`
    departures at this head's two polarity faces: tip = fall face plus the
    specified synchronizing level, blanking = rise face plus the specified
    zero; the spacing is their difference per row and g_r the same straight
    line `_read_sync_depth` fits, through `_gain_per_row` with the same
    bounds. A_ire is the mean blanking, as `sync_depth.spacing` reports it.

    THE AMOUNT enters as the fractional inverse in the log domain for the
    gain - g_r to the power of the amount, the form the corrector and
    `baseband_eq` already use - and linearly for the zero, so amount one is
    the sequence exactly and a half is the gain law's half of it.
    """
    tip = (_departure(transform, 0, head, 0, "levels", treatment, wave).real
           + standard_levels.SYNC_TIP_IRE)
    blank = (_departure(transform, 0, head, 1, "levels", treatment, wave).real
             + standard_levels.BLANKING_IRE)
    rows = np.asarray(spans["rows"], dtype=np.float64)
    lines = int(spans["lines"])
    units = float(spans["units_per_ire"])
    spacing = blank - tip
    anchor_ire = float(np.mean(blank))
    per_row = np.ones(lines, dtype=np.float64)
    reading = {"anchor_ire": anchor_ire, "spacing_ire": float(np.mean(spacing)),
               "gain": 1.0, "gain_applied": False, "zero_ire": 0.0,
               "zero_applied": False, "amount": float(amount)}
    if gain_on:
        mean_gain = float(np.mean(spacing)) / sync_depth.SPECIFIED_DEPTH_IRE
        try:
            sync_depth.correction({
                "gain": mean_gain,
                "deficit_ire": sync_depth.SPECIFIED_DEPTH_IRE
                - float(np.mean(spacing))})
        except ValueError as error:
            reading["gain_refused"] = str(error)
        else:
            if rows.size > 8 and np.ptp(rows) > 0:
                slope, intercept = np.polyfit(rows, spacing, 1)
            else:
                slope, intercept = 0.0, float(np.mean(spacing))
            per_row = _gain_per_row(float(slope), float(intercept), lines) \
                ** float(amount)
            reading.update(gain=mean_gain, gain_applied=True,
                           slope_ire_per_field=float(slope) * lines)
    zero = np.zeros(lines, dtype=np.float64)
    if zero_on:
        at_rows = per_row[rows.astype(np.intp)]
        gained = anchor_ire + (blank - anchor_ire) / at_rows
        trend = standard_levels.dominant_trend(gained)
        level = float(trend["level"])
        if abs(level) < sync_depth.SPECIFIED_DEPTH_IRE:
            zero = _zero_per_row(trend, rows, lines) * float(amount)
            reading.update(zero_ire=level, zero_applied=True,
                           trend_swing_ire=float(trend.get("trend_swing",
                                                           0.0)))
        else:
            reading["zero_refused"] = (
                "the porch baseline reads %+.2f IRE, further from the "
                "specified zero than the whole %.0f IRE the standard puts "
                "between blanking and the synchronizing level"
                % (level, sync_depth.SPECIFIED_DEPTH_IRE))
    reading["anchor"] = spans["blanking"] + anchor_ire * units
    reading["gain_per_row"] = per_row
    reading["zero_per_row_ire"] = zero
    return reading


def _realise_level_map(lines, spans, mapping):
    """The closed-form map on a float field, in place: the gain about the
    anchor in the three steps `_realise_sync_depth` takes, then the zero."""
    anchor = float(mapping["anchor"])
    lines -= anchor
    lines /= np.asarray(mapping["gain_per_row"], dtype=np.float64)[:, None]
    lines += anchor
    lines -= (np.asarray(mapping["zero_per_row_ire"], dtype=np.float64)
              * float(spans["units_per_ire"]))[:, None]


def _realise_iq_imbalance(field, uphet, transform, context, amount):
    """THE QUADRATURE IMAGE TAKEN OFF THE UP-CONVERTED CHROMA, by the amount
    it reproduces.

    The pair is the latched `image` departure at this head's chroma vertex,
    passed as `alpha = 1` and CONJUGATED - the decoder's up-conversion runs
    its oscillator above the colour-under and conjugates the baseband, the
    fact `iq_imbalance.correct` records at -252.7 dB against -16.7 - with
    `line_offset` placing this field's first row against the first picture
    row the pair was measured from, so each line receives the coefficient
    its own burst frame carries. The reference is the burst as OBSERVED on
    these lines, in `line_reference`'s convention, because `correct`
    divides the image's own wobble out of it.

    THE AMOUNT IS THE RESOLVED AMOUNT TIMES THE AGREEMENT: the image
    channel's colour contrast under REMOVE is scaled by how far the two
    banks agree about it - the R5 rule - so a pair the banks do not
    reproduce is applied by nothing, and no threshold is chosen anywhere.
    """
    if not hasattr(iq_imbalance, "correct"):
        return {"applied": False, "why": "iq_imbalance.correct is not present"}
    if "image" not in transform.channels or "colour" not in transform.axes:
        return {"applied": False, "why": "no image channel on this transform"}
    sl = transform.channels["image"]
    head = int(bool(field.isFirstField))
    try:
        beta = complex(_departure(transform, 1, head, 0, "image",
                                  transform.latched.treatment
                                  if transform.latched else None)[0])
        _contrasts, amounts = transform.contrasts("remove")
        agreement = float(np.asarray(amounts[("colour",)])[sl][0])
    except (ValueError, KeyError) as error:
        return {"applied": False, "why": "the image is not judged yet (%s)"
                % (error,)}
    gain = float(amount) * agreement
    if not np.isfinite(beta) or not abs(beta) < 1.0 or gain <= 0.0:
        return {"applied": False, "beta": beta, "agreement": agreement,
                "why": "no reproducible image to remove"}
    burst = context.get("image")
    first_row = int(burst["first_row"]) if burst and "first_row" in burst \
        else int(context["geometry"]["quiet"][0])
    window = _burst_window(field)
    if window is None:
        return {"applied": False, "why": "no burst window"}
    geom = context["geometry"]
    chroma = np.asarray(uphet, dtype=np.float64).reshape(geom["lines"],
                                                         geom["width"])
    shared = chroma.base is uphet or np.shares_memory(chroma, uphet)
    if not shared:
        chroma = np.array(chroma, dtype=np.float64)
    reference = iq_imbalance.line_reference(chroma, window)
    before = _image_reading(chroma, window)
    try:
        result = iq_imbalance.correct(chroma, reference, 1.0, np.conj(beta),
                                      amount=gain, line_offset=-first_row)
    except (ValueError, TypeError) as error:
        return {"applied": False, "beta": beta, "agreement": agreement,
                "why": "iq_imbalance.correct refused: %s" % (error,)}
    if not shared:
        uphet[:] = chroma.reshape(-1).astype(uphet.dtype, copy=False)
    after = _image_reading(chroma, window)
    return {
        "applied": True,
        "beta": beta,
        "agreement": agreement,
        "amount": gain,
        "lines": int(result["lines"]),
        "skipped": int(result["skipped"]),
        "image_rejection_db": float(result["image_rejection_db"]),
        "residual_image_rejection_db": float(
            result["residual_image_rejection_db"]),
        "before": before,
        "after": after,
    }


def _image_reading(chroma, window):
    """The module's own instrument on these lines, before and after; on the
    decoder's locked, up-converted lines it reports itself not usable, and
    that verdict is carried rather than hidden."""
    try:
        found = iq_imbalance.image_of_lines(chroma, window)
    except (ValueError, TypeError) as error:
        return {"usable": False, "why": str(error)}
    return {k: found.get(k) for k in ("usable", "image_ratio",
                                      "image_rejection_db", "over_control",
                                      "why")}


def _realise_picture(field, uphet, context, transform, store, treatment,
                     amount, wave=None):
    """EVERYTHING REALISED, composed into ONE float copy of the luma and
    written back ONCE; then the chroma.

    The order inside the composition is the legacy's where it matters and
    exact where it does not: the two predicted components measured on the
    field as it arrived - the decoder's own precursor and the locked
    colour-under estimate - are subtracted first, from the field they were
    measured on; the closed-form level map follows; and the response kernel
    last, which commutes with the map exactly, its DC being unity. The
    precursor is the decoder's own kernel and is applied in full, as
    `correct_precursor` does and for its reason; the colour-free transfer
    is this field's own fit at its node's amount, as `correct_colour_free
    _luma` applies it. The resolved amount governs the two realisations
    that come from the transform's latched departures, the kernel and the
    level map, and the image.
    """
    spans = context["spans"]
    geom = context["geometry"]
    on = context["on"]
    options = field.rf.options
    picture = field.dspicture
    head = int(bool(field.isFirstField))
    luma = np.array(picture, dtype=np.float64).reshape(spans["lines"],
                                                       spans["width"])
    applied = {"treatment": treatment, "amount": float(amount), "head": head}

    reading = context.get("precursor")
    if reading is not None:
        _realise_precursor(luma, reading)
        applied["precursor"] = {
            "peak_ire": float(np.abs(reading["removed_ire"]).max()),
            "rms_ire": float(np.sqrt(np.mean(reading["removed_ire"] ** 2))),
            "onset": int(reading["onset"]),
        }

    reading = context.get("colour_free")
    if reading is not None:
        luma = _realise_colour_free(luma, reading)
        applied["colour_free"] = {
            "transfer": reading["transfer"],
            "removed": float(reading["share"]["removed"]),
        }

    gain_on = on["sync_depth"] and not _sync_depth_superseded(field)
    zero_on = on["standard_levels"]
    if (gain_on or zero_on) and "levels" in transform.channels \
            and transform.evidence()["levels"]["used"]:
        mapping = _level_map(transform, spans, head, treatment, amount,
                             gain_on, zero_on, wave)
        _realise_level_map(luma, spans, mapping)
        applied["levels"] = {k: v for k, v in mapping.items()
                             if k not in ("gain_per_row", "zero_per_row_ire")}

    if on["source_correction"] and transform.evidence()["response"]["used"]:
        kernels = _response_kernels(transform, store, spans, geom, treatment,
                                    amount, wave,
                                    getattr(field, "field_number", None))
        entry = kernels.get(head)
        if entry is not None:
            result = _realise_source_kernel(luma, geom, entry["taps"])
            applied["response"] = {
                "taps": int(entry["taps"].size),
                "bins": entry["bins"],
                "fraction": entry["fraction"],
                "maximum_gain_db": entry["maximum_gain_db"],
                "capped_bins": entry["capped_bins"],
                "departure_db_rms": entry["departure_db_rms"],
                "lines": int(result["lines"]),
                "changed_rms": float(result["changed_rms"]),
            }
    _write_back(picture, luma)

    if uphet is not None:
        reading = context.get("chroma_head_switch")
        if reading is not None:
            _realise_chroma_head_switch(uphet, reading,
                                        options.chroma_head_switch)
            applied["chroma_head_switch"] = {
                "lines": sorted(int(k) for k in reading["phases"]),
                "degrees": {int(k): float(np.degrees(v))
                            for k, v in reading["phases"].items()},
            }
        if on["iq_imbalance"]:
            applied["iq_imbalance"] = _realise_iq_imbalance(
                field, uphet, transform, context, amount)
    return applied


def _remainders(transform, values, amount):
    """EACH FIELD'S OWN READING AGAINST THE DEPARTURE EACH TREATMENT WOULD
    REMOVE, for every channel and vertex this field filled: the exact
    inverse and the applied amount side by side, because the second is the
    first scaled and costs nothing to carry."""
    out = {}
    for treatment in single_transform.TREATMENTS:
        minimum = 2 if treatment == "remove" else 1
        if not transform.ready(minimum, treatment):
            out[treatment] = None
            continue
        per = {}
        for channel, entries in values.items():
            sl = transform.channels[channel]
            for (colour, head, polarity), x in entries.items():
                try:
                    expected = transform.departure(
                        _vertex(transform, colour, head, polarity),
                        treatment)[sl]
                except ValueError:
                    continue
                before = float(np.sqrt(np.mean(np.abs(x) ** 2)))
                exact = float(np.sqrt(np.mean(np.abs(x - expected) ** 2)))
                partial = float(np.sqrt(np.mean(
                    np.abs(x - float(amount) * expected) ** 2)))
                name = "colour%d:head%d:polarity%d" % (colour, head, polarity)
                per.setdefault(channel, {})[name] = {
                    "before": before,
                    "after_exact": exact,
                    "after_applied": partial,
                    "share_explained_exact": (1.0 - (exact / before) ** 2)
                    if before > 0 else float("nan"),
                    "share_explained_applied": (1.0 - (partial / before) ** 2)
                    if before > 0 else float("nan"),
                }
        out[treatment] = per
    return out


def _picture_field_rate(field):
    """The field rate the wave's time axis is scaled by, from the system's
    own frame rate: two fields a frame."""
    try:
        return 2.0 * float(field.rf.SysParams["FPS"])
    except (AttributeError, KeyError, TypeError, ValueError):
        return 0.0


def _record_picture_wave(transform, field, values):
    """Record this field's remainder into every vertex it filled on all
    the channels in use, and report the fold and the comparison."""
    number = getattr(field, "field_number", None)
    rate = _picture_field_rate(field)
    if number is None or not rate > 0.0:
        return None
    used = [name for name in transform.channel_names
            if transform.evidence()[name]["used"]]
    keys = None
    for name in used:
        present = set((values.get(name) or {}).keys())
        keys = present if keys is None else keys & present
    keys = keys or set()
    out = {"recorded": [], "applied": False, "folded": {}, "why": (
        "the remainder after the latch, per vertex, folded on the bits of "
        "the field index; the comparison is the remainder after the "
        "constant alone against after the constant and the admitted wave")}
    for key in sorted(keys):
        colour, head, polarity = key
        vertex = _vertex(transform, colour, head, polarity)
        vector = np.zeros(transform.bins, dtype=np.complex128)
        for name in used:
            sl = transform.channels[name]
            vector[sl] = np.asarray(values[name][key], dtype=np.complex128).reshape(-1)
        try:
            transform.record_remainder(vertex, int(number), vector, rate)
        except ValueError:
            continue
        name = "colour%d:head%d:polarity%d" % key
        out["recorded"].append(name)
        folded = transform.wave(vertex)
        if folded is None:
            continue
        value = transform.wave_at(vertex, int(number))
        index = transform.vertex_index(vertex)
        after_constant = vector - transform.latched.departures[index]
        after_wave = after_constant if value is None else after_constant - value
        comparison = {}
        for cname in used:
            sl = transform.channels[cname]
            comparison[cname] = {
                "after_constant": float(np.sqrt(np.mean(np.abs(after_constant[sl]) ** 2))),
                "after_wave": float(np.sqrt(np.mean(np.abs(after_wave[sl]) ** 2))),
            }
        out["folded"][name] = {
            "fields_used": folded["fields_used"],
            "admitted": folded["admitted"],
            "noise_floor": folded["noise_floor"],
            "value_rms": (None if value is None else
                          float(np.sqrt(np.mean(np.abs(value) ** 2)))),
            "null_space_against_wave": comparison,
        }
    return out


def _picture_wave_now(transform, field, head):
    """The admitted wave at this field for every vertex of its head, keyed
    the way `_departure` reads it; empty when no run has folded."""
    number = getattr(field, "field_number", None)
    if number is None:
        return {}
    out = {}
    colours = (0, 1) if "colour" in transform.axes else (0,)
    for colour in colours:
        for polarity in (0, 1):
            vertex = _vertex(transform, colour, head, polarity)
            try:
                value = transform.wave_at(vertex, int(number))
            except ValueError:
                value = None
            if value is not None:
                out[(colour, head, polarity)] = value
    return out


# --------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------

def transform_picture(field, uphet, amount=None):
    """THE ONE PICTURE STAGE: measure everything once, fill the vertices,
    latch, realise together.

    Node `picture_transform`, gated by its option and seeded on by its
    declaration; its parts `latch`, `causal_split`, `report`,
    `remove_residuals` and `exact_inverse` answer to
    `--stages -picture_transform.<part>` and `+...`. `uphet` may be None,
    in which case the transform runs on the luma alone with the colour axis
    a known constant of the decode. `amount` overrides the resolved amount
    where a caller passes one.

    PER FIELD, IN THIS ORDER. The field is refused where the option is off,
    the picture is missing or the field is not the shape `geometry` expects.
    `picture_context` then takes every measurement on the field as it
    arrived. The vertices are filled - once per `readloc`, so a field the
    decoder redoes, which is a new object, does not count twice - with the
    witnesses marking a transient. The transform latches once it holds
    `_LOCK_DECISION_FIELDS` fills at every vertex of every channel in use,
    the same decision count the source correction uses; nothing is applied
    before the latch and everything after it, from the LATCHED departures,
    in one write. The reading is attached as `field.picture_transform`, the
    per-field caches are dropped and the down-converted chroma released.

    BOTH TREATMENTS ARE CARRIED EVERY FIELD. The applied one is
    `substitute` unless `remove_residuals` is named; the other's remainder
    is computed beside it from the running contrasts so the decode can
    compare them, which is what Ethan asked for: "let's try both out".

    WHAT FACE DUPLICATION DOES TO THE DECLARATION, said plainly because it
    is the finding of building this. The chroma has no polarity state, so
    its response and its level are filled identically on both faces; the
    fold's polarity contrast is the average over colour of the two colours'
    fall-rise differentials, so exactly half of the luma's own polarity
    difference lands in the colour-by-polarity interaction. The picture
    declaration keeps `polarity` and not `colour:polarity`, so under
    SUBSTITUTE a luma-only polarity quantity - the sync spacing's deficit
    above all - is realised at HALF, while under REMOVE the interaction is
    admitted by its reproducibility and the deficit is realised whole.
    Measured on a synthetic decode with a planted deficit of 2 IRE, the
    luma tip's 4 IRE departure is left at 0.0003 IRE under remove and at
    0.5 IRE under substitute - a quarter of the 2 IRE fall-rise difference
    on each of the two faces, so the spacing's deficit is realised at half.
    The luma-only transform has no colour axis and no such term. The remainders under both treatments show it on every
    field; changing it is a change to `single_transform.PICTURE_SIGNAL`,
    not to this file.
    """
    options = field.rf.options
    if not getattr(options, _PICTURE_STAGE, 0):
        return None
    picture = getattr(field, "dspicture", None)
    if picture is None:
        return None
    try:
        geom = geometry(field)
    except (KeyError, AttributeError, TypeError, ValueError):
        return None
    expected = geom["lines"] * geom["width"]
    if len(picture) != expected:
        return None
    if uphet is not None and len(uphet) != expected:
        return None

    components = _picture_components(field)
    treatment = "remove" if components["remove_residuals"] else "substitute"
    amount, reason = _resolve_amount(amount, components)
    context = picture_context(field, uphet)
    spans = context["spans"]
    reading = {
        "components": components,
        "treatment": {"applied": treatment,
                      "computed": list(single_transform.TREATMENTS)},
        "amount": {"value": amount, "reason": reason},
        "on": context["on"],
        "declined": context["declined"],
        "latched": False,
        "applied": None,
    }
    if spans is None:
        _log_once(field, "picture_transform_declined",
                  "model stages: picture_transform declined - %s",
                  context["declined"]["spans"])
        field.picture_transform = reading
        return reading

    store = _picture_store(field)
    transform = _picture_transform(field, spans, uphet is not None,
                                   components["causal_split"])
    head = int(bool(field.isFirstField))
    readloc = getattr(field, "readloc", None)
    if readloc is None:
        readloc = id(field)
    folded = readloc in store["folded"]

    # THE PARITY WITNESS, decided against the previous field folded: the
    # heads alternate, so two consecutive fields of one parity are a skip or
    # a repeat, and neither is evidence about the path.
    witness = dict(context["witness"])
    last = store.get("last")
    if not folded and last is not None and last["readloc"] != readloc \
            and last["parity"] == head:
        witness["parity"] = True
    transient = any(witness.values())

    outcome, values = {}, {}
    if not folded:
        outcome, values = _fill_picture(transform, store, field, context,
                                        transient)
        store["folded"].add(readloc)
        store["last"] = {"readloc": readloc, "parity": head}
        store["fields"] += 1

    latched = None
    if components["latch"]:
        latched = transform.latch(treatment, _LOCK_DECISION_FIELDS)
        if latched is not None and store.get("latched_at") is None:
            store["latched_at"] = store["fields"]
            _log_once(field, "picture_transform_latched",
                      "model stages: picture_transform latched after %d "
                      "fields under %s at amount %.2f (%s): %s",
                      store["fields"], treatment, amount, reason,
                      latched.why)
    live = latched is not None or (
        not components["latch"]
        and transform.ready(_LOCK_DECISION_FIELDS, treatment))
    if not live:
        _name_starving_vertices(field, transform)

    # THE WAVE UNDER THE CONVERGENCE. After the latch, this field's own
    # readings less the latched departure go into each filled vertex's wave
    # (one vector per vertex, every channel in use), which folds them on the
    # bits of the field index. The comparison Ethan asked for - "what the
    # comparison is between the wave and null space" - is carried per field:
    # the remainder after the latched constant alone against the remainder
    # after the constant and the admitted wave. Applied only when the
    # component is named; folded and reported regardless.
    wave_reading = None
    wave_now = None
    if transform.latched is not None and not folded and values:
        wave_reading = _record_picture_wave(transform, field, values)
    if live and components["wave"] and transform.latched is not None:
        wave_now = _picture_wave_now(transform, field, head)
        if wave_reading is not None:
            wave_reading["applied"] = bool(wave_now)

    applied = None
    if live:
        applied = _realise_picture(field, uphet, context, transform, store,
                                   treatment, amount, wave_now)

    reading.update({
        "axes": transform.axes,
        "unreached": dict(single_transform.UNREACHED["picture"],
                          **store["unreached"]),
        "channels": {name: {
            "bins": sl.stop - sl.start,
            "used": bool(transform.evidence()[name]["used"]),
        } for name, sl in transform.channels.items()},
        "grid_hz": store["grid"],
        "folded": not folded,
        "witness": witness,
        "transient": transient,
        "fills": outcome,
        "latched": transform.latched is not None,
        "live": bool(live),
        "applied": applied,
        "wave": wave_reading,
        "measured": _measured_summary(context),
    })
    if components["report"]:
        reading["report"] = transform.report()
        reading["remainder"] = _remainders(transform, values, amount)
    else:
        reading["report"] = None
        reading["remainder"] = None

    # THE CACHES ARE THIS FIELD'S AND THE PICTURE HAS MOVED: a stale pooled
    # pulse or level series would describe the field as it arrived, not as
    # it is delivered. The down-converted chroma is released here because
    # the legacy list's `luma_beat` no longer runs to release it.
    field.__dict__.pop("_sync_pulses", None)
    field.__dict__.pop("_level_series", None)
    if getattr(field, "chroma_under_tbc", None) is not None:
        field.chroma_under_tbc = None
    field.picture_transform = reading
    return reading


def _name_starving_vertices(field, transform):
    """A latch that cannot fire is said once, with the vertex to blame: a
    channel some vertices have filled to the decision count while another
    has never been filled is a channel whose instrument declines on every
    field of this decode, and that is the decode's finding rather than a
    delay to wait out."""
    evidence = transform.evidence()
    starving = []
    for name, entry in evidence.items():
        if not entry["used"] or entry["fewest"] > 0:
            continue
        if entry["most"] < _LOCK_DECISION_FIELDS:
            continue
        for vertex, count in entry["per_vertex"].items():
            if count != 0:
                continue
            # keyed by the coordinates as text ("0,1,0") so a reading can be
            # written out; an older form keyed by the tuple itself
            coordinates = (tuple(int(v) for v in vertex.split(","))
                           if isinstance(vertex, str)
                           else tuple(int(i) for i in vertex))
            starving.append("%s at %s" % (name, dict(zip(transform.axes,
                                                         coordinates))))
    if starving:
        _log_once(field, "picture_transform_starving",
                  "model stages: picture_transform cannot latch - %s "
                  "%s never filled while the channel's other vertices have "
                  "reached the decision count, so that instrument declines "
                  "on every field of this decode",
                  "vertex" if len(starving) == 1 else "vertices",
                  "; ".join(starving))


def _measured_summary(context):
    """The absorbed nodes' own per-field readings, in the reading, so the
    transform's departures can be checked against what each instrument saw
    on this field."""
    out = {}
    faces = context.get("luma_response")
    if faces:
        out["luma_response"] = {
            polarity: None if face is None else {
                "bins": face["bins"], "delay_s": face["delay_s"],
                "is_a_delay": face["is_a_delay"]}
            for polarity, face in faces.items()}
    chroma = context.get("chroma_response")
    if chroma is not None:
        out["chroma_response"] = {
            "gain_fsc": chroma["gain_fsc"],
            "usable_bins": chroma["usable_bins"],
            "offset_range_hz": chroma["offset_range_hz"]}
    levels = context.get("luma_levels")
    if levels:
        out["luma_levels"] = {
            "reading": levels["reading"],
            "trustworthy": levels["trustworthy"],
            "why_not": levels["why_not"],
            "tip_ire": float(np.mean(levels["fall"]))
            + standard_levels.SYNC_TIP_IRE,
            "blanking_ire": float(np.mean(levels["rise"]))}
    series = context.get("levels")
    if series is not None and context["on"].get("sync_depth"):
        result = series["result"]
        out["sync_depth"] = {"spacing_ire": float(result["spacing_ire"]),
                             "gain": float(result["gain"]),
                             "clipped": bool(series["clip"]["clipped"])}
    reading = context.get("standard_levels")
    if reading is not None:
        out["standard_levels"] = {"baseline_ire": float(reading["level"]),
                                  "trend_swing_ire": float(
                                      reading["trend"]["trend_swing"])}
    reading = context.get("precursor")
    if reading is not None:
        out["precursor"] = {"peak_ire": float(np.abs(reading["removed_ire"]).max()),
                            "onset": int(reading["onset"])}
    reading = context.get("colour_free")
    if reading is not None:
        out["colour_free"] = {"transfer": reading["transfer"],
                              "removed": float(reading["share"]["removed"])}
    reading = context.get("image")
    if reading is not None:
        out["image"] = {"beta": reading.get("beta"),
                        "over_control": reading.get("over_control"),
                        "why": reading.get("why")}
    if context.get("timing") is not None:
        out["timing"] = dict(zip(_TIMING_BINS,
                                 [float(v) for v in context["timing"]]))
    return out


# --------------------------------------------------------------------------
# THE RADIO-FREQUENCY TRANSFORM'S SHIMS. Its body is `vhsdecode.rf_transform`,
# a module written beside this one; these names are the contract the call
# sites in `field.py` and `process.py` bind to, and they delegate to that
# module where it exists and answer from the published snapshot where it
# does not.
# --------------------------------------------------------------------------

_RF_TRANSFORM_ABSENT = False


def _rf_transform_module():
    """`vhsdecode.rf_transform` where it can be imported, else None. Looked
    up lazily because the module may not exist yet, and cheaply because
    `rf_transform_table` runs on the worker threads for every block: a
    module already imported is a dictionary lookup, and a failed import is
    remembered so the machinery's search is not repeated per block - while
    a module that appears in `sys.modules` afterwards is still found."""
    global _RF_TRANSFORM_ABSENT
    import sys

    module = sys.modules.get("vhsdecode.rf_transform")
    if module is not None:
        return module
    if _RF_TRANSFORM_ABSENT:
        return None
    try:
        from vhsdecode import rf_transform
    except ImportError:
        _RF_TRANSFORM_ABSENT = True
        return None
    return rf_transform


def transform_rf(field, amount=None):
    """The one radio-frequency stage, delegated to `vhsdecode.rf_transform`;
    None where that module is not present."""
    module = _rf_transform_module()
    if module is None or not hasattr(module, "transform_rf"):
        return None
    return module.transform_rf(field, amount)


def rf_transform_live(rf):
    """Whether a latched radio-frequency table has been published."""
    module = _rf_transform_module()
    if module is not None and hasattr(module, "live"):
        return module.live(rf)
    return rf.__dict__.get("_rf_transform") is not None


def rf_transform_image(rf):
    """The luma's latched quadrature image for the worker threads, or None.

    Ethan: "Apply the IQ imbalance to the luma channel as well ... correct
    it in the same existing correction pattern." The pair is measured on
    the reserved intervals' mirror tones and published beside the table in
    `Published.extras["image"]`; it is not a multiply, so it is applied to
    the analytic signal by `apply_luma_image` rather than folded into the
    table. Reads one immutable snapshot and never mutates."""
    published = rf.__dict__.get("_rf_transform")
    if published is None:
        return None
    extras = getattr(published, "extras", None) or {}
    image = extras.get("image")
    if not image:
        return None
    try:
        alpha = complex(image["alpha"])
        beta = complex(image["beta"])
        carrier = float(image["carrier_hz"])
    except (KeyError, TypeError, ValueError):
        return None
    amount = float(image.get("amount", 1.0))
    if not (abs(alpha) > abs(beta)) or amount <= 0.0:
        return None
    return {"alpha": alpha, "beta": beta, "carrier_hz": carrier,
            "amount": amount}


def apply_luma_image(hilbert, image, block_start, sample_rate_hz):
    """THE CHROMA'S CORRECTION PATTERN ON THE CARRIER'S COMPLEX BASEBAND.

    `iq_imbalance.correct_baseband` writes b = (conj(alpha) z - beta conj(z))
    / (|alpha|^2 - |beta|^2) on a baseband z = alpha b + beta conj(b). The
    analytic signal h is that baseband rotated by the carrier, h = z e^{j w
    t}, so conj(z) e^{j w t} = conj(h) e^{2 j w t} and the same map reads
    h' = (conj(alpha) h - beta conj(h) e^{2 j w t}) / (|alpha|^2 - |beta|^2),
    with t the block's absolute time from `block_start`, because the
    rotation's phase is continuous across blocks and a block that started
    the phasor at zero would put a phase step at every boundary. `amount`
    scales the change linearly, as the chroma's does. Returns a new array;
    the caller's is untouched.
    """
    if image is None or block_start is None:
        return hilbert
    n = hilbert.size
    t = (np.arange(n, dtype=np.float64) + float(block_start)) / float(sample_rate_hz)
    rotation = np.exp(1j * (4.0 * np.pi * image["carrier_hz"]) * t)
    alpha, beta = image["alpha"], image["beta"]
    scale = abs(alpha) ** 2 - abs(beta) ** 2
    corrected = (np.conj(alpha) * hilbert - beta * np.conj(hilbert) * rotation) / scale
    amount = image["amount"]
    if amount >= 1.0:
        return corrected
    return hilbert + amount * (corrected - hilbert)


def rf_transform_table(rf, block_start, n_fft):
    """The complex multiplier for one block on its full transform grid, or
    None while nothing is published. Reads one immutable snapshot and never
    mutates, because it runs on the worker threads."""
    module = _rf_transform_module()
    if module is not None and hasattr(module, "table"):
        return module.table(rf, block_start, n_fft)
    published = rf.__dict__.get("_rf_transform")
    if published is None:
        return None
    return published.table(block_start, n_fft)


def transform_wants_redo(rf):
    """True exactly once per newly published version, so the decoder's
    one-shot redemodulation repays the blocks demodulated before the latch."""
    module = _rf_transform_module()
    if module is not None and hasattr(module, "wants_redo"):
        return module.wants_redo(rf)
    published = rf.__dict__.get("_rf_transform")
    if published is None:
        return False
    done = rf.__dict__.get("_rf_transform_redone", 0)
    if published.version <= done:
        return False
    rf.__dict__["_rf_transform_redone"] = published.version
    return True
