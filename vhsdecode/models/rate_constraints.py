"""Rate constraints: no fitted parameter may vary faster than the mechanism
that owns it.

Ethan: *"Constrain parameter rates to physical time constants. Drum
29.97 Hz, head switch 59.94 Hz, capstan servo tens of Hz. Nothing
mechanical varies above a few hundred Hz; a parameter floating per-line
will absorb things it shouldn't."*

That is a statement about the TIME axis of every parameter this arc fits,
and it turns the transport model into a constraint. `transport_model`
predicts where on the time axis each rotating part can disturb the tape;
this module says which fitted parameter each part owns, and therefore the
fastest rate at which that parameter is allowed to move. A per-line sync-tip
level is one number every 63.6 us; the drum that could move it turns once
every 33 ms. Everything the per-line series carries between those two rates
is either the measurement's own noise or something the parameter has
absorbed that belongs to no mechanism - picture content, dropouts, a
line-locked structure - and a fit that follows it is fitting the wrong
thing.

FOUR INSTRUMENTS, ONE QUESTION.

  `mechanism_rates`   every rate the transport, its servos and the record
                      electronics can impose, each with its source: the
                      format specification, the deck's schematic, or an
                      assumption labelled as one.
  `allowed_rate`      the registry: which mechanism owns each fitted
                      parameter, and so its maximum rate of variation.
  `rate_test`         given a parameter's series, whether anything above
                      the mechanism's rate is there BESIDES the estimator's
                      white noise, against a Monte-Carlo white null of the
                      same estimator; where the excess sits; what is
                      line-locked (the same at the same line of every
                      field) and what moves from field to field; and the
                      slow term the halves of the record differ by.
  `constrain`         the projection onto the allowed band - a zero-phase
                      low-pass at the mechanism rate, or a harmonic model at
                      the mechanism's rate and its harmonics with the
                      per-field sweep shapes of `transport_model` - returning
                      the constrained series AND what was removed. The
                      removed part is evidence of something unmodelled and
                      is reported, never discarded.

WHAT THE RATE TEST ACTUALLY DECIDES, because the obvious version is wrong.
A per-line estimate is one sample of a noisy quantity per line, so its
series is white noise plus whatever really moves. The fraction of its power
above the mechanism rate is therefore LARGE for every honest parameter - a
white series puts 95 per cent of its power above 360 Hz when sampled at
15.7 kHz - and comparing that fraction against the white null says only how
noisy the estimator is. The question Ethan asks is different: is there
anything above the rate BESIDES noise? So the test reads the white floor
from the top of the band, where no mechanism can reach, and asks whether
the power above the allowed rate exceeds what that floor accounts for. A
slow mechanical wobble with white measurement noise passes: above the rate
its spectrum is flat. A random walk fails: its spectrum falls as one over
frequency squared, so just above the rate it stands far over the floor. A
parameter absorbing a static picture fails with a signature: the excess is
LINE-LOCKED, the same at the same line of every field.

THE SERIES THIS ARC ACTUALLY PRODUCES ARE NOT CONTIGUOUS. A per-line
parameter comes as one value per (field, line) with the vertical interval
and the head-switch window missing, and per head the fields alternate, so
the honest form of the test takes the field and line indices and works on
the two-way structure the repository already uses (`content_independence.
two_way_residual`): a per-field term (what moves every line of a field
alike - the drum, the servos, the record AGC), a per-line-position term
(what is fixed to the line's place in the field - the recovery after the
vertical interval, the tape-path sweep shapes, and on a static pattern the
content itself), and the field-varying remainder. Each part is judged
against its own white null: the remainder above the rate, field by field
with the per-field sweep shapes removed first; the line-locked term above
the rate along the line axis; and the per-field term against the slowest
things - the mechanism rates it can resolve, and the contrast between the
first and second halves of the record.

THE RATES, WITH THEIR SOURCES. Format figures come from `transport_model`
and `head_model.FORMAT_MECHANICS` (JVC VTG82063); nothing is retyped here.
The deck figures were read from the Sony SLV-777HF/778HF/788HF service
schematic (`/testdata/test_patterns/vhs/Sony Slv777Hf 778Hf 788Hf
Schematic.pdf`, MA-327 board sheet 3/8 "SERVO/SYSTEM CONTROL", printed pages
4-11 to 4-13, PDF page 3, rasterised at 300 dpi to read the part values).
IC160 is the servo/system-control microcontroller and the sheet's own
"SIGNAL PATH" table places the drum speed servo, drum phase servo, capstan
speed servo and capstan phase servo INSIDE it - so both loop filters are
firmware and the schematic carries no component values for them. What it
does carry is every tachometer and the error outputs:

    waveform 24   IC160 pin 85  CAP FG      4.4 Vp-p  1.078 kHz
    waveform 25   IC160 pin 86  DRUM PG     4.5 Vp-p  30 Hz pulses
    waveform 26   IC160 pin 87  DRUM FG     4 Vp-p    360 Hz
    waveform 17   IC160 pin 18  RF SWP      5 Vp-p    30 Hz square
    waveform 23   IC160 pin 84  C SYNC      4 Vp-p    once per H
    waveform 21   IC160 pin 77  CAP ERR     5 Vp-p    62.5 kHz  (PWM)
    waveform 22   IC160 pin 78  DRUM ERR    5 Vp-p    62.5 kHz  (PWM)

Each error output is a 62.5 kHz pulse-width modulation, smoothed by a
TWO-SECTION RC ladder before the motor's own driver: the drum's through
R116 47 k / C115 0.01 uF and then R115 47 k / C116 0.01 uF into CN101 pin 2
"D VS" of M901 DRUM MOTOR; the capstan's through R109 47 k / C111 0.01 uF
and then R110 47 k / C112 0.01 uF into CN102 pin 5 "CAP VS" of M902 CAPSTAN
MOTOR. One section corners at 1/(2 pi R C) = 339 Hz; the two-section ladder
into an open load is 1/(1 + 3 s R C + (s R C)^2), whose -3 dB point is
127 Hz (`rc_ladder_corner_hz` computes it from the parts, not from that
figure). The ladder is NOT the loop bandwidth - it is what turns the
microcontroller's PWM into a voltage - but it bounds the loop from above,
since nothing the loop commands passes it faster. The loop bandwidths
themselves remain Ethan's "tens of Hz" and are labelled as an assumption
everywhere they appear.

The 360 Hz drum FG means this deck's drum tachometer has twelve poles where
the JVC guide's generic figure (`transport_dimensions.DRUM_FG_POLES`) is
eight at 240 Hz; both are carried, the deck's winning when the deck is
named. The capstan FG disagrees between the two sources - the schematic's
waveform 24 reads 1.078 kHz, the JVC guide states 1440 Hz for SP (via
docs/RESIDUAL_LIMIT_DESIGN.md) - and both are carried with their sources;
neither sets any ceiling, because the capstan acts on the linear tape
speed, which is under one per cent of the head-to-tape speed.

THE CEILING, DERIVED TWICE. "Nothing mechanical varies above a few hundred
Hz" is reached two ways here, both from things already in the tree. The
tape-path shapes of `transport_model.sweep_shapes` are at most quadratic
along a field, and a periodic parabola's harmonics fall as one over n
squared in amplitude, so `sweep_ceiling_hz` finds the field harmonic at
which its variance has converged: 99 per cent by the third harmonic, 180 Hz.
And the fastest torque ripple on the part that sets the head-to-tape speed
is the drum motor's own FG rate: 240 Hz generic, 360 Hz on this deck. The
larger of the two is the ceiling, and it is a few hundred hertz.

MEASURED VERDICTS (2026-09-05, `tools/ringing_measure/rate_constraints_
measure.py`, the numbers it printed are reproduced at the foot of this
docstring). Read together they say one thing: on this deck every per-line
level, width, phase and envelope estimate is WHITE above the ceiling once
its line-locked part is set aside - the field-varying remainder never
exceeds its white null on any of 32 series across three captures and the
RF - so a per-line float absorbs only the estimator's noise, and the
constraint costs nothing but that noise. What is NOT white is the
line-locked part: on the levels it stands 4 to 400 times over its null,
and the locator puts it in the lines just after the vertical interval
(the recovery of the record electronics, an electronic transient that is
fixed to the line position and is no mechanism), on the sync width in the
same place, and on the back porch of the bars tape everywhere (the burst's
leakage, which is content). The slow term the tesseract found - the
halves of a record differing - is present here too and its rate is stated:
a half-contrast lives at the record's own fundamental, 1/T, and its odd
harmonics (81 per cent of its variance at 1/T), which for the 0.22 s
captures is 4.5 Hz and for a 3 s decode 0.33 Hz - the reel band and
slower, never the drum or the servos, which every half averages over many
revolutions.

    (measured table follows; see MEASURED_VERDICTS below)
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from vhsdecode.models import (head_model, sync_geometry, transport_dimensions,
                              transport_model, vcr_agc)


# --------------------------------------------------------------------------
# the deck's own figures, read from its schematic
# --------------------------------------------------------------------------

# Sony SLV-777HF/778HF/788HF, MA-327 board sheet 3/8 (SERVO/SYSTEM CONTROL),
# printed pages 4-11 to 4-13. Waveform numbers are the sheet's own circled
# references; part numbers were read from the sheet rasterised at 300 dpi.
# The playback captures under test were made on an SLV-778HF
# (/testdata/test_patterns/vhs/readme.txt), so these are the rates of the
# machine that read the tape.
DECKS: Dict[str, Dict[str, object]] = {
    "SLV-778HF": {
        "servo_ic": ("IC160 'SERVO/SYSTEM CONTROL' (Mitsubishi M37777-class "
                     "single-chip microcontroller)"),
        "servo_loops_are_firmware": True,
        "drum_fg_hz": 360.0,          # waveform 26, IC160 pin 87 DRUM FG
        "drum_pg_hz": 30.0,           # waveform 25, IC160 pin 86 DRUM PG
        "capstan_fg_hz": 1078.0,      # waveform 24, IC160 pin 85 CAP FG
        "head_switch_square_hz": 30.0,   # waveform 17, IC160 pin 18 RF SWP
        # the error outputs are pulse-width modulation at this carrier
        "error_pwm_hz": 62.5e3,       # waveforms 21/22, pins 77 CAP ERR, 78 DRUM ERR
        # each error output is smoothed by two cascaded sections of
        # 47 kilohm series and 0.01 microfarad shunt before the motor's own
        # driver: drum R116/C115 then R115/C116 into CN101 pin 2 D VS;
        # capstan R109/C111 then R110/C112 into CN102 pin 5 CAP VS
        "error_smoothing_r_ohm": 47e3,
        "error_smoothing_c_f": 0.01e-6,
        "error_smoothing_sections": 2,
        "error_smoothing_parts": {
            "drum": "R116/C115 + R115/C116 -> CN101 pin 2 'D VS' (M901)",
            "capstan": "R109/C111 + R110/C112 -> CN102 pin 5 'CAP VS' (M902)",
        },
        "source": ("Sony SLV-777HF/778HF/788HF schematic, MA-327 sheet 3/8 "
                   "'SERVO/SYSTEM CONTROL', printed pages 4-11 to 4-13 "
                   "(PDF page 3)"),
    },
}

# The JVC guide's capstan tachometer rates by tape speed (JVC VTG82063 as
# quoted in docs/RESIDUAL_LIMIT_DESIGN.md: "Capstan FG by mode: SP 1440 Hz,
# LP 720 Hz, EP 480 Hz"). A format-generic figure; the deck's schematic
# reads 1.078 kHz for this machine and wins when the deck is named.
CAPSTAN_FG_HZ_BY_MODE: Dict[str, float] = {"SP": 1440.0, "LP": 720.0,
                                           "EP": 480.0}

# Ethan's figure for the servo loops, as a range: "capstan servo tens of Hz".
# The schematic cannot refine it (see the module docstring), so it stays an
# ASSUMPTION and is labelled one wherever it is reported.
ASSUMED_SERVO_BANDWIDTH_HZ: Tuple[float, float] = (10.0, 100.0)

# The record-side video AGC's time constant, as a range. ASSUMPTION: no
# standard states it (`vcr_agc` establishes that BT.1700 and BT.1701-1 are
# both silent, and its own assumed span of 1 us to 50 ms brackets this) and
# the schematic's video sheet (2/8) shows IC201's AGC block without the
# loop's capacitor value. A keyed sync-tip AGC samples once per line and is
# built to ignore line-rate and field-rate content, so it settles over tens
# of milliseconds; the corner is 1/(2 pi tau) and the fast end of the range
# is what the registry allows.
ASSUMED_AGC_TIME_CONSTANT_S: Tuple[float, float] = (0.010, 0.100)

# How much of a sweep shape's variance its harmonics must carry before the
# shape is considered captured. A structural choice, not a specification:
# 0.99 places the quadratic bow's ceiling at its third field harmonic and
# moving it to 0.999 places it at the seventh, both still "a few hundred
# hertz" - the conclusion does not turn on it.
SWEEP_CAPTURE_SHARE = 0.99

# The white floor is read from the top of the band, and this is the share of
# the band it is read from. No mechanism reaches the top quarter of a
# per-line series (3.9 kHz and above at the line rate), so what sits there is
# the estimator's own noise. Guarded below: the floor band always starts
# above the allowed rate.
FLOOR_BAND_SHARE = 0.25

# Monte-Carlo draws for the white null and the percentile that sets the
# verdict. 200 draws resolve a 99th percentile to about one part in fifty,
# which is finer than any verdict here depends on.
NULL_DRAWS = 200
NULL_PERCENTILE = 99.0

# A first/second-half contrast is the projection of the series onto a step
# at the middle of the record. Over one record length that step is one
# period of a square wave, whose odd harmonics carry 8/(pi n)^2 of its
# variance each: the fundamental alone holds 8/pi^2 = 81 per cent.
STEP_FUNDAMENTAL_SHARE = 8.0 / np.pi ** 2


def line_rate_hz(system: str = "NTSC") -> float:
    """The line rate from the standard's own line period (ITU-R BT.1700
    Table 3 via `sync_geometry.LINE_PERIOD_US`): 15734.264 Hz for 525/60."""
    return 1e6 / float(sync_geometry.LINE_PERIOD_US[sync_geometry._system(system)])


def field_rate_hz(system: str = "NTSC") -> float:
    """The field rate from the standard: the line rate over the lines in a
    field, which is half the system's line count. 59.94006 Hz for 525/60 -
    NOT the 30 Hz drum figure the JVC guide rounds to, which is 0.1 per cent
    high and would put every predicted line 0.1 per cent off."""
    key = sync_geometry._system(system)
    return line_rate_hz(system) / (float(key) / 2.0)


def rc_ladder_response(frequency_hz, r_ohm: float, c_f: float,
                       sections: int = 1) -> np.ndarray:
    """|H| of `sections` cascaded series-R, shunt-C sections driving an open
    load, evaluated from the parts: walking back from the output node with
    unit voltage and no load current, each section adds the shunt current
    and the drop across its resistor. One section is the familiar
    1/(1 + s R C); two are 1/(1 + 3 s R C + (s R C)^2)."""
    omega = 2.0 * np.pi * np.asarray(frequency_hz, dtype=np.float64)
    voltage = np.ones_like(omega, dtype=np.complex128)
    current = np.zeros_like(omega, dtype=np.complex128)
    for _ in range(int(sections)):
        current = current + 1j * omega * float(c_f) * voltage
        voltage = voltage + float(r_ohm) * current
    return 1.0 / np.abs(voltage)


def rc_ladder_corner_hz(r_ohm: float, c_f: float, sections: int = 1
                        ) -> float:
    """The -3 dB frequency of the ladder, found on the response itself by
    bisection so that no closed-form factor is typed in."""
    single = 1.0 / (2.0 * np.pi * float(r_ohm) * float(c_f))
    low, high = single * 1e-3, single * 1e3
    target = 1.0 / np.sqrt(2.0)
    for _ in range(200):
        middle = np.sqrt(low * high)
        if rc_ladder_response(middle, r_ohm, c_f, sections) > target:
            low = middle
        else:
            high = middle
    return float(np.sqrt(low * high))


def sweep_ceiling_hz(field_rate_hz: float, line_count: int = 263,
                     share: float = SWEEP_CAPTURE_SHARE,
                     shape: str = "parabolic_bow_m") -> Dict[str, object]:
    """The field harmonic at which a sweep shape's variance has converged.

    A shape from `transport_model.sweep_shapes` repeats once per field, so
    its spectrum is a comb at the field rate and its harmonics. This tiles
    the shape over many fields, takes its spectrum, and finds the harmonic
    by which `share` of its variance is in hand. The parabolic bow converges
    fast (its harmonics fall as one over n squared in amplitude); the linear
    tilt does not, because repeated per field it is a sawtooth whose
    discontinuity is the head switch - an event, not a rate, which is why
    the tests below remove the per-field shapes field by field."""
    shapes = transport_model.sweep_shapes(int(line_count))
    if shape not in shapes:
        raise KeyError(f"unknown sweep shape {shape!r}; known: {list(shapes)}")
    one = shapes[shape] - shapes[shape].mean()
    fields = 64            # enough repeats that the comb's teeth are sharp
    tiled = np.tile(one, fields)
    power = np.abs(np.fft.rfft(tiled)) ** 2
    power[0] = 0.0
    total = float(power.sum())
    if total <= 0:
        return {"harmonic": 0, "ceiling_hz": 0.0, "shape": shape}
    # harmonic n of the field rate sits at bin n * fields
    harmonics = np.arange(1, len(one) // 2 + 1)
    bins = np.minimum(harmonics * fields, len(power) - 1)
    cumulative = np.cumsum(power[bins]) / total
    reached = int(np.searchsorted(cumulative, float(share)) + 1)
    return {
        "shape": shape,
        "harmonic": reached,
        "ceiling_hz": float(reached * field_rate_hz),
        "share_captured": float(cumulative[min(reached - 1,
                                               len(cumulative) - 1)]),
        "share_required": float(share),
        "why": ("a sweep shape repeats once per field, so its spectrum is a "
                "comb at the field rate; the harmonic at which its variance "
                "converges is the fastest rate the shape can impose"),
    }


def mechanism_rates(tape_format: str = "VHS", system: str = "NTSC",
                    tape_speed: str = "SP", deck: Optional[str] = "SLV-778HF"
                    ) -> Dict[str, Dict[str, object]]:
    """Every rate a mechanism can impose on a fitted parameter, with its
    source.

    The rotating parts come from `transport_model.rotation_rates` at the
    format's own linear speed and field rate, so nothing here retypes a
    diameter or a rotation. Added on top: the head switch (two per drum
    revolution on a two-head drum), the tachometers (format-generic from
    `transport_dimensions.DRUM_FG_POLES` and the JVC guide's capstan FG, and
    the deck's own from its schematic when the deck is named), the servo
    error smoothing (from the schematic's parts), the servo loop bandwidths
    (an assumption, labelled), the record-side AGC (an assumption,
    labelled; no standard states it - see `vcr_agc`), and the sweep-shape
    ceiling derived from `transport_model.sweep_shapes`.

    Each entry carries `kind`: "mechanical" for a rotating part or the
    tape-path geometry, "tachometer" for a motor's FG or PG - where servo
    ripple lands, "servo" for a loop or its smoothing, "electronic" for a
    record-side circuit, "format" for the standard's own rates. `certain`
    is True only for the format's own figures.
    """
    mechanics = head_model.mechanics_for(tape_format, system, tape_speed)
    if mechanics is None:
        raise KeyError(f"no mechanics for ({tape_format}, {system}, "
                       f"{tape_speed}); known: "
                       f"{list(head_model.FORMAT_MECHANICS)}")
    field_rate = field_rate_hz(system)
    linear_speed = float(mechanics["linear_tape_speed_m_s"])
    out: Dict[str, Dict[str, object]] = {}
    for part in transport_model.rotation_rates(linear_speed, field_rate):
        name = str(part["name"])
        entry = {
            "rate_hz": float(part["rate_hz"]),
            "kind": "mechanical",
            "acts_on": part.get("acts_on"),
            "after_capstan": part.get("after_capstan"),
            "certain": bool(part.get("certain", False)),
            "source": ("JVC VTG82063 via head_model.FORMAT_MECHANICS and "
                       "transport_model.TRANSPORT: " + str(part["rate_note"])),
        }
        if "rate_band_hz" in part:
            entry["band_hz"] = tuple(float(v) for v in part["rate_band_hz"])
        out[name] = entry
    drum_hz = out["head drum"]["rate_hz"]

    out["head switch"] = {
        "rate_hz": float(drum_hz * transport_model.FIELDS_PER_DRUM_REVOLUTION),
        "kind": "mechanical",
        "certain": True,
        "source": ("transport_model.FIELDS_PER_DRUM_REVOLUTION: one head per "
                   "field, so one switch per field"),
        "note": ("an EVENT at the field rate, not a smooth rate: a step "
                 "between heads whose spectrum is broadband; the per-field "
                 "shapes absorb it"),
    }
    out["drum FG (format)"] = {
        "rate_hz": float(drum_hz * transport_dimensions.DRUM_FG_POLES),
        "kind": "tachometer",
        "certain": False,
        "source": ("JVC VTG82063 section 4 via transport_dimensions."
                   "DRUM_FG_POLES: 240 Hz drum FG for NTSC"),
        "note": "where drum-servo ripple lands rather than on the drum rate",
    }
    mode = str(tape_speed).upper()
    if mode in CAPSTAN_FG_HZ_BY_MODE:
        out["capstan FG (format)"] = {
            "rate_hz": float(CAPSTAN_FG_HZ_BY_MODE[mode]),
            "kind": "tachometer", "certain": False,
            "source": ("JVC VTG82063 via docs/RESIDUAL_LIMIT_DESIGN.md: "
                       "capstan FG by mode SP 1440 / LP 720 / EP 480 Hz"),
            "note": ("acts on the LINEAR tape speed, under one per cent of "
                     "the head-to-tape speed"),
        }
    sweep = sweep_ceiling_hz(field_rate)
    out["tape-path sweep shapes"] = {
        "rate_hz": float(sweep["ceiling_hz"]),
        "kind": "mechanical",
        "certain": False,
        "harmonic": int(sweep["harmonic"]),
        "source": ("derived from transport_model.sweep_shapes: the field "
                   "harmonic at which a parabolic bow's variance reaches "
                   f"{100 * SWEEP_CAPTURE_SHARE:.0f} per cent"),
    }
    if deck is not None:
        if deck not in DECKS:
            raise KeyError(f"unknown deck {deck!r}; known: {list(DECKS)}")
        figures = DECKS[deck]
        source = str(figures["source"])
        out["drum FG (deck)"] = {
            "rate_hz": float(figures["drum_fg_hz"]), "kind": "tachometer",
            "certain": False, "source": source + ", waveform 26 (IC160 pin 87)",
            "note": (f"{float(figures['drum_fg_hz']) / drum_hz:.0f} poles on "
                     "this deck's drum tachometer"),
        }
        out["drum PG (deck)"] = {
            "rate_hz": float(figures["drum_pg_hz"]), "kind": "tachometer",
            "certain": False, "source": source + ", waveform 25 (IC160 pin 86)",
        }
        out["capstan FG (deck)"] = {
            "rate_hz": float(figures["capstan_fg_hz"]), "kind": "tachometer",
            "certain": False, "source": source + ", waveform 24 (IC160 pin 85)",
            "note": ("acts on the LINEAR tape speed, under one per cent of "
                     "the head-to-tape speed; the JVC guide's generic figure "
                     "for SP is 1440 Hz - the two sources disagree and both "
                     "are carried"),
        }
        r_ohm = float(figures["error_smoothing_r_ohm"])
        c_f = float(figures["error_smoothing_c_f"])
        sections = int(figures["error_smoothing_sections"])
        out["servo error smoothing (deck)"] = {
            "rate_hz": rc_ladder_corner_hz(r_ohm, c_f, sections),
            "section_corner_hz": rc_ladder_corner_hz(r_ohm, c_f, 1),
            "pwm_hz": float(figures["error_pwm_hz"]),
            "kind": "servo", "certain": False,
            "source": (source + ", the two-section 47 k / 0.01 uF ladders on "
                       "DRUM ERR (pin 78 -> D VS) and CAP ERR (pin 77 -> "
                       "CAP VS): " + str(figures["error_smoothing_parts"])),
            "note": ("the -3 dB point of the PWM smoothing ladder, an UPPER "
                     "bound on both loops - nothing the loops command "
                     "passes it faster; the loops themselves are firmware in "
                     + str(figures["servo_ic"]) + "; the PWM carrier it "
                     f"removes is {float(figures['error_pwm_hz']) / 1e3:.1f} kHz"),
        }
    low, high = ASSUMED_SERVO_BANDWIDTH_HZ
    for loop in ("drum servo loop", "capstan servo loop"):
        out[loop] = {
            "rate_hz": float(high), "band_hz": (float(low), float(high)),
            "kind": "servo", "certain": False, "assumed": True,
            "source": ("ASSUMPTION - Ethan: 'capstan servo tens of Hz'; not "
                       "on the schematic (firmware loops, see DECKS)"),
        }
    slow_tau, fast_tau = ASSUMED_AGC_TIME_CONSTANT_S
    out["record AGC"] = {
        "rate_hz": float(vcr_agc.corner_hz(fast_tau * 1e6)),
        "band_hz": (float(vcr_agc.corner_hz(slow_tau * 1e6)),
                    float(vcr_agc.corner_hz(fast_tau * 1e6))),
        "kind": "electronic", "certain": False, "assumed": True,
        "source": ("ASSUMPTION - ASSUMED_AGC_TIME_CONSTANT_S, tens of "
                   "milliseconds for a keyed sync-tip loop; no standard "
                   "states it (BT.1700 and BT.1701-1 are silent, see "
                   "vcr_agc) and the schematic's video sheet carries no "
                   "value for the loop's capacitor"),
        "note": ("record-side and frozen into the tape; the corner "
                 "1/(2 pi tau) is where the loop stops following"),
    }
    out["de-emphasis network"] = {
        "rate_hz": 0.0, "kind": "electronic", "certain": True,
        "source": ("fixed passive components (the format's 1.30 us luma "
                   "de-emphasis, vhsdecode.format_defs.vhs deemph_tau); only "
                   "thermal drift moves them, far below any rate a capture "
                   "resolves"),
    }
    out["field rate"] = {
        "rate_hz": float(field_rate), "kind": "format", "certain": True,
        "source": ("the standard's line rate over the lines in a field; the "
                   "drum turns at this over FIELDS_PER_DRUM_REVOLUTION"),
    }
    out["line rate"] = {
        "rate_hz": float(line_rate_hz(system)), "kind": "format",
        "certain": True,
        "source": ("ITU-R BT.1700 Table 3 via sync_geometry.LINE_PERIOD_US; "
                   "the sampling rate of a per-line parameter, not a "
                   "mechanism"),
    }
    return out


def mechanical_ceiling_hz(rates: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    """THE FASTEST RATE ANYTHING MECHANICAL CAN IMPOSE - "a few hundred
    hertz", derived rather than stated.

    Two candidates, both in `rates`: the field harmonic at which the
    tape-path sweep shapes have converged, and the drum's own tachometer
    rate - the fastest torque ripple on the part that sets the head-to-tape
    speed. The capstan's FG is deliberately NOT a candidate: it ripples the
    linear tape speed, which is under one per cent of the writing speed, so
    even a whole per cent of capstan ripple moves the head-to-tape speed by
    less than a hundred parts per million.
    """
    candidates = {name: float(entry["rate_hz"]) for name, entry in rates.items()
                  if name.startswith("drum FG")
                  or name == "tape-path sweep shapes"}
    name = max(candidates, key=candidates.get)
    return {
        "ceiling_hz": candidates[name],
        "set_by": name,
        "candidates": candidates,
        "why": ("the fastest of the drum's tachometer ripple and the field "
                "harmonic at which the tape-path shapes converge; the "
                "capstan's FG acts on under one per cent of the writing "
                "speed and cannot set it"),
    }


# --------------------------------------------------------------------------
# the registry: which mechanism owns each fitted parameter
# --------------------------------------------------------------------------

# Each parameter names the mechanism that owns it and how its allowed rate
# is read from `mechanism_rates`:
#
#   "ceiling"   the mechanical ceiling - the parameter is moved by the drum,
#               the heads and the tape path, through the time base or the
#               head-to-tape contact, and nothing mechanical is faster;
#   "agc"       the record-side AGC's corner (an assumption);
#   "constant"  a fixed network: the parameter has no mechanism to move it.
#
# `floats_at` records how the parameter is fitted TODAY, which is the point
# of the registry: every entry marked per line is allowed something far
# slower than it is given. Every module named here is read-only to this
# one; the registry describes them, it does not change them.
PARAMETERS: Dict[str, Dict[str, object]] = {
    "sync_tip_level": {
        "rule": "ceiling",
        "mechanism": "time base (drum, capstan and their servos) and the head",
        "where": ("vhsdecode/field.py level_windows + _window_levels (the "
                  "hsync/backporch ire0_adjust branch); "
                  "models/standard_levels.py affine_to_standard"),
        "floats_at": "per line, pooled per field by the decoder",
        "why": ("the demodulated tip level is the recorded tip frequency "
                "scaled by the playback speed error; only the transport and "
                "the head's channel can move it"),
    },
    "back_porch_level": {
        "rule": "ceiling",
        "mechanism": "time base (drum, capstan and their servos) and the head",
        "where": ("vhsdecode/field.py level_windows (after the burst, a "
                  "settling guard short of active video); "
                  "models/standard_levels.py porch_dimensions"),
        "floats_at": "per line, pooled per field by the decoder",
        "why": "a level in the same signal as the tip, moved by the same things",
    },
    "front_porch_level": {
        "rule": "ceiling",
        "mechanism": "time base (drum, capstan and their servos) and the head",
        "where": "models/sync_geometry.py front_porch_residual (the settled level)",
        "floats_at": "per line",
        "why": ("the settled blanking plus the relaxation from the preceding "
                "line; the level part is owned by the time base, the "
                "relaxation by the network"),
    },
    "sync_width": {
        "rule": "ceiling",
        "mechanism": "time base (drum, capstan and their servos)",
        "where": "models/standard_levels.py half_amplitude_crossings",
        "floats_at": "per line",
        "why": ("the recorded width scaled by the playback speed, plus the "
                "channel's fixed edge asymmetry"),
    },
    "front_porch_relaxation_tau": {
        "rule": "constant",
        "mechanism": "de-emphasis network",
        "where": "models/sync_geometry.py front_porch_residual (time_constant_us)",
        "floats_at": "per line",
        "why": ("a time constant of fixed passive components has no "
                "mechanism to vary it; only thermal drift, over minutes"),
    },
    "clip_depth": {
        "rule": "agc",
        "mechanism": "record AGC",
        "where": "models/clipping.py tip_clip_from_lines, clip_offset_from_porch",
        "floats_at": "per line, pooled per field",
        "why": ("the clip level is fixed in the record electronics; what "
                "varies is the level presented to it, which the record AGC "
                "sets at its own corner"),
    },
    "burst_phase": {
        "rule": "ceiling",
        "mechanism": "time base (drum, head switch, capstan) at the chroma "
                     "frequency",
        "where": "vhsdecode/chroma.py burst_complex_envelope",
        "floats_at": "per line",
        "why": ("the burst's phase against its own sync is the time-base "
                "error over the sync-to-burst interval; the format's 90 "
                "degree per-line rotation is structure, not variation"),
    },
    "head_response": {
        "rule": "ceiling",
        "mechanism": "head and drum: per head, and along the sweep",
        "where": "*_hs_impulse.npz levels_h*, width_h*, tdim_H_*",
        "floats_at": "per line within a field, per head between fields",
        "why": ("the head's channel differs per head and the head-to-tape "
                "contact varies along the sweep; a fixed channel cannot "
                "change faster than the contact does"),
    },
    "rf_envelope": {
        "rule": "ceiling",
        "mechanism": "head-to-tape contact (spacing by Wallace's law) on the drum",
        "where": ("vhsdecode/process.py demodblock 'envelope'; "
                  "tools/ringing_measure/content_independence.py envelope_tip"),
        "floats_at": "per sample, read per line at the sync tip",
        "why": ("the envelope is the head's output level, set by the "
                "spacing the contact pressure makes; the tape path and the "
                "drum set that pressure and nothing faster does"),
    },
    "agc_level": {
        "rule": "agc",
        "mechanism": "record AGC",
        "where": "models/vcr_agc.py",
        "floats_at": "per field",
        "why": "the loop's own corner is the fastest it can move the level",
    },
    "time_base": {
        "rule": "ceiling",
        "mechanism": "time base (drum, capstan, their servos and tachometers)",
        "where": "linelocs (vhsdecode/field.py, carrier_tbc.py); per-line period",
        "floats_at": "per line",
        "why": ("wow and flutter are the rotating parts and their servos; "
                "line-to-line jitter beyond the ceiling is the sync edge's "
                "measurement noise or an unmodelled event"),
    },
}


def allowed_rate(parameter_name: str, tape_format: str = "VHS",
                 system: str = "NTSC", tape_speed: str = "SP",
                 deck: Optional[str] = "SLV-778HF") -> Dict[str, object]:
    """The mechanism that owns a fitted parameter and its maximum rate of
    variation. Refuses a name it does not know, listing the ones it does."""
    if parameter_name not in PARAMETERS:
        raise KeyError(f"unknown parameter {parameter_name!r}; the registry "
                       f"knows {sorted(PARAMETERS)}")
    entry = dict(PARAMETERS[parameter_name])
    rates = mechanism_rates(tape_format, system, tape_speed, deck)
    rule = entry["rule"]
    if rule == "ceiling":
        ceiling = mechanical_ceiling_hz(rates)
        entry["allowed_hz"] = float(ceiling["ceiling_hz"])
        entry["set_by"] = ceiling["set_by"]
        entry["source"] = rates[ceiling["set_by"]]["source"]
        entry["assumed"] = False
        entry["harmonics_of"] = {
            "head drum": rates["head drum"]["rate_hz"],
            "head switch": rates["head switch"]["rate_hz"],
            "capstan": rates["capstan"]["rate_hz"],
        }
    elif rule == "agc":
        entry["allowed_hz"] = float(rates["record AGC"]["rate_hz"])
        entry["band_hz"] = rates["record AGC"]["band_hz"]
        entry["set_by"] = "record AGC"
        entry["source"] = rates["record AGC"]["source"]
        entry["assumed"] = True
    elif rule == "constant":
        entry["allowed_hz"] = 0.0
        entry["set_by"] = "de-emphasis network"
        entry["source"] = rates["de-emphasis network"]["source"]
        entry["assumed"] = False
    else:                                                   # pragma: no cover
        raise ValueError(f"unknown rule {rule!r}")
    entry["parameter"] = parameter_name
    return entry


# --------------------------------------------------------------------------
# spectral pieces shared by the tests
# --------------------------------------------------------------------------

def _fill_gaps(values: np.ndarray) -> np.ndarray:
    """Bridge NaN entries by linear interpolation (ends held)."""
    values = np.asarray(values, dtype=np.float64).copy()
    good = np.isfinite(values)
    if good.all():
        return values
    if good.sum() < 2:
        raise ValueError("fewer than two finite values")
    index = np.arange(len(values), dtype=np.float64)
    values[~good] = np.interp(index[~good], index[good], values[good])
    return values


def _remove_sweep_shapes(values: np.ndarray, segments: Sequence[Tuple[int, int]]
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Fit and remove the per-field sweep shapes of `transport_model` from
    each segment (a field, or one head's run of lines). Returns the remainder
    and the fitted shapes."""
    fitted = np.zeros_like(values)
    for start, stop in segments:
        block = values[start:stop]
        if len(block) < 4:
            continue
        shapes = transport_model.sweep_shapes(len(block))
        design = np.column_stack([shapes[name] for name in shapes])
        coefficients, *_ = np.linalg.lstsq(design, block, rcond=None)
        fitted[start:stop] = design @ coefficients
    return values - fitted, fitted


def _periodogram(values: np.ndarray, sample_rate_hz: float
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """A Hann-windowed periodogram normalised so that the bins sum to the
    windowed series' variance."""
    n = len(values)
    window = np.hanning(n)
    tapered = (values - values.mean()) * window
    power = np.abs(np.fft.rfft(tapered)) ** 2
    power *= 2.0 / max(float(np.sum(window ** 2)) * n, 1e-300)
    power[0] *= 0.5
    if n % 2 == 0:
        power[-1] *= 0.5
    return np.fft.rfftfreq(n, 1.0 / float(sample_rate_hz)), power


def _gamma_median_over_mean(count: int) -> float:
    """The median of an average of `count` exponential variables, as a
    fraction of its mean: ln 2 for one, approaching one as the average
    lengthens (Chen and Rubin 1986: median of Gamma(k, 1) is about
    k - 1/3 + 8/(405 k)). Turns a median read off a periodogram into the
    white level it estimates."""
    k = max(int(count), 1)
    if k == 1:
        return float(np.log(2.0))
    return float((k - 1.0 / 3.0 + 8.0 / (405.0 * k)) / k)


def _split(frequency: np.ndarray, power: np.ndarray, allowed_hz: float,
           floor_start_hz: float, averaged: int = 1,
           floor_override: Optional[float] = None) -> Dict[str, float]:
    """The numbers one spectrum yields: power below and above the rate, the
    white floor read from the top of the band (or handed in, when a better
    estimate exists), and the above-rate excess over what the floor
    explains - as a ratio, and per octave above the rate so the report can
    say WHERE the excess sits. The per-octave excess is the plain sum over
    the floor, never a clipped one: clipping each bin at zero credits every
    octave with 0.37 of a floor per bin and the widest octave wins by bin
    count alone."""
    above = frequency > allowed_hz
    below = (frequency <= allowed_hz) & (frequency > 0)
    total = float(power[1:].sum())
    floor_band = frequency >= floor_start_hz
    if floor_override is not None and np.isfinite(floor_override):
        floor = float(floor_override)
    else:
        floor = (float(np.median(power[floor_band]))
                 / _gamma_median_over_mean(averaged)
                 if floor_band.sum() >= 8 else float("nan"))
    power_above = float(power[above].sum())
    expected = floor * float(above.sum())
    octaves = []
    low = float(allowed_hz)
    nyquist = float(frequency[-1])
    while low < nyquist:
        high = min(2.0 * low, nyquist)
        band = (frequency > low) & (frequency <= high)
        bins = int(band.sum())
        octaves.append({"from_hz": low, "to_hz": high, "bins": bins,
                        "excess": float(power[band].sum() - floor * bins),
                        "ratio": float(power[band].sum()
                                       / max(floor * bins, 1e-300))})
        low = high
    return {
        "total": total,
        "power_below": float(power[below].sum()),
        "power_above": power_above,
        "fraction_above": power_above / max(total, 1e-300),
        "floor_per_bin": floor,
        "expected_above_if_white": expected,
        "excess_ratio": power_above / max(expected, 1e-300),
        "excess_power": max(power_above - expected, 0.0),
        "excess_by_octave": octaves,
        "bins_above": int(above.sum()),
        "bins_below": int(below.sum()),
    }


def _null_summary(fractions: np.ndarray, ratios: np.ndarray) -> Dict[str, float]:
    return {
        "fraction_above_mean": float(fractions.mean()),
        "fraction_above_low": float(np.percentile(fractions,
                                                  100.0 - NULL_PERCENTILE)),
        "excess_ratio_mean": float(ratios.mean()),
        "excess_ratio_std": float(ratios.std()),
        "excess_ratio_bound": float(np.percentile(ratios, NULL_PERCENTILE)),
        "draws": int(len(ratios)),
    }


def _verdict(numbers: Dict[str, object], null: Dict[str, float]
             ) -> Dict[str, object]:
    too_fast = bool(numbers["excess_ratio"] > null["excess_ratio_bound"])
    slow = bool(numbers["fraction_above"] < null["fraction_above_low"])
    share = float(numbers["excess_power"] / max(numbers["total"], 1e-300))
    out = dict(numbers)
    out.update({
        "floats_too_fast": too_fast,
        "slow_component_present": slow,
        "excess_share_of_variance": share,
        "excess_rms": float(np.sqrt(max(numbers["excess_power"], 0.0))),
        "sigma": float((numbers["excess_ratio"] - null["excess_ratio_mean"])
                       / max(null["excess_ratio_std"], 1e-12)),
    })
    if too_fast:
        octaves = [o for o in numbers["excess_by_octave"] if o["bins"] > 0]
        carried = max(octaves, key=lambda o: o["ratio"]) if octaves else None
        out["excess_lives_hz"] = ((carried["from_hz"], carried["to_hz"])
                                  if carried else None)
    return out


# --------------------------------------------------------------------------
# the slow end: mechanism lines and the halves of the record
# --------------------------------------------------------------------------

def mechanism_lines(values, times_s, rates: Dict[str, float]
                    ) -> List[Dict[str, object]]:
    """The predicted mechanism rates, each fitted to a series on its TRUE
    time axis (gaps and alternating heads included) as one cosine-sine pair
    plus a constant, and reported against the amplitude a white series of
    the same scatter would give by chance (`sigma = amplitude / SE`, with
    SE = residual rms times root(2/N)). The prediction comes first and the
    fit is told where to look - `transport_model.find_lines`' philosophy on
    an irregular grid, where a periodogram does not apply. The error
    charges the residual's serial correlation through
    `head_model.effective_sample_size`, so a drifting series does not make
    every slow line look significant. A rate is reported unresolvable when
    the record holds less than one of its cycles or when the sampling
    cannot reach it."""
    values = np.asarray(values, dtype=np.float64)
    times = np.asarray(times_s, dtype=np.float64)
    good = np.isfinite(values) & np.isfinite(times)
    values, times = values[good], times[good]
    out: List[Dict[str, object]] = []
    if len(values) < 8:
        return out
    duration = float(times.max() - times.min())
    spacing = float(np.median(np.diff(np.sort(times))))
    for name, rate in rates.items():
        rate = float(rate)
        entry = {"name": name, "rate_hz": rate}
        if not (0 < rate) or duration <= 0 or rate * duration < 1.0 \
                or rate > 0.5 / max(spacing, 1e-12):
            entry.update({"resolvable": False,
                          "why_not": ("slower than one cycle in the record"
                                      if rate * duration < 1.0 else
                                      "above half the sampling rate")})
            out.append(entry)
            continue
        phase = 2.0 * np.pi * rate * times
        design = np.column_stack([np.ones_like(times), np.cos(phase),
                                  np.sin(phase)])
        coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
        residual = values - design @ coefficients
        amplitude = float(np.hypot(coefficients[1], coefficients[2]))
        # the error charges the residual's serial correlation: a slowly
        # drifting series has far fewer independent samples than entries,
        # and a low-order drift projects onto any slow line
        effective = max(head_model.effective_sample_size(residual), 2.0)
        error = float(np.std(residual) * np.sqrt(2.0 / effective))
        entry.update({
            "resolvable": True,
            "amplitude": amplitude,
            "amplitude_se": error,
            "sigma": amplitude / max(error, 1e-300),
            "phase_rad": float(np.arctan2(-coefficients[2], coefficients[1])),
            "cycles_in_record": rate * duration,
        })
        out.append(entry)
    return sorted(out, key=lambda e: -e.get("sigma", -1.0))


def half_contrast(values, times_s, rates: Optional[Dict[str, float]] = None
                  ) -> Dict[str, object]:
    """THE TESSERACT'S "HALF" AXIS, AND THE RATE IT LIVES AT.

    The first-half against second-half difference of a series, with its
    error from the series' own scatter and effective sample size
    (`head_model.effective_sample_size`, which charges serial correlation),
    and then the statement the rate test owes it: a contrast between the
    halves of a record of length T is the projection onto a step at T/2,
    which over the record is one period of a square wave - so the term
    lives at the record's own fundamental 1/T (81 per cent of its variance)
    and its odd harmonics, and NOTHING faster. A mechanism at a rate f
    contributes to it only through the residue of f over 1/T; the drum, the
    head switch and the servos, at tens of hertz, complete many cycles in
    each half and average to nothing. When `rates` are given the ones that
    can live there - at or below the fundamental, the reel band and slower -
    are named.
    """
    values = np.asarray(values, dtype=np.float64)
    times = np.asarray(times_s, dtype=np.float64)
    good = np.isfinite(values) & np.isfinite(times)
    values, times = values[good], times[good]
    if len(values) < 8:
        return {"usable": False, "why": "too few samples for a contrast"}
    middle = 0.5 * (float(times.min()) + float(times.max()))
    first, second = values[times < middle], values[times >= middle]
    if len(first) < 2 or len(second) < 2:
        return {"usable": False, "why": "one half is empty"}
    difference = float(second.mean() - first.mean())
    error = float(np.sqrt(
        np.var(first, ddof=1) / max(head_model.effective_sample_size(first), 1.0)
        + np.var(second, ddof=1) / max(head_model.effective_sample_size(second),
                                       1.0)))
    duration = float(times.max() - times.min())
    fundamental = 1.0 / max(duration, 1e-12)
    out: Dict[str, object] = {
        "usable": True,
        "difference": difference,
        "se": error,
        "sigma": difference / max(error, 1e-300),
        "identified": bool(abs(difference) > 3.0 * error),
        "record_s": duration,
        "lives_at_hz": fundamental,
        "share_at_fundamental": float(STEP_FUNDAMENTAL_SHARE),
        "first_mean": float(first.mean()), "second_mean": float(second.mean()),
        "samples": (int(len(first)), int(len(second))),
        "why": ("a half-contrast is a step at the middle of the record: one "
                "period of a square wave over the record, so it lives at "
                "1/T and its odd harmonics and averages out every mechanism "
                "that completes cycles within a half"),
    }
    if rates:
        out["mechanisms_that_can_live_there"] = sorted(
            name for name, rate in rates.items() if 0 < float(rate) <= fundamental)
        out["mechanisms_averaged_out"] = sorted(
            name for name, rate in rates.items() if float(rate) > 2.0 * fundamental)
    return out


# --------------------------------------------------------------------------
# the rate test
# --------------------------------------------------------------------------

def _rate_test_contiguous(values: np.ndarray, sample_rate_hz: float,
                          allowed_hz: float, field_rate_hz: Optional[float],
                          segments: Optional[Sequence[Tuple[int, int]]],
                          draws: int, seed: int, gap_share: float
                          ) -> Dict[str, object]:
    n = len(values)
    nyquist = float(sample_rate_hz) / 2.0
    # the trend is slower than anything, so it belongs below the rate;
    # removing it keeps its leakage out of the band above
    index = np.arange(n, dtype=np.float64)
    slope, intercept = np.polyfit(index, values, 1)
    detrended = values - (slope * index + intercept)
    trend_variance = float(np.var(slope * index))
    floor_start = max((1.0 - FLOOR_BAND_SHARE) * nyquist,
                      0.5 * (allowed_hz + nyquist))

    def evaluate(x: np.ndarray) -> Dict[str, object]:
        frequency, power = _periodogram(x, sample_rate_hz)
        numbers = _split(frequency, power, allowed_hz, floor_start)
        if field_rate_hz:
            resolution = float(sample_rate_hz) / n
            picket = np.zeros(len(frequency), dtype=bool)
            for h in range(1, int(nyquist / field_rate_hz) + 1):
                centre = h * float(field_rate_hz)
                if centre <= allowed_hz:
                    continue
                picket |= np.abs(frequency - centre) <= 1.5 * resolution
            above = frequency > allowed_hz
            on = float(np.maximum(power[picket & above]
                                  - numbers["floor_per_bin"], 0.0).sum())
            off = float(np.maximum(power[~picket & above]
                                   - numbers["floor_per_bin"], 0.0).sum())
            numbers["excess_on_field_harmonics"] = on / max(on + off, 1e-300)
        return numbers

    raw = evaluate(detrended)
    variance = float(np.var(detrended)) + trend_variance
    rng = np.random.default_rng(seed)
    null_fraction = np.empty(int(draws))
    null_ratio = np.empty(int(draws))
    for k in range(int(draws)):
        white = evaluate(rng.standard_normal(n))
        null_fraction[k] = white["fraction_above"]
        null_ratio[k] = white["excess_ratio"]
    null = _null_summary(null_fraction, null_ratio)

    def signature(verdict: Dict[str, object]) -> str:
        if verdict["floats_too_fast"] and "excess_on_field_harmonics" in verdict:
            comb = float(verdict["excess_on_field_harmonics"])
            return ("field-periodic: the excess sits on harmonics of the "
                    "field rate - a picture or a per-field shape repeating "
                    "every field" if comb > 0.5 else
                    "broadband: the excess is off the field harmonics - a "
                    "random walk, coloured noise, or events")
        if verdict["floats_too_fast"]:
            return "structured above the rate"
        return ("white above the rate: only the estimator's noise is faster "
                "than the mechanism")

    result: Dict[str, object] = {
        "usable": True,
        "form": "contiguous",
        "samples": int(n),
        "sample_rate_hz": float(sample_rate_hz),
        "allowed_hz": float(allowed_hz),
        "resolution_hz": float(sample_rate_hz) / n,
        "floor_band_from_hz": float(floor_start),
        "gap_share": gap_share,
        "variance": variance,
        "trend_share": trend_variance / max(variance, 1e-300),
        "null": null,
        "raw": _verdict(raw, null),
    }
    result["raw"]["signature"] = signature(result["raw"])
    if segments:
        remainder, fitted = _remove_sweep_shapes(detrended, segments)
        result["sweep_share"] = float(np.var(fitted) / max(variance, 1e-300))
        result["remainder"] = _verdict(evaluate(remainder), null)
        result["remainder"]["signature"] = signature(result["remainder"])
        final = result["remainder"]
    else:
        final = result["raw"]
    result["floats_too_fast"] = final["floats_too_fast"]
    result["slow_component_present"] = final["slow_component_present"]
    result["signature"] = final["signature"]
    result["excess_lives_hz"] = final.get("excess_lives_hz")
    result["why"] = (
        "the fraction above the rate says how noisy the estimator is; the "
        "verdict rests on whether the power above the rate exceeds the white "
        "floor read from the top of the band, which no mechanism reaches")
    return result


def structure(values, field_index, line_index) -> Dict[str, object]:
    """Lay a per-line series out as a table of fields by line positions,
    NaN where a line is missing. Line positions present in fewer than half
    the fields and fields holding fewer than half the positions are dropped
    rather than interpolated across; what remains has its short gaps
    bridged within each field and the bridged share is reported."""
    values = np.asarray(values, dtype=np.float64).ravel()
    fields = np.asarray(field_index).ravel()
    lines = np.asarray(line_index).ravel()
    if not (len(values) == len(fields) == len(lines)):
        raise ValueError("values, field_index and line_index must align")
    field_ids = np.unique(fields)
    line_ids = np.unique(lines)
    table = np.full((len(field_ids), len(line_ids)), np.nan)
    row = np.searchsorted(field_ids, fields)
    column = np.searchsorted(line_ids, lines)
    table[row, column] = values
    present = np.isfinite(table)
    keep_lines = present.mean(axis=0) >= 0.5
    table, line_ids = table[:, keep_lines], line_ids[keep_lines]
    present = np.isfinite(table)
    keep_fields = present.mean(axis=1) >= 0.5
    table, field_ids = table[keep_fields], field_ids[keep_fields]
    gap_share = float(np.mean(~np.isfinite(table))) if table.size else 1.0
    filled = np.array([_fill_gaps(row) for row in table]) if table.size else table
    return {"table": filled, "field_ids": field_ids, "line_ids": line_ids,
            "gap_share": gap_share,
            "dropped_lines": int((~keep_lines).sum()),
            "dropped_fields": int((~keep_fields).sum())}


def decompose(table: np.ndarray) -> Dict[str, np.ndarray]:
    """The two-way decomposition of a fields-by-lines table: the per-field
    sweep shapes (`transport_model.sweep_shapes`: constant, tilt, bow, fitted
    to each field), the per-line-position term (the mean over fields of what
    is left), and the field-varying remainder. The per-field constant is the
    per-field term the slow tests read; the tilt and bow are the allowed
    tape-path shapes and are reported by their share."""
    fields, lines = table.shape
    shapes = transport_model.sweep_shapes(lines)
    design = np.column_stack([shapes[name] for name in shapes])
    coefficients, *_ = np.linalg.lstsq(design, table.T, rcond=None)
    sweep = (design @ coefficients).T
    field_term = table.mean(axis=1)
    remainder = table - sweep
    line_term = remainder.mean(axis=0)
    residual = remainder - line_term[None, :]
    names = list(shapes)
    return {"field_term": field_term, "line_term": line_term,
            "residual": residual, "sweep": sweep,
            "sweep_coefficients": {name: coefficients[k]
                                   for k, name in enumerate(names)}}


def _welch_rows(rows: np.ndarray, sample_rate_hz: float
                ) -> Tuple[np.ndarray, np.ndarray]:
    spectra = []
    for row in rows:
        frequency, power = _periodogram(row, sample_rate_hz)
        spectra.append(power)
    return frequency, np.mean(spectra, axis=0)


def _rate_test_structured(values, field_index, line_index,
                          sample_rate_hz: float, allowed_hz: float,
                          field_rate_hz: float, mechanism_hz: Optional[Dict[str, float]],
                          draws: int, seed: int) -> Dict[str, object]:
    laid = structure(values, field_index, line_index)
    table = laid["table"]
    fields, lines = table.shape
    if fields < 3 or lines < 16:
        return {"usable": False, "form": "structured",
                "why": f"too little structure: {fields} fields x {lines} lines"}
    nyquist = float(sample_rate_hz) / 2.0
    floor_start = max((1.0 - FLOOR_BAND_SHARE) * nyquist,
                      0.5 * (allowed_hz + nyquist))

    def evaluate(x: np.ndarray) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, np.ndarray]]:
        parts = decompose(x)
        frequency, residual_power = _welch_rows(parts["residual"], sample_rate_hz)
        residual = _split(frequency, residual_power, allowed_hz, floor_start,
                          averaged=x.shape[0])
        # the line-locked term is a mean over the fields of what the
        # residual holds: under a white null its variance is the residual's
        # over (fields - 1), so its floor is read from the residual's far
        # better estimate rather than from its own few top bins
        frequency, locked_power = _periodogram(parts["line_term"], sample_rate_hz)
        locked = _split(frequency, locked_power, allowed_hz, floor_start,
                        floor_override=residual["floor_per_bin"] / (x.shape[0] - 1))
        return residual, locked, parts

    residual, locked, parts = evaluate(table)
    rng = np.random.default_rng(seed)
    null_residual = np.empty((int(draws), 2))
    null_locked = np.empty((int(draws), 2))
    for k in range(int(draws)):
        white_residual, white_locked, _ = evaluate(rng.standard_normal(table.shape))
        null_residual[k] = (white_residual["fraction_above"],
                            white_residual["excess_ratio"])
        null_locked[k] = (white_locked["fraction_above"],
                          white_locked["excess_ratio"])
    residual_null = _null_summary(null_residual[:, 0], null_residual[:, 1])
    locked_null = _null_summary(null_locked[:, 0], null_locked[:, 1])
    residual_verdict = _verdict(residual, residual_null)
    locked_verdict = _verdict(locked, locked_null)

    # where along the field the line-locked structure above the rate sits:
    # high-pass the line term at the rate (zero phase) and find the span of
    # line positions holding the central 80 per cent of that energy
    term = parts["line_term"] - parts["line_term"].mean()
    spectrum = np.fft.rfft(term)
    frequency = np.fft.rfftfreq(len(term), 1.0 / float(sample_rate_hz))
    spectrum[frequency <= allowed_hz] = 0.0
    fast = np.fft.irfft(spectrum, len(term))
    energy = np.cumsum(fast ** 2)
    if energy[-1] > 0:
        energy /= energy[-1]
        span = (int(laid["line_ids"][int(np.searchsorted(energy, 0.10))]),
                int(laid["line_ids"][min(int(np.searchsorted(energy, 0.90)),
                                         len(term) - 1)]))
        first_quarter = float(energy[len(term) // 4])
    else:
        span, first_quarter = (None, None), 0.0
    locked_verdict.update({
        "lives_in_lines": span,
        "share_in_first_quarter_of_field": first_quarter,
        "share_of_line_term_variance_above_rate": float(
            np.var(fast) / max(np.var(term), 1e-300)),
    })

    # the slow end: the per-field sweep coefficients on their true time
    # axis. All three are read, because where a mechanism lands depends on
    # its phase against the field: a drum wobble whose zero crossing sits
    # mid-field is an alternating TILT with no per-field level at all
    field_times = np.asarray(laid["field_ids"], dtype=np.float64) / float(field_rate_hz)
    slow: Dict[str, object] = {
        "fields": int(fields),
        "field_times_s": field_times,
        "field_term": parts["field_term"],
        "field_term_rms": float(np.std(parts["field_term"])),
        "half_contrast": half_contrast(parts["field_term"], field_times, mechanism_hz),
    }
    if mechanism_hz:
        slow["mechanism_lines"] = {
            name: mechanism_lines(coefficient, field_times, mechanism_hz)
            for name, coefficient in parts["sweep_coefficients"].items()}
        strongest = None
        for name, found in slow["mechanism_lines"].items():
            for line in found:
                if line.get("resolvable") and (strongest is None
                                               or line["sigma"] > strongest["sigma"]):
                    strongest = dict(line, coefficient=name)
        slow["strongest_line"] = strongest
    sweep_variance = {name: float(np.var(coefficient))
                      for name, coefficient in parts["sweep_coefficients"].items()}

    too_fast = bool(residual_verdict["floats_too_fast"])
    locked_fast = bool(locked_verdict["floats_too_fast"])
    if too_fast and locked_fast:
        signature = ("structured above the rate both field to field and "
                     "line-locked: the parameter follows something no "
                     "mechanism owns, part of it fixed to the line position")
    elif too_fast:
        signature = ("field-varying structure above the rate: a walk, "
                     "coloured noise or events the parameter is following")
    elif locked_fast:
        signature = ("line-locked structure above the rate only: fixed to "
                     "the line position - a static picture, the burst, or "
                     "the electronics' recovery after the vertical interval "
                     "- while the field-to-field part is white")
    else:
        signature = ("white above the rate in both parts: only the "
                     "estimator's noise is faster than the mechanism")
    return {
        "usable": True,
        "form": "structured",
        "fields": int(fields), "lines": int(lines),
        "line_ids": (int(laid["line_ids"][0]), int(laid["line_ids"][-1])),
        "gap_share": laid["gap_share"],
        "dropped": {"fields": laid["dropped_fields"], "lines": laid["dropped_lines"]},
        "sample_rate_hz": float(sample_rate_hz),
        "allowed_hz": float(allowed_hz),
        "resolution_hz": float(sample_rate_hz) / lines,
        "floor_band_from_hz": float(floor_start),
        "variance": float(np.var(table)),
        "sweep_share": float(np.var(parts["sweep"] - parts["field_term"][:, None])
                             / max(np.var(table), 1e-300)),
        "sweep_coefficient_variance": sweep_variance,
        "field_term_share": float(np.var(parts["field_term"]) / max(np.var(table), 1e-300)),
        "line_term_share": float(np.var(parts["line_term"]) / max(np.var(table), 1e-300)),
        "residual_share": float(np.var(parts["residual"]) / max(np.var(table), 1e-300)),
        "residual": residual_verdict,
        "line_locked": locked_verdict,
        "slow": slow,
        "null": {"residual": residual_null, "line_locked": locked_null},
        "floats_too_fast": too_fast,
        "line_locked_above_rate": locked_fast,
        "slow_component_present": bool(residual_verdict["slow_component_present"]),
        "signature": signature,
        "excess_lives_hz": residual_verdict.get("excess_lives_hz"),
        "why": ("the field-varying remainder is judged above the rate against "
                "a white null of the same decomposition; the line-locked term "
                "separately, because a static picture and the vertical "
                "interval's recovery live there; the per-field term carries "
                "the slow end, read on its true time axis"),
    }


def rate_test(series, sample_rate_hz: float, allowed_hz: float,
              field_rate_hz: Optional[float] = None,
              segments: Optional[Sequence[Tuple[int, int]]] = None,
              field_index=None, line_index=None,
              mechanism_hz: Optional[Dict[str, float]] = None,
              draws: int = NULL_DRAWS, seed: int = 0) -> Dict[str, object]:
    """IS THIS PARAMETER FLOATING FASTER THAN ITS MECHANISM ALLOWS?

    Two forms. CONTIGUOUS - `series` is a uniformly sampled parameter (NaN
    where a sample is missing; gaps are bridged and their share reported),
    `sample_rate_hz` its rate, `allowed_hz` the mechanism's rate from
    `allowed_rate`. It reports the power below and above the rate, the white
    null (what a white series of the same length puts above the rate by
    chance - Monte Carlo, same estimator - as its mean and `NULL_PERCENTILE`
    bound), the white floor read from the top of the band and the EXCESS of
    the above-rate power over it (a ratio, a share of the variance, an rms
    in the series' units, and the octave it lives in); where the excess sits
    against the field harmonics when `field_rate_hz` is given; and, when
    `segments` mark the fields, the per-field sweep shapes are removed first
    and both the raw and remainder verdicts are returned.

    STRUCTURED - `field_index` and `line_index` are given alongside the
    series, one entry per measured line, and `field_rate_hz` sets the
    fields' time axis. The series is laid out as fields by line positions
    (`structure`), decomposed (`decompose`) into the per-field sweep shapes,
    the line-locked term and the field-varying remainder, and each part is
    judged: the remainder above the rate, field by field (a Welch average
    over fields) against a white null of the same decomposition; the
    line-locked term above the rate along the line axis against its own
    null, with a locator for WHERE in the field it sits; and the per-field
    term at the slow end - the mechanism rates in `mechanism_hz` it can
    resolve (`mechanism_lines`) and the first-half against second-half
    contrast (`half_contrast`) with the rate that contrast lives at.

    THE VERDICT is `floats_too_fast`: the field-varying part above the
    mechanism's rate is not the estimator's noise but something the
    parameter has absorbed. `line_locked_above_rate` says the same of the
    part fixed to the line position. `slow_component_present` says
    separately whether there is a real component BELOW the rate beyond what
    noise puts there - the thing a constrained fit would keep.
    """
    if field_index is not None or line_index is not None:
        if field_index is None or line_index is None or not field_rate_hz:
            raise ValueError("the structured form needs field_index, "
                             "line_index and field_rate_hz together")
        nyquist = float(sample_rate_hz) / 2.0
        if allowed_hz >= nyquist:
            return {"usable": False, "form": "structured",
                    "allowed_hz": float(allowed_hz), "nyquist_hz": nyquist,
                    "why": "the series is sampled too slowly to hold anything "
                           "above the allowed rate"}
        return _rate_test_structured(series, field_index, line_index,
                                     sample_rate_hz, allowed_hz, field_rate_hz,
                                     mechanism_hz, draws, seed)
    raw = np.asarray(series, dtype=np.float64).ravel()
    if len(raw) < 16:
        return {"usable": False, "why": "too few samples for a spectrum"}
    nyquist = float(sample_rate_hz) / 2.0
    if allowed_hz >= nyquist:
        return {"usable": False, "allowed_hz": float(allowed_hz),
                "nyquist_hz": nyquist,
                "why": ("the series is sampled too slowly to hold anything "
                        "above the allowed rate: nothing faster than half "
                        "the sampling rate exists in it")}
    values = _fill_gaps(raw)
    gap_share = float(np.mean(~np.isfinite(raw)))
    return _rate_test_contiguous(values, sample_rate_hz, allowed_hz,
                                 field_rate_hz, segments, draws, seed, gap_share)


# --------------------------------------------------------------------------
# the constraint
# --------------------------------------------------------------------------

def constrain(series, sample_rate_hz: float, allowed_hz: float,
              method: str = "lowpass", fundamental_hz: Optional[float] = None,
              segments: Optional[Sequence[Tuple[int, int]]] = None,
              rates: Optional[Dict[str, Dict[str, object]]] = None,
              field_rate_hz: Optional[float] = None,
              times_s=None) -> Dict[str, object]:
    """PROJECT A PARAMETER ONTO THE BAND ITS MECHANISM ALLOWS, and keep what
    was removed as evidence.

    `method="lowpass"` is the orthogonal projection onto frequencies at or
    below `allowed_hz` on the series' own uniform grid: mean and linear
    trend kept, the remainder reflected at both ends to keep the
    projection's edges honest, a raised-cosine transition one tenth of the
    allowed rate wide (never narrower than two resolution bins). It is
    applied in the frequency domain and is therefore ZERO PHASE: the group
    delay is zero at every frequency, stated in the result, and nothing is
    shifted in time. A low-pass keeps EVERYTHING slow, a random walk's slow
    part included; it is the form for a series known to carry a mechanism
    and noise, and the harmonic form is the one for a series suspected of
    carrying a walk.

    `method="harmonic"` fits the mechanism's rate `fundamental_hz` and its
    harmonics up to `allowed_hz` (cosine and sine pairs, least squares) and,
    when `segments` give the fields, the per-field sweep shapes of
    `transport_model.sweep_shapes` - constant, linear tilt and parabolic
    bow - exactly as `transport_model.fit_sweep` does. That is the form to
    use across a head switch: the switch is a step between fields, which the
    per-field constant absorbs, where a low-pass would smear it. `times_s`
    puts the fit on the true time axis when the samples are not uniform
    (alternating heads, missing fields); without it the sample index over
    `sample_rate_hz` is the time.

    Either way the result carries `constrained`, `removed`, the shares of
    variance kept and removed, the rms of what was removed in the series'
    units, and the EVIDENCE the removed part holds: its lag-one
    autocorrelation (white noise sits near zero), the mechanism rates that
    `transport_model.find_lines` still finds in it (none should survive a
    correct cut), and - when the field rate is given - the share of its
    power on field harmonics, which is where absorbed picture content lives.
    """
    values = _fill_gaps(np.asarray(series, dtype=np.float64).ravel())
    n = len(values)
    if n < 8:
        raise ValueError("too few samples to constrain")
    index = np.arange(n, dtype=np.float64)
    if method == "lowpass":
        slope, intercept = np.polyfit(index, values, 1)
        trend = slope * index + intercept
        detrended = values - trend
        # reflect at both ends so the projection sees no step at the edges
        padded = np.concatenate([detrended[::-1], detrended, detrended[::-1]])
        spectrum = np.fft.rfft(padded)
        frequency = np.fft.rfftfreq(len(padded), 1.0 / float(sample_rate_hz))
        resolution = float(sample_rate_hz) / len(padded)
        transition = max(0.1 * float(allowed_hz), 2.0 * resolution)
        gain = np.ones(len(frequency))
        ramp = (frequency > allowed_hz) & (frequency <= allowed_hz + transition)
        gain[ramp] = 0.5 * (1.0 + np.cos(np.pi * (frequency[ramp] - allowed_hz)
                                         / transition))
        gain[frequency > allowed_hz + transition] = 0.0
        smooth = np.fft.irfft(spectrum * gain, len(padded))[n: 2 * n]
        constrained = smooth + trend
        detail = {
            "method": "lowpass",
            "group_delay_s": 0.0,
            "phase": "zero (frequency-domain projection, applied without delay)",
            "transition_hz": float(transition),
            "cut_hz": float(allowed_hz),
        }
    elif method == "harmonic":
        if not fundamental_hz:
            raise ValueError("the harmonic method needs the mechanism's "
                             "fundamental rate")
        count = int(np.floor(float(allowed_hz) / float(fundamental_hz)))
        if count < 1:
            raise ValueError("the allowed rate lies below the fundamental; "
                             "nothing to fit")
        t = (np.asarray(times_s, dtype=np.float64) if times_s is not None
             else index / float(sample_rate_hz))
        if len(t) != n:
            raise ValueError("times_s must align with the series")
        columns: List[np.ndarray] = []
        names: List[str] = []
        if segments:
            for start, stop in segments:
                shapes = transport_model.sweep_shapes(stop - start)
                for name, shape in shapes.items():
                    column = np.zeros(n)
                    column[start:stop] = shape
                    columns.append(column)
                    names.append(f"{name}[{start}:{stop}]")
        else:
            columns.append(np.ones(n))
            names.append("constant")
        for h in range(1, count + 1):
            phase = 2.0 * np.pi * h * float(fundamental_hz) * t
            columns.append(np.cos(phase))
            columns.append(np.sin(phase))
            names.append(f"cos {h}x")
            names.append(f"sin {h}x")
        design = np.column_stack(columns)
        coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
        constrained = design @ coefficients
        amplitudes = {}
        phases = {}
        for h in range(1, count + 1):
            cosine = coefficients[names.index(f"cos {h}x")]
            sine = coefficients[names.index(f"sin {h}x")]
            amplitudes[h] = float(np.hypot(cosine, sine))
            phases[h] = float(np.arctan2(-sine, cosine))
        detail = {
            "method": "harmonic",
            "group_delay_s": 0.0,
            "phase": "none: a fitted model, not a filter",
            "fundamental_hz": float(fundamental_hz),
            "harmonics": count,
            "harmonic_amplitudes": amplitudes,
            "harmonic_phases_rad": phases,
            "sweep_shapes_per_segment": bool(segments),
        }
    else:
        raise ValueError(f"unknown method {method!r}: 'lowpass' or 'harmonic'")

    removed = values - constrained
    total = float(np.var(values))
    evidence: Dict[str, object] = {
        "removed_rms": float(np.std(removed)),
        "removed_share": float(np.var(removed) / max(total, 1e-300)),
        "kept_share": float(np.var(constrained) / max(total, 1e-300)),
        "removed_lag_one": head_model.lag_one(removed),
    }
    if rates:
        table = [{"name": name, "rate_hz": entry["rate_hz"]}
                 for name, entry in rates.items()
                 if entry.get("kind") == "mechanical"
                 and 0 < float(entry["rate_hz"]) < float(sample_rate_hz) / 2]
        evidence["mechanism_lines_in_removed"] = [
            line for line in transport_model.find_lines(
                removed, sample_rate_hz, table)
            if line["height_over_background"] > 10.0]
    if field_rate_hz:
        frequency, power = _periodogram(removed, sample_rate_hz)
        resolution = float(sample_rate_hz) / n
        picket = np.zeros(len(frequency), dtype=bool)
        for h in range(1, int(0.5 * sample_rate_hz / field_rate_hz) + 1):
            picket |= np.abs(frequency - h * field_rate_hz) <= 1.5 * resolution
        evidence["removed_share_on_field_harmonics"] = float(
            power[picket].sum() / max(power[1:].sum(), 1e-300))
        evidence["field_harmonic_bins_share"] = float(picket.mean())
    return {
        "constrained": constrained,
        "removed": removed,
        **detail,
        "evidence": evidence,
        "why": ("the removed part is not discarded: it is what the parameter "
                "was absorbing above its mechanism's rate, and its structure "
                "names what is still unmodelled"),
    }


# --------------------------------------------------------------------------
# the measured verdicts, reproduced from the instrument's own output
# --------------------------------------------------------------------------

# Filled by the run of tools/ringing_measure/rate_constraints_measure.py
# recorded in the module docstring; kept as data so a test can check the
# module still says what was measured.
MEASURED_VERDICTS: Dict[str, object] = {}
