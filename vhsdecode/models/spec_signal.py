"""THE EXPECTED SIGNAL: the whole reserved interval as the specification
writes it, against which the whole measured one is differentiated.

Ethan, 2026-09-07:

    "I think it's just 3d analytical component of the entire RF with the
    expected 3d anaytical spec signal for each stage. Luma, color up-het,
    will be doing all the chroma logic, and feeing back into the luma, so
    those are tied together, but this is because of the dimensionality of
    the matrix. Sync and eq pulses are one model, color (with all it's
    mappings) is a model that is connected to the time part of the sync
    model. Both are then run to generate the y and the c files."

    "What we talked about earlier except very simple and dimensional"

    "Excellent, the picture stage is what I am describing again, keep it
    there"

    "Is is the concentric rings of dimensions, like a multi fold sphere
    that has a causality dimension fixed, since we process video RF data"

SYNC AND EQUALISING PULSES ARE ONE MODEL, which is the whole of this
module. Until now the picture stage read the horizontal sync edge alone,
pooled over a field's lines. That edge is 4.7 microseconds long and by the
1/T law resolves nothing finer than 213 kHz; the vertical interval, 572
microseconds of equalising and broad pulses at half-line spacing, resolves
1.75 kHz - a hundred and twenty times further down. They are not two
probes to choose between: they are one specified waveform, and the
departure of the measured field from it is one measurement carrying every
frequency both reach.

WHAT IS COMPARED, AND WHERE. The measured field is a decoded field, every
line beginning at its own sync datum, so the specification for it is built
on the same grid: each picture row carries the specified line-sync pulse,
and the vertical-sync rows carry the specified equalising and broad pulse
train, laid down from the format's timing table by `sync_geometry` and
never from a waveform. The comparison is taken only where the mask admits
it - the vertical sync whole, each picture row's sync pulse and the tail of
its front porch, the vertical interval's test lines dropped entirely and
all active picture with them - which is the standing law that a correction
derives from the reserved intervals and never from content.

THE THREE DIMENSIONS ARE ONE READING. The same masked comparison gives all
three at once, which is what makes the sync pulse "three dimensional" in
his words rather than three separate instruments:

    AMPLITUDE   the measured blanking-to-tip spacing against the specified
                forty IRE, which is the luma's only absolute scale
    FREQUENCY   the complex log response of the masked measured signal
                against the masked expected one, a difference and never a
                ratio, bounded below by the mask's own resolution
    TIME        the delay, taken as the linear phase through zero frequency
                with a free phase reference beside it, because a ramp
                fitted about a band's centre leaves a constant behind that
                no model can express

The causal axis is the frequency one and it is not folded: the delay, the
minimum phase its own magnitude implies, and the all-pass excess are the
three parts a contrast on it splits into, and each is inverted in its own
form. The binary axes - colour, head, polarity - are the fold's, and this
module supplies what lives at each of their vertices.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from vhsdecode.models import sync_geometry

# The specified levels, from the same table every other number here comes
# from: blanking is the zero of the luma scale and the sync tip sits forty
# IRE below it (SMPTE 170M; `sync_depth` carries the same pair and the
# reason it is the only absolute scale the luma has).
BLANKING_IRE = 0.0
SYNC_TIP_IRE = -40.0


def expected_field(lines: int, width: int, sample_rate_hz: float,
                   system: str = "NTSC", first_field: bool = True,
                   vsync_lines: int = sync_geometry.VSYNC_LINES,
                   vbi_last_line: int = sync_geometry.VBI_LAST_LINE,
                   active: Optional[Sequence[int]] = None,
                   front_porch_keep_us: float = 0.2) -> Dict[str, object]:
    """THE SPECIFIED FIELD: what the format says was sent, on the decode's
    own grid, with the mask that says where it may be compared.

    Every picture row carries the line-sync pulse at its specified width and
    edge; the vertical-sync rows carry the equalising, field-sync and
    equalising sequences at half-line spacing, taken from
    `sync_geometry.pulse_train` so that the timing table is the only source.
    The two fields of a frame differ by a half line, and the second field's
    train is displaced by exactly that, which is why `first_field` is asked
    for rather than assumed.

    Returns the waveform in IRE, the mask, and the geometry each was built
    from. Nothing here reads a recording.
    """
    lines, width = int(lines), int(width)
    if lines < 1 or width < 8:
        raise ValueError("a field needs rows and a line needs samples")
    rate = float(sample_rate_hz)
    per_sample_us = 1e6 / rate
    key = sync_geometry._system(system)
    geometry = sync_geometry.interval_geometry(key)
    edge_us = sync_geometry.PULSE_EDGE_NS[key] / 1000.0
    pulse_us = sync_geometry.LINE_SYNC_US[key]

    waveform = np.full((lines, width), BLANKING_IRE, dtype=np.float64)
    columns_us = np.arange(width, dtype=np.float64) * per_sample_us

    # -- the picture rows: one specified line-sync pulse each ---------------
    line = _raised_pulse(columns_us, 0.0, pulse_us, edge_us)
    picture_row = BLANKING_IRE + (SYNC_TIP_IRE - BLANKING_IRE) * line

    # -- the vertical sync rows: the train, from the timing table -----------
    # The POSITIONS are `pulse_train`'s, which reads them off the timing
    # table alone; the edges are laid down here on the half-amplitude
    # convention above, so the whole field shares one edge treatment.
    train = sync_geometry.pulse_train(key, sample_rate_hz=rate,
                                      depth=SYNC_TIP_IRE, level=BLANKING_IRE)
    interval_rows = int(round(float(geometry["total_lines"])))
    span = np.arange(interval_rows * width, dtype=np.float64) * per_sample_us
    gate = np.zeros(span.size, dtype=np.float64)
    for start, stop, _name in train["marks"]:
        gate = np.maximum(gate, _raised_pulse(span, float(start),
                                              float(stop) - float(start),
                                              edge_us))
    block = (BLANKING_IRE + (SYNC_TIP_IRE - BLANKING_IRE)
             * gate).reshape(interval_rows, width)
    # THE HALF-LINE DIFFERENCE BETWEEN THE FIELDS DOES NOT REACH THE
    # INTERVAL'S SHAPE, and this was measured rather than assumed. Every
    # sequence of the interval puts its pulses at HALF-line spacing, so a
    # row of it is periodic in half a line and rolling the block by that
    # displacement returns the identical block. What distinguishes the two
    # fields is where the interval begins against the line, and the
    # decoder's own time base has already settled that by handing us a
    # field. `first_field` is therefore recorded and does not change the
    # waveform; a format whose interval is not half-line periodic would
    # need it, which is why it stays in the signature.

    rows = np.arange(lines)
    is_vsync = rows < int(vsync_lines)
    is_vbi = (rows >= int(vsync_lines)) & (rows <= int(vbi_last_line))
    is_picture = ~is_vsync & ~is_vbi
    waveform[is_picture, :] = picture_row
    # the VBI rows are blanking by position; they carry test signals by
    # content and are dropped by the mask, so the specification for them is
    # the line's own blanking with its sync pulse and nothing else
    waveform[is_vbi, :] = picture_row
    take = min(int(is_vsync.sum()), block.shape[0])
    if take:
        waveform[:take, :] = block[:take, :]

    mask = _mask(lines, width, per_sample_us, pulse_us, active,
                 front_porch_keep_us, vsync_lines, vbi_last_line)
    return {
        "waveform": waveform,
        "mask": mask["mask"],
        "rows": {"vsync": int(is_vsync.sum()), "vbi": int(is_vbi.sum()),
                 "picture": int(is_picture.sum())},
        "geometry": geometry,
        "sample_rate_hz": rate,
        "first_field": bool(first_field),
        "pulse_us": float(pulse_us),
        "edge_us": float(edge_us),
        "kept_samples": int(mask["mask"].sum()),
        "kept_fraction": float(mask["mask"].mean()),
        "resolution_hz": resolution(mask["mask"], rate),
        "mask_detail": {k: v for k, v in mask.items() if k != "mask"},
        "why": ("the sync and the equalising pulses are one specified "
                "waveform on the decode's own grid; the mask keeps the "
                "vertical sync whole and each picture row's pulse and "
                "front-porch tail, and drops the test lines and the "
                "picture"),
    }


def _raised_pulse(t_us: np.ndarray, start_us: float, width_us: float,
                  edge_us: float) -> np.ndarray:
    """One pulse as a gate with the specified rise time, raised-cosine,
    which is the shape a band-limited system makes of an ideal step.

    THE EDGES ARE CENTRED ON THE SPECIFIED CROSSINGS, because the format
    states every pulse width AT HALF AMPLITUDE. Placing the edges' outer
    feet on those crossings instead - which is what
    `sync_geometry.pulse_train` does - narrows every pulse by exactly one
    edge time: measured on this grid the line-sync pulse then reads 4.540
    microseconds against the specified 4.700, and the equalising pulse
    2.095 against 2.300. The difference is small and it is systematic, and
    it would be charged to the channel by every fit that used it.
    """
    half = 0.5 * max(edge_us, 1e-9)
    rising = np.clip((t_us - (start_us - half)) / max(edge_us, 1e-9), 0.0, 1.0)
    falling = np.clip(((start_us + width_us + half) - t_us)
                      / max(edge_us, 1e-9), 0.0, 1.0)
    return ((0.5 - 0.5 * np.cos(np.pi * rising))
            * (0.5 - 0.5 * np.cos(np.pi * falling)))


def _mask(lines: int, width: int, per_sample_us: float, pulse_us: float,
          active: Optional[Sequence[int]], front_porch_keep_us: float,
          vsync_lines: int, vbi_last_line: int) -> Dict[str, object]:
    """The 2-D field mask, on the rule `sync_geometry.frame_mask` states.

    It is rebuilt here rather than called, because that function takes its
    bounds from a decode's JSON file and this one is handed the geometry the
    decoder already holds; the rule, the reasons and the defaults are its.
    """
    columns = np.arange(width)
    pulse_end = int(round(pulse_us / per_sample_us))
    if active is not None:
        keep_from = int(active[1]) + int(round(front_porch_keep_us
                                               / per_sample_us))
    else:
        keep_from = width          # no active span given: the pulse alone
    line_keep = (columns >= min(keep_from, width)) | (columns < pulse_end)
    rows = np.arange(lines)
    is_vsync = rows < int(vsync_lines)
    is_vbi = (rows >= int(vsync_lines)) & (rows <= int(vbi_last_line))
    mask = np.zeros((lines, width), dtype=bool)
    mask[is_vsync, :] = True
    mask[~is_vsync & ~is_vbi, :] = line_keep
    return {
        "mask": mask,
        "line_keep": line_keep,
        "keep_from_column": int(min(keep_from, width)),
        "sync_pulse_end_column": pulse_end,
        "front_porch_samples_kept": int(max(width - keep_from, 0)),
        "vsync_rows": int(is_vsync.sum()),
        "vbi_rows_dropped": int(is_vbi.sum()),
        "picture_rows": int((~is_vsync & ~is_vbi).sum()),
    }


def resolution(mask, sample_rate_hz: float) -> Dict[str, float]:
    """WHAT THE MASK CAN RESOLVE, from the runs it keeps.

    The 1/T law applies to each continuous run of kept samples: a picture
    row's pulse reaches its own width's reciprocal, and the vertical sync
    block, kept whole across nine rows, reaches the reciprocal of its whole
    duration. The longest run is what sets the floor of the reading's band,
    and it is a hundred-fold below the pulse's own - which is the entire
    reason the interval is in the model.
    """
    flat = np.asarray(mask, dtype=bool).reshape(-1)
    if not flat.any():
        raise ValueError("the mask keeps nothing")
    edges = np.diff(np.concatenate([[0], flat.view(np.int8), [0]]))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    runs = stops - starts
    rate = float(sample_rate_hz)
    longest_us = float(runs.max()) / rate * 1e6
    shortest_us = float(runs.min()) / rate * 1e6
    return {
        "longest_run_us": longest_us,
        "shortest_run_us": shortest_us,
        "runs": int(runs.size),
        "lowest_hz": sync_geometry.resolution_hz(longest_us),
        "highest_run_hz": sync_geometry.resolution_hz(shortest_us),
        "kept_us": float(flat.sum()) / rate * 1e6,
        "pooled_hz": sync_geometry.resolution_hz(
            float(flat.sum()) / rate * 1e6),
    }


def long_runs(mask, sample_rate_hz: float, minimum_us: float = 50.0
              ) -> Dict[str, object]:
    """The mask's contiguous runs long enough to divide spectra on.

    THIS IS THE DEFECT THE MODULE WAS BUILT AROUND, AND IT IS MEASURED. A
    channel acts on the signal and the mask is applied after it, so the
    masked measurement is `mask . (h * x)` while the masked specification
    is `mask . x`: dividing the two returns `h` only where the mask's own
    edges are far apart compared with the channel's memory. Reading the
    whole mask at once - two edges on each of two hundred and forty-one
    picture rows - a planted 300 kHz one-pole came back with 0.835 nepers
    rms of error; read on the vertical interval's single run of 572
    microseconds it came back at 0.0148, fifty-six times better, and a
    contiguous window of that same block gives 0.0148 as well, so it is the
    absence of edges and not the padding that does it.

    So the interval is where the spectra may be divided, and the line sync
    pulses are read by the pooled-edge instrument that already exists and
    is matched to them. Each is used only where its own duration resolves,
    which is `sync_geometry.crossover_hz`.
    """
    flat = np.asarray(mask, dtype=bool).reshape(-1)
    rate = float(sample_rate_hz)
    edges = np.diff(np.concatenate([[0], flat.view(np.int8), [0]]))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    lengths = stops - starts
    keep = (lengths / rate * 1e6) >= float(minimum_us)
    return {
        "starts": starts[keep],
        "stops": stops[keep],
        "lengths": lengths[keep],
        "longest": int(lengths.max()) if lengths.size else 0,
        "runs": int(keep.sum()),
        "all_runs": int(lengths.size),
        "minimum_us": float(minimum_us),
        "why": ("spectra may be divided only on a run long against the "
                "channel's memory; the vertical interval is that run and "
                "the line pulses are read by the matched edge instrument"),
    }


def analytic_component(measured, expected, mask, sample_rate_hz: float,
                       band_hz: Optional[Sequence[float]] = None,
                       tip_ire: Optional[float] = None,
                       blanking_ire: Optional[float] = None,
                       edge: Optional[Dict[str, object]] = None
                       ) -> Dict[str, object]:
    """THE THREE-DIMENSIONAL ANALYTIC COMPONENT of one field against its
    specification: amplitude, frequency and time from one comparison.

    The frequency reading is taken on the mask's LONG runs - the vertical
    interval, which is one unbroken window of 572 microseconds - because
    that is where a ratio of spectra means anything (`long_runs` states the
    measurement that settles it). Where the caller passes `edge`, the
    pooled line-sync reading it already holds, the two are joined at the
    crossover the format's own durations fix: below the pulse's 213 kHz
    only the interval can speak, above it the pulse resolves detail the
    interval smears. That is Ethan's own instruction - "Use the eq pulses
    to get the longer value, and the hsync pulses to refine the higher
    frequency details" - and `sync_geometry.combined_response` performs
    the join, matching levels across it rather than fitting anything.

    The delay is fitted THROUGH ZERO with a free phase reference beside it,
    because a ramp fitted about a band's centre leaves behind a constant
    that no entry of any component key can express.

    `tip_ire` and `blanking_ire` are the field's own measured levels where
    the caller has them; without them the amplitude is read from the masked
    field itself, so the reading is complete on a field alone.
    """
    y = np.asarray(measured, dtype=np.float64)
    x = np.asarray(expected, dtype=np.float64)
    keep = np.asarray(mask, dtype=bool)
    if y.shape != x.shape or y.shape != keep.shape:
        raise ValueError("the measured field, the specification and the "
                         "mask must share a shape")
    rate = float(sample_rate_hz)

    # -- AMPLITUDE: the spacing that is the luma's only absolute scale ------
    if tip_ire is None or blanking_ire is None:
        deep = x <= (SYNC_TIP_IRE + BLANKING_IRE) / 2.0
        tip_ire = float(np.median(y[keep & deep])) if np.any(keep & deep) \
            else float("nan")
        blanking_ire = float(np.median(y[keep & ~deep])) \
            if np.any(keep & ~deep) else float("nan")
    measured_span = float(blanking_ire) - float(tip_ire)
    specified_span = BLANKING_IRE - SYNC_TIP_IRE
    amplitude = {
        "tip_ire": float(tip_ire),
        "blanking_ire": float(blanking_ire),
        "span_ire": measured_span,
        "specified_span_ire": specified_span,
        "departure_ire": measured_span - specified_span,
        "gain": (measured_span / specified_span) if specified_span else
        float("nan"),
    }

    # -- FREQUENCY: on the long runs, where a ratio of spectra means it ----
    runs = long_runs(keep, rate)
    if not runs["runs"]:
        raise ValueError("the mask keeps no run long enough to divide "
                         "spectra on; the interval is what supplies one")
    flat_y, flat_x = y.reshape(-1), x.reshape(-1)
    index = int(np.argmax(runs["lengths"]))
    a, b = int(runs["starts"][index]), int(runs["stops"][index])
    reading = sync_geometry.differential(flat_y[a:b], flat_x[a:b], rate)
    frequency = np.asarray(reading["frequency_hz"], dtype=np.float64)
    transfer = np.asarray(reading["H"], dtype=np.complex128)
    carried = np.asarray(reading["valid"], dtype=bool)
    floor_hz = float(reading["resolution_hz"])
    top = float(rate) / 2.0
    band = (floor_hz, top) if band_hz is None else (float(band_hz[0]),
                                                    float(band_hz[1]))
    inside = carried & (frequency >= max(band[0], floor_hz)) \
        & (frequency <= band[1])
    if inside.sum() < 4:
        raise ValueError("the run and the band leave too few bins to read")
    f, H = frequency[inside], transfer[inside]
    joined = None
    if edge is not None:
        joined = sync_geometry.combined_response(
            f, H, np.asarray(edge["frequency_hz"], dtype=np.float64),
            np.asarray(edge["H"], dtype=np.complex128),
            short_probe_us=sync_geometry.LINE_SYNC_US["525"],
            long_probe_us=float(runs["lengths"][index]) / rate * 1e6)
        f = np.asarray(joined["frequency_hz"], dtype=np.float64)
        H = np.asarray(joined["H"], dtype=np.complex128)
        inside_band = (f >= band[0]) & (f <= band[1])
        f, H = f[inside_band], H[inside_band]
    departure = np.log(np.abs(H) + 1e-300) + 1j * np.unwrap(np.angle(H))

    # -- TIME: the delay through zero, with its own phase reference --------
    basis = np.stack([np.ones(f.size), f], axis=1)
    coefficients, *_ = np.linalg.lstsq(basis, departure.imag, rcond=None)
    reference, slope = float(coefficients[0]), float(coefficients[1])
    residue = departure.imag - reference - slope * f
    time = {
        "delay_s": -slope / (2.0 * np.pi),
        "phase_reference_rad": reference,
        "residual_phase_rms": float(np.sqrt(np.mean(residue ** 2))),
        "why": ("the delay is the linear phase through zero and the "
                "reference is the constant beside it; fitted about the "
                "band's centre a pure delay leaves that constant behind as "
                "a false all-pass"),
    }
    return {
        "amplitude": amplitude,
        "response": departure,
        "frequency_hz": f,
        "time": time,
        "band_hz": band,
        "resolution_hz": floor_hz,
        "bins": int(f.size),
        "run_us": float(runs["lengths"][index]) / rate * 1e6,
        "runs": {k: v for k, v in runs.items()
                 if k in ("runs", "all_runs", "longest", "minimum_us")},
        "joined": None if joined is None else {
            k: joined[k] for k in ("crossover_hz", "from_long_probe",
                                   "from_short_probe", "level_offset_nepers",
                                   "overlap_disagreement_nepers")},
        "kept_samples": int(keep.sum()),
        "why": ("one comparison of the reserved intervals against their "
                "specification gives the amplitude, the frequency response "
                "and the time at once; the response is read on the long "
                "run and joined to the pooled edge at the crossover the "
                "format's own durations fix"),
    }


def edge_agreement(whole, edge, frequency_whole, frequency_edge,
                   band_hz: Optional[Sequence[float]] = None) -> Dict[str, object]:
    """THE CROSS-CHECK: the whole-mask reading against the pooled horizontal
    edge, on the band they share.

    The edge is the instrument this arc used until now, and it is kept
    precisely so that the wider reading can be held against it. They should
    agree where both have evidence - above the pulse's own resolution - and
    the whole-mask reading should carry the band below it that the edge
    cannot reach at all. A disagreement on the shared band is a defect in
    one of the two and is reported rather than averaged away.
    """
    a = np.asarray(whole, dtype=np.complex128)
    b = np.asarray(edge, dtype=np.complex128)
    fa = np.asarray(frequency_whole, dtype=np.float64)
    fb = np.asarray(frequency_edge, dtype=np.float64)
    low = max(fa.min(), fb.min()) if band_hz is None else float(band_hz[0])
    high = min(fa.max(), fb.max()) if band_hz is None else float(band_hz[1])
    grid = fb[(fb >= low) & (fb <= high)]
    if grid.size < 4:
        return {"shared_bins": int(grid.size), "coherence": float("nan"),
                "why": "the two readings share too little band to compare"}
    onto = np.interp(grid, fa, a.real) + 1j * np.interp(grid, fa, a.imag)
    other = b[(fb >= low) & (fb <= high)]
    scale = np.sqrt(np.mean(np.abs(onto) ** 2) * np.mean(np.abs(other) ** 2))
    coherence = float(np.abs(np.mean(onto * np.conj(other))) / scale) \
        if scale > 0 else float("nan")
    return {
        "shared_bins": int(grid.size),
        "band_hz": (float(low), float(high)),
        "coherence": coherence,
        "whole_rms": float(np.sqrt(np.mean(np.abs(onto) ** 2))),
        "edge_rms": float(np.sqrt(np.mean(np.abs(other) ** 2))),
        "why": ("the pooled edge and the whole mask should agree where both "
                "have evidence; below the pulse's own resolution only the "
                "interval reaches, and that band is the reason for it"),
    }
