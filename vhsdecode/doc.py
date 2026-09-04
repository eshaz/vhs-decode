from dataclasses import dataclass
import numpy as np
from numba import njit
import math

import vhsdecode.formats as vhs_formats
from vhsdecode import head_switch
from vhsdecode.rust_utils import sosfiltfilt_rust


@dataclass
class DodOptions:
    dod_threshold_p: float
    dod_threshold_a: float
    dod_hysteresis: float


@njit(cache=True)
def find_crossings(data, threshold):
    """Find where the data crosses the set threshold."""

    # We do this by constructing array where positions above
    # the threshold are marked as true, other sfalse,
    # and use diff to mark where the value changes.
    crossings = np.diff(data < threshold)
    # TODO: See if we can avoid reduntantly looking for both up and
    # down crossing when we just need one of them.
    return crossings


@njit(cache=True, nogil=True, fastmath=True)
def find_dropouts_rf(env, start_rf, end_rf, threshold, hysteresis, merge_threshold):
    # list of tuples containing start and end
    down_thresh = threshold
    up_thresh = threshold * hysteresis

    dropouts = []
    dropout_idx = -1
    
    for i in range(start_rf, end_rf):
        v = env[i]
        dropout_ended = False

        if v <= down_thresh:
            if dropout_idx == -1 or (dropout_ended := dropouts[dropout_idx][1] != -1 and i - dropouts[dropout_idx][1] > merge_threshold):
                # start dropout if none exist or distance from previous is greater than the merge threshold
                dropout_idx += 1
                dropouts.append((i, -1))
            elif dropout_ended:
                # continue existing dropout
                dropouts[dropout_idx] = (dropouts[dropout_idx][0], -1)
        elif v >= up_thresh:
            # end dropout
            if dropout_idx != -1 and dropouts[dropout_idx][1] == -1:
                dropouts[dropout_idx] = (dropouts[dropout_idx][0], i)

    if dropout_idx != -1 and dropouts[dropout_idx][1] == -1:
        # set the dropout ending to the last sample when the dropout happens at the end of the field
        dropouts[dropout_idx] = (dropouts[dropout_idx][0], end_rf)

    return dropouts


def detect_dropouts_rf(field, dod_options):
    """Look for dropouts in the input data, based on rf envelope amplitude.
    Uses either an percentage of the frame average rf level, or an absolute value.
    TODO: A more advanced algorithm with hysteresis etc.
    """
    # The envelope is carried at full bandwidth for the amplitude measurement,
    # which needs every excursion it can see. Dropout detection wants the
    # opposite: on a wide envelope a single damaged region crosses the
    # threshold repeatedly, breaking into a run of short spans with unmarked
    # gaps between them, and the concealment then covers less of the damage
    # than it should. So it is band limited here instead - once over the
    # assembled field rather than once per block.
    env = sosfiltfilt_rust(field.rf.Filters["FEnvPost"], field.data["video"]["envelope"])

    threshold_p = dod_options.dod_threshold_p
    threshold_abs = dod_options.dod_threshold_a
    hysteresis = dod_options.dod_hysteresis

    threshold = 0.0
    field_average = np.mean(env)
    # Store the average for later.
    field.rf.field_averages.rf_level.push(field_average)
    if threshold_abs is not None:
        threshold = threshold_abs
    else:
        # Generate a threshold based on the field envelope average.
        # This may not be ideal on a field with a lot of droputs,
        # so we may want to use statistics of the previous averages
        # to avoid the threshold ending too low.
        threshold = field_average * threshold_p

    start_line = field.lineoffset + 1
    end_line = min(len(field.linelocs) - 1, field.linecount + start_line + 1)

    start_rf = math.floor(field.linelocs[start_line])
    end_rf = min(len(env), math.ceil(field.linelocs[end_line]))

    dropouts_rf = find_dropouts_rf(env, start_rf, end_rf, threshold, hysteresis, vhs_formats.DOD_MERGE_THRESHOLD)

    # Drop very short dropouts that were not merged.
    # We do this after mergin to avoid removing short consecutive dropouts that
    # could be merged.
    dropouts_rf = list(filter(lambda s: s[1] - s[0] > vhs_formats.DOD_MIN_LENGTH, dropouts_rf))

    debug_plot = getattr(field.rf, "debug_plot", None)
    if debug_plot and debug_plot.is_plot_requested("luma_noise"):

        from vhsdecode.debug_plot import plot_luma_noise

        # Per-head wow panel: the per-line line-period deviation measured on
        # the carrier's own sync-edge trace.  Only fields decoded under a
        # flag that keeps the demod_raw channel can measure it; the trace of
        # the most recent opposite-parity field is kept on the rf object -
        # display state only, nothing reads it back - so the panel shows the
        # two heads side by side.
        wow_trace = None
        if field.rf.options.carrier_tbc != 0 or field.rf.options.head_switch != 0:
            from vhsdecode.luma_amplitude import sync_edge_trace

            trace = sync_edge_trace(field)
            if trace is not None and trace[1] > 0:
                edges, period = trace
                # The period between lines i and i+1, as a percentage
                # deviation from the trace's own median period, plotted at
                # line i; pairs broken by an unmeasured line stay NaN and
                # draw as gaps rather than interpolations.
                deviation = np.full(len(edges), np.nan)
                consecutive = (edges[:-1] > 0) & (edges[1:] > 0)
                deviation[:-1][consecutive] = (
                    np.diff(edges)[consecutive] / period - 1.0
                ) * 100.0
                # The trace's own hygiene, as the correction applies it: a
                # crossing latched on the wrong feature (a serration edge at
                # the field boundary, typically) reads tens of samples off,
                # which no physical wow does.  Bound at the extreme-value
                # tail of the series' own robust scale so honest flutter is
                # never touched, and blank what fails - the panel shows
                # measurements, not latch failures.
                finite = np.isfinite(deviation)
                values = deviation[finite]
                if len(values) > 2:
                    median = np.median(values)
                    sigma = 1.4826 * np.median(np.abs(values - median))
                    if sigma > 0:
                        bound = sigma * np.sqrt(2.0 * np.log(2.0 * len(values)))
                        deviation[finite & (np.abs(deviation - median) > bound)] = np.nan
                parity = bool(field.isFirstField)
                store = field.rf.__dict__.setdefault("_wow_panel_traces", {})
                store[parity] = deviation
                wow_trace = {
                    "head": parity,
                    "deviation": deviation,
                    "other_head": store.get(not parity),
                }

        video = field.data["video"]
        plot_luma_noise(
            video["envelope"],
            env,
            video["demod"],
            video["demod_raw"] if "demod_raw" in video else None,
            getattr(field, "chroma_envelope_deviation", None),
            getattr(field, "chroma_envelope_correction", None),
            getattr(field, "chroma_envelope_response", None),
            field.isFirstField,
            dropouts_rf,
            threshold,
            hysteresis,
            start_rf,
            end_rf,
            field.linelocs,
            field.rf.hztoire,
            field.rf.dod_options.dod_threshold_p,
            # Present only where `--luma_eq` is running and this plot asked
            # `demodblock` to keep it.
            demod_noeq=video.get("demod_noeq"),
            # What the color-under beat correction took out of the luma, where
            # it ran and this plot asked for it to be kept. It is made on the
            # time base corrected grid, so it comes with the RF sample position
            # of each of its samples - the same mapping the downscale itself
            # used - and lands on the shared axis with everything else.
            beat=getattr(field, "luma_beat", None),
            beat_locs=beat_sample_locations(field),
            # What the head switch correction subtracted, carried as a video
            # channel so it arrives already on this plot's sample grid; the
            # switch itself is located from the field's own deviation (its
            # sustained offset never shows in the applied trace - the phase
            # kernel nulls DC by design); the format says how many switch
            # events a field should hold.
            head_switch=video.get("head_switch"),
            head_switch_regions=(
                head_switch.locate(field)
                if field.rf.options.head_switch != 0
                else None
            ),
            head_switch_expected=field.rf.SysParams.get("head_switches_per_field", 1),
            # THE OTHER HEAD's accumulated response, so the two can be read
            # against each other. The heads differ measurably - in level, in
            # the response's slope and in its shape - and a panel that shows
            # only the field's own head cannot show that. Same convention as
            # the wow panel directly below it: this field's head solid, the
            # other faint, both named.
            other_head_response=_other_head_response(field),
            expected_response=_expected_response(field),
            # What the baseband equalizer changed, in IRE at the raw
            # demodulated site, for the folded sync-interval view.
            baseband_eq=video.get("baseband_eq"),
            # The specified carrier frequencies, which is the anchor the
            # response model is binned on - see `carrier_frequency_bins`.
            ire_to_hz=lambda ire: field.rf.iretohz(ire, spec=True),
            wow_trace=wow_trace,
        )

    export_path = getattr(field.rf, "_head_switch_export", None)
    if export_path and field.rf.options.head_switch != 0:
        # The measured transfer as of this field, for offline reconciliation.
        head_switch.export_measured_transfer(field.rf, export_path)

    if debug_plot and debug_plot.is_plot_requested("sync_step_fold"):
        from vhsdecode.debug_plot import plot_sync_step_fold

        # The decoded line folded over the field's lines against the
        # level-adjusted specified sync pulse, with what the baseband
        # equalizer changed (zeros when it is not running) - the residual
        # against the ideal is the measurement the equalizer's closed loop
        # refines on.
        video = field.data["video"]
        plot_sync_step_fold(
            video["demod"],
            video.get("baseband_eq"),
            field.linelocs,
            start_rf,
            end_rf,
            field.rf.hztoire,
            field.rf.SysParams,
            field.rf.freq_hz,
            field.isFirstField,
        )

    if debug_plot and debug_plot.is_plot_requested("luma_averaging"):
        from vhsdecode.debug_plot import plot_luma_averaging

        # Accumulated over the decode rather than measured on this field, so
        # what it draws improves as fields go by. Nothing here feeds back.
        rf = field.rf
        plot_luma_averaging(
            {
                "luma": rf.__dict__.get("_luma_averaging_probe", {}),
                "chroma": rf.__dict__.get("_chroma_amount_probe", {}),
            },
            rf.dod_options.dod_threshold_p,
        )

    return map_dropouts_rf_to_tbc(dropouts_rf, start_line, end_line, field.linelocs, field.outlinelen, field.lineoffset)

def map_dropouts_rf_to_tbc(errlist, start_line_idx, end_line_idx, linelocs, outlinelen, lineoffset):
    rv_lines = []
    rv_starts = []
    rv_ends = []

    line_idx = start_line_idx
    line_start_rf = linelocs[line_idx]
    line_end_rf = linelocs[line_idx + 1]

    for (start_rf, end_rf) in errlist:
        while line_idx < end_line_idx:
            # find the line that contains start of the dropout
            line_len = line_end_rf - line_start_rf
            if (start_rf >= line_start_rf or line_idx == start_line_idx) and start_rf < line_end_rf and line_len > 0:
                rv_lines.append(line_idx - lineoffset)
                
                # scale down to tbc line position
                start_rf_linepos = start_rf - line_start_rf
                start_linepos = math.floor(start_rf_linepos / line_len * outlinelen)

                rv_starts.append(max(0, start_linepos))
                break
            else:
                line_idx += 1
                line_start_rf = linelocs[line_idx]
                line_end_rf = linelocs[line_idx + 1]

        while line_idx < end_line_idx:
            line_len = line_end_rf - line_start_rf
            if end_rf < line_end_rf and line_len > 0:
                # dropout is contained within this line
                # scale down to tbc line position
                end_rf_linepos = end_rf - line_start_rf
                end_linepos = math.ceil(end_rf_linepos / line_len * outlinelen)

                rv_ends.append(min(outlinelen, end_linepos))
                break
            else:
                # dropout spans multiple lines
                rv_ends.append(outlinelen)
                line_idx += 1

                if line_idx < end_line_idx:
                    # continue the dropout to the next line
                    line_start_rf = linelocs[line_idx]
                    line_end_rf = linelocs[line_idx + 1]
                    
                    rv_starts.append(0)
                    rv_lines.append(line_idx - lineoffset)

    return rv_lines, rv_starts, rv_ends


def _expected_response(field):
    """THIS head's accumulated luma response - the model the residual is
    measured against.

    The field's own `ResponseModel` carries the model that was in force when
    that field was measured, and under the limit the first field's is empty
    by construction: a field is refined against the model the fields before
    it built, and the first has none. The panel draws the first field, so
    without this the expected response - the one quantity the residual is a
    residual OF - is the one curve missing from it.
    """
    from vhsdecode.luma_amplitude import measured_response

    try:
        return measured_response(field.rf, field.isFirstField)
    except Exception:                                            # noqa: BLE001
        return None


def _other_head_response(field):
    """The OTHER head's accumulated luma response, for the plot.

    The heads differ measurably - in level, in the response's slope and in
    its shape - so a panel that draws only the field's own head cannot show
    that difference. Imported locally, as `sync_edge_trace` is, because
    this module is on the dropout path and the plot is the only caller.
    """
    from vhsdecode.luma_amplitude import measured_response

    try:
        return measured_response(field.rf, not field.isFirstField)
    except Exception:                                            # noqa: BLE001
        return None


def beat_sample_locations(field):
    """Where each sample of the beat correction sits on the RF sample grid.

    The correction is made after the chroma is processed, so it is on the time
    base corrected picture's grid while the rest of the plot is on the RF one.
    `computewow_scaled` already holds the mapping between them - it is what the
    downscale interpolated against - and caches it on the field, so reading it
    back costs nothing and cannot disagree with where the samples actually came
    from.
    """
    if getattr(field, "luma_beat", None) is None:
        return None
    locations = getattr(field, "interpolated_pixel_locs", None)
    if locations is None:
        locations = field.computewow_scaled()[0]
    start = (field.lineoffset + 1) * field.outlinelen
    return locations[start: start + len(field.luma_beat)]
