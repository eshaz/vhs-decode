import sys


def to_db_power(input_data):
    import numpy as np

    return 20 * np.log10(input_data)


# Plots that describe the pipeline's STRUCTURE rather than any field's
# data. They need no decode state, so they do not need the decode
# serialised - and forcing a single worker thread for one of them would
# both cost the user their throughput and, if the plot ever reported
# timing, profile a pipeline they are not running.
STRUCTURAL_PLOTS = frozenset({"pipeline_graph"})


class DebugPlot:
    def __init__(self, stuff_to_plot: str):
        self.__stuff_to_plot = stuff_to_plot.casefold().split()

    def is_plot_requested(self, requested_info: str):
        return requested_info in self.__stuff_to_plot

    def requested(self):
        return tuple(self.__stuff_to_plot)

    def wants_serialised_decode(self):
        """Whether any requested plot actually needs one worker thread.

        Every debug plot used to force `threads=0`, structural ones
        included. A plot that draws the pipeline's declaration reads no
        field and cannot benefit from it."""
        return any(name not in STRUCTURAL_PLOTS
                   for name in self.__stuff_to_plot)


def plot_data_and_pulses(
    demod_video,
    raw_pulses=None,
    linelocs=None,
    pulses=None,
    extra_lines=None,
    vblank_lines=None,
    threshold=None,
):
    import matplotlib.pyplot as plt
    from matplotlib import rc_context

    with rc_context(
        {"figure.figsize": (14, 10), "figure.constrained_layout.use": True}
    ):
        # Make sure we have enough plots
        # Surely there is a cleaner way to handle this.
        plots = 1
        if raw_pulses is not None:
            plots += 1
        if linelocs is not None:
            plots += 1
        if pulses is not None:
            plots += 1
        if extra_lines is not None:
            plots += 1
        if vblank_lines is not None:
            plots += 1

        plots = max(2, plots)

        ax_number = 0

        fig, ax = plt.subplots(plots, 1, sharex=True)
        ax[ax_number].plot(demod_video)
        ax[ax_number].set_title("Video")
        if threshold is not None:
            ax[ax_number].axhline(threshold)

        if raw_pulses is not None:
            ax_number += 1
            ax[ax_number].set_title("Raw pulses")
            for raw_pulse in raw_pulses:
                ax[ax_number].axvline(raw_pulse.start, color="#910000")
                ax[ax_number].axvline(raw_pulse.start + raw_pulse.len, color="#090909")

        if pulses is not None:
            ax_number += 1
            ax[ax_number].set_title("Validated pulses")
            for pulse in pulses:
                color = "#FF0000" if pulse[0] == 2 else "#00FF00"
                ax[ax_number].axvline(pulse[1][0], color=color)
                ax[ax_number].axvline(pulse[1][0] + pulse[1][1], color="#009900")

        if linelocs is not None:
            ax_number += 1
            ax[ax_number].set_title("Calculated line locations")
            for ll in linelocs:
                ax[ax_number].axvline(ll)

        if vblank_lines is not None:
            ax_number += 1
            ax[ax_number].set_title("V-blanking Pulses")
            for ll in vblank_lines:
                ax[ax_number].axvline(ll)

        if extra_lines is not None:
            ax_number += 1
            colors = [
                "#FF0000",
                "#00FF00",
                "#0000FF",
                "#FF0000",
                "#FFFF00",
                "#000000",
                "#00FFFF",
                "#888888",
            ]
            for extra_line, color in zip(extra_lines, colors):
                ax[ax_number].axvline(extra_line, color=color)

        # to_right_edge = self.usectoinpx(self.rf.SysParams["hsyncPulseUS"]) + (
        #    2.25 * (self.rf.freq / 40.0)
        # )

        plt.show()


def plot_magnitude_density(
    raw_data,
    filtered_data,
    rfdecode,
):
    import matplotlib.pyplot as plt
    from matplotlib import rc_context
    import numpy as np
    import scipy.signal as sps
    from vhsdecode.hilbert import unwrap_hilbert

    raw_h = sps.hilbert(raw_data)
    fil_h = sps.hilbert(filtered_data)

    raw_m = np.abs(raw_h)
    fil_m = np.abs(fil_h)

    raw_f = unwrap_hilbert(raw_h, rfdecode.freq_hz)
    fil_f = unwrap_hilbert(fil_h, rfdecode.freq_hz)

    with rc_context(
        {"figure.figsize": (14, 10), "figure.constrained_layout.use": True}
    ):
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        ax1.hist2d(
            raw_f,
            raw_m,
            bins=(256, 256),
            range=[[0, rfdecode.freq_hz // 2], [0, len(raw_m) // 16]],
            cmap=plt.cm.jet,
        )
        ax1.set_title("Raw data")
        ax2.hist2d(
            fil_f,
            fil_m,
            bins=(256, 256),
            range=[[0, rfdecode.freq_hz // 2], [0, len(fil_m) // 16]],
            cmap=plt.cm.jet,
        )
        ax2.set_title("Filtered data")

        plt.show()


def plot_input_data(
    raw_data,
    filtered_data,
    raw_fft,
    filtered_fft,
    env,
    env_mean,
    demod_video,
    filtered_video,
    chroma,
    rf_filter,
    rfdecode,
    plot_db=True,
    plot_demod_fft=False,
    plot_chroma_fft=False,
):
    import matplotlib.pyplot as plt
    from matplotlib import rc_context
    import numpy as np

    with rc_context(
        {"figure.figsize": (14, 10), "figure.constrained_layout.use": True}
    ):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)

        ire0 = rfdecode.sysparams_const.ire0
        ire100 = rfdecode.sysparams_const.ire0 + (100 * rfdecode.sysparams_const.hz_ire)
        sync_hz = rfdecode.sysparams_const.vsync_hz
        cc_freq = rfdecode.DecoderParams["color_under_carrier"]

        # ax1.plot((20 * np.log10(self.Filters["Fdeemp"])))
        #        ax1.plot(hilbert, color='#FF0000')
        blocklen = len(raw_data)
        ax1.plot(raw_data, color="#00FF00")
        ax1.plot(filtered_data, color="#FF0000")
        ax1.plot(env, label="Envelope", color="#0000FF")
        if rfdecode.dod_options.dod_threshold_a:
            ax1.axhline(
                rfdecode.dod_options.dod_threshold_a,
                label="DOD Threshold (absolute)",
                color="#001100",
            )
        else:
            ax1.axhline(
                rfdecode.dod_options.dod_threshold_p * env_mean,
                label="DOD Threshold",
                color="#110000",
            )
            ax1.axhline(
                rfdecode.dod_options.dod_threshold_p
                * env_mean
                * rfdecode.dod_options.dod_hysteresis,
                label="DOD Hysteresis threshold",
                ls="--",
                color="#000011",
            )
        ax1.set_title("Raw data")
        ax1.legend()
        ax2.plot(demod_video)
        ax2.set_title("Demodulated video")
        ax3.plot(filtered_video, color="#00FF00")
        ax3.axhline(rfdecode.iretohz(0), label="0 IRE", color="#000000")
        ax3.axhline(rfdecode.iretohz(100), label="100 IRE", color="#000000", ls="--")
        ax3.axhline(rfdecode.iretohz(0), label="0 IRE", color="#000000")
        ax3.set_title("Output video (Deemphasized and filtered) ")
        ax3.axvline(0, ls="dotted")
        ax3.axvline(rfdecode.linelen, ls="dotted", label="Length of one line")
        ax3.legend()
        ax4.plot(chroma)
        ax4.set_title("Chroma signal")

        half_size = (blocklen // 2) + 1
        freq_array = (np.arange(blocklen) / blocklen * rfdecode.freq_hz)[:half_size]

        def to_plot(a):
            return to_db_power(a) if plot_db else abs(a)

        if plot_demod_fft:
            fig2, (ax4, ax5) = plt.subplots(2, 1, sharex=True)
            ax5 = fig2.add_subplot(2, 1, 2)
            ax5.plot(
                freq_array, to_db_power(abs(np.fft.rfft(demod_video))), color="#00FF00"
            )
            ax5.plot(
                freq_array,
                to_db_power(abs(np.fft.rfft(filtered_video))),
                color="#FF0000",
            )
        elif plot_chroma_fft:
            fig2, (ax4, ax5) = plt.subplots(2, 1, sharex=True, sharey=False)
            # ax5 = fig2.add_subplot(2, 1, 2)
            ax5.plot(
                freq_array,
                to_plot(raw_fft[:half_size]),
                color="#00FF00",
                label="Raw input",
            )
            ax5.plot(
                freq_array,
                to_db_power(abs(np.fft.rfft(chroma))),
                color="#FF0000",
                label="Filtered chroma",
            )
        else:
            fig2, ax4 = plt.subplots(1, 1, sharey=False)

        ax4.sharey = False
        ax4.plot(
            freq_array, to_plot(raw_fft[:half_size]), color="#00FF00", label="Raw input"
        )
        ax5 = ax4.twinx()
        ax5.plot(
            freq_array,
            to_plot(filtered_fft[:half_size]),
            color="#FF0000",
            label="After rf filtering",
        )
        ax5.set_ylim(bottom=-60)
        ax6 = ax4.twinx()
        ax6.plot(
            freq_array,
            to_plot(rf_filter[:half_size]),
            color="#0000FF",
            label="rf filter",
        )
        ax6.set_ylim(bottom=-60)
        ax6.axhline(0, label="0db filter", ls="--")
        ax4.set_xlabel("frequency")
        if plot_db:
            ax4.set_ylabel("dB power")
        ax4.set_title("frequency spectrum of rf input")
        ax4.axvline(ire0, label="0 IRE", color="#000000")
        ax4.axvline(ire100, label="100 IRE", ls="--")
        ax4.axvline(sync_hz, label="sync tip", ls="-.")
        ax4.axvline(cc_freq, label="Chroma carrier", color="#6F6F00")
        ax4.legend()

        plt.show()


def plot_luma_rf(rf, rf_luma_filter):
    #    import math
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(0, 40, rf_luma_filter.size)

    mag = 20 * np.log10(np.absolute(rf_luma_filter))
    ph = np.angle(rf_luma_filter, deg=True)

    fig, ax1 = plt.subplots()

    col = "tab:red"
    ax1.set_xlabel("Frequenzy (MHz)")
    ax1.set_ylabel("Magnetude (dB)", color=col)
    ax1.plot(x, mag, color=col)
    ax1.tick_params(axis="y", labelcolor=col)

    ax2 = ax1.twinx()

    col = "tab:green"
    ax2.set_ylabel("Phase (°)", color=col)
    ax2.plot(x, ph, color=col)
    ax2.tick_params(axis="y", labelcolor=col)

    fig.tight_layout()
    plt.show()


def plot_env_filter(env_filter1, env_filter2):
    #    import math
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(0, 40, env_filter1.size)

    mag = 20 * np.log10(np.absolute(env_filter1))
    mag2 = 20 * np.log10(np.absolute(env_filter2))
    # ph = np.angle(env_filter1, deg=True)

    fig, ax1 = plt.subplots()

    col = "tab:red"
    ax1.set_xlabel("Frequenzy (MHz)")
    ax1.set_ylabel("Magnetude (dB)", color=col)
    ax1.plot(x, mag, color=col)
    ax1.tick_params(axis="y", labelcolor=col)

    ax2 = ax1.twinx()

    col = "tab:green"
    ax2.set_ylabel("Phase (°)", color=col)
    ax2.plot(x, mag2, color=col)
    ax2.tick_params(axis="y", labelcolor=col)

    fig.tight_layout()
    plt.show()


def plot_deemphasis(rf, filter_video_lpf, decoder_params, filter_deemp):
    import math
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.signal as sps
    from vhsdecode.addons.FMdeemph import FMDeEmphasisB

    corner_freq = 1 / (math.pi * 2 * decoder_params["deemph_tau"])

    db, da = FMDeEmphasisB(
        rf.freq_hz, decoder_params["deemph_gain"], decoder_params["deemph_mid"]
    ).get()

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    w1, h1 = sps.freqz(db, da, fs=rf.freq_hz)

    # VHS eyeballed freqs.
    test_arr = np.array(
        [
            [
                0.04,
                0.05,
                0.07,
                0.1,
                corner_freq / 1e6,
                0.2,
                0.3,
                0.4,
                0.5,
                0.7,
                1,
                2,
                3,
                4,
                5,
            ],
            [
                0.4,
                0.6,
                1.2,
                2.2,
                3,
                5.25,
                7.5,
                9.2,
                10.5,
                11.75,
                12.75,
                13.5,
                13.8,
                13.9,
                14,
            ],
        ]
    )

    # Betamax pal eyeballed freqs for -3 dB
    _test_arr = np.array(
        [
            [
                0.05,
                # corner_freq / 1e6,
                0.2,
                0.5,
                1.0,
                2.0,
                4.0,
            ],
            [
                0.2,
                # 3,
                4.6,
                8.5,
                11.7,
                13.3,
                11.6,
            ],
        ]
    )

    # Video8 freqs for -3 dB
    _test_arr = np.array(
        [
            [
                0.05,
                0.1,
                # corner_freq / 1e6,
                0.2,
                0.5,
                1.0,
                2.0,
                4.0,
            ],
            [
                0.9,
                2.6,
                # 3,
                6.4,
                11.2,
                13.1,
                13.8,
                13.9,
            ],
        ]
    )

    test_arr[0] *= 1000000.0

    ax1.plot(test_arr[0], test_arr[1], color="#000000")
    ax1.plot(w1, -20 * np.log10(h1))
    ax1.axhline(-3)
    ax1.axhline(-7)
    ax1.axvline(corner_freq)

    blocklen_half = rf.blocklen // 2
    freqs = np.linspace(0, rf.freq_hz_half, blocklen_half)
    ax2.set_ylim(bottom=-60)
    ax2.plot(
        freqs, 20 * np.log10(filter_deemp[:blocklen_half]), label="Deemphasis only"
    )
    ax2.plot(freqs, 20 * np.log10(filter_video_lpf[:blocklen_half]), label="lpf")
    ax2.plot(
        freqs,
        20 * np.log10(rf.Filters["FVideo"][:blocklen_half]),
        label="Deemphasis + lpf",
    )
    ax2.axhline(-3, label="-3 db", ls="--", color="#000000")
    ax3 = ax2.twinx()
    ax3.plot(
        freqs,
        rf.Filters["FVideo"][:blocklen_half].imag,
        label="Deemphasis + lpf phase",
        color="#990000",
    )
    ax2.legend()
    ax3.legend()
    plt.show()
    sys.exit()


def plot_final_chroma_field(input_chroma, final_chroma) -> None:
    import math
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.signal as sps

    import matplotlib.pyplot as plt
    from matplotlib import rc_context
    import numpy as np

    with rc_context(
        {"figure.figsize": (14, 10), "figure.constrained_layout.use": True}
    ):
        fig, (ax1, ax2) = plt.subplots(2, 1)  # , sharex=True)

        ax1.plot(input_chroma)
        ax1.plot(final_chroma, color="#AA0000")
        ax2.plot(to_db_power(np.fft.rfft(input_chroma)))
        ax2.plot(to_db_power(np.fft.rfft(final_chroma)), color="#AA0000")

        ax1.legend()
        ax2.legend()
        plt.show()
        sys.exit()


def _fold_lines(signal, change, linelocs, start_rf, end_rf, rate_hz=40e6):
    """Mean line of `signal` and of `signal - change`, on the lines' own grid.

    Each line between two consecutive line positions is resampled onto a
    common grid of one median line length and averaged; the result is the
    field's line as the eye sees it in a static picture, and the difference
    between the two folds is what the change did to it. None where fewer
    than two whole lines fit the window.
    """
    import numpy as np

    locs = linelocs[np.isfinite(linelocs)]
    locs = locs[(locs >= start_rf) & (locs < end_rf - 1)]
    if len(locs) < 3 or len(signal) != len(change):
        return None
    lengths = np.diff(locs)
    period = float(np.median(lengths))
    if not period > 0:
        return None
    count = int(round(period))
    grid = np.arange(count) / count
    x = np.arange(len(signal), dtype=float)
    on_sum = np.zeros(count)
    off_sum = np.zeros(count)
    used = 0
    for a, length in zip(locs[:-1], lengths):
        # A line whose length strays far from the median is not a line - a
        # missed or doubled sync - and is left out rather than smeared in.
        if abs(length - period) > 0.02 * period:
            continue
        at = a + grid * length
        on_sum += np.interp(at, x, signal)
        off_sum += np.interp(at, x, signal - change)
        used += 1
    if used < 2:
        return None
    return on_sum / used, off_sum / used, period / rate_hz * 1e6


def plot_luma_noise(
    envelope,
    detection_envelope,
    demod,
    carrier_hz,
    deviation,
    correction,
    response,
    is_first_field,
    dropouts,
    threshold,
    hysteresis,
    start_rf,
    end_rf,
    linelocs,
    hz_to_ire,
    dropout_fraction,
    demod_noeq=None,
    ire_to_hz=None,
    beat=None,
    beat_locs=None,
    head_switch=None,
    head_switch_regions=None,
    head_switch_expected=1,
    other_head_response=None,
    expected_response=None,
    wow_trace=None,
    baseband_eq=None,
):
    """Luma carrier amplitude beside the luma it was demodulated from.

    The carrier is frequency modulated, so its amplitude should be constant and
    everything in the top trace is the tape rather than the picture. That makes
    it a continuous noise profile: where it dips, the demodulator's output noise
    rises and the color-under recorded alongside it lost signal at the same
    instant.

    The top panel is the correction itself - what actually multiplies the
    color-under, wet/dry mix included. It is the carrier amplitude with the
    path's response to the carrier's own frequency divided out, then scaled
    across by wavelength; dividing that response out is the whole difficulty of
    the measurement, so the right hand panel scores it. Carrier amplitude,
    deviation and correction are binned against luma level: the amplitude leans
    across the range because the path responds to frequency, and the other two
    should not. Whatever slope is left in the correction is picture being
    imposed on the chroma.

    Panels down the left cover the whole field on one shared axis, so they line
    up sample for sample and zooming or panning any of them moves the others
    with it. Nothing is zoomed in advance - finding where the interesting
    excursions are is what the plot is for.

    Shaded spans are the dropouts the decoder detected from this same envelope,
    so the plot shows both what the threshold caught and the shallower
    excursions it did not.

    Where `--luma_beat` is running, a panel carries what the color-under beat
    correction took out. The color-under amplitude modulates the luma FM
    carrier, and what survives the demodulator lands in the picture at the
    color-under's own frequency, where it is energy that has nothing to do with
    the luma's image. It is drawn against the color-under beside it, because the
    two should rise and fall together - the beat is the chroma, arriving where
    it does not belong.

    Where `--luma_eq` is running, the luma panel carries the demodulated field
    twice - once as the equalizer left it and once as it would have been
    without - and a panel below gives the difference. Both come from the same
    post-demodulation chain, so what separates them is the equalizer alone.

    Where `--head_switch` is running, the switch is LOCATED on the field's
    own amplitude deviation - the sustained per-head offset, the signature
    the phase kernel deliberately nulls and the applied trace can therefore
    never show - and drawn as a span with a dashed entry line across the
    shared axes. A panel below carries what the correction subtracted - the
    phase artifact predicted from the carrier amplitude residual, in IRE -
    so the located region and the removed transient are judged against the
    envelope step and the luma beside them.

    Where the carrier's sync-edge trace is available (`--carrier_tbc` or
    `--head_switch` carries the channel it needs), a panel gives the
    per-line line-period deviation that trace measured - the wow and
    flutter itself, per head: this field's head beside the most recent
    field the other head wrote, never averaged, because the two heads'
    paths differ and an average would describe a machine that does not
    exist. It is the one panel drawn against line number rather than RF
    sample, so it alone does not pan with the others.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    # R5's declared floor: derived and tested in one place rather
    # than retyped at each call site
    from vhsdecode.models import residual_floor

    envelope = np.asarray(envelope, dtype=float)
    demod = np.asarray(demod, dtype=float)
    samples = np.arange(start_rf, end_rf)
    window = envelope[start_rf:end_rf]
    reference = float(np.median(window)) if len(window) else 1.0

    # Two envelopes, and the difference between them is the point: the
    # correction measures the carrier at full bandwidth, while dropout
    # detection band limits it first, because a wide envelope crosses the
    # threshold repeatedly inside one damaged region. The thresholds drawn
    # below belong to the band limited trace, not the wide one.
    detection_window = (
        np.asarray(detection_envelope, dtype=float)[start_rf:end_rf]
        if detection_envelope is not None
        else None
    )

    # the same weight the color-under amplitude correction uses to decide how
    # far to trust itself, so the plot shows where that correction backs off
    relative = window / reference if reference > 0 else window
    squared = relative * relative
    half = dropout_fraction * dropout_fraction
    confidence = squared * (1.0 + half) / (squared + half)

    # The correction and the response check are only available when the
    # correction ran; without them this is the plot it always was.
    def slice_of(a):
        return np.asarray(a, dtype=float)[start_rf:end_rf] if a is not None else None

    correction_window = slice_of(correction)
    deviation_window = slice_of(deviation)
    luma_window = hz_to_ire(np.asarray(demod, dtype=float)[start_rf:end_rf])
    noeq_window = (
        hz_to_ire(np.asarray(demod_noeq, dtype=float)[start_rf:end_rf])
        if demod_noeq is not None
        else None
    )
    has_correction = correction_window is not None
    can_score = deviation_window is not None and carrier_hz is not None

    # The modelled beat, already in IRE. It is made on the time base corrected
    # picture's grid rather than this one, because that is the only place the
    # decoded saturation it scales with exists - so it arrives with the RF
    # sample position of each of its samples and is drawn against those, which
    # puts it on the same axis as everything else.
    beat_window = np.asarray(beat, dtype=float) if beat is not None else None
    beat_at = np.asarray(beat_locs, dtype=float) if beat_locs is not None else None
    if beat_window is not None and (
        beat_at is None or len(beat_at) != len(beat_window)
    ):
        beat_window = None

    switch_window = slice_of(head_switch)

    # One row per panel, tallest where the detail is. Named rather than
    # indexed: the arithmetic that did this broke every time a panel was added.
    # Ordered by the signal each panel belongs to, in the order the process
    # produces them: the carrier's amplitude and what is derived from it
    # (the deviation, the chroma gain it drives, that gain's confidence),
    # then the demodulated luma and what acts on it (the head switch
    # removal, the luma itself, the equalizer's change, the beat removal),
    # then the two per-line panels on their own axes. Packed tightly: the
    # shared axis is the point, so the panels read as one strip.
    panels = [("env", 2)]
    if deviation_window is not None:
        panels.append(("delta", 2))
    if has_correction:
        panels.append(("corr", 2))
    panels.append(("conf", 1))
    if switch_window is not None:
        panels.append(("switch", 2))
    panels.append(("luma", 3))
    if noeq_window is not None:
        panels.append(("eqdelta", 2))
    if beat_window is not None:
        panels.append(("beat", 2))
    if wow_trace is not None:
        panels.append(("wow", 2))
    # The baseband equalizer's eye: the field's raw demodulated luma folded
    # over its lines, with and without what the equalizer changed, so the
    # sync pulse and porches are read against the specified waveform. Needs
    # the raw channel (pre-de-emphasis, where the equalizer acts) and the
    # line positions; per LINE-TIME like the wow panel, so its own axis.
    eye = None
    if baseband_eq is not None:
        eye = _fold_lines(
            hz_to_ire(np.asarray(demod, dtype=float)),
            np.asarray(baseband_eq, dtype=float),
            np.asarray(linelocs, dtype=float), start_rf, end_rf,
        )
    if eye is not None:
        panels.append(("eye", 2))

    # Two strips: the panels sharing the RF-sample axis, packed tight so they
    # read as one, then the per-line panels (wow per line number, the eye per
    # line time) on their own axes below, with room between the strips.
    shared = [(name, h) for name, h in panels if name not in ("wow", "eye")]
    per_line = [(name, h) for name, h in panels if name in ("wow", "eye")]
    units = sum(h for _, h in panels)
    fig = plt.figure(figsize=(16 if can_score else 14, max(9.0, 0.85 * units + 1.5)))
    outer = fig.add_gridspec(
        2 if per_line else 1,
        2 if can_score else 1,
        width_ratios=[3, 1] if can_score else [1],
        height_ratios=[sum(h for _, h in shared)] + ([sum(h for _, h in per_line)] if per_line else []),
        hspace=0.16,
        left=0.07, right=0.98, top=0.95, bottom=0.05,
    )
    grid = outer  # the right-hand column spans both strips
    strip = outer[0, 0].subgridspec(len(shared), 1, height_ratios=[h for _, h in shared], hspace=0.08)
    axes, anchor = {}, None
    for position, (name, _) in enumerate(shared):
        axes[name] = fig.add_subplot(
            strip[position], **({"sharex": anchor} if anchor is not None else {})
        )
        if anchor is None:
            anchor = axes[name]
    if per_line:
        below = outer[1, 0].subgridspec(len(per_line), 1, height_ratios=[h for _, h in per_line], hspace=0.7)
        for position, (name, _) in enumerate(per_line):
            axes[name] = fig.add_subplot(below[position])
    # One x label for the whole shared strip, on its last panel; the inner
    # panels keep their ticks but drop the labels.
    for name, _ in shared[:-1]:
        axes[name].tick_params(labelbottom=False)
    axes[shared[-1][0]].set_xlabel("RF sample")
    ax_corr = axes.get("corr")
    ax_beat = axes.get("beat")
    ax_switch = axes.get("switch")
    ax_wow = axes.get("wow")
    ax_eye = axes.get("eye")
    ax_delta = axes.get("eqdelta")
    ax_departure = axes.get("delta")
    ax_env, ax_luma, ax_conf = axes["env"], axes["luma"], axes["conf"]
    if can_score:
        right = grid[:, 1].subgridspec(2, 1, height_ratios=[3, 2], hspace=0.35)
        ax_resp = fig.add_subplot(right[0])
        ax_model = fig.add_subplot(right[1])
    else:
        ax_resp = ax_model = None

    if ax_corr is not None:
        ax_corr.plot(
            samples, correction_window, color="tab:red", linewidth=0.5,
            label="gain applied to the color-under",
        )
        ax_corr.axhline(1.0, color="tab:grey", linestyle=":", linewidth=1)
        ax_corr.set_ylabel("chroma gain")
        ax_corr.legend(loc="lower left", fontsize="small")

    ax_env.plot(
        samples, window, color="tab:blue", linewidth=0.5, alpha=0.45,
        label="carrier amplitude, full bandwidth (drives the correction)",
    )
    if detection_window is not None:
        ax_env.plot(
            samples, detection_window, color="black", linewidth=0.7,
            label="band limited (drives dropout detection)",
        )
    ax_env.axhline(
        reference, color="tab:grey", linestyle=":", linewidth=1, label="field median"
    )
    ax_env.axhline(
        threshold, color="tab:red", linestyle="--", linewidth=1,
        label="dropout threshold",
    )
    ax_env.axhline(
        threshold * hysteresis, color="tab:orange", linestyle="--", linewidth=1,
        label="dropout recovery (hysteresis)",
    )
    ax_env.set_ylabel("carrier amplitude")
    ax_env.legend(loc="lower left", fontsize="small", ncol=2)
    # Which head wrote this field. The two differ measurably - in level, in the
    # response's slope, and in its shape - so the parity belongs on the plot.
    head = "first field (head A)" if is_first_field else "second field (head B)"
    ax_env.set_title(
        f"Luma carrier amplitude, the color-under correction it drives, "
        f"and the demodulated luma   [{head}]"
    )

    if ax_departure is not None:
        # What the carrier amplitude does that the PICTURE does not explain.
        # The amplitude leans across the field because the path responds to the
        # carrier's own frequency, and the carrier's frequency is the picture;
        # divide that out and what is left is the tape's, which is the whole
        # object the correction acts on. Unity means the carrier sits exactly
        # where its own frequency says it should.
        ax_departure.plot(samples, deviation_window, color="tab:red",
                          linewidth=0.5, label="amplitude with the picture removed")
        ax_departure.axhline(1.0, color="tab:grey", linestyle=":", linewidth=1)
        ax_departure.set_ylabel("deviation")
        ax_departure.legend(loc="lower left", fontsize="small")
        ax_departure.annotate(
            f"{100.0 * float(np.std(deviation_window)):.2f}% rms",
            xy=(0.995, 0.94), xycoords="axes fraction", ha="right", va="top",
            fontsize="x-small",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

    if noeq_window is not None:
        # Drawn first, so the equalized trace sits on top of it.
        ax_luma.plot(
            samples, noeq_window, color="tab:grey", linewidth=0.6, alpha=0.8,
            label="equalizer off",
        )
    ax_luma.plot(
        samples, luma_window, color="tab:green", linewidth=0.5,
        label="equalizer on" if noeq_window is not None else "demodulated luma",
    )
    for level in (0, 100):
        ax_luma.axhline(level, color="tab:grey", linestyle=":", linewidth=1)
    ax_luma.set_ylabel("luma (IRE)")
    ax_luma.legend(loc="lower left", fontsize="small")

    if ax_delta is not None:
        # What the equalizer did, on the same axis as the luma above. It acts
        # at transitions - on a carrier that is not sweeping the demodulator
        # discards an amplitude change entirely - so this should sit at zero
        # across flat content and lift at every edge.
        delta = luma_window - noeq_window
        ax_delta.plot(samples, delta, color="tab:red", linewidth=0.5)
        ax_delta.axhline(0.0, color="tab:grey", linewidth=0.8)
        ax_delta.set_ylabel("equalizer\ndelta (IRE)")
        ax_delta.annotate(
            f"{np.std(delta):.4f} IRE rms, peak {np.max(np.abs(delta)):.3f}",
            xy=(0.995, 0.94), xycoords="axes fraction", ha="right", va="top",
            fontsize="x-small",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

    if ax_beat is not None:
        # What the color-under put into the luma, and was taken back out. It is
        # drawn on the luma's own scale so its size can be read directly against
        # the picture above it.
        #
        # The envelope is the diagnostic rather than the waveform: the beat is a
        # carrier at the color-under's frequency, far too fine to resolve across
        # a field, and what carries the meaning is how tall it stands. It should
        # rise on saturated colour and fall to nothing where there is none, and
        # anywhere it stands tall over neutral content is the model reaching for
        # something that is not there.
        ax_beat.plot(beat_at, beat_window, color="tab:green", linewidth=0.4,
                     alpha=0.55, label="modelled color-under beat")
        envelope_span = max(len(beat_window) // 400, 8)
        lifted = np.abs(
            np.convolve(
                np.abs(beat_window), np.ones(envelope_span) / envelope_span, mode="same"
            )
        ) * (np.pi / 2.0)
        ax_beat.plot(beat_at, lifted, color="tab:red", linewidth=0.9,
                     label="its envelope")
        ax_beat.plot(beat_at, -lifted, color="tab:red", linewidth=0.9)
        ax_beat.axhline(0.0, color="tab:grey", linewidth=0.8)
        ax_beat.set_ylabel("beat removed\n(IRE)")
        ax_beat.legend(loc="lower left", fontsize="small", ncol=2)
        ax_beat.annotate(
            f"{np.std(beat_window):.4f} IRE rms, peak {np.max(np.abs(beat_window)):.3f}",
            xy=(0.995, 0.94), xycoords="axes fraction", ha="right", va="top",
            fontsize="x-small",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

    if head_switch_regions:
        # The switch as LOCATED on the field's own deviation. Its robust
        # signature is a sustained amplitude offset, which the phase kernel
        # nulls by design - so the applied trace can never show it, and the
        # marker must come from the measurement instead. Span and line on
        # the shared axes so the region reads against the envelope step
        # above and the luma beside it.
        for i, (r_start, r_end, _off, _sig) in enumerate(head_switch_regions):
            for ax in (ax_env, ax_departure, ax_luma, ax_switch):
                if ax is None:
                    continue
                ax.axvspan(r_start, r_end, color="tab:brown", alpha=0.15,
                           linewidth=0)
                ax.axvline(
                    r_start, color="tab:brown", linestyle="--", linewidth=1,
                    label="head switch" if (i == 0 and ax is ax_env) else None,
                )
        ax_env.legend(loc="lower left", fontsize="small")

    if ax_switch is not None:
        # What the correction subtracted, on the luma's own scale. Zero
        # wherever the residual sat inside the block's own background; what
        # stands in it is the transient side of the story, judged against
        # the located region above.
        ax_switch.plot(samples, switch_window, color="tab:red", linewidth=0.6,
                       label="head switch / dropout transient removed")
        ax_switch.axhline(0.0, color="tab:grey", linewidth=0.8)
        ax_switch.set_ylabel("switch removed\n(IRE)")
        ax_switch.legend(loc="lower left", fontsize="small")

        if head_switch_regions:
            # The switch is the LAST region positionally, not the strongest:
            # the strongest was measured to spread over ten lines while the
            # last stands within one (see `head_switch.locate`).
            r_start, r_end, off_db, sigma = head_switch_regions[-1]
            lines = np.searchsorted(np.asarray(linelocs, dtype=float),
                                    [r_start, r_end])
            found = (
                f"switch: lines {lines[0]}-{lines[1]}, {off_db:+.2f} dB, "
                f"{sigma:.0f} sigma; {len(head_switch_regions)} region(s), "
                f"format expects {head_switch_expected}"
            )
        elif head_switch_regions is not None:
            found = "no switch located on this field's deviation"
        else:
            found = f"format expects {head_switch_expected}/field"
        ax_switch.annotate(
            f"{found} | removed {np.std(switch_window):.4f} IRE rms, "
            f"peak {np.max(np.abs(switch_window)) if len(switch_window) else 0.0:.3f}",
            xy=(0.995, 0.94), xycoords="axes fraction", ha="right", va="top",
            fontsize="x-small",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

    if ax_wow is not None:
        # The per-line line-period deviation off the carrier's sync-edge
        # trace - the wow and flutter, drawn per head and never averaged.
        # Same colours and names as `plot_luma_averaging`, so a head keeps
        # its identity across the two plots.
        colours = {True: "tab:blue", False: "tab:red"}
        names = {True: "head A", False: "head B"}
        head = bool(wow_trace["head"])
        other = wow_trace.get("other_head")
        if other is not None:
            other = np.asarray(other, dtype=float)
            ax_wow.plot(
                np.arange(len(other)), other, color=colours[not head],
                linewidth=0.8, alpha=0.55,
                label=f"{names[not head]} (previous field)",
            )
        current = np.asarray(wow_trace["deviation"], dtype=float)
        ax_wow.plot(
            np.arange(len(current)), current, color=colours[head],
            linewidth=0.9, label=f"{names[head]} (this field)",
        )
        ax_wow.axhline(0.0, color="tab:grey", linewidth=0.8)
        ax_wow.set_ylabel("line period\ndeviation (%)")
        ax_wow.set_xlabel("line number")
        ax_wow.legend(loc="lower left", fontsize="small", ncol=2)
        finite = current[np.isfinite(current)]
        if len(finite):
            ax_wow.annotate(
                f"{np.std(finite):.4f}% rms over {len(finite)} lines",
                xy=(0.995, 0.94), xycoords="axes fraction", ha="right", va="top",
                fontsize="x-small",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
            )

    if ax_eye is not None:
        # Off and on over one line time, the difference magnified so a
        # fraction of an IRE reads. Two views of the same fold: the sync
        # interval - pulse, porches and burst, the first fifth of the line,
        # where the equalizer's low-frequency phase shows first - beside
        # the whole line.
        on, off, period_us = eye
        t = np.arange(len(on)) * (period_us / len(on))
        change = on - off
        spec = ax_eye.get_subplotspec()
        ax_eye.remove()
        halves = spec.subgridspec(1, 2, width_ratios=[2, 3], wspace=0.12)
        ax_sync = fig.add_subplot(halves[0])
        ax_line = fig.add_subplot(halves[1])
        for ax in (ax_sync, ax_line):
            ax.plot(t, off, color="tab:grey", linewidth=0.8, label="decoded line, equalizer off")
            ax.plot(t, on, color="tab:blue", linewidth=0.8, label="equalizer on")
            ax.plot(t, change * 10.0, color="tab:green", linewidth=0.7, label="(on - off) x10")
            ax.axhline(0.0, color="tab:grey", linewidth=0.6, linestyle=":")
            ax.set_xlabel("microseconds from the line start")
        sync = t <= period_us / 5.0
        shown = np.concatenate((off[sync], on[sync], change[sync] * 10.0))
        pad = 0.1 * (np.max(shown) - np.min(shown) or 1.0)
        ax_sync.set_xlim(0.0, period_us / 5.0)
        ax_sync.set_ylim(np.min(shown) - pad, np.max(shown) + pad)
        ax_sync.set_ylabel("folded line\n(IRE)")
        ax_sync.set_title("sync interval", fontsize="small", loc="left")
        ax_line.set_title("whole line", fontsize="small", loc="left")
        ax_line.legend(loc="lower right", fontsize="small", ncol=3)
        ax_line.annotate(
            f"equalizer changed {np.std(change):.3f} IRE rms over the folded line, "
            f"peak {np.max(np.abs(change)):.3f}",
            xy=(0.995, 0.94), xycoords="axes fraction", ha="right", va="top",
            fontsize="x-small",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

    ax_conf.plot(samples, confidence, color="tab:purple", linewidth=0.5)
    ax_conf.set_ylim(0, 1.05)
    ax_conf.set_ylabel("chroma correction\nconfidence")

    if ax_resp is not None:
        # Both right hand panels are against the carrier's own FREQUENCY. The
        # response is a function of frequency and of nothing else, and the luma
        # level axis these used to carry is that same axis scaled - the carrier
        # is linear in level - so nothing is lost and the quantity is read where
        # it lives.
        carrier_window = np.asarray(carrier_hz, dtype=float)[start_rf:end_rf]

        # The format's own reference levels, on the anchor the response model
        # is binned against - the SPECIFIED carrier frequencies, not the levels
        # this decode has tracked. `luma_amplitude` anchors there deliberately,
        # and on a tape decoded with `--ire0_adjust` the two have been measured
        # 36 IRE apart, so marking the running ones here would put sync tip in a
        # different place from where the model put it.
        if ire_to_hz is None:
            # `hz_to_ire` is affine, so two points invert it exactly.
            f_low, f_high = 3.4e6, 4.4e6
            ire_low, ire_high = hz_to_ire(f_low), hz_to_ire(f_high)
            slope_hz = (f_high - f_low) / (ire_high - ire_low)

            def ire_to_hz(ire):
                return f_low + (ire - ire_low) * slope_hz

        hz_per_ire = ire_to_hz(1.0) - ire_to_hz(0.0)

        # Binned at twice the module's own resolution - it describes the curve
        # at one point per IRE, and two at a time leaves each bin enough samples
        # to take a median of.
        edges = np.arange(ire_to_hz(-40.0), ire_to_hz(122.0), 2.0 * hz_per_ire)
        index = np.digitize(carrier_window, edges) - 1
        centres, before, after, spread, noise, counts = [], [], [], [], [], []
        for b in range(len(edges) - 1):
            sel = index == b
            if np.count_nonzero(sel) < 64:
                continue
            centres.append((edges[b] + edges[b + 1]) / 2.0)
            counts.append(np.count_nonzero(sel))
            before.append(np.median(window[sel]) / reference)
            after.append(np.median(deviation_window[sel]))
            lo, hi = np.percentile(deviation_window[sel], (25, 75))
            spread.append((lo, hi))
            # robust, because dropouts sit in the low tail and would otherwise
            # set the scale
            here = deviation_window[sel]
            noise.append(residual_floor.robust_sigma(here))

        centres = np.asarray(centres)
        before, after = np.asarray(before), np.asarray(after)
        counts = np.asarray(counts, dtype=float)
        # The standard error of each bin's own median - R5's DECLARED FLOOR,
        # now taken from `residual_floor` where the two factors are derived
        # from the normal distribution and tested, rather than typed here.
        # It was implemented twice in this tree with no test holding either
        # to the relation, which is the wrong place for the rule that
        # decides when the algorithm has finished.
        #
        # Bins differ in population by a factor of a hundred here, so a trace
        # drawn without this reads its thinnest bin as response when it is
        # sampling: measured, the best sampled third of the bins is flat to
        # 0.26% rms where the thinnest third swings 7.5% peak to peak.
        error = (residual_floor.MEDIAN_STANDARD_ERROR_FACTOR
                 * np.asarray(noise) / np.sqrt(np.maximum(counts, 1.0)))
        model_total = model_line = after_line = None
        if len(centres):
            mhz = centres / 1e6
            spread = np.array(spread)

            # The whole model that was divided out, straight from the two
            # measured traces and needing nothing else to state it: `before` is
            # A(f)/median and `after` is A(f)/model(f), so their ratio is
            # model(f)/median directly. Nothing further is normalised into it -
            # the point of drawing it is that the measured trace divided by this
            # one IS the corrected trace, and any extra scaling here would break
            # that relationship and hide where the model departs from the
            # amplitude it is supposed to describe.
            model_total = before / np.maximum(after, 1e-9)
            roll_off = (
                f"{response.separation[2] * 1e6:+.2f}/MHz"
                if response is not None else "fitted"
            )

            # The roll-off's share of that model. Inside the described range
            # the model IS the measurement, so the dense term is the departure
            # itself; dividing it out leaves the line alone, on the same
            # reference as everything else.
            model_line = model_total.copy()
            model_db = None
            if response is not None and len(getattr(response, "described", ())):
                from vhsdecode.luma_amplitude import _LOG_TO_DB

                described_hz = np.asarray(response.described_hz, dtype=float)
                centre_hz, level_at_centre, slope = response.separation
                line = level_at_centre + slope * (described_hz - centre_hz)
                described = np.asarray(response.described, dtype=float)
                # The model in force. Inside the described range it IS the
                # measurement - nothing is held back - so the two traces below
                # coincide there by design, and where they part is the edge of
                # the range, past which the line takes over.
                departure = described - line
                dense = departure
                model_db = _LOG_TO_DB * dense
                inside = (centres >= described_hz[0]) & (centres <= described_hz[-1])
                model_line[inside] = model_total[inside] / np.exp(
                    np.interp(centres[inside], described_hz, dense)
                )
            after_line = before / np.maximum(model_line, 1e-9)

            ax_resp.plot(
                mhz, before, color="tab:blue", marker=".", markersize=4,
                label="measured luma frequency response",
            )
            ax_resp.plot(
                mhz, model_line, color="tab:blue", linestyle="--", linewidth=1.2,
                label=f"fitted roll-off ({roll_off})",
            )
            ax_resp.plot(
                mhz, model_total, color="tab:cyan", linewidth=1.4,
                # Named for what it is. This curve is the measured trace divided
                # by the corrected one, so it cannot disagree with them and is
                # not independent evidence that the model is right.
                label="model removed (= measured / corrected)",
            )
            ax_resp.plot(
                mhz, after, color="tab:red", marker=".", markersize=4,
                label="corrected frequency response",
            )
            ax_resp.fill_between(
                mhz, spread[:, 0], spread[:, 1], color="tab:red", alpha=0.13,
                linewidth=0, label="corrected, interquartile range",
            )
            if response is not None and len(getattr(response, "described", ())):
                # The dense term on its own axis, in dB, because what it
                # describes is the DEPARTURE from the response's straight line -
                # and because the measurement it is built from has the decoder's
                # own RF filter divided out of it, so it shares no reference
                # with the traces above and laying it over them would compare
                # two different quantities.
                #
                # What is drawn is the model the correction APPLIES, beside the
                # measurement it was built from and the part it left behind. An
                # echo fit stood here and was removed: it described the ripple
                # closely and was applied to nothing, which is exactly how a
                # panel comes to show a model working when it is not.
                if model_db is not None:
                    ripple_ax = ax_resp.twinx()
                    # ONE curve, because inside the described range the model and
                    # the measurement are the same array - `dense` is `departure`.
                    # Drawing both put an invisible trace under an identical one
                    # and invited the reading that they agreed about something.
                    # If a residual is ever wanted here it has to be computed;
                    # nothing in this function holds one.
                    ripple_ax.plot(
                        described_hz / 1e6, model_db,
                        color="tab:purple", linewidth=1.4,
                        label=f"measured departure = model applied, in full "
                              f"({np.std(model_db):.3f} dB rms)",
                    )
                    ripple_ax.set_ylabel("departure (dB)")
                    ripple_ax.legend(loc="lower left", fontsize="x-small")
            # THE EXPECTED RESPONSE, which is the quantity the residual is
            # a residual OF, and which the panel did not draw at all.
            #
            # Under the limit a field is refined against the model the
            # fields before it built, so the first field - the one drawn
            # here - has none of its own, and the model taken from its
            # `ResponseModel` is empty. What is wanted is the model the
            # decode converged on, which is the accumulated response for
            # this head, and that is what is passed in.
            if expected_response is not None:
                expected_curve, expected_dense = expected_response
                if expected_dense is not None and expected_curve is not None:
                    expected_hz, expected_ln, _ = expected_dense
                    centre_hz, level_at_centre, line_slope = expected_curve
                    expected_line = (level_at_centre
                                     + line_slope * (expected_hz - centre_hz))
                    shown = np.exp(expected_ln - expected_line)
                    scale = np.interp(expected_hz, centres,
                                      np.maximum(model_total, 1e-9)
                                      if model_total is not None
                                      else np.ones(len(centres)))
                    ax_resp.plot(
                        expected_hz / 1e6, shown * scale,
                        color="tab:green", linewidth=1.4, linestyle=(0, (4, 2)),
                        label="expected response (the model in force)",
                    )

            # BOTH HEADS. The two paths differ - in level, in slope and in
            # shape - so a panel drawing only the field's own head cannot
            # show the difference the equalizer is pooling over. Same
            # convention as the wow panel below: this field's head named and
            # solid, the other faint, so a head keeps its identity down the
            # whole figure.
            if other_head_response is not None:
                other_curve, other_dense = other_head_response
                this_name = "head A" if is_first_field else "head B"
                that_name = "head B" if is_first_field else "head A"
                that_colour = "tab:red" if is_first_field else "tab:blue"
                if other_dense is not None:
                    other_hz, other_log, _ = other_dense
                    # On the same reference as `before`: the accumulated log
                    # response, exponentiated and normalised at its own median,
                    # which is what `before` already is.
                    other_rel = np.exp(
                        np.asarray(other_log, dtype=float)
                        - np.median(np.asarray(other_log, dtype=float))
                    )
                    ax_resp.plot(
                        np.asarray(other_hz, dtype=float) / 1e6, other_rel,
                        color=that_colour, linewidth=1.0, alpha=0.55,
                        # The two heads' accumulated responses, drawn to be
                        # read against each other.
                        #
                        # CONTENT CANNOT MAKE THESE DIFFER, and the reason is
                        # the physics rather than an averaging argument: the
                        # response is a function of CARRIER FREQUENCY, and
                        # what the picture encodes decides which frequencies
                        # get visited and how often - the sampling - not the
                        # value at a frequency. So content changes how WELL
                        # each frequency is measured and never what is
                        # measured there. Corroborated where no picture
                        # reaches at all: the per-head gain on the sync tip,
                        # one fixed carrier and a constant envelope, reads
                        # +0.98 dB at 161 sigma.
                        #
                        # The honest caveat is about the SHAPE, not the
                        # existence. The departure carries measurement
                        # scatter from the sweep mix - the carrier arrives at
                        # a frequency by different trajectories depending on
                        # the picture, and the collapse compensates that only
                        # to within its product term, itself 0.058 to 0.088
                        # dB. The signature is in the bands: across two
                        # contents the response reproduces at r = +0.607 in
                        # the sync band, where the carrier is nearly
                        # stationary, and only +0.347 over the picture range
                        # where the sweep mix is widest.
                        label=f"{that_name} accumulated response",
                    )
                ax_resp.annotate(
                    f"this field: {this_name}",
                    xy=(0.015, 0.965), xycoords="axes fraction",
                    fontsize="x-small", va="top", ha="left",
                    color="tab:blue" if is_first_field else "tab:red",
                    bbox={"boxstyle": "round", "facecolor": "white",
                          "alpha": 0.85},
                )

        ax_resp.axhline(1.0, color="tab:grey", linestyle=":", linewidth=1)
        ax_resp.set_ylabel("relative amplitude")
        ax_resp.set_title("Luma frequency response and the model removed",
                          fontsize="medium")
        # Sync tip and blanking, the format's own two fixed levels, so the
        # curves can be read against something that is defined rather than
        # pictorial.
        for level, name in ((-40.0, "sync tip"), (0.0, "blanking"), (100.0, "peak white")):
            ax_resp.axvline(ire_to_hz(level) / 1e6, color="tab:grey",
                            linestyle="-.", linewidth=1, alpha=0.7)
            ax_resp.annotate(name, xy=(ire_to_hz(level) / 1e6, 0.0),
                             xycoords=("data", "axes fraction"),
                             xytext=(2, 3), textcoords="offset points",
                             fontsize="x-small", color="tab:grey",
                             rotation=90, va="bottom", ha="left")
        # Room made for the legend rather than the legend laid over the traces:
        # every curve in here matters and one hidden behind a box is a curve
        # that gets misread.
        low, high = ax_resp.get_ylim()
        ax_resp.set_ylim(low, high + 0.42 * (high - low))
        ax_resp.legend(loc="upper right", fontsize="small", framealpha=0.95)
        ax_resp.grid(alpha=0.3)

    if ax_model is not None and len(centres):
        # What is left at each stage of the removal. Flat is the objective:
        # anything sloping here is response the model did not describe, and it
        # is corrected onto the color-under as though it were tape noise.
        #
        # THESE TWO LABELS WERE THE WRONG WAY ROUND, and the mistake mattered:
        # read off the old labels, the smoother trace looked like the full
        # model beating the line, which argues for dropping the line - the
        # opposite of the truth.
        #
        # The deviation divides by the LINE'S response only. The accumulated
        # table is deliberately withheld from it (`measure_amplitude_deviation`
        # passes `dense=None`, measured worth -0.016/-0.037/-0.041 per cent on
        # the three conditions), so `after` IS the fitted roll-off's own
        # result. And `after_line = after * exp(departure)` inside the
        # described range - the accumulated table's departure put BACK on,
        # which is a trace of what the correction would look like if the table
        # were not withheld. It is drawn because that comparison is the whole
        # argument for withholding it, not because it is applied to anything.
        mhz = centres / 1e6
        stages = (
            (before, "tab:grey", "uncorrected"),
            (after_line, "tab:orange",
             "with the accumulated table re-imposed (not applied)"),
            (after, "tab:red", "after the fitted roll-off (the model in force)"),
        )
        for values, colour, name in stages:
            if values is None:
                continue
            swing = float(np.ptp(values))
            # Peak to peak is set by the single worst bin, which is reliably the
            # emptiest one; the population weighted rms is what the response
            # actually is.
            centred = values - np.average(values, weights=counts)
            weighted = float(np.sqrt(np.average(centred ** 2, weights=counts)))
            ax_model.plot(mhz, values, color=colour, linewidth=1.2, marker=".",
                          markersize=3,
                          label=f"{name} ({swing * 100:.1f}% p-p, "
                                f"{weighted * 100:.2f}% rms by evidence)")
        if after is not None and len(error) == len(after):
            ax_model.fill_between(
                mhz, after - error, after + error, color="tab:red", alpha=0.18,
                linewidth=0, label="corrected, one standard error of the median",
            )
        ax_model.axhline(1.0, color="tab:grey", linestyle=":", linewidth=1)
        ax_model.set_ylabel("deviation")

        # Beside it, how noisy the measurement is at each frequency - a property
        # of the path rather than of the picture, which is what makes it safe to
        # read where the residual mean is not.
        twin = ax_model.twinx()
        twin.plot(mhz, np.asarray(noise) * 8.686, color="tab:purple",
                  linewidth=0.9, alpha=0.8, label="measurement noise")
        twin.set_ylabel("noise (dB)")
        twin.set_ylim(bottom=0.0)
        tlow, thigh = twin.get_ylim()
        twin.set_ylim(tlow, thigh + 0.42 * (thigh - tlow))

        handles, labels = ax_model.get_legend_handles_labels()
        extra = twin.get_legend_handles_labels()
        ax_model.legend(handles + extra[0], labels + extra[1],
                        loc="upper left", fontsize="small", framealpha=0.95)
        low, high = ax_model.get_ylim()
        ax_model.set_ylim(low, high + 0.30 * (high - low))
        for level in (-40.0, 0.0, 100.0):
            ax_model.axvline(ire_to_hz(level) / 1e6, color="tab:grey",
                             linestyle="-.", linewidth=1, alpha=0.7)
        ax_model.set_xlabel("carrier frequency (MHz)")
        ax_model.set_title("Deviation from the instantaneous frequency",
                           fontsize="medium")
        ax_model.grid(alpha=0.3)

    for start, end in dropouts or []:
        if end < 0:
            end = end_rf
        for ax in (
            ax_corr, ax_env, ax_departure, ax_luma, ax_beat, ax_switch,
            ax_delta, ax_conf,
        ):
            if ax is not None:
                ax.axvspan(start, end, color="tab:red", alpha=0.3, linewidth=0)

    ax_env.set_xlim(start_rf, end_rf)
    fig.tight_layout()
    plt.show()


def _ideal_line(t_us, sys_params, porch, tip):
    """The specified line at the measured levels, where it is specified.

    Time from the sync fall's 50% crossing. The sync pulse is an erf-edged
    pulse of the specified width and 10-90% transition time from the porch
    level to the tip level; the back porch runs to the start of active
    video and the front porch fills the specified width before the next
    fall. Returns the ideal and a mask of where it is defined - the active
    picture is content, not specification. Levels are the field's own
    measured porch and tip, so the shape is judged, not the recorder's
    level setting (which is a separate finding).
    """
    import numpy as np
    from scipy.special import erf

    width = float(sys_params["hsyncPulseUS"])
    transition = float(sys_params["syncTransitionUS"])
    # The 10-90% width of an error-function edge is 2.563 sigma.
    sigma = max(transition, 1e-3) / 2.563
    active_start = float(sys_params["activeVideoUS"][0])
    period = float(sys_params["line_period"])
    front = float(sys_params["frontPorchUS"])
    edge = lambda t0: 0.5 * (1.0 + erf((t_us - t0) / (np.sqrt(2.0) * sigma)))
    pulse = edge(0.0) - edge(width)
    ideal = porch + (tip - porch) * pulse
    defined = (t_us < active_start) | (t_us >= period - front)
    return ideal, defined


def plot_sync_step_fold(demod, change, linelocs, start_rf, end_rf, hz_to_ire, sys_params,
                        rate_hz, is_first_field, show=True):
    """The decoded line folded over the field's lines against the
    level-adjusted specified sync pulse, equalizer off and on.

    Top: the sync interval - the fold with the equalizer off and on, and
    the ideal at the field's own measured porch and tip levels. Middle: the
    residuals against that ideal wherever it is specified (sync pulse,
    back porch to active video, front porch), with the equalizer's change
    magnified. Bottom: the whole folded line. The residual after correction
    is the quantity the equalizer's closed loop refines on: measured shape
    against the expected shape.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    signal = hz_to_ire(np.asarray(demod, dtype=float))
    delta = (
        np.asarray(change, dtype=float) if change is not None else np.zeros(len(signal))
    )
    fold = _fold_lines(signal, delta, np.asarray(linelocs, dtype=float), start_rf, end_rf, rate_hz)
    if fold is None:
        return
    on, off, period_us = fold
    t = np.arange(len(on)) * (period_us / len(on))
    width = float(sys_params["hsyncPulseUS"])
    transition = float(sys_params["syncTransitionUS"])
    active_start = float(sys_params["activeVideoUS"][0])
    # Levels from the fold itself: tip over the pulse interior, porch over
    # the back porch clear of both edges and of the active picture.
    tip_zone = (t > 4 * transition) & (t < width - 4 * transition)
    porch_zone = (t > width + 4 * transition) & (t < active_start - 4 * transition)
    levels = {}
    for name, line in (("off", off), ("on", on)):
        levels[name] = (float(np.median(line[porch_zone])), float(np.median(line[tip_zone])))
    ideal_off, defined = _ideal_line(t, sys_params, *levels["off"])
    ideal_on, _ = _ideal_line(t, sys_params, *levels["on"])
    spec_depth = -float(sys_params["vsync_ire"])

    fig, (ax_sync, ax_resid, ax_line) = plt.subplots(
        3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 2, 2], "hspace": 0.28}
    )
    head = "first field (head A)" if is_first_field else "second field (head B)"
    fig.suptitle(f"Sync pulse: the folded decoded line against the level-adjusted specification   [{head}]")

    sync = t < active_start + 1.0
    ax_sync.plot(t[sync], off[sync], color="tab:grey", linewidth=1.0, label="decoded line, equalizer off")
    ax_sync.plot(t[sync], on[sync], color="tab:blue", linewidth=1.0, label="equalizer on")
    ax_sync.plot(t[sync], ideal_on[sync], color="tab:green", linewidth=1.0, linestyle="--",
                 label="specified pulse at the measured levels")
    ax_sync.axhline(levels["on"][0], color="tab:grey", linewidth=0.5, linestyle=":")
    ax_sync.set_ylabel("IRE")
    ax_sync.legend(loc="lower right", fontsize="small")
    ax_sync.annotate(
        f"tip depth below porch: off {levels['off'][0] - levels['off'][1]:.2f}, "
        f"on {levels['on'][0] - levels['on'][1]:.2f}, spec {spec_depth:.0f} IRE",
        xy=(0.01, 0.94), xycoords="axes fraction", ha="left", va="top", fontsize="x-small",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    resid_off = np.where(defined, off - ideal_off, np.nan)
    resid_on = np.where(defined, on - ideal_on, np.nan)
    ax_resid.plot(t[sync], resid_off[sync], color="tab:grey", linewidth=0.9, label="residual, equalizer off")
    ax_resid.plot(t[sync], resid_on[sync], color="tab:blue", linewidth=0.9, label="residual, equalizer on")
    ax_resid.plot(t[sync], (on - off)[sync] * 10.0, color="tab:green", linewidth=0.7, label="(on - off) x10")
    ax_resid.axhline(0.0, color="tab:grey", linewidth=0.6)
    ax_resid.set_ylabel("residual against\nthe specification (IRE)")
    ax_resid.set_xlabel("microseconds from the sync fall")
    ax_resid.legend(loc="lower right", fontsize="small", ncol=3)
    good = defined & np.isfinite(resid_off) & np.isfinite(resid_on)
    ax_resid.annotate(
        f"rms over the specified intervals: off {np.sqrt(np.mean(resid_off[good] ** 2)):.3f}, "
        f"on {np.sqrt(np.mean(resid_on[good] ** 2)):.3f} IRE",
        xy=(0.01, 0.94), xycoords="axes fraction", ha="left", va="top", fontsize="x-small",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    ax_line.plot(t, off, color="tab:grey", linewidth=0.8, label="equalizer off")
    ax_line.plot(t, on, color="tab:blue", linewidth=0.8, label="equalizer on")
    ax_line.plot(t, (on - off) * 10.0, color="tab:green", linewidth=0.7, label="(on - off) x10")
    ax_line.axhline(0.0, color="tab:grey", linewidth=0.6, linestyle=":")
    ax_line.set_ylabel("whole line (IRE)")
    ax_line.set_xlabel("microseconds from the sync fall")
    ax_line.legend(loc="lower right", fontsize="small", ncol=3)
    if show:
        plt.show()


def plot_luma_averaging(probe, dod_threshold_p, source="", show=True):
    """What the response model's averaging is actually doing, per head.

    Six panels, one per open question, all measured with the correction running
    unchanged - nothing here feeds back into the decode. Read together they say
    whether the accumulation's assumptions hold on the material being decoded
    rather than on the material they were designed against.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    log_to_db = 20.0 / np.log(10.0)
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    figure.suptitle(
        f"Response model averaging{(' - ' + source) if source else ''}"
        f"   (dropout threshold {dod_threshold_p:g})"
    )
    colours = {True: "tab:blue", False: "tab:red"}
    names = {True: "head A", False: "head B"}

    # 1. How much evidence stands behind each described bin. A graded shrinkage
    #    was drawn here and is gone with the blend it measured - admission is
    #    the only gate now, so what matters is which bins clear it and by how
    #    far, not a share that is always one.
    ax = axes[0][0]
    from vhsdecode.luma_amplitude import CURVE_MINIMUM_POPULATION

    for head, state in sorted(probe["luma"].items()):
        if "evidence" not in state:
            continue
        ax.semilogy(state["evidence_hz"] / 1e6, np.maximum(state["evidence"], 1.0),
                    color=colours[head], linewidth=1.1,
                    label=f"{names[head]} ({state['fields']} fields)")
    ax.axhline(CURVE_MINIMUM_POPULATION, color="tab:grey", linestyle=":", linewidth=1)
    ax.annotate("admission gate", xy=(0.02, CURVE_MINIMUM_POPULATION),
                xycoords=("axes fraction", "data"),
                fontsize="x-small", color="tab:grey", va="bottom")
    ax.set_xlabel("carrier frequency (MHz)")
    ax.set_ylabel("samples accumulated")
    ax.set_title("1. Evidence behind each described bin")
    ax.legend(fontsize="x-small")
    ax.grid(alpha=0.3)

    # 2. The step the model takes where the described range ends, against the
    #    deviation's own spread - the scale at which a step matters at all.
    ax = axes[0][1]
    for head, state in sorted(probe["luma"].items()):
        if not state["edge_low_db"]:
            continue
        deviation = float(np.mean(state["deviation_db"]))
        for values, style, edge in ((state["edge_low_db"], "-", "low"),
                                    (state["edge_high_db"], "--", "high")):
            ax.plot(values, style, color=colours[head], linewidth=1.1,
                    label=f"{names[head]} {edge} edge")
        ax.axhline(deviation, color=colours[head], linestyle=":", linewidth=1)
        ax.axhline(-deviation, color=colours[head], linestyle=":", linewidth=1)
    ax.axhline(0.0, color="tab:grey", linewidth=0.8)
    ax.set_xlabel("field")
    ax.set_ylabel("step (dB)")
    ax.set_title("2. Step at the edges of the described range\n(dotted = that head's deviation rms)")
    ax.legend(fontsize="x-small")
    ax.grid(alpha=0.3)

    # 3. Stationary or drifting. A bin's field-to-field difference is a moving
    #    average of order one under a random walk plus measurement noise, so the
    #    lag-one term separates the walk from the noise where a variance cannot.
    ax = axes[0][2]
    for head, state in sorted(probe["luma"].items()):
        count = state["difference_count"]
        seen = count >= 3
        if not seen.any():
            continue
        variance = state["difference_square"][seen] / count[seen]
        lag = state["difference_lag"][seen] / np.maximum(count[seen] - 1, 1)
        noise = np.maximum(-lag, 0.0)
        walk = np.maximum(variance - 2.0 * noise, 0.0)
        ax.scatter(log_to_db * np.sqrt(noise), log_to_db * np.sqrt(walk),
                   s=8, alpha=0.5, color=colours[head],
                   label=f"{names[head]} ({int(seen.sum())} bins)")
    limit = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, limit], [0, limit], color="tab:grey", linestyle=":", linewidth=1)
    ax.annotate("walk = noise", xy=(limit * 0.55, limit * 0.6), fontsize="x-small",
                color="tab:grey", rotation=45)
    ax.set_xlabel("measurement noise per field (dB)")
    ax.set_ylabel("drift step per field (dB)")
    ax.set_title("3. Drift against noise, per bin\n(above the line: an unbounded mean is wrong)")
    ax.legend(fontsize="x-small")
    ax.grid(alpha=0.3)

    # 4. Whether the deviation ever reaches the bounds the dropout threshold
    #    sets for it.
    ax = axes[1][0]
    width = 0.35
    for offset, (head, state) in enumerate(sorted(probe["luma"].items())):
        low = 100.0 * float(np.mean(state["clamp_low"]))
        high = 100.0 * float(np.mean(state["clamp_high"]))
        # Drawn as two bars rather than one call with a list of alphas -
        # matplotlib takes a scalar there, so `[0.9, 0.45][0]` silently gave
        # both bars the same shade and the high bar was then redrawn over
        # itself to correct it.
        ax.bar([offset - width / 2], [low], width, color=colours[head], alpha=0.9)
        ax.bar([offset + width / 2], [high], width, color=colours[head], alpha=0.45)
        ax.annotate(f"{low:.4f}%", xy=(offset - width / 2, low), ha="center",
                    va="bottom", fontsize="x-small")
        ax.annotate(f"{high:.4f}%", xy=(offset + width / 2, high), ha="center",
                    va="bottom", fontsize="x-small")
    ax.set_xticks(range(len(probe["luma"])))
    ax.set_xticklabels([f"{names[h]}\nlow / high" for h, _ in sorted(probe["luma"].items())],
                       fontsize="x-small")
    ax.set_ylabel("samples at the clamp (%)")
    ax.set_title(f"4. Deviation clamp [{dod_threshold_p:g}, {1.0 / dod_threshold_p:.2f}]")
    ax.grid(alpha=0.3, axis="y")

    # 5. The amount the regression measures, against the amount the track-width
    #    model asks for. Unity means the two agree.
    ax = axes[1][1]
    for head, records in sorted(probe["chroma"].items()):
        level = np.array([r["ungated_level"] for r in records])
        variance = np.array([r["ungated_var"] for r in records])
        model = np.array([r["model_exponent"] for r in records])
        good = np.isfinite(level) & np.isfinite(variance) & (variance > 0.0)
        if not good.any():
            continue
        amount = level[good] / (2.0 * model[good])
        error = np.sqrt(variance[good]) / (2.0 * model[good])
        ax.errorbar(np.arange(good.sum()), amount, yerr=error, fmt="o", markersize=3,
                    color=colours[head], alpha=0.7, linewidth=0.8,
                    label=f"{names[head]}")
        precision = 1.0 / variance[good]
        pooled = float((level[good] * precision).sum() / precision.sum()) / (
            2.0 * float(np.mean(model[good]))
        )
        ax.axhline(pooled, color=colours[head], linestyle="--", linewidth=1.1)
        ax.annotate(f"{pooled:.3f}", xy=(0.99, pooled), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize="x-small", color=colours[head])
    ax.axhline(1.0, color="tab:grey", linestyle=":", linewidth=1)
    ax.annotate("what the model applies", xy=(0.02, 1.0), xycoords=("axes fraction", "data"),
                fontsize="x-small", color="tab:grey", va="bottom")
    ax.set_xlabel("measurement")
    ax.set_ylabel("measured amount / model amount")
    ax.set_title("5. The amount, measured against the model")
    ax.legend(fontsize="x-small")
    ax.grid(alpha=0.3)

    # 6. How often the transfer resolves at all. Where it does not, the shaping
    #    falls back to no roll-off and the whole amount is applied wideband.
    ax = axes[1][2]
    for offset, (head, records) in enumerate(sorted(probe["chroma"].items())):
        resolved = 100.0 * float(np.mean([r["resolved"] for r in records]))
        passing = float(np.mean([r["passing"] for r in records]))
        measurable = float(np.mean([r["measurable"] for r in records]))
        ax.bar([offset - width / 2], [resolved], width, color=colours[head],
               label=f"{names[head]} resolved")
        ax.bar([offset + width / 2], [100.0 * passing / max(measurable, 1.0)], width,
               color=colours[head], alpha=0.45)
        ax.annotate(f"{resolved:.0f}%", xy=(offset - width / 2, resolved), ha="center",
                    va="bottom", fontsize="x-small")
        ax.annotate(f"{passing:.1f}/{measurable:.0f}\nbands",
                    xy=(offset + width / 2, 100.0 * passing / max(measurable, 1.0)),
                    ha="center", va="bottom", fontsize="x-small")
    ax.set_xticks(range(len(probe["chroma"])))
    ax.set_xticklabels([f"{names[h]}\nresolved / bands" for h, _ in sorted(probe["chroma"].items())],
                       fontsize="x-small")
    ax.set_ylabel("per cent")
    ax.set_ylim(0, 110)
    ax.set_title("6. Transfer resolution\n(0% = unshaped fallback every field)")
    ax.grid(alpha=0.3, axis="y")

    figure.tight_layout()
    if show:
        plt.show()
    return figure


def plot_single_transform(picture, rf, title: str = ""):
    """The two transforms as one figure: every contrast against its floor,
    the causal split, the remainders, and the wave against the null space.

    Ethan, 2026-09-07: "I do want to see what the comparison is between the
    wave and null space." Six panels, all read from the readings the two
    stages attach to the field (`field.picture_transform`,
    `field.rf_transform`); nothing here feeds back into the decode. A
    reading that is absent leaves its panel labelled so, rather than empty.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(17, 9))
    figure.suptitle(("The single transform" + (" - " + title if title else "")))
    readings = {"picture": picture, "radio frequency": rf}

    def report_of(reading):
        return (reading or {}).get("report") or {}

    # 1 and 2: the contrasts of each transform, magnitude rms per channel,
    # the kept ones in colour and the amount applied written beside each
    for column, (name, reading) in enumerate(readings.items()):
        ax = axes[0][column]
        report = report_of(reading)
        contrasts = report.get("contrasts") or {}
        if not contrasts:
            ax.set_title("%s: no latched contrasts yet" % name)
            ax.axis("off")
            continue
        kept = set(report.get("kept") or [])
        names = list(contrasts)
        channels = [c for c in report.get("channels", {})]
        width = 0.8 / max(len(channels), 1)
        for k, channel in enumerate(channels):
            values = [contrasts[n].get(channel, {}).get("magnitude_rms", 0.0)
                      for n in names]
            ax.bar(np.arange(len(names)) + k * width, values, width,
                   label=channel)
        for i, n in enumerate(names):
            amount = np.mean([contrasts[n].get(c, {}).get("amount", 1.0)
                              for c in channels]) if channels else 1.0
            ax.text(i + 0.4, 0, "%s\n%.2f" % ("kept" if (n in kept or n == "mean")
                                              else "null", amount),
                    ha="center", va="bottom", fontsize=7)
        ax.set_xticks(np.arange(len(names)) + 0.4)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("magnitude rms (channel units)")
        ax.set_title("%s: contrasts (%s)" % (
            name, (report.get("latched") or {}).get("treatment", "unlatched")))
        ax.legend(fontsize=7)

    # 3: the floors, both transforms, in each channel's own power
    ax = axes[0][2]
    rows, labels = [], []
    for name, reading in readings.items():
        floors = report_of(reading).get("floors") or {}
        for channel, entry in floors.items():
            if not entry.get("used"):
                continue
            rows.append([entry.get(k, float("nan")) for k in
                         ("instrument", "banks", "null_pool", "hypercube")])
            labels.append("%s %s" % (name[:3], channel))
    if rows:
        rows = np.asarray(rows, dtype=float)
        x = np.arange(len(labels))
        for k, key in enumerate(("instrument", "banks", "null_pool", "hypercube")):
            ax.bar(x + 0.2 * k, np.where(rows[:, k] > 0, rows[:, k], np.nan),
                   0.2, label=key)
        ax.set_yscale("log")
        ax.set_xticks(x + 0.3)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.legend(fontsize=7)
        ax.set_title("the three floors, and the instrument's own")
    else:
        ax.set_title("floors: not yet measurable")
        ax.axis("off")

    # 4: the causal split of every latched contrast
    ax = axes[1][0]
    drawn = False
    for name, reading in readings.items():
        contrasts = report_of(reading).get("contrasts") or {}
        for n, entry in contrasts.items():
            causal = entry.get("causal")
            if not causal:
                continue
            ax.scatter(1e9 * causal.get("delay_s", 0.0), causal.get("allpass_rms", 0.0),
                       label="%s %s (min-phase share %.2f)" % (
                           name[:3], n, causal.get("minimum_phase_share", float("nan"))))
            drawn = True
    if drawn:
        ax.set_xlabel("delay, ns")
        ax.set_ylabel("all-pass, rad rms")
        ax.legend(fontsize=6)
        ax.set_title("the causal split per contrast")
    else:
        ax.set_title("causal split: no response contrasts latched")
        ax.axis("off")

    # 5: this field's remainders under each treatment, before and after
    ax = axes[1][1]
    drawn = False
    for name, reading in readings.items():
        remainders = (reading or {}).get("remainders") or {}
        for treatment, rows in remainders.items():
            if not isinstance(rows, dict):
                continue
            for key, entry in rows.items():
                if not isinstance(entry, dict) or "before" not in entry:
                    continue
                after = entry.get("exact_inverse", entry.get("after"))
                applied = entry.get("applied")
                ax.plot([0, 1, 2], [entry["before"], applied if applied is not None
                                     else np.nan, after],
                        marker="o", label="%s %s %s" % (name[:3], treatment, key))
                drawn = True
    if drawn:
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["before", "applied", "exact inverse"])
        ax.set_yscale("log")
        ax.legend(fontsize=6)
        ax.set_title("this field's remainder")
    else:
        ax.set_title("remainders: nothing latched yet")
        ax.axis("off")

    # 6: the wave against the null space
    ax = axes[1][2]
    drawn = False
    for name, reading in readings.items():
        wave = (reading or {}).get("wave") or {}
        for vertex, entry in (wave.get("folded") or {}).items():
            comparison = entry.get("null_space_against_wave") or {}
            for channel, pair in comparison.items():
                ax.plot([0, 1], [pair["after_constant"], pair["after_wave"]],
                        marker="o", label="%s %s %s (admitted %s)" % (
                            name[:3], vertex, channel,
                            ", ".join(entry.get("admitted") or ["none"])))
                drawn = True
    if drawn:
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["after the constant (null space)", "after the wave"])
        ax.set_yscale("log")
        ax.legend(fontsize=6)
        ax.set_title("the wave against the null space, this field")
    else:
        ax.set_title("wave: no run folded yet")
        ax.axis("off")
    figure.tight_layout()
    return figure
