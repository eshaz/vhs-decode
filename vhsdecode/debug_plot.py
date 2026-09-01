import sys


def to_db_power(input_data):
    import numpy as np

    return 20 * np.log10(input_data)


class DebugPlot:
    def __init__(self, stuff_to_plot: str):
        self.__stuff_to_plot = stuff_to_plot.casefold().split()

    def is_plot_requested(self, requested_info: str):
        return requested_info in self.__stuff_to_plot


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
    """
    import matplotlib.pyplot as plt
    import numpy as np

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
    has_correction = correction_window is not None
    can_score = deviation_window is not None and carrier_hz is not None

    rows = 4 if has_correction else 3
    fig = plt.figure(figsize=(16, 9) if can_score else (14, 9))
    grid = fig.add_gridspec(
        rows,
        2 if can_score else 1,
        width_ratios=[3, 1] if can_score else [1],
        height_ratios=[2, 3, 3, 1] if has_correction else [3, 3, 1],
    )
    row = 0
    ax_corr = None
    if has_correction:
        ax_corr = fig.add_subplot(grid[row, 0])
        row += 1
    ax_env = fig.add_subplot(grid[row, 0], **({"sharex": ax_corr} if ax_corr else {}))
    anchor = ax_corr or ax_env
    ax_luma = fig.add_subplot(grid[row + 1, 0], sharex=anchor)
    ax_conf = fig.add_subplot(grid[row + 2, 0], sharex=anchor)
    if can_score:
        right = grid[:, 1].subgridspec(2, 1, height_ratios=[3, 2], hspace=0.35)
        ax_resp = fig.add_subplot(right[0])
        ax_model = fig.add_subplot(right[1])
    else:
        ax_resp = ax_model = None

    if ax_corr is not None:
        ax_corr.plot(
            samples, correction_window, color="tab:red", linewidth=0.5,
            label="correction applied to the color-under",
        )
        ax_corr.axhline(1.0, color="tab:grey", linestyle=":", linewidth=1)
        ax_corr.set_ylabel("chroma gain")
        ax_corr.legend(loc="lower left", fontsize="small")

    ax_env.plot(
        samples, window, color="tab:blue", linewidth=0.5, alpha=0.45,
        label="carrier amplitude (what the correction measures)",
    )
    if detection_window is not None:
        ax_env.plot(
            samples, detection_window, color="black", linewidth=0.7,
            label="band limited (what the thresholds act on)",
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
        label="recovery (hysteresis)",
    )
    ax_env.set_ylabel("carrier amplitude")
    ax_env.legend(loc="lower left", fontsize="small", ncol=2)
    # Which head wrote this field. The two differ measurably - in level, in the
    # response's slope, and in its shape - so the parity belongs on the plot.
    head = "first field (head A)" if is_first_field else "second field (head B)"
    (ax_corr or ax_env).set_title(
        f"Color-under correction, the luma carrier amplitude it came from, "
        f"detected dropouts, and the demodulated luma    [{head}]"
    )

    ax_luma.plot(
        samples, hz_to_ire(demod[start_rf:end_rf]), color="tab:green",
        linewidth=0.5, label="demodulated luma",
    )
    for level in (0, 100):
        ax_luma.axhline(level, color="tab:grey", linestyle=":", linewidth=1)
    ax_luma.set_ylabel("luma (IRE)")
    ax_luma.legend(loc="lower left", fontsize="small")

    ax_conf.plot(samples, confidence, color="tab:purple", linewidth=0.5)
    ax_conf.set_ylim(0, 1.05)
    ax_conf.set_ylabel("correction\nconfidence")
    ax_conf.set_xlabel("RF sample")

    if ax_resp is not None:
        # Both quantities against the luma level that produced them. The
        # response the correction removes shows as a lean in the amplitude;
        # what is left in the deviation is luma that survived the removal.
        carrier_window = np.asarray(carrier_hz, dtype=float)[start_rf:end_rf]
        level = hz_to_ire(carrier_window)
        edges = np.arange(-40.0, 122.0, 2.0)
        index = np.digitize(level, edges) - 1
        centres, before, after, spread = [], [], [], []
        bin_hz, noise = [], []
        for b in range(len(edges) - 1):
            sel = index == b
            if np.count_nonzero(sel) < 64:
                continue
            centres.append((edges[b] + edges[b + 1]) / 2.0)
            before.append(np.median(window[sel]) / reference)
            after.append(np.median(deviation_window[sel]))
            lo, hi = np.percentile(deviation_window[sel], (25, 75))
            spread.append((lo, hi))
            bin_hz.append(np.median(carrier_window[sel]))
            # robust, because dropouts sit in the low tail and would otherwise
            # set the scale
            here = deviation_window[sel]
            noise.append(np.median(np.abs(here - np.median(here))) * 1.4826)
        if centres:
            spread = np.array(spread)
            ax_resp.fill_between(
                centres, spread[:, 0], spread[:, 1], color="tab:red", alpha=0.15,
                linewidth=0, label="deviation, interquartile",
            )
            ax_resp.plot(
                centres, before, color="tab:blue", marker=".",
                label="carrier amplitude / median",
            )
            ax_resp.plot(
                centres, after, color="tab:red", marker=".",
                label="deviation (response removed)",
            )
            # The whole response that was actually divided out, straight from
            # the two traces above and needing no model to state: the decoder's
            # own RF path and the separation loss fitted on top of it. Each
            # curve is shown against its own mean, so what is comparable is the
            # shape rather than a level.
            # The model, on the SAME reference as the carrier amplitude above:
            # `before` is A(f)/median and `after` is A(f)/model(f), so their
            # ratio is model(f)/median directly. Nothing further is normalised
            # into it - the point of drawing it is that the blue trace divided
            # by this one IS the red trace, and any extra scaling here would
            # break that relationship and hide where the model departs from the
            # amplitude it is supposed to describe.
            removed_total = np.asarray(before) / np.maximum(np.asarray(after), 1e-9)
            ax_resp.plot(
                centres, removed_total, color="tab:cyan",
                linestyle="-", linewidth=1.2,
                label="response removed (blue / this = red)",
            )
            if response is not None and len(getattr(response, "described", ())):
                # What the model is actually fitted from: the steadiest samples
                # only, pooled per head across the decode. The blue trace above
                # is every sample, which is a different population - they are
                # not expected to agree where a level is only passed through.
                described_ire = hz_to_ire(np.asarray(response.described_hz, dtype=float))
                # Drawn as what it is fitted from DIVIDED BY what the model
                # made of it, so unity means the correction followed its own
                # measurement and a departure marks a frequency where something
                # else won. Its own scale would not answer that: this is the
                # FLATTENED response, with the decoder's RF filter already
                # divided out, so against blue and cyan - which still carry that
                # filter - it shares no reference and its shape is mostly the
                # filter's, not the path's.
                #
                # Every point is `share * described + (1 - share) * line`, so
                # the ratio is exactly `exp((1 - share) * (described - line))`:
                # the line's pull, and nothing else, since inside this range the
                # model has no other term.
                from vhsdecode.luma_amplitude import CURVE_MINIMUM_POPULATION

                centre_hz, level_at_centre, slope = response.separation
                line = level_at_centre + slope * (
                    np.asarray(response.described_hz, dtype=float) - centre_hz
                )
                weight = np.asarray(response.described_weight, dtype=float)
                share = weight / (weight + CURVE_MINIMUM_POPULATION)
                followed = np.exp(
                    (1.0 - share) * (np.asarray(response.described, dtype=float) - line)
                )
                ax_resp.plot(
                    described_ire, followed,
                    color="tab:green", linewidth=1.1, alpha=0.8,
                    label="model / what it is fitted from (1 = followed)",
                )
            if response is not None:
                # And the fitted part of it on its own. Separation loss is
                # exp(-2 pi d / wavelength) with the recorded wavelength going
                # as 1/frequency, so the whole of it is a straight line in the
                # logarithm - and a line is all that survives contact with the
                # measurement, because anything freer starts fitting the
                # picture rather than the path.
                centre_hz, level_at_centre, slope = response.separation
                fitted = np.exp(slope * (np.asarray(bin_hz) - centre_hz))
                ax_resp.plot(
                    centres, fitted / fitted.mean(), color="tab:blue",
                    linestyle="--", linewidth=1.2,
                    label=f"separation loss fitted ({slope * 1e6:+.2f}/MHz)",
                )
        ax_resp.axhline(1.0, color="tab:grey", linestyle=":", linewidth=1)
        ax_resp.set_ylabel("relative amplitude")
        ax_resp.set_title(
            f"Response removal - {'head A' if is_first_field else 'head B'}"
        )
        # Sync tip and blanking, the format's own two fixed levels, so the
        # curves can be read against something that is defined rather than
        # pictorial.
        for level, name in ((-40.0, "sync tip"), (0.0, "blanking")):
            ax_resp.axvline(level, color="tab:grey", linestyle="-.", linewidth=1,
                            alpha=0.7)
            ax_resp.annotate(name, xy=(level, 0.0), xycoords=("data", "axes fraction"),
                             xytext=(2, 3), textcoords="offset points",
                             fontsize="x-small", color="tab:grey",
                             rotation=90, va="bottom", ha="left")
        if centres:
            # How much of the level-dependent swing the fit actually took out.
            # Peak to peak rather than a slope, because what is left is not
            # necessarily a straight line - a quadratic cannot follow every
            # shape the path has.
            swing_before = float(np.ptp(before))
            swing_after = float(np.ptp(after))
            removed = 1.0 - swing_after / swing_before if swing_before else 0.0
            ax_resp.annotate(
                f"level-dependent swing\n{swing_before * 100:.1f}% -> "
                f"{swing_after * 100:.1f}% p-p\n{removed * 100:.0f}% removed",
                xy=(0.03, 0.03),
                xycoords="axes fraction",
                fontsize="small",
                va="bottom",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )
        # Room made for the legend rather than the legend laid over the traces:
        # every curve in here matters and one hidden behind a box is a curve
        # that gets misread.
        low, high = ax_resp.get_ylim()
        ax_resp.set_ylim(low, high + 0.42 * (high - low))
        ax_resp.legend(loc="upper right", fontsize="small", framealpha=0.95)
        ax_resp.grid(alpha=0.3)

    if ax_model is not None:
        # Not what the correction leaves behind - that is the deviation, drawn
        # in red above. This is the departure of the measurement from the LINE,
        # which is exactly the part the dense fit supplies on top of it, so it
        # shows how much of the response a straight line would have missed.
        # Beside it, how noisy the measurement is at each level, which is a
        # property of the path rather than of the picture.
        if centres:
            noise_db = np.asarray(noise) * 8.686
            ax_model.plot(
                centres, noise_db, color="tab:purple", marker=".",
                label="noise (spread of the deviation) - follows the path",
            )
            ax_model.set_ylabel("noise (dB)")
            ax_model.set_ylim(bottom=0.0)
            if response is not None and len(response.residual):
                # Held to the levels the panel above covers; beyond them the
                # picture barely goes and a bin there is one excursion wide.
                residual_ire = hz_to_ire(
                    np.asarray(response.residual_hz, dtype=float)
                )
                inside = (residual_ire >= min(centres)) & (residual_ire <= max(centres))
                twin = ax_model.twinx()
                twin.plot(
                    residual_ire[inside],
                    np.asarray(response.residual, dtype=float)[inside] * 8.686,
                    color="tab:orange", linewidth=0.9,
                    label="what the dense fit adds to the line",
                )
                twin.axhline(0.0, color="tab:grey", linestyle=":", linewidth=1)
                twin.set_ylabel("residual (dB)")
                tlow, thigh = twin.get_ylim()
                twin.set_ylim(tlow, thigh + 0.42 * (thigh - tlow))
                handles = ax_model.get_legend_handles_labels()[0] + \
                    twin.get_legend_handles_labels()[0]
                labels = ax_model.get_legend_handles_labels()[1] + \
                    twin.get_legend_handles_labels()[1]
                ax_model.legend(handles, labels, loc="upper left", fontsize="small")
            else:
                ax_model.legend(loc="upper left", fontsize="small")
        low, high = ax_model.get_ylim()
        ax_model.set_ylim(low, high + 0.42 * (high - low))
        for level in (-40.0, 0.0):
            ax_model.axvline(level, color="tab:grey", linestyle="-.", linewidth=1,
                             alpha=0.7)
        ax_model.set_xlabel("luma level (IRE)")
        ax_model.set_title("What the line alone would miss", fontsize="medium")
        ax_model.grid(alpha=0.3)

    for start, end in dropouts or []:
        if end < 0:
            end = end_rf
        for ax in (ax_corr, ax_env, ax_luma, ax_conf):
            if ax is not None:
                ax.axvspan(start, end, color="tab:red", alpha=0.3, linewidth=0)

    ax_env.set_xlim(start_rf, end_rf)
    fig.tight_layout()
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

    # 1. The shrinkage in force. It cannot fall below a half: the range gate
    #    admits a bin only at the same population that sets the blend's knee.
    ax = axes[0][0]
    for head, state in sorted(probe["luma"].items()):
        if "share" not in state:
            continue
        ax.plot(state["share_hz"] / 1e6, state["share"], color=colours[head],
                linewidth=1.1, label=f"{names[head]} ({state['fields']} fields)")
    ax.axhline(0.5, color="tab:grey", linestyle=":", linewidth=1)
    ax.annotate("floor of the range gate", xy=(0.02, 0.5), xycoords=("axes fraction", "data"),
                fontsize="x-small", color="tab:grey", va="bottom")
    ax.set_xlabel("carrier frequency (MHz)")
    ax.set_ylabel("share of the measurement")
    ax.set_title("1. Shrinkage actually in force")
    ax.set_ylim(0.0, 1.05)
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
        ax.bar([offset - width / 2, offset + width / 2], [low, high], width,
               color=[colours[head], colours[head]], alpha=[0.9, 0.45][0])
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
