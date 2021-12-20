import sys
import signal
import traceback
import argparse
import lddecode.utils as lddu
from lddecode.utils_logging import init_logging
from vhsdecode.process import VHSDecode
from vhsdecode.cmdcommons import (
    common_parser,
    select_sample_freq,
    select_system,
    get_basics,
    get_rf_options,
    get_extra_options,
)

supported_tape_formats = {"VHS", "SVHS", "UMATIC"}


def main(use_gui=False):
    import vhsdecode.formats as f
    parser, debug_group = common_parser("Extracts video from raw VHS rf captures", use_gui=use_gui)
    if not use_gui:
        parser.add_argument(
            "--doDOD",
            dest="dodod",
            action="store_true",
            default=False,
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            "-U",
            "--u-matic",
            dest="umatic",
            action="store_true",
            default=False,
            help=argparse.SUPPRESS,
        )
    parser.add_argument(
        "-tf",
        "--tape_format",
        type=str.upper,
        metavar="tape_format",
        default="VHS",
        choices=supported_tape_formats,
        help="Tape format, currently VHS (Default), SVHS or UMATIC are supported.",
    )
    luma_group = parser.add_argument_group("Luma decoding options")
    luma_group.add_argument(
        "-L",
        "--level_adjust",
        dest="level_adjust",
        metavar="IRE Multiplier",
        type=float,
        default=0.1,
        help="Multiply top/bottom IRE in json by 1 +/- this value (used to avoid clipping on RGB conversion in chroma decoder).",
    )
    luma_group.add_argument(
        "--high_boost",
        metavar="High frequency boost multiplier",
        type=float,
        default=None,
        help="Multiplier for boost to high rf frequencies, uses default if not specified. Subject to change.",
    )
    luma_group.add_argument(
        "-nodd",
        "--no_diff_demod",
        dest="disable_diff_demod",
        action="store_true",
        default=False,
        help="Disable diff demod",
    )
    luma_group.add_argument(
        "-noclamp",
        "--no_clamping",
        dest="disable_dc_offset",
        action="store_true",
        default=False,
        help="Disable blanking DC offset clamping/compensation",
    )
    luma_group.add_argument(
        "-nld",
        "--non_linear_deemphasis",
        dest="nldeemp",
        action="store_true",
        default=False,
        help="Enable non-linear deemphasis, can help reduce ringing and oversharpening. (WIP).",
    )
    chroma_group = parser.add_argument_group("Chroma decoding options")
    chroma_group.add_argument(
        "-cafc",
        "--chroma_AFC",
        dest="cafc",
        action="store_true",
        default=False,
        help="Enable downconverted chroma carrier AFC (Automatic freq. control), implies --recheck_phase",
    )
    chroma_group.add_argument(
        "-T",
        "--track_phase",
        metavar="Track phase",
        type=int,
        default=None,
        help="If set to 0 or 1, force use of video track phase. (No effect on U-matic)",
    )
    chroma_group.add_argument(
        "--recheck_phase",
        dest="recheck_phase",
        action="store_true",
        default=False,
        help="Re-check chroma phase on every field. (No effect on U-matic)",
    )
    chroma_group.add_argument(
        "-nocomb",
        "--no_comb",
        dest="disable_comb",
        action="store_true",
        default=False,
        help="Disable internal chroma comb filter.",
    )
    plot_options = "demodblock"
    debug_group.add_argument(
        "-dp",
        "--debug_plot",
        dest="debug_plot",
        help="Do a plot for the requested data, separated by whitespace. Current options are: " + plot_options + "."
    )
    debug_group.add_argument(
        "-sclip",
        "--sync_clip",
        dest="sync_clip",
        action="store_true",
        default=False,
        help="Enables sync clipping",
    )
    dodgroup = parser.add_argument_group("Dropout detection options")
    dodgroup.add_argument(
        "--noDOD",
        dest="nodod",
        action="store_true",
        default=False,
        help="Disable dropout detector.",
    )
    dodgroup.add_argument(
        "-D",
        "--dod_t",
        "--dod_threshold_p",
        dest="dod_threshold_p",
        metavar="value",
        type=float,
        default=None,
        help="RF level fraction threshold for dropouts as percentage of average (in decimal).",
    )
    dodgroup.add_argument(
        "--dod_t_abs",
        "--dod_threshold_abs",
        dest="dod_threshold_a",
        metavar="value",
        type=float,
        default=None,
        help="RF level threshold absolute value. Note that RF levels vary greatly between tapes and recording setups.",
    )
    dodgroup.add_argument(
        "--dod_h",
        "--dod_hysteresis",
        dest="dod_hysteresis",
        metavar="value",
        type=float,
        default=f.DEFAULT_HYSTERESIS,
        help="Dropout detection hysteresis, the rf level needs to go above the dropout threshold multiplied by this for a dropout to end.",
    )

    args = parser.parse_args()

    filename, outname, firstframe, req_frames = get_basics(args)

    system = select_system(args)
    sample_freq = select_sample_freq(args)

    try:
        loader = lddu.make_loader(filename, sample_freq)
    except ValueError as e:
        print(e)
        exit(1)

    dod_threshold_p = f.DEFAULT_THRESHOLD_P_DDD
    if args.cxadc or args.cxadc3 or args.cxadc_tenbit or args.cxadc3_tenbit:
        dod_threshold_p = f.DEFAULT_THRESHOLD_P_CXADC

    rf_options = get_rf_options(args)
    rf_options["dod_threshold_p"] = dod_threshold_p
    rf_options["dod_threshold_a"] = args.dod_threshold_a
    rf_options["dod_hysteresis"] = args.dod_hysteresis
    rf_options["track_phase"] = args.track_phase
    rf_options["recheck_phase"] = args.recheck_phase
    rf_options["high_boost"] = args.high_boost
    rf_options["disable_diff_demod"] = args.disable_diff_demod
    rf_options["disable_dc_offset"] = args.disable_dc_offset
    rf_options["disable_comb"] = args.disable_comb
    rf_options["nldeemp"] = args.nldeemp
    rf_options["cafc"] = args.cafc
    rf_options["sync_clip"] = args.sync_clip

    extra_options = get_extra_options(args, not use_gui)

    # Wrap the LDdecode creation so that the signal handler is not taken by sub-threads,
    # allowing SIGINT/control-C's to be handled cleanly
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

    logger = init_logging(outname + ".log")

    if not use_gui and args.umatic:
        tape_format = "UMATIC"
    else:
        tape_format = args.tape_format.upper()
    if tape_format not in supported_tape_formats:
        logger.warning("Tape format %s not supported! Defaulting to VHS", tape_format)

    if not use_gui and args.dodod:
        logger.warning("--doDOD is deprecated, dod is on by default, use noDOD to turn off.")

    debug_plot = None
    if args.debug_plot:
        from vhsdecode.debug_plot import DebugPlot
        debug_plot = DebugPlot(args.debug_plot)

    # Initialize VHS decoder
    # Note, we pass 40 as sample frequency, as any other will be resampled by the
    # loader function.
    vhsd = VHSDecode(
        filename,
        outname,
        loader,
        logger,
        system=system,
        tape_format=tape_format,
        doDOD=not args.nodod,
        threads=args.threads,
        inputfreq=40,
        level_adjust=args.level_adjust,
        rf_options=rf_options,
        extra_options=extra_options,
        debug_plot=debug_plot,
    )

    signal.signal(signal.SIGINT, original_sigint_handler)

    if args.start_fileloc != -1:
        vhsd.roughseek(args.start_fileloc, False)
    else:
        vhsd.roughseek(firstframe * 2)


    if system == "NTSC" and not args.ntscj:
        vhsd.blackIRE = 7.5

    done = False

    jsondumper = lddu.jsondump_thread(vhsd, outname)

    def cleanup():
        jsondumper.put(vhsd.build_json(vhsd.curfield))
        vhsd.close()
        jsondumper.put(None)

    while not done and vhsd.fields_written < (req_frames * 2):
        try:
            f = vhsd.readfield()
        except KeyboardInterrupt:
            print("Terminated, saving JSON and exiting")
            cleanup()
            exit(1)
        except Exception as err:
            print(
                "\nERROR - please paste the following into a bug report:", file=sys.stderr
            )
            print("current sample:", vhsd.fdoffset, file=sys.stderr)
            print("arguments:", args, file=sys.stderr)
            print("Exception:", err, " Traceback:", file=sys.stderr)
            traceback.print_tb(err.__traceback__)
            cleanup()
            exit(1)

        if f is None:
            # or (args.ignoreleadout == False and vhsd.leadOut == True):
            done = True

        if vhsd.fields_written < 100 or ((vhsd.fields_written % 500) == 0):
            jsondumper.put(vhsd.build_json(vhsd.curfield))

    print("saving JSON and exiting")
    cleanup()
    exit(0)