import os
import time
import numpy as np
import traceback
import scipy.signal as sps
import threading
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import lddecode.core as ldd

# from lddecode.core import npfft
# Use numpy fft rather than scipy fft as is imported in lddecode core as it seems to be slightly faster.
import numpy.fft as npfft

import lddecode.utils as lddu
import vhsdecode.utils as utils
from vhsdecode.utils import StackableMA, filtfft
from vhsdecode.chroma import chroma_color_under_filter, TRANSFER_AVERAGE_FIELDS
from vhsdecode import head_switch
from vhsdecode import baseband_eq
from vhsdecode import channel_eq
from vhsdecode import luma_amplitude
from vhsdecode import luma_transient
from vhsdecode import model_stages

import vhsdecode.formats as vhs_formats

from vhsdecode.addons.chromasep import ChromaSepClass
from vhsdecode.addons.chromaAFC import ChromaAFC

from vhsdecode.demod import replace_spikes, unwrap_hilbert, smooth_spikes

from vhsdecode.field import field_class_from_formats
from vhsdecode.video_eq import VideoEQ
from vhsdecode.doc import DodOptions
from vhsdecode.field_averages import FieldAverage
from vhsdecode.load_params_json import override_params
from vhsdecode.nonlinear_filter import sub_deemphasis
from vhsdecode.compute_video_filters import (
    gen_video_main_deemp_fft_params,
    gen_video_lpf_params,
    gen_nonlinear_bandpass_params,
    gen_nonlinear_amplitude_lpf,
    gen_custom_video_filters,
    create_sub_emphasis_params,
    gen_video_lpf_supergauss_params,
    gen_bpf_supergauss,
    gen_fm_audio_notch_params,
    NONLINEAR_AMP_LPF_FREQ_DEFAULT,
    ENVELOPE_LPF_FREQ_DEFAULT,
    CHROMA_AUDIO_NOTCH_Q,
)
from vhsdecode import compute_video_filters as cvf
from vhsdecode.demodcache import DemodCacheTape
from vhsdecode.rust_utils import sosfiltfilt_rust
from vhsdecode.dbwriter import DBWriter


def is_secam(system: str):
    return system == "SECAM" or system == "MESECAM"


def _computefilters_dummy(self):
    self.Filters = {}
    # Needs to be defined here as it's referenced in constructor.
    self.Filters["F05_offset"] = 32


# HACK - override this in a hacky way for now to skip generating some filters we don't use.
# including one that requires > 20 mhz sample rate.
ldd.RFDecode.computefilters = _computefilters_dummy


def _demodcache_dummy(self, *args, **kwargs):
    self.ended = True
    pass


# Superclass to override laserdisc-specific parts of ld-decode with stuff that works for VHS
#
# We do this simply by using inheritance and overriding functions. This results in some redundant
# work that is later overridden, but avoids altering any ld-decode code to ease merging back in
# later as the ld-decode is in flux at the moment.
def _resolve_stage_selection(text):
    """The parsed `--stages` selection, or None. Resolved once, here, so
    every consumer reads the same answer and none of them re-parses."""
    if not text:
        return None
    try:
        from vhsdecode import pipeline_graph

        return pipeline_graph.parse_selection(text, pipeline_graph.load())
    except Exception:                                        # noqa: BLE001
        return None


class VHSDecode(ldd.LDdecode):
    def __init__(
        self,
        fname_in,
        fname_out,
        freader,
        logger,
        system="NTSC",
        tape_format="VHS",
        doDOD=True,
        threads=1,
        inputfreq=40,
        level_adjust=0,
        rf_options={},
        extra_options={},
        debug_plot=None,
        field_order_action="detect",
    ):

        # monkey patch init with a dummy to prevent calling set_start_method twice on macos
        # and not create extra threads.
        # This is kinda hacky and should be sorted in a better way ideally.
        temp_init = ldd.DemodCache.__init__
        ldd.DemodCache.__init__ = _demodcache_dummy
        self._processing_thread_pool = ThreadPoolExecutor(max_workers=threads + 1)

        if system == "405":
            sys_params_pal_temp = ldd.SysParams_PAL.copy()
            # If we are using 405-line we need to override this so the superclasses are initialized with the right values.
            ldd.SysParams_PAL = vhs_formats.get_sys_params_405()
        elif system == "819":
            sys_params_pal_temp = ldd.SysParams_PAL.copy()
            # If we are using 819-line we need to override this so the superclasses are initialized with the right values.
            ldd.SysParams_PAL = vhs_formats.get_sys_params_819()

        # We pass None as output filename to avoid superclass creating output file and database here.
        super(VHSDecode, self).__init__(
            fname_in,
            None,
            freader,
            logger,
            analog_audio=False,
            system=vhs_formats.parent_system(system),
            doDOD=doDOD,
            threads=threads,
            inputfreq=inputfreq,
            extra_options=extra_options,
        )

        if system == "819":
            # We need a larger buffer for 819-line input
            # TODO: Is this useful for normal formats too?
            self.readlen = self.rf.linelen * 500
        # else:
        #    self.readlen = int(self.readlen * 1.1)

        # Adjustment for output to avoid clipping.
        self.level_adjust = level_adjust
        # Overwrite the rf  with the VHS-altered one
        self.rf = VHSRFDecode(
            processing_thread_pool=self._processing_thread_pool,
            system=system,
            tape_format=tape_format,
            inputfreq=inputfreq,
            rf_options=rf_options,
            extra_options=extra_options,
            debug_plot=debug_plot,
        )

        if system == "405":
            # TODO: oln 18/03/2026 This wasn't reset correctly, not sure if it's relevant or not.
            ldd.SysParams_PAL = sys_params_pal_temp

        # Store reference to ourself in the rf decoder - needed to access data location for track
        # phase, may want to do this in a better way later.
        self.rf.decoder = self
        self.FieldClass = field_class_from_formats(system, tape_format)

        # Restore init functino now that superclass constructor is finished.
        ldd.DemodCache.__init__ = temp_init

        self.demodcache = DemodCacheTape(
            self.rf,
            self.infile,
            self.freader,
            self.rf_opts,
            num_worker_threads=self.numthreads,
        )

        self._db_writer = DBWriter(fname_out) if extra_options.get("write_db") else None
        self.dbconn = None
        if self._db_writer:
            self.dbconn = self._db_writer.db_connection
            self.create_db_schema()

        # self._io_thread_pool = ThreadPoolExecutor(2)

        self.outfile_chroma = None

        self.fname_out = fname_out

        if fname_out is not None and self.rf.options.write_chroma:
            if extra_options.get("orc"):
                self.outfile_video = open(fname_out + ".tbcy", "wb")
                self.outfile_chroma = open(fname_out + ".tbcc", "wb")
            else:
                self.outfile_video = open(fname_out + ".tbc", "wb")
                self.outfile_chroma = open(fname_out + "_chroma.tbc", "wb")
        elif fname_out:
            self.outfile_video = open(fname_out + ".tbc", "wb")

        self.debug_plot = debug_plot
        self.field_order_action = field_order_action
        if tape_format == "TYPEC":
            # Since typec usually lacks vsync set this to none to avoid dropping fields.
            self.field_order_action = "none"
        self.duplicate_prev_field = True

        # For tape, it is recommended to use `--ire0_adjust` to fix brightness variations between lines
        # This method usually gives false positives for noisy signals, so smooth the correction out by an entire field to avoid banding
        if self.wow_level_adjust_smoothing is None:
            if self.rf.options.carrier_tbc != 0:
                # With the carrier-refined time base the wow estimate is clean
                # enough that the level adjust may follow the drum-rate
                # flutter instead of averaging it away.  A two-head helical
                # drum lays one field per head pass, so it revolves once per
                # FRAME and the flutter's fundamental period is frame_lines
                # output lines.  The level adjust is smoothed by a single-pole
                # IIR whose time constant is this value in lines, and a
                # single pole's -3 dB corner sits at 1/(2*pi*tau) - so
                # tau = frame_lines/(2*pi) puts the passband edge exactly at
                # the drum fundamental: drum-rate flutter passes while
                # line-rate estimation noise, hundreds of times above the
                # corner, is still attenuated by the same factor.
                self.wow_level_adjust_smoothing = self.rf.SysParams[
                    "frame_lines"
                ] / (2 * np.pi)
            else:
                self.wow_level_adjust_smoothing = self.rf.SysParams["frame_lines"] / 2

        # Needs to be overridden since this is overwritten for 405-line.
        # self.output_lines = (self.rf.SysParams["frame_lines"] // 2) + 1
        # Not modified as of now but may be tweaked for 405-line later
        # self.outwidth = self.rf.SysParams["outlinelen"]

    # Override to avoid NaN in JSON.
    def calcsnr(self, f, snrslice, psnr=False):
        # if dspicture isn't converted to float, this underflows at -40IRE
        data = f.output_to_ire(f.dspicture[snrslice].astype(float))

        signal = np.mean(data) if not psnr else 100
        noise = np.std(data)
        # Make sure signal is positive so we don't try to do log on a negative value.
        if signal < 0.0:
            ldd.logger.info(
                "WARNING: Negative mean for SNR, changing to absolute value."
            )
            signal = abs(signal)
        if noise == 0:
            return 0
        return 20 * np.log10(signal / noise)

    def buildmetadata(self, f, check_phase=True):
        """returns field information JSON and whether to duplicate or drop the field

        THE PHASE CHECK IS PERFORMED HERE, not inherited. This override never
        calls `super().buildmetadata()`, so turning the parameter on did
        nothing until the check itself was ported into the body below - see
        the comment there. The base class defaults it True
        (`lddecode/core.py:4529`) and this override used to turn it off,
        silencing the one contract that verifies `fieldPhaseID` steps by one
        with wrap. With it silent, 45 of the 683 NTSC decodes in this
        repository's own output carry a non-ascending four-field sequence and
        nothing said so.

        One reason it was disabled is addressed: PAL's `fieldPhaseID` was a
        constant 1 - it read an `rf.field_number` that is never incremented -
        and now advances. The other is not, and the check is on anyway
        precisely so it says so: NTSC's framing is still a free-running
        counter, because the colour-under provably cannot carry the colour
        frame (`chroma.colour_frame_parity` has the arithmetic), and a
        counter is seek-dependent.

        The check only logs and sets decode-fault bit 2; it never aborts. So
        the cost of it being on is a warning on genuinely disturbed tape,
        which is what a warning is for.
        """
        prevfi_1 = self.fieldinfo[-1] if len(self.fieldinfo) else None
        prevfi_2 = self.fieldinfo[-2] if len(self.fieldinfo) > 1 else None

        # Not calulated and used for tapes at the moment
        # bust_median = lddu.roundfloat(np.nan_to_num(f.burstmedian)) #lddu.roundfloat(f.burstmedian if not math.isnan(f.burstmedian) else 0.0)
        # "medianBurstIRE": bust_median,

        fi = {
            "isFirstField": True if f.isFirstField else False,
            "detectedFirstField": True if f.isFirstField else False,
            "isDuplicateField": False,
            # burstStartLine description:
            # -1                    -> Color killer is active, no color for entire field
            #  0                    -> Color killer is inactive, color for the entire field
            #  1 to num_field_lines -> Color killer is active until this line, then it is deactivated and color is returned for this and all following lines
            "burstStartLine": f.burst_detected_line,
            "syncConf": f.compute_syncconf(),
            "seqNo": len(self.fieldinfo) + 1,
            "diskLoc": np.round((f.readloc / self.bytes_per_field) * 10) / 10,
            "fileLoc": int(np.floor(f.readloc)),
        }

        if f.fieldPhaseID is None:
            fi["fieldPhaseID"] = {
                (1, 0): 1,
                (0, 1): 2,
                (1, 1): 3,
                (0, 0): 4,
            }[(fi["isFirstField"], (fi["seqNo"] // 2) % 2)]
        else:
            fi["fieldPhaseID"] = f.fieldPhaseID

        write_field = True

        if self.doDOD:
            dropout_lines, dropout_starts, dropout_ends = f.dropout_detect()
            if len(dropout_lines):
                fi["dropOuts"] = {
                    "fieldLine": dropout_lines,
                    "startx": dropout_starts,
                    "endx": dropout_ends,
                }

        # This is a bitmap, not a counter
        # docs for this mysterious bitmap???

        decode_faults = 0

        # THE PHASE CHECK, PORTED RATHER THAN INHERITED. Setting the
        # `check_phase` parameter alone did nothing at all: this override
        # never calls `super().buildmetadata()` and carried no sequence test,
        # so the flag was accepted and ignored - a docstring claiming the
        # check was back on while a 20-field decode with 13 non-ascending
        # transitions logged nothing. That is worse than the honest `False`
        # it replaced, and this is the eight lines from
        # `lddecode/core.py:4556-4568` that make the parameter mean something.
        #
        # It is the guard that catches a wrong colour frame on the first
        # field, and it only warns and sets bit 2 - it never aborts.
        if check_phase and prevfi_1 is not None:
            phases = self.rf.SysParams.get("fieldPhases")
            ascends = (
                (fi["fieldPhaseID"] == 1
                 and prevfi_1["fieldPhaseID"] == phases)
                or fi["fieldPhaseID"] == prevfi_1["fieldPhaseID"] + 1)
            if phases and not ascends:
                ldd.logger.warning(
                    "At field #%d, Field phaseID sequence mismatch (%s->%s) "
                    "(player may be paused)",
                    len(self.fieldinfo), prevfi_1["fieldPhaseID"],
                    fi["fieldPhaseID"])
                decode_faults |= 2

        fi["vitsMetrics"] = self.computeMetrics(self.fieldstack[0], self.fieldstack[1])
        # interlaced video requires alternating fields, handle cases where fields are repeated
        #   this can happen due to breaks in recordings between fields, i.e. home recordings, and
        #   progressive content, such as video game, osd, computer output, etc.
        if prevfi_1 is not None and prevfi_1["isFirstField"] == fi["isFirstField"]:
            distance_from_previous_field = fi["diskLoc"] - prevfi_1["diskLoc"]
            if (
                # there are three (this one, and two previous) repeating field orders in a row
                # should be impossible for valid interlaced video, so maybe it's progressive??
                # progressive examples needed to test this!
                prevfi_1["detectedFirstField"] == fi["detectedFirstField"]
                and prevfi_2 is not None
                and prevfi_2["detectedFirstField"] == prevfi_1["detectedFirstField"]
                # and this field is within a reasonable distance to be valid
                and lddu.inrange(distance_from_previous_field, 0.9, 1.1)
                # Skip on TYPEC since we expect to have missing vsync there and we don't
                # expect progressive video.
                and self.rf.options.tape_format != "TYPEC"
            ):
                # treat this as progressive, and manually flip the field order
                ldd.logger.error(
                    "Detected progressive video content..., manually flipping the field order to compensate"
                )
                decode_faults |= 1
                fi["syncConf"] = 10
                fi["isFirstField"] = not prevfi_1["isFirstField"]
            else:
                if self.field_order_action == "duplicate":
                    self.duplicate_prev_field = True
                elif self.field_order_action == "drop":
                    self.duplicate_prev_field = False
                elif self.field_order_action == "detect":
                    # duplicated field order was detected more than 1.1 fields away from the previous field, possibly a gap
                    if distance_from_previous_field > 1.1:
                        self.duplicate_prev_field = True
                    # duplicated field order was detected less than 0.9 fields away from the previous field, probably overlaped end of last field
                    elif distance_from_previous_field < 0.9:
                        self.duplicate_prev_field = False
                    # next field is close enough to be a valid field, duplicating or dropping is valid, alternate to avoid too many duplicates or drops
                    else:
                        self.duplicate_prev_field = not self.duplicate_prev_field

                if self.field_order_action == "none":
                    if self.rf.options.tape_format != "TYPEC":
                        ldd.logger.error(
                            "Possibly skipped field (Two fields with same isFirstField in a row), manually flipping the field order to compensate"
                        )
                    decode_faults |= 4
                    fi["syncConf"] = 0
                    fi["isFirstField"] = not prevfi_1["isFirstField"]
                elif self.duplicate_prev_field:
                    ldd.logger.error(
                        "Possibly skipped field (Two fields with same isFirstField in a row), duplicating the last field to compensate..."
                    )
                    decode_faults |= 4
                    fi["syncConf"] = 0
                    fi["isDuplicateField"] = True
                else:
                    ldd.logger.error(
                        "Possibly skipped field (Two fields with same isFirstField in a row), dropping the last field to compensate..."
                    )
                    decode_faults |= 4
                    write_field = False
                    fi["syncConf"] = 0

            if decode_faults != 0:
                # Only write this if it's anything else than 0, to save a little space in the json,
                # since it's not used for anything atm anyhow.
                fi["decodeFaults"] = decode_faults

            return fi, fi["isDuplicateField"], write_field

        self.frameNumber = None
        if f.isFirstField:
            self.firstfield = f
        else:
            # use a stored first field, in case we start with a second field
            if self.firstfield is not None:
                # process VBI frame info data
                self.frameNumber = None

                rawloc = np.floor((f.readloc / self.bytes_per_field) / 2)

                tape_format = (
                    self.rf.options.tape_format
                )  # "CLV" if self.isCLV else "CAV"

                try:
                    if self.est_frames is not None:
                        outstr = f"Frame {(self.fields_written//2)+1}/{int(self.est_frames)}: File Frame {int(rawloc)}: {tape_format} "
                    else:
                        outstr = f"File Frame {int(rawloc)}: {tape_format} "

                    self.logger.status(outstr)
                except Exception:
                    ldd.logger.warning("file frame %d : VBI decoding error", rawloc)
                    traceback.print_exc()

        return fi, fi["isDuplicateField"], write_field

    # Again ignored for tapes
    def checkMTF(self, field, pfield=None):
        return True

    def writeout(self, dataset):
        f, fi, (picturey, picturec), audio, efm = dataset

        # Remove fields that are currently not used to cut down on space usage.
        # the qt tools will load them as 0 with the current code
        # if they don't exist.
        if "audioSamples" in fi:
            del fi["audioSamples"]

        self.fieldinfo.append(fi)

        if self._db_writer:
            if not self.capture_id:
                self.build_sqlite_metadata()
            self._db_writer.write_field(fi, self.doDOD, self.capture_id)
            # NOTE: this calls commit so we don't call it in dbwriter.write_field.
            self.build_sqlite_metadata()

        self.outfile_video.write(picturey)
        if self.rf.options.write_chroma:
            self.outfile_chroma.write(picturec)
        self.fields_written += 1

    def close(self):
        if self.decodethread and self.decodethread.is_alive():
            self.decodethread.join()
            self.decodethread = None
        if self.rf.options.write_chroma:
            setattr(self, "outfile_chroma", None)

        if self._processing_thread_pool is not None:
            self._processing_thread_pool.shutdown(wait=True)
        super(VHSDecode, self).close()

    def computeMetricsPAL(self, metrics, f, fp=None):
        return None

    def computeMetricsNTSC(self, metrics, f, fp=None):
        return None

    def build_json(self):
        try:
            # if not f:
            #    # Make sure we don't fail if the last attempted field failed to decode
            #    # Might be better to fix this elsewhere.
            #    f = self.prevfield
            jout = super(VHSDecode, self).build_json()

            black = jout["videoParameters"]["black16bIre"]
            white = jout["videoParameters"]["white16bIre"]

            if self.rf.color_system == "PAL_M" or self.rf.color_system == "NLINHA":
                # jout["videoParameters"]["isSourcePal"] = True
                # jout["videoParameters"]["isSourcePalM"] = True
                jout["videoParameters"]["system"] = "PAL-M"

            jout["videoParameters"]["black16bIre"] = black * (1 - self.level_adjust)
            jout["videoParameters"]["white16bIre"] = white * (1 + self.level_adjust)

            jout["videoParameters"]["tapeFormat"] = self.rf.options.tape_format
            return jout
        except TypeError as e:
            if self.rf.debug:
                traceback.print_exc()
                ldd.logger.error("Error! Cannot build json: %s" % e)
            ldd.logger.error(
                "Error! Something went wrong when decoding or building json!"
            )
            return None

    def readfield(self, initphase=False):
        done = False
        adjusted = False
        redo = None
        df_args = None
        f = None
        offset = 0

        if len(self.fieldstack) >= 2:
            ## Done in main files
            # XXX: Need to cut off the previous field here, since otherwise
            # it'll leak for now.
            # if self.fieldstack[-1]:
            #    self.fieldstack[-1].prevfield = None
            self.fieldstack.pop(-1)

        while done is False:
            if self.second_decode is None and self.fields_written:
                self.second_decode = time.time()

            if redo:
                # Drop existing thread
                self.decodethread = None

                f, offset = self.decodefield(
                    redo, self.mtf_level, self.fieldstack[0], initphase, redo
                )
                # Only allow one redo, no matter what
                done = True
                redo = None

            else:
                if self.decodethread and self.decodethread.ident:
                    self.decodethread.join()
                    self.decodethread = None

                # In non-threaded mode self.threadreturn was filled earlier...
                # ... but if the first call, this is empty
                if len(self.threadreturn) > 0:
                    f, offset = self.threadreturn["field"], self.threadreturn["offset"]

            # Start new thread
            self.threadreturn = {}
            if f and f.valid:
                prevfield = f
                toffset = self.fdoffset + offset
            else:
                prevfield = None
                toffset = self.fdoffset

                if offset:
                    toffset += offset

            df_args = (
                toffset,
                self.mtf_level,
                prevfield,
                initphase,
                False,
                self.threadreturn,
            )

            # decode the next field in a thread so the result is ready for the next iteration
            if self.numthreads != 0:
                self.decodethread = threading.Thread(
                    target=self.decodefield, args=df_args
                )
                self.decodethread.start()
            else:
                self.decodefield(*df_args)

            # process previous run
            if f:
                self.fdoffset += offset
            elif offset is None:
                # Probable end, so push an empty field
                self.fieldstack.insert(0, None)

            if f and f.valid:
                picture, audio, efm = f.downscale(
                    linesout=self.output_lines,
                    final=True,
                    audio=self.analog_audio,
                    lastfieldwritten=self.lastFieldWritten,
                )

                _ = self.computeMetrics(f, None, verbose=True)
                # if "blackToWhiteRFRatio" in metrics and adjusted is False:
                #    keep = 900 if self.isCLV else 30
                #    self.bw_ratios.append(metrics["blackToWhiteRFRatio"])
                #    self.bw_ratios = self.bw_ratios[-keep:]

                redo = f.needrerun
                if redo:
                    redo = self.fdoffset - offset

                # Perform AGC changes on first fields only to prevent luma mismatch intra-field
                if self.useAGC and f.isFirstField and f.sync_confidence > 80:
                    # TODO: actuall test this after changes
                    sync_hz, ire0_hz, ire100_hz = self.detectLevels(f)

                    actualwhiteIRE = f.rf.hztoire(ire100_hz)

                    sync_ire_diff = lddu.nb_abs(
                        self.rf.hztoire(sync_hz) - self.rf.DecoderParams["vsync_ire"]
                    )
                    whitediff = lddu.nb_abs(self.rf.hztoire(ire100_hz) - actualwhiteIRE)
                    ire0_diff = lddu.nb_abs(self.rf.hztoire(ire0_hz))

                    acceptable_diff = 2 if self.fields_written else 0.5

                    if max((whitediff, ire0_diff, sync_ire_diff)) > acceptable_diff:
                        hz_ire = (ire100_hz - ire0_hz) / 100
                        vsync_ire = (sync_hz - ire0_hz) / hz_ire

                        if vsync_ire > -20:
                            ldd.logger.warning(
                                "At field #{0}, Auto-level detection malfunction (vsync IRE computed at {1}, nominal ~= -40), possible disk skipping".format(
                                    len(self.fieldinfo), np.round(vsync_ire, 2)
                                )
                            )
                        else:
                            redo = self.fdoffset - offset

                            self.rf.DecoderParams["ire0"] = ire0_hz
                            # Note that vsync_ire is a negative number, so (sync_hz - ire0_hz) is correct
                            self.rf.DecoderParams["hz_ire"] = hz_ire
                            self.rf.DecoderParams["vsync_ire"] = vsync_ire

                # One-shot re-demodulation once the amplitude stage's block
                # model first exists: the demod prefetch ran ~4 fields ahead
                # of the first measured field, so every block so far was
                # demodulated before the head switch correction could act -
                # the first field came out uncorrected and its debug trace
                # empty. Redo this first field from re-demodulated blocks
                # (FFTs are cached, only the demod is repaid) and reset the
                # correction's calibration so redone blocks do not vote
                # twice.
                if (
                    self.rf.options.head_switch != 0
                    and not self.rf.__dict__.get("_head_switch_redone", False)
                    and luma_amplitude.block_model(self.rf) is not None
                ):
                    self.rf.__dict__["_head_switch_redone"] = True
                    self.rf.__dict__.pop("_head_switch_cal", None)
                    if not redo:
                        redo = self.fdoffset - offset

                # The same one-shot for the RF transform, for the same
                # reason: the demod prefetch runs ahead of field assembly,
                # so the blocks in flight when the transform latched a table
                # were demodulated without it. `transform_wants_redo`
                # answers True exactly once per published version, and the
                # redo repays those blocks exactly as the head-switch
                # one-shot does - FFTs cached, only the demod repaid. It is
                # asked BEFORE the pending-redo test so that a redo already
                # owed for another reason consumes the request as well: that
                # redo flushes and re-demodulates the same blocks against
                # the table that is now live.
                if (
                    self.rf.options.rf_transform != 0
                    and model_stages.transform_wants_redo(self.rf)
                    and not redo
                ):
                    redo = self.fdoffset - offset

                if adjusted is False and redo:
                    # No direct flush_demod() here: decodefield passes the
                    # redo offset as forceredo, and read() performs the
                    # flush under the cache lock - the unlocked flush this
                    # branch used to make raced against the worker.
                    adjusted = True
                    self.fdoffset = redo
                else:
                    done = True
                    self.fieldstack.insert(0, f)

            if f is None and offset is None:
                # EOF, probably
                return None

            if self.decodethread and not self.decodethread.ident and not redo:
                self.decodethread.start()

        if f is None or f.valid is False:
            return None

        if f is not None and self.fname_out is not None:
            # Only write a FirstField first
            if len(self.fieldinfo) == 0 and not f.isFirstField:
                return f

            fi, duplicateField, writeField = self.buildmetadata(f)

            if writeField:
                self.lastvalidfield[f.isFirstField] = (f, fi, picture, audio, efm)

            if duplicateField:
                if self.lastvalidfield[not f.isFirstField] is not None:
                    self.writeout(self.lastvalidfield[not f.isFirstField])
                    self.writeout(self.lastvalidfield[f.isFirstField])

                # If this is the first field to be written, don't write anything
                return f

            if writeField:
                self.lastFieldWritten = (self.fields_written, f.readloc)
                self.writeout(self.lastvalidfield[f.isFirstField])

        return f


class VHSRFDecode(ldd.RFDecode):
    def __init__(
        self,
        processing_thread_pool=None,
        inputfreq=40,
        system="NTSC",
        tape_format="VHS",
        rf_options={},
        extra_options={},
        debug_plot=None,
    ):

        # First init the rf decoder normally.
        super(VHSRFDecode, self).__init__(
            inputfreq,
            vhs_formats.parent_system(system),
            decode_analog_audio=False,
            has_analog_audio=False,
            extra_options=extra_options,
        )
        if processing_thread_pool is None:
            processing_thread_pool = ThreadPoolExecutor(max_workers=1)
        self._processing_thread_pool = processing_thread_pool

        # Store a separate setting for *color* system as opposed to 525/625 line here.
        # TODO: Fix upstream so we don't have to fake tell ld-decode code that we are using ntsc for
        # palm to avoid it throwing errors.
        self._color_system = system

        self._dod_options = DodOptions(
            dod_threshold_p=rf_options.get(
                "dod_threshold_p", vhs_formats.DEFAULT_THRESHOLD_P_DDD
            ),
            dod_threshold_a=rf_options.get("dod_threshold_a", None),
            dod_hysteresis=rf_options.get(
                "dod_hysteresis", vhs_formats.DEFAULT_HYSTERESIS
            ),
        )

        self._chroma_trap = rf_options.get("chroma_trap", False)
        # Offline measurements of the playback channel's baseband response,
        # consumed by `baseband_eq` and by `head_switch`'s regularization.
        self._baseband_lf_response = rf_options.get("baseband_lf_response", None)
        self._sync_step_response = rf_options.get("sync_step_response", None)
        self._head_switch_export = rf_options.get("head_switch_export", None)
        self._time_base_response = rf_options.get("time_base_response", None)
        self._baseband_eq_declared = bool(
            self._baseband_lf_response or self._sync_step_response
        )
        # The identified RF channel response applied by `channel_eq`, and the
        # directory the residual channels are exported to, downscaled to
        # 4fsc on the final time base (`residual_channels`).
        self._channel_response = rf_options.get("channel_response", None)
        self._channel_eq_declared = bool(self._channel_response)
        self._residual_channels_dir = rf_options.get("residual_channels", None)
        # TODO: integrate this under chroma_trap later
        self._use_fsc_notch_filter = (
            tape_format == "BETAMAX" or tape_format == "BETAMAX_HIFI"
        )
        track_phase = None if is_secam(system) else rf_options.get("track_phase", None)
        high_boost = rf_options.get("high_boost", None)
        self._notch = rf_options.get("notch", None)
        self._notch_q = rf_options.get("notch_q", 10.0)
        self._disable_diff_demod = rf_options.get("disable_diff_demod", False)
        self.useAGC = extra_options.get("useAGC", False)
        self.debug = extra_options.get("debug", False)

        # cafc measures a single carrier peak, which doesn't exist in the
        # line-alternating two-carrier SECAM FM chroma signal.
        requested_cafc = rf_options.get("cafc", False)
        if requested_cafc and is_secam(system):
            ldd.logger.warning(
                "cafc is not supported for SECAM systems, ignoring. "
                "(For ME-SECAM, the carrier servo tunes the conversion LO instead.)"
            )
            requested_cafc = False

        # Enable cafc for betamax until proper track detection for it is implemented.
        self._do_cafc = (
            True
            if (tape_format == "BETAMAX" and system != "NTSC")
            else requested_cafc
        )

        self.track_phase = None
        if track_phase == 0 or track_phase == 1:
            self.track_phase = track_phase
        elif track_phase is not None:
            raise Exception("Track phase can only be 0, 1 or None")

        self.hsync_tolerance = 0.8

        self.field_number = 0
        self.last_raw_loc = None

        self.SysParams, self.DecoderParams = vhs_formats.get_format_params(
            system,
            tape_format,
            vhs_formats.parse_tape_speed(rf_options.get("tape_speed", "sp")),
            ldd.logger,
        )

        params_file = extra_options.get("params_file", None)
        if params_file:
            override_params(self.SysParams, self.DecoderParams, params_file, ldd.logger)

        # Make (intentionally) mutable copies of HZ<->IRE levels
        # (NOTE: used by upstream functions, we use a namedtuple to keep const values already)
        self.DecoderParams["ire0"] = self.SysParams["ire0"]
        self.DecoderParams["hz_ire"] = self.SysParams["hz_ire"]
        self.DecoderParams["vsync_ire"] = self.SysParams["vsync_ire"]
        self.DecoderParams["track_ire0_offset"] = self.SysParams.get(
            "track_ire0_offset", [0, 0]
        )

        export_raw_tbc = rf_options.get("export_raw_tbc", False)
        ire0_adjust_raw = rf_options.get("ire0_adjust", "")
        if isinstance(ire0_adjust_raw, str):
            ire0_adjust = tuple(
                mode.strip().lower()
                for mode in ire0_adjust_raw.split(",")
                if mode.strip()
            )
        else:
            ire0_adjust = tuple()
        is_color_under = vhs_formats.is_color_under(tape_format)
        write_chroma = (
            is_color_under
            and not export_raw_tbc
            and not rf_options.get("skip_chroma", False)
            and not (system == "405")
        )

        # THE DECLARATION SUPPLIES THE DEFAULTS, AND THE SELECTION OVERRIDES
        # THEM, both folded into the option values BEFORE the Options
        # namedtuple is built - so every existing gate reads its own option
        # and there is still exactly one gate per stage.
        #
        # Ethan: "Let's retire all the one-off options and have everything we
        # have done so far live in the graph." A stage declared with no flag
        # behind it takes its default from `pipeline/stages.toml` and is
        # turned on or off by name with `--stages`; a stage that still has a
        # flag is untouched, because `apply_defaults` only writes keys that
        # are absent and `main.py` writes every flag-backed key.
        _stage_selection_value = _resolve_stage_selection(
            rf_options.get("stages"))
        # `ldd.logger` is None until a decode initialises it, and a unit test
        # that constructs this class never does. Logging what the declaration
        # supplied must not be the thing that breaks the demodulator.
        _log = getattr(ldd, "logger", None)
        try:
            from vhsdecode import pipeline_graph as _pipeline_graph

            _declared = _pipeline_graph.load()
            for _line in _pipeline_graph.apply_defaults(rf_options, _declared):
                if _log is not None:
                    _log.debug("pipeline: %s", _line)
            if _stage_selection_value is not None:
                for _line in _pipeline_graph.apply_selection(
                        rf_options, _stage_selection_value, _declared):
                    if _log is not None:
                        _log.info("--stages: %s", _line)
            # A node that SUPERSEDES another must not run beside it. Refused
            # here rather than warned about: the two would apply the same
            # correction twice and the second would be fitted against a
            # picture the first had already emptied, so the decode would
            # complete and be wrong rather than fail.
            _conflicts = _pipeline_graph.supersessions(rf_options, _declared)
            if _conflicts:
                raise ValueError(" ".join(_conflicts))
        except ValueError:
            raise
        except Exception as _error:                          # noqa: BLE001
            if _log is not None:
                _log.warning("the pipeline declaration could not be "
                             "applied: %s", _error)

        # No idea if this is a common pythonic way to accomplish it but this gives us values that
        # can't be changed later.
        # first depends on IRE/Hz so has to be set after that is properly set.
        # TODO: May want to split this up eventually
        self._options = namedtuple(
            "Options",
            [
                "diff_demod_check_value",
                "tape_format",
                "disable_comb",
                "nldeemp",
                "subdeemp",
                "disable_right_hsync",
                "disable_dc_offset",
                "fallback_vsync",
                "field_order_confidence",
                "saved_levels",
                "y_comb",
                "write_chroma",
                "color_under",
                "chroma_deemphasis_filter",
                "skip_hsync_refine",
                "hsync_refine_use_threshold",
                "export_raw_tbc",
                "fm_audio_notch",
                "chroma_audio_notch",
                "chroma_offset",
                "cagc_fields",
                "chroma_env_gain",
                "chroma_env_phase",
                "luma_transient",
                "luma_beat",
                # The model-based stages (vhsdecode/model_stages.py). All
                # default to zero, so an unflagged decode is unchanged.
                "chroma_head_switch",
                "colour_free_luma",
                # The only one of these on the SOURCE side of the record
                # head: it corrects the luma response the signal already
                # carried when the recording VCR received it, rather than
                # anything the tape or either deck did.
                "source_correction",
                # And the ones that MEASURE and apply nothing. Their gates
                # exist for the same reason the corrections' do - every stage
                # has to be callable by name from `--stages` - and each of
                # them leaves the decode byte-identical whether it is on or
                # off, writing what it found to the log instead.
                #
                # `colour_under` here is the colour-under CHANNEL MEASUREMENT
                # and is not `color_under` five lines above, which is the
                # boolean saying whether this format records its chrominance
                # heterodyned down at all. Two different things one letter
                # apart, so they are named together rather than left to be
                # confused at a call site.
                "burst_instrument",
                "chroma_leakage",
                # `tape_bias` measures the third-order product the luma FM's
                # own bias leaves in the delivered luma at 80 f_H. It stands
                # beside `chroma_leakage` and reads something else: that one
                # measures the colour under SURVIVING THE SEPARATION, which
                # a filter owns, and this one a product WRITTEN ON THE TAPE
                # that no separation filter can reach.
                "tape_bias",
                "colour_framing",
                "colour_under",
                "iq_imbalance",
                "vectorscope",
                # THE SYNC AND TIMING GROUP. Two corrections and four
                # measurements, every one of them read on the sync pulses
                # and the reserved intervals alone. `precursor` and
                # `sync_depth` write samples; `sync_shape`,
                # `burst_sync_lock`, `vertical_interval` and `tape_speed`
                # leave the decode byte-identical and write what they found
                # to the log. All default to zero.
                "sync_shape",
                "precursor",
                "sync_depth",
                "burst_sync_lock",
                "vertical_interval",
                # `tape_speed_stage`, not `tape_speed`: the latter is the
                # existing SP / LP / EP option and is a non-empty string, so
                # a gate sharing its name would read as permanently on.
                "tape_speed_stage",
                # THE LEVELS, GAIN AND SOURCE-SIDE GROUP. One correction and
                # six measurements, every one of them read on the sync
                # pulses, the reserved intervals or the colour burst.
                # `standard_levels` writes samples - it drives the back
                # porch to the specified zero with the two known chroma
                # carriers projected out of it first; the other six leave
                # the decode byte-identical and write what they found to
                # the log. All default to zero.
                #
                # EVERY NAME HERE WAS CHECKED AGAINST THE LIST ABOVE before
                # it was added, for the reason `tape_speed_stage` records:
                # a gate whose option name collides with an existing
                # non-empty option reads as permanently on and the stage
                # runs on every decode.
                "standard_levels",
                "level_from_frequency",
                "source_agc",
                "vcr_agc",
                "composite_channel",
                "picture_stage",
                "multipath",
                # THE RADIO-FREQUENCY, CAPTURE AND TRANSPORT GROUP. Eight
                # measurements, every one of them read on the RAW radio
                # frequency the capture card delivered rather than on the
                # demodulated picture, and confined to the sync tips, the
                # burst and the intervals above the video band where no
                # picture exists. Not one of them writes a sample, so a
                # decode is byte-identical whether they are on or off and
                # the finding goes to the log. All default to zero.
                #
                # EVERY NAME HERE WAS CHECKED AGAINST THE LISTS ABOVE
                # before it was added, for the reason `tape_speed_stage`
                # records: a gate whose option name collides with an
                # existing non-empty option reads as permanently on and the
                # stage runs on every decode. `head_switch_pair` in
                # particular is NOT `head_switch`, which is the amplitude
                # correction ten lines below.
                "capture_profile",
                "capture_filter",
                "rf_stages",
                "filter_model",
                "transport_model",
                "band_delay",
                "head_switch_pair",
                "interference",
                # THE HEAD, THE MEDIUM AND THE PATH. Five more measurements
                # of the same kind, read in the synchronizing pulse's flat
                # interior, the back porch after the burst and the colour
                # burst - reserved intervals every one, so no sample of
                # active picture reaches a level, a fit or a verdict. Not
                # one of them writes a sample. All default to zero.
                #
                # EVERY NAME HERE WAS CHECKED AGAINST THE LISTS ABOVE, for
                # the reason `tape_speed_stage` records. `magnetic` is not
                # `magnetic_circuit`, and neither is `head_model` the
                # `filter_model` twelve lines up: three different modules
                # whose names share a word.
                "head_model",
                "head_differential",
                "magnetic",
                "magnetic_circuit",
                "tape_path",
                # THE TWO TRANSFORM STAGES. `picture_transform` is the one
                # picture stage that stands in for the sync, head-and-medium
                # and levels groups above and for the field-level chroma
                # stages; `rf_transform` is the one RF stage that stands in
                # for the radio-frequency group and publishes the per-head
                # table the demodulator's equalizer site multiplies by.
                # Both are seeded ON by their declaration and have no flag
                # of their own; `--stages -rf_transform` turns one off.
                "picture_transform",
                "rf_transform",
                "luma_eq",
                "head_switch",
                "carrier_tbc",
                "baseband_eq",
                "head_switch_prior",
                "channel_eq",
                "inverse_eq",
                "lti_gain",
                "cti_mix",
                "cti_width",
                # The stage and component selection, resolved once at init.
                # NOTE: this list and the VALUE list below are positional and
                # nothing checks their alignment - a new entry must go in at
                # the SAME index in both.
                "stage_selection",
                "ire0_adjust",
                "gnrc_afe",
                "relaxed_line0",
                "detect_chroma_track_phase",
                "enable_color_killer",
                "disable_burst_hsync",
                "disable_phase_correction",
                "secam_carrier_servo",
            ],
        )(
            self.iretohz(100) * 2,
            tape_format,
            rf_options.get("disable_comb", False) or is_secam(system),
            rf_options.get("nldeemp", False),
            self.DecoderParams.get("use_sub_deemphasis", False)
            or rf_options.get("subdeemp", False),
            rf_options.get("disable_right_hsync", False),
            rf_options.get("disable_dc_offset", False),
            # Always use this if we are decoding TYPEC since it doesn't have normal vsync.
            # also enable by default with EIAJ since that was typically used with a primitive sync gen
            # which output not quite standard vsync.
            rf_options.get("fallback_vsync", False)
            or tape_format == "TYPEC"
            or tape_format == "EIAJ"
            or system == "405"
            or system == "819",
            rf_options.get("field_order_confidence", False),
            rf_options.get("saved_levels", False),
            rf_options.get("y_comb", 0) * self.SysParams["hz_ire"],
            write_chroma,
            is_color_under,
            tape_format == "VIDEO8" or tape_format == "HI8",
            rf_options.get("skip_hsync_refine", False),
            # hsync_refine_use_threshold - use detected level for hsync refine
            # TODO: This should be used for everything eventually but needs proper testing
            True,
            export_raw_tbc,
            # Optional on VHS/Beta/video8
            # always enable for hi8 since should pretty much always have the second audio
            # channel. May want to enable for video8 as well.
            rf_options.get("fm_audio_notch", 0) or (tape_format == "HI8"),
            self.DecoderParams.get("chroma_audio_notch_freq", 0) > 0,
            int(self.DecoderParams.get("chroma_offset", 5) * (self.freq / 40.0)),
            rf_options.get("cagc_fields", 0),
            rf_options.get("chroma_env_gain", 0),
            rf_options.get("chroma_env_phase", 0),
            rf_options.get("luma_transient", 0),
            rf_options.get("luma_beat", 0),
            rf_options.get("chroma_head_switch", 0),
            rf_options.get("colour_free_luma", 0),
            rf_options.get("source_correction", 0),
            rf_options.get("burst_instrument", 0),
            rf_options.get("chroma_leakage", 0),
            rf_options.get("tape_bias", 0),
            rf_options.get("colour_framing", 0),
            rf_options.get("colour_under", 0),
            rf_options.get("iq_imbalance", 0),
            rf_options.get("vectorscope", 0),
            rf_options.get("sync_shape", 0),
            rf_options.get("precursor", 0),
            rf_options.get("sync_depth", 0),
            rf_options.get("burst_sync_lock", 0),
            rf_options.get("vertical_interval", 0),
            rf_options.get("tape_speed_stage", 0),
            rf_options.get("standard_levels", 0),
            rf_options.get("level_from_frequency", 0),
            rf_options.get("source_agc", 0),
            rf_options.get("vcr_agc", 0),
            rf_options.get("composite_channel", 0),
            rf_options.get("picture_stage", 0),
            rf_options.get("multipath", 0),
            rf_options.get("capture_profile", 0),
            rf_options.get("capture_filter", 0),
            rf_options.get("rf_stages", 0),
            rf_options.get("filter_model", 0),
            rf_options.get("transport_model", 0),
            rf_options.get("band_delay", 0),
            rf_options.get("head_switch_pair", 0),
            rf_options.get("interference", 0),
            rf_options.get("head_model", 0),
            rf_options.get("head_differential", 0),
            rf_options.get("magnetic", 0),
            rf_options.get("magnetic_circuit", 0),
            rf_options.get("tape_path", 0),
            rf_options.get("picture_transform", 0),
            rf_options.get("rf_transform", 0),
            rf_options.get("luma_eq", 0),
            rf_options.get("head_switch", 0),
            rf_options.get("carrier_tbc", 0),
            rf_options.get("baseband_eq", 0),
            rf_options.get("head_switch_prior", 0),
            rf_options.get("channel_eq", 0),
            rf_options.get("inverse_eq", -1),
            rf_options.get("lti_gain", None),
            rf_options.get("cti_mix", 1),
            rf_options.get("cti_width", 2),
            _stage_selection_value,
            ire0_adjust,
            rf_options.get("gnrc_afe", False),
            rf_options.get("relaxed_line0", False),
            rf_options.get("detect_chroma_track_phase", False),
            rf_options.get("enable_color_killer", False),
            # SECAM has no phase-locked burst; the "burst" is an FM carrier
            # whose phase carries no timing information, so locking hsync to
            # it just injects sub-pixel jitter into both planes.
            rf_options.get("disable_burst_hsync", False) or is_secam(system),
            rf_options.get("disable_phase_correction", False),
            rf_options.get("secam_carrier_servo", True),
        )

        # As agc can alter these sysParams values, store a copy to then
        # initial value for reference.
        self._sysparams_const = namedtuple(
            "SysparamsConst", "hz_ire vsync_hz vsync_ire ire0 vsync_pulse_us"
        )(
            self.SysParams["hz_ire"],
            self.iretohz(self.SysParams["vsync_ire"]),
            self.SysParams["vsync_ire"],
            self.SysParams["ire0"],
            self.SysParams["vsyncPulseUS"],
        )

        #
        self._sub_emphasis_params = create_sub_emphasis_params(
            self.DecoderParams,
            self.SysParams,
            self._sysparams_const.hz_ire,
            self._sysparams_const.vsync_ire,
        )

        self.debug_plot = debug_plot

        # Lastly we re-create the filters with the new parameters.
        self._computevideofilters_b()

        if self._channel_eq_declared and self.options.channel_eq != 0:
            # The identified channel response, built once on the block grid
            # the filters above use and never mutated (the demodulation
            # thread runs ahead of field assembly).
            channel_eq.load(self, self._channel_response, self.options.channel_eq)

        # THE STAGE SELECTION REACHES THE RINGING STAGE HERE. It lives on
        # the Options namedtuple, but `process_field` is handed only its
        # own `shared_state` dict and no rf reference, so a component gate
        # inside that stage cannot see the selection unless it is seeded -
        # exactly as `channel_eq` seeds `channel_eq_active` into the same
        # dict (channel_eq.py:219). Without this the flag parses, validates
        # and is silently ignored, which is how the first version of the
        # relaxation component shipped inert: the decode was byte-identical
        # with the flag ON, and that reads like a working no-harm gate.
        if getattr(self.options, "stage_selection", None):
            self.__dict__.setdefault("_ringing_state", {})["stage_selection"] \
                = self.options.stage_selection

        DP = self.DecoderParams

        self._high_boost = (
            high_boost if high_boost is not None else DP["boost_bpf_mult"]
        )

        # controls the sharpness EQ gain
        sharpness_level = (
            rf_options.get("sharpness", vhs_formats.DEFAULT_SHARPNESS) / 100
        )

        self._video_eq = None
        if sharpness_level != 0:
            self._video_eq = VideoEQ(DP, sharpness_level, self.freq_hz)

        # Heterodyning / chroma wave related filter part

        self._chroma_afc = ChromaAFC(
            self.freq_hz,
            DP["chroma_bpf_upper"] / DP["color_under_carrier"],
            self.SysParams,
            self.DecoderParams["color_under_carrier"],
            self.DecoderParams.get("chroma_bpf_order", 4),
            tape_format=tape_format,
            do_cafc=self._do_cafc,
            chroma_bpf_lower=self.DecoderParams.get("chroma_bpf_lower", 60000),
            conversion_lo_freq=self.DecoderParams.get("chroma_conversion_lo", None),
            carrier_mult=self.DecoderParams.get("chroma_carrier_mult", None),
        )

        self.Filters["FVideoBurst"] = (
            self._chroma_afc.get_chroma_bandpass()
            if self._options.color_under
            else self._chroma_afc.get_chroma_bandpass_final(False)
        )

        if self.options.chroma_deemphasis_filter:
            from vhsdecode.addons.biquad import peaking

            out_freq_half = self._chroma_afc.getOutFreqHalf()

            b, a = peaking(
                self.sys_params["fsc_mhz"] / out_freq_half,
                3.4,
                BW=0.5 / out_freq_half,
                type="constantq",
            )
            self.Filters["chroma_deemphasis"] = (b, a)

        if self._notch is not None:
            video_notch_filter = sps.iirnotch(
                self._notch / self.freq_half, self._notch_q
            )

            # Chroma notch filter
            if self._do_cafc:
                self.Filters["FVideoNotch"] = sps.iirnotch(
                    self._notch / self._chroma_afc.getOutFreqHalf(), self._notch_q
                )
            else:
                self.Filters["FVideoNotch"] = video_notch_filter

            # Luma notch filter
            self.Filters["FVideoNotchF"] = abs(
                utils.filtfft(video_notch_filter, self.blocklen)
            )
        else:
            self.Filters["FVideoNotch"] = None, None

        if self._options.chroma_audio_notch:
            if self._do_cafc:
                self.Filters["FChromaAudioNotch"] = sps.iirnotch(
                    DP["chroma_audio_notch_freq"]
                    / (self._chroma_afc.getOutFreqHalf() * 1e6),
                    CHROMA_AUDIO_NOTCH_Q,
                )
            else:
                self.Filters["FChromaAudioNotch"] = sps.iirnotch(
                    DP["chroma_audio_notch_freq"] / (self.freq_hz_half),
                    CHROMA_AUDIO_NOTCH_Q,
                )

        # The following filters are for post-TBC:
        # The output sample rate is 4fsc
        self.Filters["FChromaFinal"] = self._chroma_afc.get_chroma_bandpass_final(
            self._options.color_under
        )

        if is_color_under:
            self.chroma_heterodyne = self._chroma_afc.getChromaHet()
            self.fsc_wave, self.fsc_cos_wave = self._chroma_afc.getFSCWaves()

        if self._chroma_afc.carrier_mult is not None:
            # SECAM method 1: post-TBC band-pass around the under carriers
            # ahead of the x4 phase multiplication.
            self.Filters["FSecamUnder"] = self._chroma_afc.get_secam_under_bandpass()
            # Porch carrier pair sanity check over the first fields, to catch
            # tapes that were actually recorded with the ME-SECAM method.
            self.secam_method_diag = {
                "fields": 0,
                "method1": 0,
                "mesecam": 0,
                "done": False,
            }

        # Long-term average of the measured ME-SECAM rest carrier pair offset,
        # used to trim the chroma up-conversion LO.
        self.secam_servo_avg = utils.StackableMA(min_watermark=2, window_average=60)
        lo_trim_seed = rf_options.get("secam_lo_trim", None)
        if lo_trim_seed is not None and system == "MESECAM":
            # Seed with a known trim (e.g. from a two-pass calibration decode)
            # so it applies from the first field. With the servo enabled it
            # keeps adapting from here; with it disabled this is a fixed trim.
            for _ in range(3):
                self.secam_servo_avg.push(float(lo_trim_seed))

        # Increase the cutoff at the end of blocks to avoid edge distortion from filters
        # making it through.
        self.blockcut_end = 1024

        if self._chroma_trap:
            self.chromaTrap = ChromaSepClass(
                self.freq_hz, self.SysParams["fsc_mhz"], ldd.logger
            )

        if self.useAGC:
            self.AGClevels = StackableMA(
                window_average=self.SysParams["FPS"] / 2
            ), StackableMA(window_average=self.SysParams["FPS"] / 2)

        self._field_averages = FieldAverage(
            self._options.cagc_fields,
            TRANSFER_AVERAGE_FIELDS,
        )

        # TODO: This should be managed elsewhere.
        self._compute_linelocs_issues = False

    @property
    def sysparams_const(self):
        return self._sysparams_const

    @property
    def sys_params(self):
        return self.SysParams

    @property
    def options(self):
        return self._options

    @property
    def notch(self):
        return self._notch

    @property
    def chroma_afc(self):
        return self._chroma_afc

    @property
    def do_cafc(self):
        return self._do_cafc

    @property
    def color_system(self):
        return self._color_system

    @property
    def dod_options(self):
        return self._dod_options

    @property
    def field_averages(self):
        return self._field_averages

    @property
    def compute_linelocs_issues(self):
        return self._compute_linelocs_issues

    @compute_linelocs_issues.setter
    def compute_linelocs_issues(self, value):
        self._compute_linelocs_issues = value

    def computefilters(self):
        # Override the stuff used in lddecode to skip generating filters we don't use.
        self.computevideofilters()
        self.computedelays()

    def computevideofilters(self):
        self.Filters = {}
        # Needs to be defined here as it's referenced in constructor.
        self.Filters["F05_offset"] = 32

    def _computevideofilters_b(self):
        # Use some shorthand to compact the code.
        SF = self.Filters
        DP = self.DecoderParams

        SF["hilbert"] = lddu.build_hilbert(self.blocklen)

        if DP.get("video_bpf_supergauss", False):
            self.Filters["RFVideo"] = gen_bpf_supergauss(
                DP["video_bpf_low"],
                DP["video_bpf_high"],
                DP["video_bpf_order"],
                self.freq_hz_half,
                self.blocklen,
            )[:-1]
            # Mirror to negative frequencies
            self.Filters["RFVideo"] = np.concatenate(
                (self.Filters["RFVideo"], np.flip(self.Filters["RFVideo"]))
            )
        else:
            # Filter for rf before demodulating.
            # Only use bpf if order defined - otherwise skip
            if DP.get("video_bpf_order", None):
                y_fm = utils.filtfft(
                    sps.butter(
                        DP["video_bpf_order"],
                        [
                            DP["video_bpf_low"] / self.freq_hz_half,
                            DP["video_bpf_high"] / self.freq_hz_half,
                        ],
                        btype="bandpass",
                    ),
                    self.blocklen,
                )
            else:
                y_fm = None

            # Gen fft filter from sos filter
            # TODO: Move this elsewhere
            def sosfiltfft(filter_value, block_len):
                return sps.sosfreqz(filter_value, block_len, whole=True)[1]

            y_fm_lowpass = sosfiltfft(
                sps.butter(
                    DP["video_lpf_extra_order"],
                    DP["video_lpf_extra"] / self.freq_hz_half,
                    btype="lowpass",
                    output="sos",
                ),
                self.blocklen,
            )

            y_fm_highpass = sosfiltfft(
                sps.butter(
                    DP["video_hpf_extra_order"],
                    DP["video_hpf_extra"] / self.freq_hz_half,
                    btype="highpass",
                    output="sos",
                ),
                self.blocklen,
            )

            if y_fm is not None:
                # Only use this if defined
                self.Filters["RFVideo"] = (
                    abs(y_fm) * abs(y_fm_lowpass) * abs(y_fm_highpass)
                )
            else:
                self.Filters["RFVideo"] = abs(y_fm_lowpass) * abs(y_fm_highpass)

        if DP.get("video_rf_peak_freq", False):
            # Add optional rf peaking filter
            from vhsdecode.addons.biquad import peaking

            peaking_filter = utils.filtfft(
                peaking(
                    DP["video_rf_peak_freq"] / self.freq_hz_half,
                    DP.get("video_rf_peak_gain", 3),
                    BW=DP.get("video_rf_peak_bandwidth", 2.5e6) / self.freq_hz_half,
                    type="constantq",
                ),
                self.blocklen,
            )
            self.Filters["RFVideo"] *= abs(peaking_filter)

        # b, a = ([1, -1], [1])
        # rf_eq = filtfft((b, a), self.blocklen)
        # self.Filters["rf_eq"] = b, a
        # self.Filters["rf_eq_fft"] = abs(rf_eq)

        # self.Filters["RFVideo"] *= abs(rf_eq)

        # Make sure this is an int in case it could be passed in as a string via the gui.
        if int(self.options.fm_audio_notch) > 0:
            if "fm_audio_channel_0_freq" in DP and "fm_audio_channel_1_freq" in DP:
                # Optionally enable double notch filter on fm audio channel frequencies.
                # This is mainly useful on VHS (and possibly PAL betamax with hifi?)
                # The hifi carriers on vhs are depth-multiplexed and read by a different head
                # but they still sometimes are picked up strongly enough by the video heads to
                # interfere with the video signal. The carrier for the upper channel especially since
                # it sits high enough that it it overlaps with the lower video sideband.
                # On formats where audio and video share the same heads (8mm, betamax NTSC hifi) the audio and
                # video bands are set up to be separatated more cleanly but for vhs cutting off the video sideband
                # above the audio carrier cuts off too much so use this approach instead and only if needed.
                audio_fm_notch_filter = gen_fm_audio_notch_params(
                    DP, self.options.fm_audio_notch, self.freq_hz_half, self.blocklen
                )
                self.Filters["RFVideo"] *= abs(audio_fm_notch_filter)
            else:
                ldd.logger.warning(
                    "Audio frequencies are not specified for this format, audio fm notch filters not enabled!"
                )

        if DP.get("boost_rf_linear_0", None) is not None:
            ramp = cvf.gen_ramp_filter_params(
                DP,
                self.freq_hz_half,
                self.blocklen,
            )

            self.Filters["RFVideo"] *= ramp
            if DP.get("boost_rf_linear_double", False):
                self.Filters["RFVideo"] *= ramp

        self.Filters["RFTop"] = sps.butter(
            1,
            [
                DP["boost_bpf_low"] / self.freq_hz_half,
                DP["boost_bpf_high"] / self.freq_hz_half,
            ],
            btype="bandpass",
            output="sos",
        )

        # Video (luma) main de-emphasis
        filter_deemp = gen_video_main_deemp_fft_params(DP, self.freq_hz, self.blocklen)

        if DP.get("video_lpf_supergauss", False) is True:
            filter_video_lpf = gen_video_lpf_supergauss_params(
                DP, self.freq_hz_half, self.blocklen
            )
        else:
            _, filter_video_lpf = gen_video_lpf_params(
                DP, self.freq_hz_half, self.blocklen
            )

        if DP.get("video_custom_luma_filters", None) is not None:
            self.Filters["FCustomVideo"] = gen_custom_video_filters(
                DP["video_custom_luma_filters"],
                self.freq_hz,
                self.blocklen,
            )
        else:
            self.Filters["FCustomVideo"] = 1.0

        # additional filters:  0.5mhz, used for sync detection.
        # Using an FIR filter here to get a known delay
        F0_5 = sps.firwin(65, [0.5 / self.freq_half], pass_zero=True)
        filter_05 = filtfft((F0_5, [1.0]), self.blocklen, False)

        # SF["F05"] = utils.filtfft((F0_5, [1.0]), self.blocklen)
        # Defined earlier
        # SF["F05_offset"] = 32

        # This filter is simple enough that we can get away with single precision
        # sections and thus do the filtering in sngle precision.
        # On higher order filters this is not viable as it tends to alter the filter too much.
        #
        # Applied by the dropout detector in `doc.py`, and by nothing else: the
        # envelope carried on the field is deliberately left at full bandwidth,
        # because a wide envelope crosses a dropout threshold repeatedly inside
        # one damaged region while the amplitude correction wants every bit of
        # the tape's own noise it can get.
        self.Filters["FEnvPost"] = sps.butter(
            1,
            DP.get("envelope_lpf_freq", ENVELOPE_LPF_FREQ_DEFAULT) / self.freq_hz_half,
            btype="lowpass",
            output="sos",
        )

        self.Filters["NLAmplitudeLPF"] = gen_nonlinear_amplitude_lpf(
            DP.get("nonlinear_amp_lpf_freq", NONLINEAR_AMP_LPF_FREQ_DEFAULT),
            self.freq_hz_half,
        )

        if self._use_fsc_notch_filter:
            self.Filters["fsc_notch"] = sps.iirnotch(
                self.sys_params["fsc_mhz"] / self.freq_half, 2
            )

        self.Filters["FDeemp"] = filter_deemp

        self.Filters["FVideo"] = (
            filter_deemp * filter_video_lpf * self.Filters["FCustomVideo"]
        )

        SF["FVideo05"] = filter_video_lpf * filter_deemp * filter_05

        # SF["YNRHighPass"] = sps.butter(
        #     1,
        #     (0.5e6) / self.freq_hz_half,
        #     btype="highpass",
        #     output="sos",
        # )

        if self.options.nldeemp or self.options.subdeemp:
            SF["NLHighPassF"] = gen_nonlinear_bandpass_params(
                DP, self.freq_hz_half, self.blocklen
            )

        if self.debug_plot and self.debug_plot.is_plot_requested("rf_luma"):
            from vhsdecode.debug_plot import plot_luma_rf

            plot_luma_rf(self, self.Filters["RFVideo"])

        if (
            self.debug_plot
            and self.debug_plot.is_plot_requested("nldeemp")
            and self.options.subdeemp
        ):
            from vhsdecode.nonlinear_filter import test_filter

            test_filter(
                self.Filters,
                self.freq_hz,
                self.blocklen,
                (self._sysparams_const.hz_ire * 143.0),
                self._sub_emphasis_params,
            )

        if self.debug_plot and self.debug_plot.is_plot_requested("deemphasis"):
            from vhsdecode.debug_plot import plot_deemphasis

            plot_deemphasis(self, filter_video_lpf, DP, filter_deemp)

    def computedelays(self, mtf_level=0):
        """Override computedelays
        It's normally used for dropout compensation, but the dropout compensation implementation
        in ld-decode assumes composite color. This function is called even if it's disabled, and
        seems to break with the VHS setup, so we disable it by overriding it for now.
        """
        # Set these to 0 for now, the metrics calculations look for them.
        self.delays = {}
        self.delays["video_sync"] = 0
        self.delays["video_white"] = 0

    def _demodulate_to_video(self, hilbert, envelope=None, calibrate=True):
        """Analytic RF to de-emphasised video - the whole post-demodulation chain.

        Factored out so it can be run a second time on the signal as it was
        BEFORE `--luma_eq` shaped it, when a debug plot asks to see what that
        stage changed. Both passes then go through the same spike replacement,
        video equalizer, chroma trap and de-emphasis stages, so what separates
        the two results is the equalizer and nothing else.

        `envelope` is the pre-equalization carrier amplitude, handed in where
        `--head_switch` wants its residual measured against the raw RF.

        Returns the de-emphasised video, the raw demodulated frequency, the
        latter's spectrum (which the caller needs for the half-bandwidth
        copy), and what the head switch correction subtracted, for the plot.
        """
        # FM demodulator
        # test1 = np.angle(hilbert)
        # from vhsd_rust import complex_angle_py
        # test2 = hilbert
        # print(test1 - test2)
        # np.savez_compressed("hilbert_data", data=hilbert)
        demod = unwrap_hilbert(hilbert, self.freq_hz)

        # The phase partner of the carrier amplitude residual - AM-induced
        # noise, with head switch steps and dropout edges as its coherent
        # extremes - cancelled first of all: the artifact is introduced
        # after the tape is read, so it is removed before every stage that
        # corrects what lies underneath it. See `head_switch`.
        head_switch_trace = None
        if self.options.head_switch != 0:
            head_switch_trace = head_switch.correct(
                self, demod, envelope, self.options.head_switch,
                calibrate=calibrate,
            )

        # The playback channel's measured low-frequency response - amplitude
        # and phase - inverted next: the electronics act on the signal after
        # the tape is read, so their response is undone before any stage
        # that reshapes the waveform and before the nonlinear ones. Stateless
        # and file-derived; see `baseband_eq`.
        baseband_eq_trace = None
        before_baseband_eq = None
        if self.options.baseband_eq != 0 and self._baseband_eq_declared:
            wants_trace = bool(
                self.debug_plot
                and (
                    self.debug_plot.is_plot_requested("luma_noise")
                    or self.debug_plot.is_plot_requested("sync_step_fold")
                )
            )
            before_baseband_eq = demod.copy() if wants_trace else None
            baseband_eq.correct(self, demod, self.options.baseband_eq)

        # The luma path's transient artifact, taken out where it is created and
        # before anything else has touched the signal. Everything below this
        # point - the spike replacement, the video equalizer, the chroma trap,
        # the de-emphasis stages - either reshapes the artifact or, in the case
        # of the nonlinear sub-de-emphasis, stops a model derived here from
        # composing at all. See `luma_transient`.
        if self.options.luma_transient != 0:
            luma_transient.correct(self, demod, self.options.luma_transient)

        # If there are obviously out of bounds values, do an extra demod on a diffed waveform and
        # replace the spikes with data from the diffed demod. (Which in practice is an extra EQed signal)
        if not self._disable_diff_demod:
            check_value = self.options.diff_demod_check_value

            if np.max(demod[20:-20]) > check_value:
                demod_b = unwrap_hilbert(
                    np.ediff1d(hilbert, to_begin=0), self.freq_hz
                ).real

                demod = replace_spikes(demod, demod_b, check_value)
                del demod_b
                # Not used yet, needs more testing.
                # 2.2 seems to be a sweet spot between reducing spikes and not causing
                # more
                if False:
                    demod = smooth_spikes(demod, check_value * 2.2)

        # Disabled if sharpness level is zero (default).
        # TODO: This should be done after the deemphasis steps
        if self._video_eq:
            # applies the video EQ
            demod = self._video_eq.filter_video(demod)

        # TODO: This should be done after the deemphasis steps
        if self._chroma_trap:
            # applies the Subcarrier trap
            demod = self.chromaTrap.work(demod)

        # applies main deemphasis filter
        demod_fft = npfft.rfft(demod)
        out_video_fft = demod_fft * self.Filters["FVideo"]
        if before_baseband_eq is not None:
            # What the baseband equalizer changed, carried through the same
            # de-emphasis as the picture and in IRE, so the plots fold it
            # against the decoded line - the site the sync-step measurement
            # is referenced to. One extra inverse transform, plot-only.
            out_before = npfft.irfft(
                npfft.rfft(before_baseband_eq) * self.Filters["FVideo"]
            ).real
            baseband_eq_trace = (
                (npfft.irfft(out_video_fft).real - out_before) / self.SysParams["hz_ire"]
            ).astype(np.float32)
        out_video = npfft.irfft(out_video_fft).real

        if self.options.nldeemp:
            # Extract the high frequency part of the signal
            hf_part = npfft.irfft(out_video_fft * self.Filters["NLHighPassF"])
            # Limit it to preserve sharp transitions
            np.clip(
                hf_part,
                self.DecoderParams["nonlinear_highpass_limit_l"],
                self.DecoderParams["nonlinear_highpass_limit_h"],
                out=hf_part,
            )

            # And subtract it from the output signal.
            out_video -= hf_part

        if self.options.subdeemp:
            out_video = sub_deemphasis(
                out_video,
                out_video_fft,
                self.Filters,
                self._sub_emphasis_params.deviation,
                self._sub_emphasis_params.exponential_scaling,
                self._sub_emphasis_params.scaling_1,
                self._sub_emphasis_params.scaling_2,
                self._sub_emphasis_params.logistic_mid,
                self._sub_emphasis_params.logistic_rate,
                self._sub_emphasis_params.static_factor,
            )

        del out_video_fft

        if self._use_fsc_notch_filter:
            out_video = sps.filtfilt(
                self.Filters["fsc_notch"][0], self.Filters["fsc_notch"][1], out_video
            )

        return out_video, demod, demod_fft, head_switch_trace, baseband_eq_trace

    def demodblock(
        self,
        data=None,
        mtf_level=0,
        fftdata=None,
        cut=False,
        thread_benchmark=False,
        block_start=None,
    ):
        """Demodulate one RF block.

        `block_start` is the block's absolute start sample on the capture,
        which the cache's worker supplies as `blocknum * blocksize` and which
        the RF transform picks a head's table by. None where a caller has no
        position to give; the transform then answers for an unplaced block.
        """
        rv = {}
        demod_block_debug = False
        demod_start_time = time.time()
        if fftdata is not None:
            indata_fft = fftdata
        elif data is not None:
            indata_fft = npfft.fft(data[: self.blocklen])
        else:
            raise Exception("demodblock called without raw or FFT data")

        if data is None:
            data = npfft.ifft(indata_fft).real

        if self.debug_plot and self.debug_plot.is_plot_requested("demodblock"):
            demod_block_debug = True
            # If we're doing a plot make a copy of the input to be able to plot it since we
            # are modifying the data in place.
            indata_fft_copy = indata_fft.copy()

        if self._notch is not None:
            indata_fft *= self.Filters["FVideoNotchF"]

        # Applies RF filters
        indata_fft *= self.Filters["RFVideo"]

        # The hilbert mask makes an analytic signal, so its magnitude is the
        # instantaneous carrier amplitude directly. Taking the magnitude of the
        # real part instead is a full wave rectification, which carries a
        # component at twice the carrier that the envelope would then have to be
        # filtered to remove. No delay is introduced either way - every filter in
        # this chain is zero phase - so there is none to compensate for.
        #
        # This is the same analytic signal the demodulator runs on, so it is kept
        # rather than transformed a second time. It is rebuilt further down only
        # if something between here and there has written to `indata_fft`.
        hilbert = npfft.ifft(indata_fft * self.Filters["hilbert"])

        # Single precision: the envelope drives dropout detection and the
        # color-under amplitude correction, and neither needs more.
        env = np.abs(hilbert).astype(np.single)
        env_mean = np.mean(env)

        # Boost high frequencies in areas where the signal is weak to reduce missed zero crossings
        # on sharp transitions. Using filtfilt to avoid phase issues.
        boosted = False
        if env.min() > 0:  # checks for zeroes on env
            if self._high_boost is not None:
                data_filtered = npfft.ifft(indata_fft).real
                high_part = sosfiltfilt_rust(self.Filters["RFTop"], data_filtered) * (
                    (env_mean * 0.9) / env
                )
                del data_filtered
                indata_fft += npfft.fft(high_part * self._high_boost)
                boosted = True
        else:
            ldd.logger.warning("RF signal is weak. Is your deck tracking properly?")

        # The luma path's own equalization, applied here and not with the RF
        # filters above, so that the envelope has already been taken from the
        # uncorrected signal. The envelope is a MEASUREMENT of the tape - it is
        # what the color-under correction reads the head-to-tape loss from, and
        # what dropout detection thresholds against - so a correction applied to
        # the luma must not reach it. Measured, folding this in with `RFVideo`
        # instead costs 17% of the chroma correction's benefit to buy 1.8% on
        # the luma.
        luma_eq = self.Filters.get("LumaPathEQ")
        # THE RF TRANSFORM'S TABLE stands between the two equalizers this
        # site already knows: an identified channel response the user
        # supplied (`ChannelEQ`, an offline file) outranks it, and it
        # outranks the luma path equalizer - whose measurement keeps running
        # underneath it on every field and is what feeds the transform.
        # Chosen per block by the head schedule from the block's absolute
        # start sample, on this block's own FFT grid; None until a table has
        # been published, and the luma equalizer then stands as before. It
        # is carried in `luma_eq` to the line below, whose text the
        # declaration anchors the `rf_eq` node on.
        if self.options.rf_transform != 0 and self.Filters.get("ChannelEQ") is None:
            transform_table = model_stages.rf_transform_table(
                self, block_start, indata_fft.size
            )
            if transform_table is not None:
                luma_eq = transform_table
        # The identified channel response (`channel_eq`) takes the luma
        # equalizer's place at this site while it is declared; with neither
        # present, nothing below touches the signal.
        rf_eq = self.Filters.get("ChannelEQ", luma_eq)
        unequalized = None
        if rf_eq is not None:
            if self._residual_channels_dir is not None or (
                self.debug_plot and self.debug_plot.is_plot_requested("luma_noise")
            ):
                # The spectrum the equalizer is about to shape, demodulated
                # separately below so the plot can show what it changed - and
                # so the frequency-axis residual channel can carry it. Taken
                # after the high frequency boost, if that fired, so the two
                # differ by the equalizer alone.
                unequalized = (
                    npfft.ifft(indata_fft * self.Filters["hilbert"])
                    if boosted
                    else hilbert
                )
            indata_fft = indata_fft * rf_eq

        # Only the two branches above can have moved `indata_fft` since the
        # analytic signal was taken; where neither did, it still holds.
        if boosted or rf_eq is not None:
            hilbert = npfft.ifft(indata_fft * self.Filters["hilbert"])

        # THE LUMA'S QUADRATURE IMAGE, in the chroma's own correction
        # pattern: a conjugate map on the carrier's complex baseband, which
        # no table can carry, applied to the analytic signal once the
        # transform has latched and published the pair. The envelope above
        # was taken before it, as before every luma correction at this site.
        if self.options.rf_transform != 0 and block_start is not None:
            luma_image = model_stages.rf_transform_image(self)
            if luma_image is not None:
                hilbert = model_stages.apply_luma_image(
                    hilbert, luma_image, block_start, self.freq_hz
                )

        if not demod_block_debug:
            del indata_fft

        out_video, demod, demod_fft, head_switch_trace, baseband_eq_trace = (
            self._demodulate_to_video(hilbert, envelope=env)
        )

        # The same field as it would have been without the equalizer, for the
        # plot that draws the two against each other. Built only where that
        # plot asked `unequalized` to be kept. It receives the same head
        # switch correction as the main pass, so the difference between the
        # two stays the equalizer's alone - but it must not feed the
        # correction's calibration: a differently equalized channel voting
        # in the same regression would bias the measured gain.
        if unequalized is not None:
            unequalized = self._demodulate_to_video(
                unequalized, envelope=env, calibrate=False
            )[0]

        out_video05 = npfft.irfft(demod_fft * self.Filters["FVideo05"]).real
        out_video05 = np.roll(out_video05, -self.Filters["F05_offset"])

        # Filter out the color-under signal from the raw data.
        chroma_source = data if self.options.color_under else out_video
        out_chroma = (
            chroma_color_under_filter(
                chroma_source,
                self.Filters["FVideoBurst"],
                self.blocklen,
                self.Filters["FVideoNotch"],
                self._notch,
                move=int(self.options.chroma_offset),
                audio_notch=self.Filters.get("FChromaAudioNotch", None),
                # TODO: Do we need to tweak move elsewhere too?
                # if cafc is enabled, this filtering will be done after TBC
            )
            if not self._do_cafc
            else data[: self.blocklen]
        )

        if self.debug_plot and self.debug_plot.is_plot_requested("magdens"):
            from vhsdecode.debug_plot import plot_magnitude_density

            plot_magnitude_density(
                raw_data=data[: self.blocklen],
                filtered_data=npfft.ifft(indata_fft).real,
                rfdecode=self,
            )

        if demod_block_debug:
            from vhsdecode.debug_plot import plot_input_data

            plot_input_data(
                raw_data=data,
                filtered_data=npfft.ifft(indata_fft).real,
                env=env,
                env_mean=env_mean,
                raw_fft=indata_fft_copy,
                filtered_fft=indata_fft,
                demod_video=demod,
                filtered_video=out_video,
                chroma=out_chroma,
                rf_filter=self.Filters["RFVideo"],
                rfdecode=self,
                plot_chroma_fft=True,
            )

        if self.options.export_raw_tbc:
            out_video = demod

        # demod_burst is a bit misleading, but keeping the naming for compatability.
        if (
            self.options.chroma_env_gain > 0
            or self.options.luma_eq != 0
            or self.options.luma_transient != 0
            # --head_switch consumes the amplitude stage's block model, which
            # is fed by the field-level measurement, which needs this channel.
            or self.options.head_switch != 0
            # --carrier_tbc refines the time base from the carrier's own
            # sync-edge trace, which is measured on this channel.
            or self.options.carrier_tbc != 0
            # The RF transform is fed by the same field-level amplitude
            # measurement `luma_eq` is, and that measurement reads this
            # channel.
            or self.options.rf_transform != 0
            # --baseband_eq's head_switch prior and plots read this channel.
            or (self.options.baseband_eq != 0 and self._baseband_eq_declared)
            # --channel_eq's plots and residual channels read it too.
            or (self.options.channel_eq != 0 and self._channel_eq_declared)
            # --residual_channels carries it as the amplitude channel's
            # abscissa (the envelope against the instantaneous carrier).
            or self._residual_channels_dir is not None
        ):
            # The color-under amplitude correction models the carrier amplitude
            # as a function of the instantaneous carrier frequency, so it needs
            # the demodulated frequency before de-emphasis - the de-emphasised
            # output understates deviation above the de-emphasis corner, which
            # is well inside the bandwidth the envelope is measured over.
            # "demod_raw" is the name lddecode's own demodblock already uses for
            # this signal. Only carried when the correction is enabled, so the
            # extra channel costs nothing otherwise.
            video_out = {
                "demod": out_video,
                "demod_raw": demod.astype(np.float32),
                "demod_05": out_video05,
                "demod_burst": out_chroma,
                "envelope": env,
            }
        else:
            video_out = {
                "demod": out_video,
                "demod_05": out_video05,
                "demod_burst": out_chroma,
                "envelope": env,
            }

        if unequalized is not None:
            # Carried as a channel so it is cut and assembled with the rest and
            # arrives at the plot on the same sample grid as `demod`.
            video_out["demod_noeq"] = unequalized

        if (
            self.options.head_switch != 0
            and self.debug_plot
            and self.debug_plot.is_plot_requested("luma_noise")
        ):
            # What the head switch correction subtracted, in IRE. A channel
            # for the same reason as `demod_noeq`: it is cut and assembled
            # with the rest, so the plot draws it against the very samples it
            # corrected. Every block must carry it once the flag and the plot
            # are on - the block assembly concatenates by name - so a block
            # where nothing was subtracted (or the model does not exist yet)
            # carries zeros.
            video_out["head_switch"] = (
                head_switch_trace
                if head_switch_trace is not None
                else np.zeros(len(out_video), dtype=np.single)
            )

        if (
            self.options.baseband_eq != 0
            and self._baseband_eq_declared
            and self.debug_plot
            and (
                self.debug_plot.is_plot_requested("luma_noise")
                or self.debug_plot.is_plot_requested("sync_step_fold")
            )
        ):
            # What the baseband equalizer changed, in IRE - a channel for the
            # same reason as `head_switch`, zeros where nothing was applied.
            video_out["baseband_eq"] = (
                baseband_eq_trace
                if baseband_eq_trace is not None
                else np.zeros(len(out_video), dtype=np.single)
            )

        if self._residual_channels_dir is not None:
            # The residual channels ride the video dict to the field, where
            # they are downscaled to 4fsc on the final time base
            # (`residual_channels`). The stages' own traces are carried where
            # those stages run and no plot already carries them; zeros where
            # a block had nothing applied, as their plot channels do.
            for name, trace, active in (
                ("head_switch", head_switch_trace, self.options.head_switch != 0),
                (
                    "baseband_eq",
                    baseband_eq_trace,
                    self.options.baseband_eq != 0 and self._baseband_eq_declared,
                ),
            ):
                if active and name not in video_out:
                    video_out[name] = (
                        trace
                        if trace is not None
                        else np.zeros(len(out_video), dtype=np.single)
                    )

        rv["video"] = (
            {
                name: channel[self.blockcut : -self.blockcut_end]
                for name, channel in video_out.items()
            }
            if cut
            else video_out
        )

        demod_end_time = time.time()
        if thread_benchmark:
            ldd.logger.debug(
                "Demod thread %d, work done in %.02f msec"
                % (os.getpid(), (demod_end_time - demod_start_time) * 1e3)
            )

        return rv
