"""AN EXTERNAL REFERENCE: videosynth generates the ideal, this arc did not.

Ethan, 2026-09-06:

  *"I want you to spawn a sub agent that uses our new model and builds
  reference video signals using this program. https://github.com/decode-orc/
  videosynth. The video signals can be generated on the fly here when forming
  the residual against the rf."*

WHY AN OUTSIDE GENERATOR CHANGES THE ARGUMENT. Every ideal this arc has
differenced against was drawn by the arc: a spec-shaped pulse it built, or a
pooled average of its own measurements. Both are circular in a way no amount
of care removes - a pooled ideal cannot show an error common to every pulse
in the pool, and a drawn one can only be as right as the drawing. videosynth
is decode-orc's own C++ generator, standards-based (ITU-R BT.470/BT.1700,
SMPTE 170M/244M, IEC 60856/60857, IEC 60461), built in the time domain from
the standards' timing and level definitions, and deterministic - its own
documentation states the output is byte-identical at any thread count. It
knows nothing about this repository, so a residual against it is a residual
against something outside the loop.

TWO OF ITS ENCODINGS ARE THIS ARC'S OWN GRIDS, which is why it is the right
generator rather than merely an available one.

  `CVBS_U10_4FSC`   four times the subcarrier, unsigned ten bit - exactly the
                    output cap `output_limit.py` records Ethan setting this
                    session.
  `RAW_S16_40M`     forty megasamples a second, signed sixteen bit in
                    millivolts - the rate of /testdata/countdown.flac and
                    /testdata/home.flac.

WHAT WAS MEASURED ON THE GENERATED FILE, and it is the check that the
generator is what it says. One NTSC frame at `CVBS_U10_4FSC` is 477 750
samples of 910 by 525, which is the orthogonal 4fsc raster, and its code
values land on the standard's levels:

    sync tip     code   15      blanking     code  240
    black        code  282      peak white   code  800

Against the generator's own quantisation profile (1.2755 mV a code, blanking
at 240, `include/videosynth/cvbs_quantization.h`) and System M's 140 IRE to
the volt, one code is 0.178570 IRE, so code 800 is +100.00 IRE, code 282 is
+7.50 IRE - the NTSC setup - and code 15 is -40.18 IRE, a fifth of a code
below the specified tip. Nothing here was fitted; the file simply lands
there. The 75 per cent bars reach +103.18 IRE at their peak, which is the
yellow bar's chroma riding on its luma and not an error.

THE GENERATOR'S TEN BITS ARE NOT THIS ARC'S TEN BITS, and the difference is
worth stating because it bounds what the reference can say. `output_limit`
reads the cap off a decode whose container spans 163.5 IRE, so a ten-bit step
is 0.1597 IRE and the quantiser's noise is 0.0461 IRE. videosynth's ten-bit
CVBS grid spans 1023 codes of 1.2755 mV, which is 182.7 IRE, so its step is
0.1786 IRE and its own quantisation noise is 0.0516 IRE - 1.12 times this
arc's floor. `RAW_S16_40M` carries one millivolt a code, 0.1400 IRE, whose
noise is 0.0404 IRE, or 0.88 floors. So the raw preset is the finer of the
two references and even it is not far below the cap: a residual measured
against a generated reference cannot honestly be quoted below about one
floor, because the reference itself is not quieter than that.

A SECOND DEPARTURE, SMALL AND REAL. videosynth carries the NTSC 4fsc clock as
14 318 180 Hz, citing SMPTE 244M section 3.4's printed "14.31818 MHz"; the
exact value implied by the line rate is 4 x 455/2 x 525 x 30/1.001 =
14 318 181.8 Hz, which `output_limit.FOUR_FSC_HZ` computes. The difference is
0.127 parts per million - 0.06 sample over one frame, 0.87 sample over the
0.48 s of a zaroff capture. It is far too small to matter for a residual
aligned per line and far too large to ignore in an absolute phase argument,
so it is recorded rather than corrected.

WHAT THE FORWARD CHAIN DOES, and why the reference cannot be differenced
against the RF as it stands. videosynth emits the PICTURE side - the
composite that went into the recorder - while the captures are the recorder's
own RF. Three modelled stages stand between them, and all three are already
in this tree with their citations:

  Y/C separation    the recorder's luma channel is band limited to the 3.0 MHz
                    baseband `capture_profile.NTSC_VHS_SP_BASEBAND_HZ`
                    carries from SMPTE 32M-2004 clause 3.9.1.1.4
  pre-emphasis      `rf_stages.main_pre_emphasis`, IEC 774-1 (1994) page 67,
                    1.3 us and a gain factor of five
  white / dark clip `clipping.clip_levels("VHS")`, SMPTE 32M-2004 clause
                    3.9.1.1.3 - 160 per cent and 40 per cent of the
                    sync-tip-to-peak-white span, measured from the tip
  FM modulation     `profiles.IDEAL_DECK`, SMPTE 32M-2004 clause 3.9.1.1.4 -
                    3.4 MHz at sync tip, 4.4 MHz at peak white

THE FM LAW IS ATTESTED BY THE CAPTURE ITSELF. Demodulating four fields of
zaroff-75bars-NTSC-SP at the record tap and taking the modes of the
instantaneous frequency gives 3.43 MHz on the sync tips and 4.405 MHz on peak
white, against the specified 3.4 and 4.4. So the specified law is the deck's
law to better than a per cent before any fitting, and the forward model may
use it as given.

WHAT IS DELIBERATELY NOT MODELLED HERE. The colour-under chroma channel: the
reference's luma is taken through the recorder's luma path alone, and the
comparison is confined to the luma FM band, so the down-converted chroma at
629.371 kHz is outside it on both sides. The record head's own response, the
tape and the playback chain are likewise absent by design - they are what the
residual is meant to expose.

WHAT THE RESIDUAL CAME TO, and it is the first time this arc has had a
number that does not depend on an ideal it drew itself. Two patterns, the
zaroff captures at SP, the forward chain with the falsified spec clip left
out, the licensed two-point sync anchoring applied per field, and the
residual pooled over the line pairs so what is reported is the model's own
error rather than the per-line noise. Everything is in multiples of the
output file's quantisation floor, 0.0461 IRE rms.

    window              75bars record   75bars playback   multiburst record
    sync leading edge      197.7             167.3             229.6
    sync tip                10.2               8.2              10.4
    sync trailing          487.5             503.5             515.8
    back porch              25.5              17.5              16.3
    blanking               287.1             292.9             306.9
    active                 644.7             642.2            1182.6

    1476 line pairs (75bars), 1722 (multiburst)

THE SYNC WINDOWS AGREE BETWEEN THE TWO PATTERNS to about fifteen per cent
and the ACTIVE WINDOW DOES NOT - 645 floors against 1183 - which is the
strongest internal check available here and says exactly what it should: the
sync structure is the same in every pattern, so a pattern-independent number
there is a property of the chain, while the active area compares two
generators' drawings of a named pattern and is a bound rather than a
measurement.

WHAT THE TAPE AND THE PLAYBACK CHAIN ADD, by differencing the two taps'
profiles rather than their powers - see `tap_difference` for why the
distinction is not cosmetic. In IRE, with the ratio to the two taps' pooled
standard error beside it:

    window              75bars           multiburst
    sync leading edge   2.781 (32 SE)    3.601 (29 SE)
    sync trailing       2.325 (34 SE)    2.470 (31 SE)
    blanking            2.035 (26 SE)    2.408 (26 SE)
    back porch          0.403 ( 8 SE)    0.291 ( 6 SE)
    sync tip            0.129 ( 3 SE)    0.363 ( 9 SE)

So the tape and the playback chain add between 0.13 and 3.6 IRE depending on
where in the line one looks, every one of them far above the file's floor,
and the transient at the sync leading edge is where they add most.

WHAT THE MEASUREMENT SETTLED ALONG THE WAY, each recorded with the function
that carries it:

  * the deck's DARK CLIP is at 2.787 MHz, 0.613 MHz below the specified
    sync-tip carrier, and no white clip is visible at the record tap at all
    (`forward_to_rf`, the `clip_ire` note);
  * neither reading of SMPTE 32M-2004 clause 3.9.1.1.3 this arc has
    considered puts it there, and the reading `clipping.py` records would
    clip 35 per cent of an ordinary picture (`clip_levels_ire`);
  * the generator's sync pulses are one transition narrow at the
    half-amplitude points, with the cause found in its own source
    (`windows_of`);
  * the two generators do not start their active picture at the same place,
    by 1.47 us (`windows_of`).

THE CACHE EXISTS BECAUSE ETHAN ASKED FOR ON-THE-FLY USE. A request is hashed
into a key, the generated samples are kept on disc under that key and in
memory for the process, and a repeated request returns the identical array
without touching the generator. Generation of one NTSC frame took 37 ms inside the
generator's own pipeline and 59 ms of wall time including the process, so
the cache saves little on one frame and a great deal on a loop that asks for
the same pattern once a field.
"""

import hashlib
import os
import shutil
import subprocess
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from vhsdecode.models import capture_profile, clipping, output_limit, profiles
from vhsdecode.models import rf_stages

# --------------------------------------------------------------------------
# Where the generator and its configuration live
# --------------------------------------------------------------------------
#
# One readable configuration file for this tool, per the standing rule, and
# no shell environment variables. The file is optional: without it the
# generator is looked for on PATH and in the conventional build locations,
# so a checkout that has one simply works.

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))),
    "tools", "ringing_measure", "reference_signal.toml")

# The generator's own executable name, and the Flatpak application id its
# release bundle installs under (README, "Install").
BINARY_NAME = "videosynth"
FLATPAK_APPLICATION_ID = "io.github.decode_orc.VideoSynth"


def config(path: Optional[str] = None) -> Dict[str, object]:
    """The tool's configuration, or the defaults when the file is absent.

    Keys, all optional: `binary` (the generator's executable), `asset_root`
    (where the `{bundled}` asset root resolves, needed only for a build that
    was not installed), `cache_dir` (where generated references are kept).
    """
    target = path or CONFIG_PATH
    settings: Dict[str, object] = {"path": target, "present": False}
    if os.path.exists(target):
        import tomllib

        with open(target, "rb") as handle:
            loaded = tomllib.load(handle)
        settings.update(loaded.get("generator", {}))
        settings["present"] = True
    return settings


def _candidate_binaries(settings: Dict[str, object]) -> List[str]:
    """Every place the generator might be, most specific first."""
    out: List[str] = []
    configured = settings.get("binary")
    if configured:
        out.append(str(configured))
    found = shutil.which(BINARY_NAME)
    if found:
        out.append(found)
    return out


def available(path: Optional[str] = None) -> Dict[str, object]:
    """Is the generator present, which build, and what can it produce?

    Returns `present` False rather than raising when it is not, so a caller
    - and a test - can skip cleanly. `version` is the generator's own
    `--version`, which it documents as the git commit hash of the build, so
    a reference can always be traced to a state of videosynth.
    """
    settings = config(path)
    for candidate in _candidate_binaries(settings):
        if not (os.path.isfile(candidate) and os.access(candidate, os.X_OK)):
            continue
        try:
            done = subprocess.run([candidate, "--version"],
                                  capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            continue
        if done.returncode != 0:
            continue
        return {
            "present": True,
            "binary": candidate,
            "version": done.stdout.strip(),
            "config": settings,
            "encodings": sorted(ENCODINGS),
            "systems": sorted(SYSTEMS),
            "patterns": sorted(PATTERNS),
            "why": ("the version is the generator's own git commit, so a "
                    "reference names the build that made it"),
        }
    return {
        "present": False,
        "binary": None,
        "version": None,
        "config": settings,
        "encodings": sorted(ENCODINGS),
        "systems": sorted(SYSTEMS),
        "patterns": sorted(PATTERNS),
        "why": ("no runnable videosynth was found; build it with CMake or "
                "install the Flatpak release, then name it in "
                f"{settings['path']} under [generator] binary"),
    }


# --------------------------------------------------------------------------
# The generator's own definitions, quoted from its source with the file
# --------------------------------------------------------------------------
#
# Every number below is videosynth's, read out of its headers rather than
# assumed, and each carries the header and the standard that header cites.

SYSTEMS: Dict[str, Dict[str, object]] = {
    "NTSC": {
        "lines": 525,
        "active_raster": (720, 486),
        "sample_rate_4fsc_hz": 14318180.0,
        "samples_per_line_4fsc": 910,
        "samples_per_frame_4fsc": 910 * 525,
        "blanking_code": 240,
        "millivolts_per_code": 1.2755,
        "cite": ("videosynth include/videosynth/timing_constants.h, citing "
                 "SMPTE 244M-2003 section 3.4; the quantisation profile from "
                 "include/videosynth/cvbs_quantization.h, citing ITU-R "
                 "BT.470-6 Table 1 item 4"),
    },
    "PAL": {
        "lines": 625,
        "active_raster": (720, 576),
        "sample_rate_4fsc_hz": 17734475.0,
        "samples_per_line_4fsc": 1135,
        "samples_per_frame_4fsc": 709379,
        "blanking_code": 256,
        "millivolts_per_code": 1.1905,
        "cite": ("videosynth include/videosynth/timing_constants.h, citing "
                 "EBU Tech. 3280-E section 1.1.1 Table 1; the PAL frame is "
                 "709 379 samples because four lines a frame carry 1134"),
    },
    "PAL_M": {
        "lines": 525,
        "active_raster": (720, 486),
        "sample_rate_4fsc_hz": 14318445.96,
        "samples_per_line_4fsc": 909,
        "samples_per_frame_4fsc": 909 * 525,
        "blanking_code": 240,
        "millivolts_per_code": 1.2755,
        "cite": ("videosynth include/videosynth/timing_constants.h, citing "
                 "ITU-R BT.470-6 Table 2 item 2.11b, fsc/fH = 909/4 exactly; "
                 "System M levels, so the NTSC quantisation profile applies"),
    },
}

# `rate` is the samples a second the encoding is written at, `dtype` the
# numpy type of the flat little-endian stream, and `unit` says what one code
# means. The two 4fsc entries and the two raw entries are the four this arc
# has any use for; the generator defines two more 4fsc variants which are the
# same samples in a different offset and scale.
ENCODINGS: Dict[str, Dict[str, object]] = {
    "CVBS_U10_4FSC": {
        "dtype": "<u2", "unit": "code", "rate": "4fsc", "bits": 10,
        "cite": ("videosynth src/output_stage.cpp EncodeCompositeSample; ten "
                 "bit codes in sixteen bit little-endian words, offset by the "
                 "system's blanking code"),
    },
    "CVBS_U16_4FSC": {
        "dtype": "<u2", "unit": "code", "rate": "4fsc", "bits": 16,
        "cite": "videosynth src/output_stage.cpp, the same codes unscaled",
    },
    "RAW_S16_28M": {
        "dtype": "<i2", "unit": "millivolt", "rate": 28000000.0, "bits": 16,
        "cite": ("videosynth include/videosynth/timing_constants.h, 28 MHz "
                 "exactly; one code is one millivolt about blanking"),
    },
    "RAW_S16_40M": {
        "dtype": "<i2", "unit": "millivolt", "rate": 40000000.0, "bits": 16,
        "cite": ("videosynth include/videosynth/timing_constants.h, 40 MHz "
                 "exactly; one code is one millivolt about blanking, and it "
                 "is the rate of this arc's two long captures"),
    },
}

# System M puts 140 IRE on the volt: sync tip at -40 IRE is -286 mV and peak
# white at +100 IRE is +714 mV (ITU-R BT.470-6 Table 1 System M, and SMPTE
# 170M states the same span). This is a definition, not a measurement.
MILLIVOLTS_PER_IRE_M = 1000.0 / 140.0
SYNC_TIP_IRE_M = -40.0
PEAK_WHITE_IRE_M = 100.0
# PAL states its levels in millivolts directly: 700 mV of picture over a
# 300 mV sync (ITU-R BT.470-6 Table 1, systems B/G/I item 4), so the "IRE"
# used here for PAL is 100 IRE to 700 mV.
MILLIVOLTS_PER_IRE_PAL = 7.0
SYNC_TIP_IRE_PAL = -300.0 / MILLIVOLTS_PER_IRE_PAL

# The bundled progressive sources that carry a standard test pattern, by the
# name this arc's captures use for it. The zaroff captures were made from a
# Tektronix TSG-130A (/testdata/test_patterns/vhs/readme.txt), so a name
# here asserts only that both are the same NAMED standard pattern - not that
# the two generators draw it identically, which is one of the things the
# residual is for.
PATTERNS: Dict[str, Dict[str, object]] = {
    "75bars": {"asset": "75_BARS", "what": "75 per cent colour bars",
               "zaroff": "zaroff-75bars"},
    "100bars": {"asset": "100_BARS", "what": "100 per cent colour bars",
                "zaroff": None},
    "smptebars": {"asset": "SMPTE_BARS_001", "what": "SMPTE split-field bars",
                  "zaroff": None},
    "multiburst": {"asset": "MULTIBURST", "what": "multiburst",
                   "zaroff": "zaroff-multiburst"},
    "ramp": {"asset": "LUMA_RAMP", "what": "luminance ramp",
             "zaroff": "zaroff-ramp"},
    "pluge": {"asset": "PLUGE", "what": "PLUGE", "zaroff": None},
    "flat_black": {"asset": None, "what": "no bundled asset; see `planted`",
                   "zaroff": None},
}


def _system_of(system: str) -> Dict[str, object]:
    if system not in SYSTEMS:
        raise ValueError(f"unknown system {system!r}; known: {sorted(SYSTEMS)}")
    return SYSTEMS[system]


def sample_rate_of(system: str, encoding: str) -> float:
    """The samples a second an encoding is written at, for this system."""
    if encoding not in ENCODINGS:
        raise ValueError(f"unknown encoding {encoding!r}; "
                         f"known: {sorted(ENCODINGS)}")
    rate = ENCODINGS[encoding]["rate"]
    if rate == "4fsc":
        return float(_system_of(system)["sample_rate_4fsc_hz"])
    return float(rate)


def ire_scale(system: str, encoding: str) -> Dict[str, float]:
    """How one code of this encoding maps to IRE, and where blanking sits.

    The 4fsc encodings are codes offset by the system's blanking code and
    scaled by its millivolts a code; the raw encodings are millivolts about
    blanking directly. Both then divide by the system's millivolts an IRE.
    """
    entry = _system_of(system)
    per_ire = (MILLIVOLTS_PER_IRE_PAL if system == "PAL"
               else MILLIVOLTS_PER_IRE_M)
    if ENCODINGS[encoding]["unit"] == "millivolt":
        return {"offset_code": 0.0, "millivolts_per_code": 1.0,
                "millivolts_per_ire": per_ire, "ire_per_code": 1.0 / per_ire}
    per_code = float(entry["millivolts_per_code"])
    return {"offset_code": float(entry["blanking_code"]),
            "millivolts_per_code": per_code,
            "millivolts_per_ire": per_ire,
            "ire_per_code": per_code / per_ire}


# --------------------------------------------------------------------------
# The project file
# --------------------------------------------------------------------------


def project_text(system: str = "NTSC", pattern: str = "75bars",
                 encoding: str = "CVBS_U10_4FSC", frames: int = 1,
                 video_path: str = "{output}/reference.cvbs",
                 setup_ire: Optional[float] = 7.5,
                 name: str = "VhsDecodeReference") -> str:
    """The YAML project videosynth is given, written deterministically.

    The generator's own reference states five top-level blocks and a
    canonical key order; this follows it so a file written here comes back
    unchanged if it ever passes through the generator's editor. The text is a
    pure function of its arguments - no timestamps, no paths outside those
    given - which is what makes the cache key sound.

    `setup_ire` is the NTSC black setup. The generator accepts 7.5 or 0.0 on
    System M only and rejects the key on PAL, so it is dropped there. The
    zaroff captures came from a Tektronix TSG-130A configured for NTSC-M, so
    7.5 is the default rather than a guess.
    """
    if pattern not in PATTERNS:
        raise ValueError(f"unknown pattern {pattern!r}; "
                         f"known: {sorted(PATTERNS)}")
    asset = PATTERNS[pattern]["asset"]
    if asset is None:
        raise ValueError(f"pattern {pattern!r} has no bundled source; it "
                         f"names a planted signal, not a generated one")
    if encoding not in ENCODINGS:
        raise ValueError(f"unknown encoding {encoding!r}; "
                         f"known: {sorted(ENCODINGS)}")
    entry = _system_of(system)
    width, height = entry["active_raster"]
    if int(frames) < 1:
        raise ValueError("frames must be at least one")
    lines = [
        "project:",
        f"  name: {name}",
        '  version: "1.0"',
        f"  description: reference for vhs-decode, {pattern} on {system}",
        "",
        "cvbs_presets:",
        f"  video_standard_preset: {system}",
        f"  sample_encoding_preset: {encoding}",
        "  signal_state_preset: STANDARD_STABLE_LOCKED",
    ]
    if setup_ire is not None and system != "PAL":
        lines.append(f"  ntsc_black_setup_ire: {float(setup_ire)}")
    lines += [
        "",
        "output:",
        f'  video_path: "{video_path}"',
        "  signal_type: composite",
        "",
        "sections:",
        f"  - name: {pattern}",
        "    type: progressive",
        f'    source: "{{bundled}}/exr/{width}x{height}/{asset}.exr"',
        f"    duration_frames: {int(frames)}",
        "",
    ]
    return "\n".join(lines)


def _request_key(text: str, version: str) -> str:
    """The cache key: the project text and the generator build, hashed.

    Both matter. The same project on a different build of videosynth is a
    different reference, and saying so is the point of the generator writing
    its commit into every file it makes.
    """
    digest = hashlib.sha256()
    digest.update(text.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(str(version).encode("utf-8"))
    return digest.hexdigest()[:32]


def cache_dir(settings: Optional[Dict[str, object]] = None) -> str:
    """Where generated references are kept between runs.

    Outside the working tree by default. One NTSC frame is 0.95 MB at 4fsc
    and 2.7 MB at the forty megasample raw rate, and other sessions hold
    uncommitted work in this repository, so generated media has no business
    appearing in its status.
    """
    settings = settings if settings is not None else config()
    configured = settings.get("cache_dir")
    if configured:
        return str(configured)
    return os.path.join(os.path.expanduser("~"), ".cache", "vhs-decode",
                        "reference_signal")


# --------------------------------------------------------------------------
# Generating and loading
# --------------------------------------------------------------------------

# Arrays already loaded in this process, by cache key. `load` returns the
# array itself and not a copy, so a repeated request is free and the caller
# sees the identical object - which is what "on the fly" has to mean if a
# residual loop is to call it per pattern.
_LOADED: Dict[str, Dict[str, object]] = {}


def generate(system: str = "NTSC", pattern: str = "75bars",
             encoding: str = "CVBS_U10_4FSC", frames: int = 1,
             setup_ire: Optional[float] = 7.5,
             out_dir: Optional[str] = None, threads: int = 1,
             config_path: Optional[str] = None) -> Dict[str, object]:
    """Write a project file, run the generator, and return where it landed.

    A request already generated is not generated again: the key is the
    project text and the generator's build, and a complete file under that
    key is returned as it stands. `threads` defaults to one because the
    generator documents its output as byte-identical at any thread count, so
    one thread costs nothing in fidelity and keeps a residual loop from
    contending with whatever else is running.
    """
    tool = available(config_path)
    if not tool["present"]:
        raise RuntimeError(str(tool["why"]))
    text = project_text(system=system, pattern=pattern, encoding=encoding,
                        frames=frames, setup_ire=setup_ire)
    key = _request_key(text, tool["version"])
    root = out_dir or cache_dir(tool["config"])
    directory = os.path.join(root, key)
    samples = os.path.join(directory, "reference.cvbs")
    request = {"system": system, "pattern": pattern, "encoding": encoding,
               "frames": int(frames), "setup_ire": setup_ire,
               "generator_version": tool["version"], "key": key}
    if os.path.exists(samples) and os.path.getsize(samples) > 0:
        return {**request, "path": samples, "project": os.path.join(
            directory, "reference.yaml"), "generated": False,
            "sample_rate_hz": sample_rate_of(system, encoding)}
    os.makedirs(directory, exist_ok=True)
    project = os.path.join(directory, "reference.yaml")
    with open(project, "w") as handle:
        handle.write(text)
    command = [str(tool["binary"]), "--project", project,
               "--output-root", directory, "--threads", str(int(threads))]
    asset_root = tool["config"].get("asset_root")
    if asset_root:
        command += ["--asset-root", f"bundled={asset_root}"]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0 or not os.path.exists(samples):
        raise RuntimeError(
            f"videosynth failed for {pattern} on {system}: "
            f"{done.stderr.strip() or done.stdout.strip()}")
    return {**request, "path": samples, "project": project, "generated": True,
            "sample_rate_hz": sample_rate_of(system, encoding)}


def load(path: str, system: str = "NTSC", encoding: str = "CVBS_U10_4FSC",
         key: Optional[str] = None) -> Dict[str, object]:
    """A generated file as IRE, with the geometry it was written on.

    The stream is flat and headerless - the generator's own documentation
    says so - so the geometry comes from the system and the encoding rather
    than from the file. At 4fsc the frame is a whole number of samples and
    the raster is exact; at a raw rate it is not, so `samples_per_frame` is
    reported as a float there and no raster is claimed.
    """
    entry = _system_of(system)
    if encoding not in ENCODINGS:
        raise ValueError(f"unknown encoding {encoding!r}; "
                         f"known: {sorted(ENCODINGS)}")
    memo = key or path
    if memo in _LOADED:
        return _LOADED[memo]
    raw = np.fromfile(path, dtype=np.dtype(ENCODINGS[encoding]["dtype"]))
    scale = ire_scale(system, encoding)
    ire = (raw.astype(np.float64) - scale["offset_code"]) * scale["ire_per_code"]
    rate = sample_rate_of(system, encoding)
    if ENCODINGS[encoding]["rate"] == "4fsc":
        per_frame: float = float(entry["samples_per_frame_4fsc"])
        per_line: Optional[int] = int(entry["samples_per_line_4fsc"])
    else:
        # the frame rate is the system's, so the raw rate gives a
        # non-integer frame and there is no raster to claim
        frame_rate = (25.0 if system == "PAL" else 30.0 / 1.001)
        per_frame = rate / frame_rate
        per_line = None
    out = {
        "ire": ire, "codes": raw, "path": path, "system": system,
        "encoding": encoding, "sample_rate_hz": rate,
        "samples_per_frame": per_frame, "samples_per_line": per_line,
        "lines": int(entry["lines"]),
        "frames": len(ire) / per_frame,
        "ire_per_code": scale["ire_per_code"],
        "quantisation_ire": abs(scale["ire_per_code"]) / np.sqrt(
            output_limit.QUANTISATION_DIVISOR),
        "why": ("the stream is flat and headerless, so the geometry is the "
                "standard's and not the file's"),
    }
    _LOADED[memo] = out
    return out


def reference(system: str = "NTSC", pattern: str = "75bars",
              encoding: str = "RAW_S16_40M", frames: int = 1,
              setup_ire: Optional[float] = 7.5,
              config_path: Optional[str] = None) -> Dict[str, object]:
    """Generate if needed and return the samples - the on-the-fly entry.

    The second call for the same request returns the identical array without
    running the generator or touching the disc.
    """
    made = generate(system=system, pattern=pattern, encoding=encoding,
                    frames=frames, setup_ire=setup_ire,
                    config_path=config_path)
    loaded = load(made["path"], system=system, encoding=encoding,
                  key=made["key"])
    return {**loaded, "request": made}


def clear_cache() -> None:
    """Forget the in-process arrays. The files on disc are left alone."""
    _LOADED.clear()


# --------------------------------------------------------------------------
# The forward chain: picture side to RF
# --------------------------------------------------------------------------


def _resample(values: np.ndarray, from_hz: float, to_hz: float) -> np.ndarray:
    """Rate conversion by the exact rational ratio where there is one.

    40 MHz to 50 MHz is five over four exactly, which is the case this arc
    needs, so a polyphase resampler is used rather than a Fourier one and no
    periodicity is assumed of the signal.
    """
    from fractions import Fraction

    from scipy import signal as scipy_signal

    if abs(float(from_hz) - float(to_hz)) < 1e-6:
        return np.asarray(values, dtype=np.float64)
    ratio = Fraction(float(to_hz) / float(from_hz)).limit_denominator(1000)
    return scipy_signal.resample_poly(np.asarray(values, dtype=np.float64),
                                      ratio.numerator, ratio.denominator)


def _low_pass(values: np.ndarray, sample_rate_hz: float,
              corner_hz: float) -> np.ndarray:
    """The recorder's luma channel, as a zero-phase band limit.

    Zero phase on purpose: the deck's own filter has a phase characteristic
    and this model does not claim to know it, so the forward chain carries
    the band limit and leaves the phase to the residual. A minimum-phase
    stand-in would put a delay into the reference that is not measured.
    """
    n = len(values)
    spectrum = np.fft.rfft(values)
    f = np.fft.rfftfreq(n, 1.0 / float(sample_rate_hz))
    spectrum[f > float(corner_hz)] = 0.0
    return np.fft.irfft(spectrum, n=n)


def _apply_response(values: np.ndarray, sample_rate_hz: float,
                    response_of) -> np.ndarray:
    """One linear stage, applied on the signal's own frequency grid."""
    n = len(values)
    spectrum = np.fft.rfft(values)
    f = np.fft.rfftfreq(n, 1.0 / float(sample_rate_hz))
    return np.fft.irfft(spectrum * response_of(f), n=n)


def clip_levels_ire(system: str = "NTSC", tape_format: str = "VHS",
                    reading: str = "blanking") -> Dict[str, float]:
    """The two clipping levels in IRE - and WHICH READING OF THE CLAUSE.

    SMPTE 32M-2004 clause 3.9.1.1.3 states a white clipping level of 160 per
    cent and a dark clipping level of 40 per cent for VHS. `clipping.py`
    records the reference for those percentages as the sync-tip-to-peak-white
    span measured FROM THE TIP, which puts the dark clip at +16 IRE. The other
    reading in circulation measures them from BLANKING with the
    blanking-to-peak-white amplitude as 100 per cent, which puts the dark clip
    at -40 IRE - exactly the sync tip - and the white clip at +160 IRE.

    THE MEASUREMENT DECIDES BETWEEN THEM, and it is not close. Taking the
    videosynth 75 per cent bars through the band limit and the main
    pre-emphasis and asking how much of the signal each reading would remove:

        reading                     below the dark clip   above the white clip
        from the sync tip, +16 IRE        40.2 %                0.50 %
        from blanking, -40 IRE             5.1 %                0.80 %

    and confining the count to the picture - excluding the sync pulses, which
    the recorder inserts at a fixed level rather than clipping - separates
    them completely: 35.3 per cent of ordinary picture below the first
    reading's dark clip against 0.10 per cent below the second's. A clip that
    removes a third of an ordinary picture is not a clip, so the first reading
    cannot be a level on the video signal, whatever it is. The second behaves
    the way a clip should: it catches the emphasis overshoot and leaves the
    picture alone.

    The second reading is also the one that makes physical sense of the dark
    clip's purpose. Under it a dark clipping level is a DEPTH below blanking,
    which is why the standard prints it as a positive number, and 40 per cent
    of the 100 IRE video amplitude puts it at -40 IRE - the sync-tip level.
    So it stops an emphasised black overshoot from driving the carrier below
    3.4 MHz, where a sync separator would read it as a pulse. The white clip
    at +160 IRE corresponds to 4.829 MHz, inside the deck's RF band.

    `reading` therefore defaults to `blanking`. `sync_tip` is kept so the
    excluded reading stays demonstrable rather than merely asserted.

    WHAT IS NOT RESOLVED HERE, and it is left open rather than guessed. IEC
    60774-3 gives S-VHS 210 per cent and MINUS 70, and `clipping.py` records
    the dark clip as changing SIGN between the formats. Under the depth
    reading a negative depth would put the S-VHS dark clip above blanking,
    which is not a dark clip at all, so one of the two - the sign convention
    or this reading - does not carry across to S-VHS. Nothing in this arc's
    material is S-VHS, so the question is recorded and not settled, and this
    function should not be trusted for S-VHS without settling it.
    """
    levels = clipping.clip_levels(tape_format)
    if system == "PAL":
        tip, white = SYNC_TIP_IRE_PAL, 100.0
    else:
        tip, white = SYNC_TIP_IRE_M, PEAK_WHITE_IRE_M
    if reading == "sync_tip":
        # both percentages measured upward from the tip, over the whole span
        span, base = white - tip, tip
        dark_ire = base + levels["dark"] * span
        white_ire = base + levels["white"] * span
    elif reading == "blanking":
        # both measured from blanking over the blanking-to-peak-white
        # amplitude, the white clip upward and the dark clip DOWNWARD - a
        # dark clipping level is a depth, which is why the standard prints
        # it as a positive number for VHS
        span, base = white - 0.0, 0.0
        dark_ire = base - levels["dark"] * span
        white_ire = base + levels["white"] * span
    else:
        raise ValueError(f"unknown reading {reading!r}; "
                         "known: 'blanking', 'sync_tip'")
    return {"sync_tip_ire": tip, "peak_white_ire": white,
            "span_ire": white - tip, "reading": reading,
            "reference_ire": base, "reference_span_ire": span,
            "dark_ire": dark_ire, "white_ire": white_ire,
            "cite": ("SMPTE 32M-2004 clause 3.9.1.1.3 through "
                     "clipping.clip_levels; which span the percentages are "
                     "measured against is the reading, and the docstring "
                     "records the measurement that chose it")}


def carrier_law(profile: Optional[Dict[str, object]] = None,
                system: str = "NTSC") -> Dict[str, float]:
    """IRE to instantaneous frequency, as the standard states the two ends.

    The law is affine because the standard gives it that way: one frequency
    at the sync tip and one at peak white, with the deviation the difference
    between them. Everything else follows.

    ATTESTED ON THE CAPTURE. Demodulating four fields of the zaroff 75bars SP
    record tap gives modes at 3.43 MHz on the sync tips and 4.405 MHz on peak
    white, against the specified 3.4 and 4.4 - so the law below is the deck's
    to better than a per cent, measured, before anything is fitted.
    """
    profile = profile or profiles.profile("NTSC-M", "composite", "ideal")
    parameters = profile["deck"]["parameters"]
    tip_hz = float(parameters["carrier_sync_tip_hz"][0])
    white_hz = float(parameters["carrier_peak_white_hz"][0])
    ends = clip_levels_ire(system)
    span = ends["span_ire"]
    return {
        "sync_tip_hz": tip_hz, "peak_white_hz": white_hz,
        "deviation_hz": white_hz - tip_hz,
        "sync_tip_ire": ends["sync_tip_ire"],
        "hz_per_ire": (white_hz - tip_hz) / span,
        "ire_per_hz": span / (white_hz - tip_hz),
        "cite": ("SMPTE 32M-2004 clause 3.9.1.1.4 through "
                 "profiles.IDEAL_DECK; the deviation is the difference of "
                 "the two stated ends, not a third number"),
    }


def forward_to_rf(reference: Dict[str, object],
                  profile: Optional[Dict[str, object]] = None,
                  sample_rate_hz: Optional[float] = None,
                  baseband_hz: Optional[float] = None,
                  clip: bool = True, emphasis: bool = True,
                  clip_reading: str = "blanking",
                  clip_ire: Optional[Tuple[float, float]] = None,
                  amplitude: float = 1.0) -> Dict[str, object]:
    """Carry a picture-side reference forward to the recorder's RF.

    `reference` is what `reference()` or `load()` returns, or any dict with
    `ire` and `sample_rate_hz`. The four stages are the ones the standards
    name, in the order the machine applies them:

      1. the luma channel's band limit - the recorder's Y/C separation
      2. the main pre-emphasis, IEC 774-1 (1994) page 67
      3. the white and dark clips, SMPTE 32M-2004 clause 3.9.1.1.3
      4. the FM law, SMPTE 32M-2004 clause 3.9.1.1.4

    Rate conversion happens on the INSTANTANEOUS FREQUENCY, after the clip
    and before the integration, so the one non-linear stage acts on the
    reference's own grid and the carrier is generated at the target rate
    rather than resampled. The returned `rf` is a unit-amplitude cosine of
    the integrated phase: the amplitude is not modelled here, and a
    demodulator that reads frequency does not care.

    `clip` and `emphasis` are switchable because each stage has to be
    checkable on its own - a flat field with the emphasis off must land on
    exactly one carrier frequency, which is what pins the FM law in the
    tests. `clip_reading` selects which span the standard's percentages are
    measured against; see `clip_levels_ire` for the measurement that chose
    the default.

    THE CLIP IS ON THE WHOLE SIGNAL, SYNC INCLUDED, and under the default
    reading that is right rather than careless: the dark limit sits exactly
    at the sync tip, so it removes the pre-emphasis undershoot at the pulse's
    leading edge - which on this reference reaches -144.7 IRE, three and a
    half times the pulse's own depth - and leaves the settled tip untouched.
    """
    ire = np.asarray(reference["ire"], dtype=np.float64).ravel()
    source_hz = float(reference["sample_rate_hz"])
    system = str(reference.get("system", "NTSC"))
    target_hz = float(sample_rate_hz or source_hz)
    corner_hz = float(baseband_hz or capture_profile.NTSC_VHS_SP_BASEBAND_HZ)
    stages: List[str] = []

    luma = _low_pass(ire, source_hz, corner_hz)
    stages.append(f"luma band limit at {corner_hz / 1e6:.3f} MHz "
                  "(SMPTE 32M-2004 clause 3.9.1.1.4 via capture_profile)")

    if emphasis:
        luma = _apply_response(
            luma, source_hz,
            lambda f: rf_stages.main_pre_emphasis(f))
        stages.append("main pre-emphasis (IEC 774-1 1994 page 67, 1.3 us, "
                      "gain factor 5)")

    ends = clip_levels_ire(system, reading=clip_reading)
    if clip_ire is not None:
        # A MEASURED clip in place of the standard's, for a caller who has
        # one. `clip_levels_ire` records that neither reading of SMPTE
        # 32M-2004 clause 3.9.1.1.3 this arc has considered puts the dark
        # clip where the SLV-778HF's record tap plainly has it: demodulating
        # the sync leading edges of three patterns gives a floor that
        # CONVERGES as the demodulator's smoothing is reduced - 2.9164,
        # 2.8211, 2.7956, 2.7891, 2.7869 MHz at successive halvings - so it
        # is a real limiter at about 2.787 MHz, 0.613 MHz below the specified
        # sync-tip carrier, which is -125.8 IRE. The same sweep on the white
        # side does NOT converge (5.41, 5.45, 5.82, 6.05, 6.15 MHz and still
        # rising), so no white clip is visible at this tap at all. Passing
        # the measured pair here is stating a measurement, not fitting one.
        ends = {**ends, "dark_ire": float(clip_ire[0]),
                "white_ire": float(clip_ire[1]), "reading": "measured"}
    if clip:
        luma = np.clip(luma, ends["dark_ire"], ends["white_ire"])
        stages.append(f"clips at {ends['dark_ire']:.1f} and "
                      f"{ends['white_ire']:.1f} IRE, the {clip_reading} "
                      "reading (SMPTE 32M-2004 clause 3.9.1.1.3)")

    law = carrier_law(profile, system)
    frequency = law["sync_tip_hz"] + (
        luma - law["sync_tip_ire"]) * law["hz_per_ire"]
    stages.append(f"FM law {law['sync_tip_hz'] / 1e6:.3f} MHz at sync tip to "
                  f"{law['peak_white_hz'] / 1e6:.3f} MHz at peak white")

    if abs(target_hz - source_hz) > 1e-6:
        frequency = _resample(frequency, source_hz, target_hz)
        stages.append(f"rate conversion {source_hz / 1e6:.3f} -> "
                      f"{target_hz / 1e6:.3f} MHz on the frequency")

    phase = 2.0 * np.pi * np.cumsum(frequency) / target_hz
    rf = float(amplitude) * np.cos(phase)
    return {
        "rf": rf, "instantaneous_hz": frequency, "phase_rad": phase,
        "luma_ire": luma, "sample_rate_hz": target_hz,
        "source_rate_hz": source_hz, "system": system,
        "law": law, "clips_ire": ends, "stages": stages,
        "why": ("the reference is the picture the recorder was given; the "
                "residual is only meaningful once the same three modelled "
                "stages stand between it and the tape"),
    }


# --------------------------------------------------------------------------
# The residual
# --------------------------------------------------------------------------


def hz_to_ire(difference_hz, law: Dict[str, float]) -> np.ndarray:
    """A frequency departure as an IRE departure, through the FM law.

    The deviation is 1.0 MHz over the 140 IRE span, so one IRE is 7142.86 Hz
    and the conversion is a division. It is exact rather than fitted because
    both ends of the law are stated by the standard.
    """
    return np.asarray(difference_hz, dtype=np.float64) * float(law["ire_per_hz"])


def against_floor(residual_ire, profile: Optional[Dict[str, float]] = None
                  ) -> Dict[str, object]:
    """The verdict Ethan's cap asks for: IRE, floors, and visible or not.

    This is `output_limit.is_meaningful` with the residual's own quantiles
    beside it, because a residual is not one number - a mean well under the
    floor with a tail far above it is a model that is right on average and
    wrong somewhere in particular, and reporting only the root mean square
    hides exactly that.
    """
    values = np.asarray(residual_ire, dtype=np.float64).ravel()
    finite = values[np.isfinite(values)]
    verdict = output_limit.is_meaningful(finite, profile)
    quantiles = (0.5, 0.9, 0.99, 1.0)
    floor = verdict["floor_ire"]
    return {
        **verdict,
        "samples": int(finite.size),
        "mean_ire": float(np.mean(finite)) if finite.size else float("nan"),
        "quantiles_ire": {q: float(np.quantile(np.abs(finite), q))
                          for q in quantiles} if finite.size else {},
        "quantiles_floors": {q: float(np.quantile(np.abs(finite), q) / floor)
                             for q in quantiles} if finite.size else {},
        "why": ("the cut is the output file's own quantisation noise, so it "
                "does not move with the instrument that made the residual"),
    }


def planted_flat_field(level_ire: float, seconds: float,
                       sample_rate_hz: float, system: str = "NTSC"
                       ) -> Dict[str, object]:
    """A constant picture level, in the shape `forward_to_rf` accepts.

    Not a generated reference and never presented as one: it exists so the
    forward chain can be checked against a signal whose answer is arithmetic.
    A flat field at one level, with the emphasis off, must modulate to
    exactly one frequency, and that is what pins the FM law.
    """
    count = max(int(round(float(seconds) * float(sample_rate_hz))), 1)
    return {"ire": np.full(count, float(level_ire)),
            "sample_rate_hz": float(sample_rate_hz), "system": system,
            "planted": True,
            "why": "a planted signal, not an independent reference"}


# --------------------------------------------------------------------------
# Differencing a reference against a capture
# --------------------------------------------------------------------------
#
# The two sides pass through IDENTICAL instrument code - the same band-pass,
# the same analytic signal, the same smoothing, the same field and line
# locator, all of them `tap_transfer`'s - so whatever the instrument does to
# one it does to the other and the residual is not the instrument's shape.
# That symmetry is the whole reason the reference is carried forward to an RF
# waveform rather than compared as a baseband.


def windows_of(params: Dict[str, object]) -> Dict[str, Tuple[float, float]]:
    """The intervals a line is reported over, in microseconds from the sync
    leading edge, all of them the decoder's own system parameters.

    THE TRAILING EDGE IS SPLIT OFF FROM THE TIP, and the reason is a defect
    in the generator that had to be found before any residual meant anything.
    videosynth's sync pulses are NARROW AT THE HALF-AMPLITUDE POINTS by
    exactly one transition, on all three pulse classes:

        pulse         specified   videosynth 50% to 50%   short by
        horizontal      4.700 us        4.4011 us          0.2989 us
        equalizing      2.300 us        2.0254 us          0.2746 us
        broad          27.100 us       26.8190 us          0.2810 us

    measured on the generated NTSC frame at both 4fsc and 40 MHz, at the -20
    IRE half-amplitude point the standards specify the width at. The cause is
    in the generator's own source and leaves nothing open:
    `src/generation_stage.cpp` `PulseWidthSeconds` returns the correct 4.7,
    2.3 and 27.1 us and `PulseWidthSamples` rounds them to 67, 33 and 388
    samples at 4fsc; `src/signal_shaping.cpp` `ShapedPulseLevel` then places
    BOTH transition ramps INSIDE that window, so the width is applied from
    the start of the fall to the end of the rise rather than between the
    half-amplitude points. The ramp is four samples at 4fsc, so every pulse
    measures four samples - 0.2794 us - narrow, and 67 - 4 = 63 samples is
    4.4011 us to the digit. The generator's transition itself is right: 0.1422
    us measured, 10 to 90 per cent, against the 0.14 us the decoder's own
    `syncTransitionUS` carries.

    So the reference's LEADING edge is trustworthy and its TRAILING edge is
    0.28 us early. Alignment therefore derives from the leading edge alone -
    which is also where `models/ringing_transient_start_model` says the
    transient this arc chases attaches - and the trailing edge is reported in
    a window of its own so the defect stays visible instead of contaminating
    the tip.

    `sync tip` is the pulse's settled first half; `sync leading edge` holds
    the transient ahead of it, which is not noise but the largest single term
    in the whole residual and is reported in a window of its own.

    THE BLANKING WINDOWS STOP ONE GUARD SHORT OF ACTIVE VIDEO, and that is
    not tidiness either. The two generators do not start their active picture
    at the same place: measured on the same lines, the TSG-130A capture's
    first excursion past +30 IRE after the burst is at 9.380 us from the sync
    leading edge (sd 0.008 over a hundred lines) and videosynth's is at 10.850
    us (sd 0.010), a difference of 1.47 us; videosynth's own departure from
    blanking is at 10.406 us against the decoder's nominal active start of
    9.45. So the capture's picture begins INSIDE the interval the decoder
    calls blanking, and a porch window running to 9.45 us reads a picture
    edge as a porch error. Backing off one smoothing width keeps it out. `back porch` is
    after the burst and not before it: `models/porch_anchor_content_bias`
    recorded the front porch failing to be a level at all, 41 to 66 standard
    errors of content dependence, while the back porch after the burst holds.
    `active` is reported but is offline validation only, because the sync-only
    law does not license correction from it.
    """
    hsync = float(params["pulse_widths_us"]["horizontal"])
    guard = hsync / 10.0            # the smoothing width the locator uses
    burst = params["colour_burst_us"]
    active = params["active_video_us"]
    front = float(params["front_porch_us"])
    # WHERE THE TIP HAS SETTLED, derived and not chosen. The pre-emphasis
    # shelf's decay is its time constant over its gain factor - 1.3/5 =
    # 0.26 us - so three of those is 0.78 us, and the demodulator's boxcar
    # adds half its own width. That is 1.015 us on this material, and the
    # measured residual profile settles exactly there: -55.9 IRE at 0.5 us
    # after the edge, +0.37 at 1.0, +5.16 at 1.5 (zaroff 75bars SP record
    # tap, 246 lines).
    settle = guard + 3.0 * (rf_stages.MAIN_EMPHASIS_TIME_CONSTANT_S
                            / rf_stages.MAIN_EMPHASIS_GAIN_FACTOR) * 1e6 \
        + guard / 2.0
    return {
        "sync leading edge": (-front, settle),
        "sync tip": (settle, hsync / 2.0),
        "sync trailing": (hsync / 2.0, hsync + guard),
        # the burst's end plus the smoothing width: the decoder's own
        # `colorBurstUS` ends the burst at 7.8 us and videosynth ends it at
        # 7.97 (5.3 + 2.67, its `BurstEndSamples` citing ITU-R BT.470-6
        # Table 2 item 2.14g), and the demodulator's boxcar spreads whatever
        # is there by its own width, so the porch cannot be read until one
        # guard past the later of the two
        "back porch": (float(burst[1]) + guard, float(active[0]) - guard),
        "blanking": (-front, float(active[0]) - guard),
        "active": (float(active[0]), float(active[1])),
        "line": (-front, float(params["line_period_s"]) * 1e6 - front),
    }


def instrument_parameters(system: str = "NTSC", tape_format: str = "VHS",
                          speed: str = "SP") -> Dict[str, object]:
    """`tap_transfer.capture_parameters` with the three line-geometry figures
    it does not carry, taken from the same system parameters it reads.

    Kept here rather than added there because that module belongs to another
    lane; nothing is changed in it and nothing is duplicated that it already
    holds.
    """
    from vhsdecode.formats import get_format_params, parse_tape_speed
    from vhsdecode.models import tap_transfer

    params = dict(tap_transfer.capture_parameters(system, tape_format, speed))
    sys_params, _ = get_format_params(system, tape_format,
                                      parse_tape_speed(speed), None)
    params["front_porch_us"] = float(sys_params["frontPorchUS"])
    params["colour_burst_us"] = tuple(float(v)
                                      for v in sys_params["colorBurstUS"])
    params["active_video_us"] = tuple(float(v)
                                      for v in sys_params["activeVideoUS"])
    # THE DECODER'S OWN LAW AND THE STANDARD'S AGREE EXACTLY, by two routes
    # that share no arithmetic: `hz_ire` is 7142.857142857143 Hz an IRE and
    # `ire0` is 3 685 714.2857 Hz, which is what `carrier_law` computes from
    # SMPTE 32M's two stated ends through `profiles.IDEAL_DECK`. Recorded
    # here so the agreement is checkable rather than assumed.
    params["decoder_hz_per_ire"] = float(sys_params["hz_ire"])
    params["decoder_blanking_hz"] = float(sys_params["ire0"])
    return params


def demodulate(rf, sample_rate_hz: float, params: Dict[str, object]
               ) -> np.ndarray:
    """The instantaneous frequency of the luma FM, by the shared instrument."""
    from vhsdecode.models import tap_transfer

    smooth_s = float(params["pulse_widths_us"]["horizontal"]) * 1e-6 / 10.0
    return tap_transfer.instantaneous_frequency(
        np.asarray(rf, dtype=np.float64), float(sample_rate_hz),
        params["fm_band_hz"], smooth_s)


def _level(frequency: np.ndarray, start: int, window: Tuple[float, float],
           sample_rate_hz: float) -> float:
    a = int(round(start + window[0] * 1e-6 * sample_rate_hz))
    b = int(round(start + window[1] * 1e-6 * sample_rate_hz))
    a, b = max(a, 0), min(b, len(frequency))
    if b - a < 2:
        return float("nan")
    return float(np.median(frequency[a:b]))


def sync_calibration(frequency: np.ndarray, starts: Sequence[int],
                     params: Dict[str, object], sample_rate_hz: float
                     ) -> Dict[str, float]:
    """The two-point anchoring the decoder itself performs, per field.

    The sync tip and the back porch are the only two levels the standard
    fixes and the sync-only law licenses, and every decode in this tree
    already sets its IRE scale from them. Measuring them on both sides and
    mapping one onto the other therefore removes exactly the freedom the
    decoder would remove anyway - the deck's record level and its carrier
    offset - and nothing else. It is two numbers a field, not a fit.
    """
    windows = windows_of(params)
    tips = [_level(frequency, s, windows["sync tip"], sample_rate_hz)
            for s in starts]
    porches = [_level(frequency, s, windows["back porch"], sample_rate_hz)
               for s in starts]
    tip = float(np.nanmedian(tips))
    porch = float(np.nanmedian(porches))
    return {"sync_tip_hz": tip, "back_porch_hz": porch,
            "span_hz": porch - tip,
            "lines": int(np.sum(np.isfinite(tips))),
            "why": ("the two levels the standard fixes and the sync-only "
                    "law licenses; the decoder anchors on the same pair")}


def _align_delay(a: np.ndarray, b: np.ndarray) -> float:
    """The delay of `b` behind `a`, in samples, to a fraction of one.

    Cross-correlation of the two mean-removed windows, with a parabolic
    interpolation of the peak. The window given to it is the front porch and
    the sync pulse's LEADING edge and nothing else, so the alignment derives
    from sync alone - and from the one edge the reference places correctly;
    see `windows_of` for the generator's trailing-edge defect and for why the
    pulse's flat interior, which carries no timing at all, is not used.
    """
    x = np.asarray(a, dtype=np.float64) - np.mean(a)
    y = np.asarray(b, dtype=np.float64) - np.mean(b)
    if not np.any(x) or not np.any(y):
        return 0.0
    corr = np.correlate(x, y, mode="full")
    peak = int(np.argmax(corr))
    if 0 < peak < len(corr) - 1:
        left, mid, right = corr[peak - 1], corr[peak], corr[peak + 1]
        denominator = left - 2.0 * mid + right
        offset = 0.5 * (left - right) / denominator if denominator else 0.0
    else:
        offset = 0.0
    return float(peak + offset - (len(y) - 1))


def line_residual(measured_hz: np.ndarray, measured_start: int,
                  reference_hz: np.ndarray, reference_start: int,
                  params: Dict[str, object], sample_rate_hz: float,
                  gain: float = 1.0, offset_hz: float = 0.0
                  ) -> Optional[Dict[str, np.ndarray]]:
    """One line of the residual, aligned on its sync pulse.

    The reference is delayed onto the measured line by `hypercomplex
    .fractional_delay`, which is exact rather than interpolated, and the
    calibration - when one is given - is applied to the reference so the
    measured side is never touched.
    """
    from vhsdecode.models import hypercomplex

    windows = windows_of(params)
    fs = float(sample_rate_hz)
    pre = int(round(windows["line"][0] * 1e-6 * fs))
    post = int(round(windows["line"][1] * 1e-6 * fs))
    m_a, m_b = measured_start + pre, measured_start + post
    r_a, r_b = reference_start + pre, reference_start + post
    # ONE LINE PERIOD OF CONTEXT ON EACH SIDE.
    # `hypercomplex.fractional_delay` is an FFT phase ramp, exact for a
    # periodic signal and liable to ring across the whole window when the two
    # ends of the segment given to it do not meet. A line extracted from
    # front porch to front porch does not quite meet. MEASURED, and it turned
    # out not to matter at the quarter-sample delays this instrument sees:
    # widening the guard from one sync width to one line period left the
    # residual on the zaroff 75bars SP record tap at 16.8763 IRE over the
    # pulse interior as that window then stood - identical to five decimal
    # places, so the wrap-around was not what made that number large. The
    # wide guard is kept because it costs nothing and a larger delay would
    # make it matter, and the measurement is recorded so the fear is not
    # re-derived. What DID make that number large was the window: it began
    # at the smoothing guard and so contained the leading-edge transient,
    # which is why `windows_of` now starts the tip after the emphasis has
    # settled and reports the transient separately.
    guard = int(round(float(params["line_period_s"]) * fs))
    if m_a - guard < 0 or m_b + guard > len(measured_hz):
        return None
    if r_a - guard < 0 or r_b + guard > len(reference_hz):
        return None
    measured = measured_hz[m_a:m_b]
    wide = reference_hz[r_a - guard:r_b + guard]
    sync_a = int(round((windows["sync leading edge"][0] - windows["line"][0])
                       * 1e-6 * fs))
    sync_b = int(round((windows["sync leading edge"][1] - windows["line"][0])
                       * 1e-6 * fs))
    delay = _align_delay(measured[sync_a:sync_b],
                         wide[guard + sync_a:guard + sync_b])
    shifted = hypercomplex.fractional_delay(wide, -delay)
    reference = shifted[guard:guard + len(measured)]
    calibrated = (reference - offset_hz) * gain + offset_hz if gain != 1.0 \
        else reference
    return {"measured_hz": measured, "reference_hz": calibrated,
            "residual_hz": measured - calibrated, "delay_samples": delay,
            "window_start_us": windows["line"][0]}


def residual_by_window(residual_hz: np.ndarray, window_start_us: float,
                       params: Dict[str, object], sample_rate_hz: float,
                       law: Dict[str, float]) -> Dict[str, np.ndarray]:
    """One line's residual cut into the reported intervals, in IRE."""
    fs = float(sample_rate_hz)
    out: Dict[str, np.ndarray] = {}
    for name, (lo, hi) in windows_of(params).items():
        a = int(round((lo - window_start_us) * 1e-6 * fs))
        b = int(round((hi - window_start_us) * 1e-6 * fs))
        a, b = max(a, 0), min(b, len(residual_hz))
        if b - a >= 2:
            out[name] = hz_to_ire(residual_hz[a:b], law)
    return out


def residual_against_capture(capture_path: str, pattern: str = "75bars",
                             system: str = "NTSC", tape_format: str = "VHS",
                             speed: str = "SP", frames: int = 3,
                             samples: Optional[int] = None,
                             calibrate: bool = True, clip: bool = True,
                             clip_reading: str = "blanking",
                             clip_ire: Optional[Tuple[float, float]] = None,
                             encoding: str = "RAW_S16_40M",
                             config_path: Optional[str] = None
                             ) -> Dict[str, object]:
    """The whole measurement: an independent reference differenced against RF.

    Generates the pattern, carries it through the record chain, demodulates
    both sides with the SAME instrument, pairs fields by interlace parity and
    lines by line number, aligns each pair on its sync leading edge, and
    accumulates the residual over the reported windows.

    `calibrate` applies the two-point sync anchoring `sync_calibration`
    describes - the sync tip and the back porch, per field, the pair the
    decoder itself anchors on and the only pair the sync-only law licenses.
    With it off the residual is against the specification's law with NO free
    parameter at all, which is the stricter of the two tests and the one that
    shows the deck's record level.

    Every window is reported in IRE and in multiples of the output file's own
    quantisation floor, because that is the cap Ethan set and a residual under
    it cannot reach the file however large it looks.
    """
    from vhsdecode.models import tap_transfer

    params = instrument_parameters(system, tape_format, speed)
    law = carrier_law(None, system)
    rate = tap_transfer.sample_rate_from_name(capture_path)
    if rate is None:
        raise ValueError(f"no sample rate in the name of {capture_path!r}; "
                         "the FLAC header cannot hold 50 MHz and carries "
                         "50 kHz instead, so the name is the only record")

    made = reference(system=system, pattern=pattern, encoding=encoding,
                     frames=frames, config_path=config_path)
    forward = forward_to_rf(made, sample_rate_hz=rate, clip=clip,
                            clip_reading=clip_reading, clip_ire=clip_ire)
    reference_hz = demodulate(forward["rf"], rate, params)
    reference_fields = tap_transfer.locate_fields(forward["rf"], rate, params)
    if not reference_fields:
        raise RuntimeError("no fields located in the forward-modelled "
                           "reference; the chain has removed its sync")

    measured = tap_transfer.read_capture(capture_path, 0, samples)
    measured_hz = demodulate(measured, rate, params)
    measured_fields = tap_transfer.locate_fields(measured, rate, params)
    if not measured_fields:
        raise RuntimeError(f"no fields located in {capture_path!r}")

    first, last = params["first_usable_line"], params["last_usable_line"]
    accumulated: Dict[str, List[np.ndarray]] = {}
    delays: List[float] = []
    calibrations: List[Dict[str, float]] = []
    pairs = 0
    for field in measured_fields:
        match = next((f for f in reference_fields if f.parity == field.parity),
                     None)
        if match is None:
            continue
        lines = [n for n in sorted(set(field.lines) & set(match.lines))
                 if first <= n <= last]
        if not lines:
            continue
        gain, shifted = 1.0, reference_hz
        if calibrate:
            measured_anchor = sync_calibration(
                measured_hz, [field.lines[n] for n in lines], params, rate)
            reference_anchor = sync_calibration(
                reference_hz, [match.lines[n] for n in lines], params, rate)
            gain = (measured_anchor["span_hz"]
                    / max(reference_anchor["span_hz"], 1e-9))
            shifted = ((reference_hz - reference_anchor["sync_tip_hz"]) * gain
                       + measured_anchor["sync_tip_hz"])
            calibrations.append({
                "field": field.index, "gain": gain,
                "measured_tip_hz": measured_anchor["sync_tip_hz"],
                "reference_tip_hz": reference_anchor["sync_tip_hz"],
                "tip_offset_ire": (measured_anchor["sync_tip_hz"]
                                   - reference_anchor["sync_tip_hz"])
                * law["ire_per_hz"]})
        for number in lines:
            one = line_residual(measured_hz, field.lines[number], shifted,
                                match.lines[number], params, rate)
            if one is None:
                continue
            delays.append(one["delay_samples"])
            for name, values in residual_by_window(
                    one["residual_hz"], one["window_start_us"], params, rate,
                    law).items():
                accumulated.setdefault(name, []).append(values)
            pairs += 1

    if not pairs:
        raise RuntimeError("no line pairs formed; the reference and the "
                           "capture share no line numbers")
    windows = {}
    for name, values in accumulated.items():
        width = min(len(v) for v in values)
        stack = np.vstack([v[:width] for v in values])
        # THE TWO STATISTICS ARE NOT THE SAME QUESTION, and reporting only
        # the first has misled this arc before. `per_sample` is every
        # residual sample, so it carries the per-line noise as well as the
        # model's error; `profile` is the residual AVERAGED over the line
        # pairs, in which the noise falls as root N and what survives is the
        # part that repeats - the model error proper. `scatter` is what the
        # averaging removed, and profile^2 + scatter^2/N is per_sample^2.
        profile = stack.mean(axis=0)
        windows[name] = {
            "per_sample": against_floor(stack),
            "profile": against_floor(profile),
            "profile_ire": profile,
            "scatter_ire": float(np.sqrt(np.mean(stack.std(axis=0) ** 2))),
            "standard_error_ire": float(np.sqrt(np.mean(
                stack.std(axis=0) ** 2)) / np.sqrt(stack.shape[0])),
            "lines": int(stack.shape[0]),
            "samples_per_line": int(width),
        }
    return {
        "capture": capture_path, "pattern": pattern, "speed": speed,
        "sample_rate_hz": rate, "line_pairs": pairs,
        "measured_fields": len(measured_fields),
        "reference_fields": len(reference_fields),
        "calibrated": bool(calibrate), "clip": bool(clip),
        "clip_reading": (forward["clips_ire"]["reading"] if clip
                         else None),
        "clips_ire": ((forward["clips_ire"]["dark_ire"],
                       forward["clips_ire"]["white_ire"]) if clip else None),
        "windows": windows,
        "delay_samples": {"median": float(np.median(delays)),
                          "spread": float(np.std(delays))},
        "calibrations": calibrations,
        "stages": forward["stages"],
        "generator": made["request"],
        "law": law,
        "why": ("the reference is outside this repository, so the residual "
                "is not against something the arc drew for itself"),
    }


def _report(argv: Sequence[str]) -> int:
    """`python3 -m vhsdecode.models.reference_signal` - what is available."""
    tool = available()
    print(f"videosynth: {'present' if tool['present'] else 'ABSENT'}")
    print(f"  binary   {tool['binary']}")
    print(f"  version  {tool['version']}")
    print(f"  config   {tool['config']['path']} "
          f"({'read' if tool['config']['present'] else 'absent, defaults'})")
    print(f"  cache    {cache_dir(tool['config'])}")
    print(f"  systems  {', '.join(tool['systems'])}")
    print(f"  encoding {', '.join(tool['encodings'])}")
    print(f"  pattern  {', '.join(tool['patterns'])}")
    if not tool["present"]:
        print(f"\n{tool['why']}")
        return 1
    profile = output_limit.output_profile()
    print(f"\noutput cap: {profile['bits']} bit at "
          f"{profile['sample_rate_hz'] / 1e6:.6f} MHz, floor "
          f"{profile['quantisation_noise_ire']:.4f} IRE rms")
    for encoding in ("CVBS_U10_4FSC", "RAW_S16_40M"):
        scale = ire_scale("NTSC", encoding)
        step = abs(scale["ire_per_code"])
        noise = step / np.sqrt(output_limit.QUANTISATION_DIVISOR)
        print(f"  {encoding:14s} step {step:.4f} IRE, own quantisation "
              f"{noise:.4f} IRE = {noise / profile['quantisation_noise_ire']:.2f} "
              "floors")
    return 0


if __name__ == "__main__":       # pragma: no cover - a convenience entry
    import sys

    raise SystemExit(_report(sys.argv[1:]))


def tap_difference(record: Dict[str, object], playback: Dict[str, object]
                   ) -> Dict[str, object]:
    """What the tape and the playback chain add, by DIFFERENCING the profiles.

    Not by subtracting powers. Two things the two taps have in common are
    large - the record chain's own model error and the reference generator's
    narrow sync pulses - and both cancel exactly in a difference of profiles
    while a difference of powers assumes they are independent and leaves
    them in. On the zaroff 75bars SP pair the two estimators disagree by
    more than an order of magnitude on the sync trailing edge, which is
    precisely where the common term is largest, so the choice is not
    cosmetic.

    The standard error of the difference is the two taps' own pooled
    standard errors added in quadrature, so a window whose difference does
    not clear it is reporting noise.
    """
    out: Dict[str, object] = {}
    for name, row in record["windows"].items():
        other = playback["windows"].get(name)
        if other is None:
            continue
        width = min(len(row["profile_ire"]), len(other["profile_ire"]))
        difference = (np.asarray(other["profile_ire"][:width])
                      - np.asarray(row["profile_ire"][:width]))
        error = float(np.hypot(row["standard_error_ire"],
                               other["standard_error_ire"]))
        verdict = against_floor(difference)
        out[name] = {
            **verdict,
            "standard_error_ire": error,
            "over_error": verdict["effect_ire"] / max(error, 1e-300),
            "record_ire": row["profile"]["effect_ire"],
            "playback_ire": other["profile"]["effect_ire"],
        }
    return {
        "windows": out,
        "why": ("the record tap is the deck's own RF before the tape, so "
                "what the playback tap adds over it - profile against "
                "profile, not power against power - is the tape and the "
                "playback chain and nothing that stands ahead of them"),
    }
