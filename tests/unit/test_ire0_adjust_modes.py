"""Every ire0_adjust mode the decoder ACTS on must be one the parser ACCEPTS.

THE BUG THIS EXISTS TO CATCH SHIPPED. `declip` was added to
`vhsdecode/field.py` and left out of `_SUPPORTED_IRE0_ADJUST_VALUES`, and the
failure was silent rather than loud: `_normalize_ire0_adjust_args` treats an
unparseable value as "not a value for this flag", drops it, and rewrites the
invocation - so `--ire0_adjust hsync,declip` became `--ire0_adjust backporch`
with a stray positional, the user lost the mode they asked for AND the one
they would have had, and nothing said so.

A whitelist that the code consuming it can drift away from is not a
whitelist. This pins the two together.
"""

import re
from pathlib import Path

import pytest

from vhsdecode import main as vhs_main

FIELD = Path(__file__).resolve().parents[2] / "vhsdecode" / "field.py"


def _modes_the_decoder_acts_on():
    """Every literal tested against `options.ire0_adjust` in field.py."""
    source = FIELD.read_text()
    return set(re.findall(r'["\']([a-z_]+)["\']\s+in\s+self\.rf\.options\.ire0_adjust',
                          source))


def test_every_mode_the_decoder_acts_on_is_accepted_by_the_parser():
    acting = _modes_the_decoder_acts_on()
    assert acting, "the scan found no modes at all - the pattern has drifted"
    unreachable = acting - vhs_main._SUPPORTED_IRE0_ADJUST_VALUES
    assert not unreachable, (
        f"these modes are checked in field.py but the parser rejects them, so "
        f"they are silently dropped and never run: {sorted(unreachable)}")


def test_declip_specifically_survives_argv_normalisation():
    """The mode is only reachable if it survives BOTH the type check and the
    argv rewrite, which is where it was lost."""
    assert vhs_main._parse_ire0_adjust("hsync,declip") == "hsync,declip"
    argv = vhs_main._normalize_ire0_adjust_args(
        ["--ire0_adjust", "backporch,hsync,declip", "in.flac", "out"])
    assert argv == ["--ire0_adjust", "backporch,hsync,declip", "in.flac",
                    "out"], "the value must not be dropped or rewritten"


def test_an_unknown_mode_is_still_refused_and_the_message_is_derived():
    """The message must come FROM the whitelist, or it drifts the moment a
    mode is added - which is exactly what had happened."""
    import argparse
    with pytest.raises(argparse.ArgumentTypeError) as raised:
        vhs_main._parse_ire0_adjust("nonsense")
    for mode in vhs_main._SUPPORTED_IRE0_ADJUST_VALUES:
        assert mode in str(raised.value), mode
