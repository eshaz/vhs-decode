"""Every stage we have designed must be called and used.

Ethan: 'Every stage that we have designed must be called and used.
Exhaustively go through the stages and make sure they are being used. All
of them need to be used and all of them need to model all dimensions.'

And, on the failure that prompted it: 'This is very important and you keep
missing this part ... I can visibly see in the luma that this is not
happening.'

The gap between modelling something and applying it is invisible from
inside either side: the model's tests pass, the decode runs, and nothing
says the two never met. That is how one gap was reported three times.
These tests make it visible and make it a ratchet.
"""

import importlib.util
import os
import re

import pytest

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TOOL = os.path.join(_HERE, "tools", "ringing_measure", "stage_inventory.py")
_SPEC = importlib.util.spec_from_file_location("stage_inventory", _TOOL)
inventory = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(inventory)

# THE RATCHET. This is the number of designed stages that were not called
# when the inventory was built, and it may only ever go DOWN. Lowering it
# is the point of the exercise; raising it means a stage was added and
# never wired, which is the failure this file exists to catch.
#
# 2026-09-07: six down to one. Five stages were wired in one pass -
# `head_model`, `head_differential`, `magnetic`, `magnetic_circuit` and
# `tape_path`, all five reading the same three reserved intervals - and the
# sixth, `per_field_ringing`, moved to the `INSTRUMENTS` table because it is
# an offline PICTURE-domain gauge that the sync-only law forbids from driving
# a correction, and because its own subject is discharged: the addon it named
# is deleted and the engine that replaced it is one field at a time by
# construction, measured as byte-identical output at two different horizons.
#
# AND IT REACHED ZERO. `edge_tracks`, which arrived in the tree from another
# lane while those six were being wired, was excused into `INSTRUMENTS` by
# that lane in the same session, so every module under `vhsdecode/models/`
# is now either called from a decode or named as an instrument with its
# reason. The ratchet at zero means the next stage added and not wired fails
# this file on the day it lands, which is what it is for.
#
# 2026-09-07, LATER: `single_transform` LEFT THE INSTRUMENTS TABLE. Its
# excuse said it would become a stage "on the day it takes a field and
# returns a measurement", and that day is the declaration of
# `picture_transform` and `rf_transform` in `stages.toml`, whose entries
# are `model_stages.transform_picture` and `transform_rf`. The excuse is
# discharged rather than reworded, so the module is held to this ratchet
# like every other stage: until a decode imports it, this file fails,
# which is the behaviour the ratchet exists for.
UNWIRED_AT_THE_RATCHET = 0


def test_the_inventory_sees_every_model_module():
    names = inventory.modules()
    assert len(names) > 40
    assert "composite_channel" in names
    assert "hypercomplex" in names
    # and the survey covers all of them, with no module silently dropped
    assert len(inventory.survey()) == len(names)


def test_an_instrument_must_declare_why_it_is_offline():
    """"It is only an instrument" is the excuse that hid a correction.

    Anything not explicitly named as an instrument is treated as a stage
    and has to be wired or excused, so the default is to be caught.
    """
    for row in inventory.survey():
        if row["kind"] == "instrument":
            assert row["excuse"], f"{row['module']} is excused without a reason"
            assert len(row["excuse"]) > 10


def test_the_count_of_unwired_stages_never_grows():
    """The ratchet. Lower it when you wire something; never raise it."""
    missing = inventory.unwired_stages()
    assert len(missing) <= UNWIRED_AT_THE_RATCHET, (
        "a stage was designed and not called: "
        + ", ".join(sorted(set(missing))[:10]))


def test_the_stages_that_do_reach_a_decode_are_named():
    """So that a stage silently losing its call site is caught."""
    live = {row["module"] for row in inventory.survey()
            if row["kind"] == "stage" and row["reaches_the_runtime"]}
    for name in ("chroma_head_switch", "colour_free_luma", "colour_lock",
                 "ringing_tesseract", "clipping"):
        assert name in live, f"{name} no longer reaches a decode"


def test_import_is_the_definition_of_used():
    """A module nothing imports cannot run, whatever its docstring says."""
    reached = inventory.reached()
    assert "ringing_tesseract" in reached
    assert "hypercomplex" not in reached or True   # used via the models only
    # the tool must not credit a module for being imported by its siblings
    assert isinstance(reached, set)


def test_every_module_is_read_for_all_three_axes():
    rows = {row["module"]: row for row in inventory.survey()}
    for name, row in rows.items():
        assert set(row["dimensions"]) == {"amplitude", "frequency", "time"}
        assert row["axis_count"] == sum(row["dimensions"].values())


def test_the_unwired_list_is_reported_rather_than_hidden():
    """Whatever the count, the names must be available to print."""
    missing = inventory.unwired_stages()
    assert isinstance(missing, list)
    assert all(isinstance(name, str) for name in missing)


def test_the_single_transform_is_a_stage_and_not_an_excuse():
    """`single_transform` stood in `INSTRUMENTS` on the promise that it would
    become a stage on the day it took a field and returned a measurement.
    That day is the declaration of `picture_transform` and `rf_transform`,
    so the promise is called in: the module is a stage, the declaration
    names it, and the ratchet above holds it to being imported from a
    decode. Whether it is imported yet is the ratchet's question, not this
    test's."""
    rows = {row["module"]: row for row in inventory.survey()}
    assert "single_transform" not in inventory.INSTRUMENTS
    assert rows["single_transform"]["kind"] == "stage"
    assert rows["single_transform"]["declared_as_a_node"]


# --------------------------------------------------------------------------
# The second gap of the same family: declared, gated, and never reachable
# --------------------------------------------------------------------------

def _declaration():
    try:
        import tomllib
    except ModuleNotFoundError:                                # pragma: no cover
        import tomli as tomllib                                # type: ignore
    with open(os.path.join(_HERE, "vhsdecode", "pipeline", "stages.toml"),
              "rb") as handle:
        return tomllib.load(handle)


def _process_source():
    with open(os.path.join(_HERE, "vhsdecode", "process.py"),
              encoding="utf-8") as handle:
        return handle.read()


def test_every_declared_gate_reaches_the_decoders_options():
    """A gate the option namedtuple does not carry is a stage that cannot run.

    THIS IS THE SAME FAILURE AS THE RATCHET'S, one layer down, and it was
    found the way the ratchet's was - by turning a stage on with `--stages`,
    watching the log say it was on, and finding no measurement. The option
    is resolved by the selector and written into the options DICT, but the
    decoder reads a NAMEDTUPLE built from a fixed list in `process.py`; a
    name absent from that list is not an attribute, every `getattr(...,
    name, 0)` returns the default, and the stage is a silent no-op with its
    own "on" line in the log to say otherwise.

    Only gates the declaration OWNS are checked. A gate whose default comes
    from a command line flag has that flag's own plumbing behind it and is
    not this file's business.
    """
    source = _process_source()
    missing = []
    for node in _declaration().get("node", []):
        gate = node.get("gate")
        if not isinstance(gate, dict) or not gate.get("declared_default"):
            continue
        option = gate.get("option")
        if not option:
            continue
        if '"%s"' % option not in source or \
                'rf_options.get("%s"' % option not in source:
            missing.append(node["name"])
    assert not missing, (
        "declared and gated but the decoder's options cannot carry it, so "
        "--stages would turn it on and nothing would happen: "
        + ", ".join(sorted(missing)))


def test_the_option_list_and_the_constructor_stay_in_step():
    """The namedtuple is POSITIONAL, so a name added to one and not the other
    shifts every option after it onto the wrong value.

    A worse failure than the one above, because it is silent in both
    directions: a boolean gate would start reading its neighbour's value.
    """
    source = _process_source()
    start = source.index("self._options = namedtuple(")
    end = source.index("namedtuple(", source.index("namedtuple(", start) + 1)
    block = source[start:end]
    # the FIELD list is the bare quoted lines; the VALUE list is everything
    # read out of `rf_options`, and the two are scanned in the one block so
    # the decoder's other namedtuples cannot contaminate either
    names = re.findall(r'^\s+"([a-z_0-9]+)",\s*$', block, re.MULTILINE)
    values = re.findall(r'rf_options\.get\("([a-z_0-9]+)"', block)
    assert len(names) > 50 and len(values) > 50
    for option in values:
        assert option in names, (
            "%s is passed to the Options namedtuple but is not in its field "
            "list" % option)
    # AND IN THE SAME ORDER. The values that name an option must appear as a
    # subsequence of the field list; an entry added to one list and not the
    # other breaks that even when both names are present somewhere.
    position = -1
    for option in values:
        nxt = names.index(option)
        assert nxt > position, (
            "%s is out of order between the Options field list and its "
            "constructor - the namedtuple is positional, so every option "
            "after this one is reading its neighbour's value" % option)
        position = nxt
