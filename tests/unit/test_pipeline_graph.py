"""The pipeline declaration, and the graph rendered from it.

The declaration is the single source of truth: the graph renders it and the
executor will run it. These tests exist mostly to keep that property true -
a declaration that no longer matches the tree it describes is worse than
none, because a diagram gets trusted.
"""

import re
import types

import pytest

from vhsdecode import pipeline_graph as pg
from vhsdecode.debug_plot import STRUCTURAL_PLOTS, DebugPlot


@pytest.fixture(scope="module")
def declared():
    return pg.load()


def _options(**overrides):
    """A stand-in for the decoder's options, everything off by default."""
    base = dict(notch=None, high_boost=None, channel_eq=0, luma_eq=0,
                head_switch=0, baseband_eq=0, luma_transient=0,
                disable_diff_demod=False, sharpness=0, chroma_trap=False,
                nldeemp=False, subdeemp=False, color_under=True,
                chroma_env_gain=1, carrier_tbc=0, y_comb=0, inverse_eq=-1,
                cti_mix=1, luma_beat=0, write_chroma=True,
                residual_channels=None)
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _declaration(*nodes, collapse=()):
    return {"meta": {"version": 1},
            "collapse": [{"name": c, "title": c, "note": ""} for c in collapse],
            "node": list(nodes)}


def _node(name, **fields):
    node = {"name": name, "site": "vhsdecode/x.py:1", "why": "because"}
    node.update(fields)
    return node


class TestTheShippedDeclaration:
    def test_it_loads(self, declared):
        assert len(declared["node"]) > 20
        assert declared["collapse"]

    def test_every_node_cites_a_site_that_actually_resolves(self, declared):
        """The site is FOUND, not stored.

        This test used to check only that a site LOOKED like one, and every
        stored line number went stale the moment the files were edited -
        after one session they were off by a uniform +36 in the RF scope,
        about +55 in the field scope, and by 315 to 550 in the chroma path.
        A citation that drifts silently is worse than none, because the
        declaration is trusted. Now each node names its file and an anchor
        in its own code, and this asserts the anchor is still there and
        still unique."""
        from vhsdecode import pipeline_graph

        unresolved = [n["name"] for n in declared["node"]
                      if pipeline_graph.resolve_site(n) is None]
        assert unresolved == [], unresolved

    def test_no_node_stores_a_line_number(self, declared):
        """A stored line is a line that will be wrong."""
        for node in declared["node"]:
            assert ":" not in node.get("site", ""), node["name"]
            assert node.get("anchor"), node["name"]

    def test_every_node_says_why_it_exists(self, declared):
        for node in declared["node"]:
            assert node.get("why", "").strip(), node["name"]

    def test_the_collapse_boundary_is_the_demodulator(self, declared):
        """The arc's central claim, asserted. Everything before the
        demodulator is LTI on the RF and composes into one response; the
        demodulator is not linear, so nothing composes across it."""
        boundary = [n["name"] for n in declared["node"]
                    if n.get("collapse") == "boundary"]
        assert boundary == ["demod"]

    def test_the_envelope_is_a_measurement_frozen_before_the_equalizer(
            self, declared):
        """It is taken before the equalizer deliberately - folding the
        equalizer in ahead of it cost 17% of the chroma correction to buy
        1.8% on the luma - so every amplitude consumer binds to the point,
        not to wherever the signal has got to."""
        envelope = declared["by_name"]["envelope"]
        assert envelope["kind"] == "measurement"
        assert envelope["frozen_before"] == "rf_eq"

    def test_no_dangling_read(self, declared):
        """Every channel read is written by some node.

        This is the property whose absence hid the whole luma amplitude
        measurement: `luma_eq_update` read `per_head_response` and
        `luma_transient_observe` read `amplitude_deviation`, nothing wrote
        either, and the stage the entire arc refines was simply not in the
        graph. `residual_limit.order` then resolved no parent for anything
        and returned a set of roots where a traversal was meant to be.

        A channel may only be exempt if it genuinely enters the pipeline
        from outside the declaration, and each exemption is named below
        with the reason. The comparison is an equality rather than a subset
        so that closing one of these also fails, loudly, with the line to
        delete - an exemption list nobody has to revisit is how the hole
        reopens.
        """
        # Genuinely outside: the capture itself, and the filter tables the
        # decoder builds at construction time. No stage produces these.
        # Genuinely outside the signal graph: the capture itself. That is
        # now the whole list.
        #
        # The decoder's filter tables stood here too - `Filters.RFVideo`,
        # `Filters.ChannelEQ` - because two namespaces were mixed in the
        # channel space and a reads/writes check had to step over the
        # second one. They are declared under `state_reads` / `state_writes`
        # now, so the exemption is gone rather than carried: an exemption
        # list nobody revisits is how the hole reopens.
        originates_outside = {
            "raw_data",           # the RF capture, as read from the file
        }
        # CLOSED. `demod_raw` stood here: it is `demod` written into the
        # field's record array under a second name, and this test named it
        # rather than blessing it. The `demod` node now declares it among
        # its `writes`, so the exemption is gone - which is what an
        # exemption naming a real hole is for.
        undeclared_writer = set()

        written = set()
        for node in declared["node"]:
            written.update(node.get("writes", []))
        dangling = {channel for node in declared["node"]
                    for channel in node.get("reads", [])
                    if channel not in written}
        assert dangling == originates_outside | undeclared_writer

    def test_the_luma_amplitude_measurement_is_in_the_graph(self, declared):
        """The stage this arc refines, declared: it applies nothing, and
        every amplitude-axis consumer binds to it."""
        node = declared["by_name"]["luma_amplitude"]
        assert node["kind"] == "measurement"
        assert set(node["writes"]) == {"per_head_response", "amplitude_deviation"}
        assert [m["node"] for m in node["measures"]] == ["envelope"]
        # the chroma path both triggers the measurement and consumes it
        assert "amplitude_deviation" in declared["by_name"]["chroma_env"]["reads"]

    def test_cti_runs_after_everything_that_reads_the_chroma(self, declared):
        cti = declared["by_name"]["cti"]
        assert "luma_beat" in cti["must_follow"]

    def test_luma_beat_is_a_join_back_into_the_luma(self, declared):
        """Not a tail stage: the chroma branch re-enters the luma, which is
        what makes the graph look cyclic if the two are collapsed."""
        beat = declared["by_name"]["luma_beat"]
        assert "uphet" in beat["reads"] and "dspicture" in beat["reads"]
        assert beat["writes"] == ["dspicture"]


class TestValidation:
    def test_a_measurement_edge_must_name_a_measurement(self):
        """Stops a data edge being recorded as a measurement edge, which
        would license the executor to reorder across it."""
        broken = _declaration(
            _node("a", writes=["s"]),
            _node("b", measures=[{"node": "a", "as": "s"}]))
        with pytest.raises(ValueError, match="not declared as a measurement"):
            pg.validate(broken)

    def test_a_measurement_edge_must_name_an_existing_node(self):
        broken = _declaration(_node("b", measures=[{"node": "ghost"}]))
        with pytest.raises(ValueError, match="unknown node"):
            pg.validate(broken)

    def test_an_unknown_collapse_group_is_refused(self):
        broken = _declaration(_node("a", collapse="nowhere"))
        with pytest.raises(ValueError, match="unknown collapse group"):
            pg.validate(broken)

    def test_ordering_is_verified_not_assumed(self):
        """`must_precede` is asserted rather than implied by file position,
        so a declaration whose order contradicts its own assertion is a
        bug the loader catches."""
        # "second" is declared AFTER "first" and yet claims to precede it
        broken = _declaration(_node("first"),
                              _node("second", must_precede=["first"]))
        with pytest.raises(ValueError, match="must precede"):
            pg.validate(broken)

        # and the same assertion the right way round is accepted
        fine = _declaration(_node("first", must_precede=["second"]),
                            _node("second"))
        assert pg.validate(fine)["by_name"].keys() == {"first", "second"}

    def test_duplicate_names_are_refused(self):
        with pytest.raises(ValueError, match="duplicate"):
            pg.validate(_declaration(_node("a"), _node("a")))


class TestGateResolution:
    def test_always_is_on(self, declared):
        assert pg.resolve_gate(declared["by_name"]["demod"])["state"] == "on"

    def test_a_flag_off_reads_off(self, declared):
        gate = pg.resolve_gate(declared["by_name"]["head_switch"], _options())
        assert gate["state"] == "off"

    def test_a_flag_on_reads_on_and_shows_its_value(self, declared):
        gate = pg.resolve_gate(declared["by_name"]["head_switch"],
                               _options(head_switch=1.0))
        assert gate["state"] == "on"
        assert "1.0" in gate["label"]

    def test_a_compound_gate_is_reported_as_conditional(self, declared):
        """`luma_beat` needs its flag AND a burst present. Reporting that
        as "on" would be the comfortable answer and the wrong one."""
        gate = pg.resolve_gate(declared["by_name"]["luma_beat"],
                               _options(luma_beat=1.0))
        assert gate["state"] == "conditional"

    def test_an_absent_default_is_off_not_on(self, declared):
        """TOML has no null, so an absent default is the string "none".
        Left unnormalised it is a non-empty string and every `is not None`
        gate reads as enabled."""
        for name in ("notch", "residual_channels"):
            gate = pg.resolve_gate(declared["by_name"][name])
            assert gate["state"] == "off", name

    def test_options_may_be_a_plain_mapping(self, declared):
        """The command line builds a dict before the options namedtuple
        exists, and the graph is rendered from it."""
        gate = pg.resolve_gate(declared["by_name"]["head_switch"],
                               {"head_switch": 1.0})
        assert gate["state"] == "on"


class TestRendering:
    def test_it_renders_every_declared_node(self, declared):
        text = pg.render_mermaid(declared, _options())
        for node in declared["node"]:
            assert pg._identifier(node["name"]) in text, node["name"]

    def test_subgraphs_are_balanced(self, declared):
        lines = [line.strip() for line in
                 pg.render_mermaid(declared, _options()).splitlines()]
        assert (sum(1 for l in lines if l.startswith("subgraph"))
                == sum(1 for l in lines if l == "end"))

    def test_no_edge_points_at_an_undeclared_node(self, declared):
        text = pg.render_mermaid(declared, _options())
        drawn = set(re.findall(r"\b(n_[A-Za-z0-9_]+)\[", text))
        referenced = set(re.findall(r"\b(n_[A-Za-z0-9_]+)\b", text))
        assert referenced - drawn == set()

    def test_the_three_edge_kinds_are_drawn_distinctly(self, declared):
        text = pg.render_mermaid(declared, _options())
        assert pg.MEASUREMENT_EDGE in text
        assert pg.FEED_FORWARD_EDGE in text
        assert pg.DATA_EDGE in text

    def test_enabled_and_disabled_are_styled_apart(self, declared):
        """The most useful thing the plot can say: almost every correction
        is default-off, so a reader must see at a glance which boxes are
        live on THIS decode."""
        text = pg.render_mermaid(declared, _options(head_switch=1.0))
        assert "classDef on" in text and "classDef off" in text
        on_line = [l for l in text.splitlines() if l.strip().startswith("class ")
                   and l.strip().endswith(" on;")][0]
        assert pg._identifier("head_switch") in on_line

    def test_a_label_cannot_break_the_syntax(self, declared):
        """Mermaid labels are delimited by quotes and brackets."""
        assert pg._escape('a "quoted" [bracketed]\nlabel') == \
            "a 'quoted' (bracketed) label"


class TestStructuralPlot:
    def test_the_graph_does_not_serialise_the_decode(self):
        """Every debug plot used to force one worker thread, structural
        ones included. This one reads no field."""
        assert "pipeline_graph" in STRUCTURAL_PLOTS
        assert not DebugPlot("pipeline_graph").wants_serialised_decode()

    def test_a_data_plot_still_does(self):
        assert DebugPlot("luma_noise").wants_serialised_decode()
        assert DebugPlot("pipeline_graph luma_noise").wants_serialised_decode()


class TestSummary:
    def test_the_default_decode_enables_few_stages(self, declared):
        """Most corrections are default-off; that is what makes the
        enabled-versus-available distinction worth drawing."""
        summary = pg.summarise(declared, _options())
        assert len(summary["enabled"]) < len(declared["node"]) / 2

    def test_declared_races_are_reported(self, declared):
        """Every shared accumulator is declared, so the reproducibility
        audit can find them without grepping."""
        races = dict((key, race) for _, key, race
                     in pg.summarise(declared)["races"])
        assert races["_head_switch_cal"] == "accepted"


class TestStageSelection:
    """`--stages` : on and off by name, never an amount.

    Ethan's ruling: "Just do an on-off and remove the amount. The amount
    will be derived using the residual removal process."
    """

    def test_it_parses_stages_and_components(self, declared):
        chosen = pg.parse_selection("+head_switch,-cti,-ringing.ghost", declared)
        assert chosen["stages"] == {"head_switch": True, "cti": False}
        assert chosen["components"] == {("ringing", "ghost"): False}

    def test_a_bare_name_means_on(self, declared):
        assert pg.parse_selection("carrier_tbc", declared)["stages"] == {
            "carrier_tbc": True}

    def test_an_unknown_stage_is_refused_with_the_valid_list(self, declared):
        with pytest.raises(ValueError, match="Known stages"):
            pg.parse_selection("+nosuchstage", declared)

    def test_an_unknown_component_is_refused(self, declared):
        with pytest.raises(ValueError, match="no component"):
            pg.parse_selection("ringing.nosuchpart", declared)

    def test_turning_a_stage_off_sets_its_own_gate(self, declared):
        """The selection is mapped ONTO the existing gates rather than
        added as a second gate at every site - a parallel mechanism would
        be a new way for the two to disagree."""
        options = {"cti_mix": 1, "head_switch": 0}
        changed = pg.apply_selection(
            options, pg.parse_selection("-cti,+head_switch", declared), declared)
        assert options["cti_mix"] == 0
        assert options["head_switch"] == 1.0
        assert any("cti off" in line for line in changed)

    def test_a_stage_that_always_runs_says_so_rather_than_failing(self, declared):
        options = {}
        changed = pg.apply_selection(
            options, pg.parse_selection("-demod", declared), declared)
        assert any("cannot be switched" in line for line in changed)

    def test_a_file_gated_stage_cannot_be_switched_on_by_name(self, declared):
        """Naming a stage cannot conjure the measurement file it needs, and
        saying so is better than pretending it was enabled."""
        options = {}
        changed = pg.apply_selection(
            options, pg.parse_selection("+baseband_eq", declared), declared)
        assert any("cannot be switched" in line for line in changed)

    def test_the_graph_draws_the_selection(self, declared):
        """The same declaration decides what runs and what is drawn, so the
        picture cannot disagree with the decode."""
        text = pg.render_mermaid(declared, {"stages": "-cti,+head_switch"})
        off = [l for l in text.splitlines()
               if l.strip().startswith("class ") and l.strip().endswith(" off;")]
        on = [l for l in text.splitlines()
              if l.strip().startswith("class ") and l.strip().endswith(" on;")]
        assert any(pg._identifier("cti") in l for l in off)
        assert any(pg._identifier("head_switch") in l for l in on)

    def test_a_raw_selection_string_resolves(self, declared):
        """`rf_options` carries the flag's TEXT, and the renderer is handed
        it directly - reading that as a parsed selection is how the render
        first failed."""
        gate = pg.resolve_gate(declared["by_name"]["cti"], {"stages": "-cti"})
        assert gate["state"] == "off"

    def test_components_default_to_on(self, declared):
        chosen = pg.parse_selection("-ringing.ghost", declared)
        assert pg.component_enabled(chosen, "ringing", "ghost") is False
        assert pg.component_enabled(chosen, "ringing", "smear") is True
        assert pg.component_enabled(None, "ringing", "ghost") is True

    def test_every_declared_component_is_named_and_explained(self, declared):
        for node in declared["node"]:
            for part in node.get("components", []):
                assert part.get("name"), node["name"]
                assert part.get("note", "").strip(), (node["name"], part)
