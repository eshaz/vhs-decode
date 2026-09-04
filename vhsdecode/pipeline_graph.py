"""The correction pipeline's declaration, and the graph rendered from it.

`vhsdecode/pipeline/stages.toml` declares every correction stage: what it
reads and writes, what it measures, what it feeds forward to the next
field, the flag that gates it, and which collapse group it belongs to.

TWO CONSUMERS, ONE SOURCE. This module renders that declaration as a graph;
the executor, when it lands, runs the same declaration. A stage absent from
the file can neither run nor be drawn. That is the whole point: a diagram
maintained beside the pipeline drifts from it and is then trusted anyway,
which is worse than having none. Three pieces of exactly that drift are
already in this tree - a docstring naming a store that does not exist, a
plot menu listing nine plots where the code requests twelve, and two
modules stating contradictory reproducibility guarantees.

WHAT THE GRAPH SHOWS beyond structure, which is what makes it a debug tool
rather than documentation:

  * which stages are ENABLED for this decode against merely available,
    with the resolved gate value printed - almost every correction is
    flag-gated and default-off, so on a typical decode most boxes are dark
    and a reader should be able to see that at a glance;
  * the AXIS each acts on - amplitude, frequency, time;
  * the three kinds of edge drawn distinctly, because conflating them is
    how a well-meaning reordering breaks a correction;
  * where the chain COLLAPSES, and where it stops collapsing.

THE COLLAPSE is the arc's central claim and the reason the graph is worth
drawing at all. Every filter before the demodulator is LTI on the RF, so
they compose into one H_RF(f) - which is why the correction belongs there,
where one response fixes every level at once. The demodulator is not
linear, so nothing either side of it composes across it. Measured: the
averaged sync pulse needs 34 of 120 DCT coefficients, and after alignment
no coefficient stands up consistently across cases - one template carries
the pulse. The collapse holds, and the graph draws where it ends.
"""

import os
import tomllib
from typing import Any, Dict, List, Optional

DECLARATION = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "pipeline", "stages.toml")

DATA_EDGE = "-->"
MEASUREMENT_EDGE = "-.->"
FEED_FORWARD_EDGE = "==>"

# How a gate's predicate is read against the resolved option value. Anything
# not here is a derived or compound condition the renderer reports as
# conditional rather than pretending to resolve.
PREDICATES = {
    "!= 0": lambda v: v not in (0, None, False),
    "> 0": lambda v: v is not None and v > 0,
    "> -1": lambda v: v is not None and v > -1,
    "is not None": lambda v: v is not None,
    "is True": lambda v: v is True,
    "is False": lambda v: v is False,
}


_CURRENT: Dict[str, Any] = {}
"""The most recently loaded declaration, so a gate can resolve a raw
selection string without the caller threading one in."""


def load(path: str = DECLARATION) -> Dict[str, Any]:
    """The declaration, validated. Raises rather than returning a graph
    that cannot be right - a malformed pipeline is cheap to catch here and
    expensive to discover in a decode."""
    with open(path, "rb") as handle:
        declared = validate(tomllib.load(handle), path)
    _CURRENT.clear()
    _CURRENT.update(declared)
    return declared


def validate(declared: Dict[str, Any], path: str = "<declaration>"
             ) -> Dict[str, Any]:
    """Check a declaration and index it. Separate from `load` so the rules
    can be tested against a hand-built declaration rather than only against
    the shipped file, which by construction passes them."""
    nodes = declared.get("node", [])
    if not nodes:
        raise ValueError("%s declares no nodes" % path)

    by_name = {}
    for node in nodes:
        name = node.get("name")
        if not name:
            raise ValueError("a node has no name")
        if name in by_name:
            raise ValueError("duplicate node %r" % name)
        by_name[name] = node
    declared["by_name"] = by_name

    groups = {c["name"]: c for c in declared.get("collapse", [])}
    declared["collapse_by_name"] = groups

    # The solving blocks. A block is the unit of SOLVING - which is not the
    # scope a stage executes in, nor the group it composes with - so a
    # member may live in either scope and still belong here.
    blocks = {b["name"]: b for b in declared.get("block", [])}
    declared["block_by_name"] = blocks
    for block in declared.get("block", []):
        for member in block.get("members", []):
            if member not in by_name:
                raise ValueError("block %s names unknown stage %r"
                                 % (block["name"], member))

    for node in nodes:
        where = node["name"]
        group = node.get("collapse")
        if group is not None and group not in groups:
            raise ValueError("%s: unknown collapse group %r" % (where, group))
        # A measurement edge whose target does not exist would let a
        # consumer silently bind to nothing.
        for measured in node.get("measures", []):
            target = measured.get("node")
            if target not in by_name:
                raise ValueError("%s measures unknown node %r" % (where, target))
            if by_name[target].get("kind") != "measurement":
                raise ValueError(
                    "%s measures %r, which is not declared as a measurement "
                    "- a data edge must not be recorded as one" % (where, target))
        for forward in node.get("feeds_forward", []):
            if forward.get("target") not in by_name:
                raise ValueError("%s feeds forward to unknown node %r"
                                 % (where, forward.get("target")))
        for other in node.get("must_precede", []) + node.get("must_follow", []):
            if other not in by_name:
                raise ValueError("%s orders against unknown node %r"
                                 % (where, other))
    _check_ordering(declared)
    return declared


def _check_ordering(declared: Dict[str, Any]) -> None:
    """`must_precede` / `must_follow` are asserted, not implied by position
    in the file, so they are verified against it."""
    order = {node["name"]: i for i, node in enumerate(declared["node"])}
    for node in declared["node"]:
        here = order[node["name"]]
        for later in node.get("must_precede", []):
            if order[later] < here:
                raise ValueError("%s must precede %s but is declared after it"
                                 % (node["name"], later))
        for earlier in node.get("must_follow", []):
            if order[earlier] > here:
                raise ValueError("%s must follow %s but is declared before it"
                                 % (node["name"], earlier))


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_site(node: Dict[str, Any], root: str = _REPO_ROOT) -> Optional[str]:
    """Where this stage runs, as `file:line` - FOUND, not stored.

    Line numbers were stored here and every one of them went stale the
    moment anyone edited the file: after one session's work they were off
    by a uniform +36 in the RF scope, about +55 in the field scope, and by
    between +315 and +550 in the chroma path. A citation that drifts
    silently is worse than none, because the declaration is trusted, and
    the test guarding them only checked that they LOOKED like sites.

    So each node names its file and an ANCHOR - a distinctive line of its
    own code - and the line is found at read time. If the code moves, the
    anchor still finds it. If the code is deleted or duplicated, the test
    below fails loudly, which is the behaviour a citation should have.
    """
    site = node.get("site", "")
    anchor_text = node.get("anchor")
    if not site or not anchor_text:
        return site or None
    path = os.path.join(root, site)
    try:
        with open(path, "r") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return None
    hits = [i + 1 for i, line in enumerate(lines) if anchor_text in line]
    if len(hits) != 1:
        return None
    return "%s:%d" % (site, hits[0])


def parse_selection(text: str, declared: Dict[str, Any]) -> Dict[str, Any]:
    """`--stages` : which stages and components this decode runs.

    Ethan: "Just do an on-off and remove the amount. The amount will be
    derived using the residual removal process." So this is a set of names,
    never a set of levels - a stage is on or it is not, and how much of its
    correction to apply is a question the residual answers, not the command
    line.

    The grammar is a comma-separated list of `+name` / `-name`, where a
    name is a stage or `stage.component`. Names are validated against the
    declaration, so a typo fails at startup with the valid list rather than
    silently disabling a correction and producing a decode nobody can
    explain.

    Returns {"stages": {name: bool}, "components": {(stage, part): bool}}.
    """
    by_name = declared["by_name"]
    stages: Dict[str, bool] = {}
    components: Dict[Any, bool] = {}
    for raw in str(text).split(","):
        token = raw.strip()
        if not token:
            continue
        enable = True
        if token[0] in "+-":
            enable = token[0] == "+"
            token = token[1:].strip()
        if not token:
            continue
        stage, _, part = token.partition(".")
        stage = stage.strip().lower()
        part = part.strip().lower()
        if stage not in by_name:
            raise ValueError(
                "unknown stage %r. Known stages: %s"
                % (stage, ", ".join(sorted(by_name))))
        if part:
            known = {c["name"] for c in by_name[stage].get("components", [])}
            if part not in known:
                raise ValueError(
                    "stage %r has no component %r. It has: %s"
                    % (stage, part, ", ".join(sorted(known)) or "none"))
            components[(stage, part)] = enable
        else:
            stages[stage] = enable
    return {"stages": stages, "components": components}


# What value turns a gate off, and what turns it on, per predicate. Turning
# a stage off is always possible; turning one ON can only be done where an
# enabling value can be synthesised - a stage gated on a measurement FILE
# cannot be switched on by naming it, and saying so is better than pretending.
GATE_OFF = {"!= 0": 0, "> 0": 0, "> -1": -1, "is not None": None,
            "is True": False, "is False": True}
GATE_ON = {"!= 0": 1.0, "> 0": 1.0, "> -1": 0, "is True": True,
           "is False": False}


def apply_selection(options: Dict[str, Any], selection: Dict[str, Any],
                    declared: Dict[str, Any]) -> List[str]:
    """Fold a `--stages` selection into the option values, in place.

    The selection is mapped ONTO THE EXISTING GATES rather than added as a
    second gate at every site: a stage turned off has its own option set to
    the value its own predicate reads as off, so every existing `if` in the
    decoder keeps working unchanged and there is still one gate per stage.
    A second, parallel gating mechanism would be a new way for the two to
    disagree.

    Returns the human-readable list of what it changed, for the log.
    """
    changed: List[str] = []
    if not selection:
        return changed
    by_name = declared["by_name"]
    for name, enable in selection.get("stages", {}).items():
        node = by_name.get(name)
        gate = node.get("gate") if node else None
        if not isinstance(gate, dict):
            changed.append("%s cannot be switched by name (it always runs)" % name)
            continue
        option = gate.get("option")
        predicate = gate.get("predicate", "!= 0")
        # A compound gate - "!= 0 and burst present" - cannot be turned ON
        # by naming it, because the extra condition is not ours to satisfy.
        # But it can always be turned OFF: the extra conditions only ADD
        # requirements, so making the flag itself false is sufficient
        # whatever else is true. Refusing to switch it off was needlessly
        # strict and left `--stages -luma_beat` reporting a failure for a
        # thing it could have done.
        base = predicate.split(" and ")[0].strip() if not enable else predicate
        table = GATE_ON if enable else GATE_OFF
        predicate = base
        if predicate not in table:
            changed.append(
                "%s cannot be switched by name (its gate is %r, which needs "
                "more than a value)" % (name, predicate))
            continue
        options[option] = table[predicate]
        changed.append("%s %s (--%s = %r)"
                       % (name, "on" if enable else "off", option,
                          table[predicate]))
    return changed


def component_enabled(selection, stage: str, part: str,
                      default: bool = True) -> bool:
    """Whether one component of one stage runs. Absent from the selection
    means the stage's own default, which is on."""
    if not selection:
        return default
    return selection.get("components", {}).get((stage, part), default)


def resolve_gate(node: Dict[str, Any], options: Any = None) -> Dict[str, Any]:
    """Whether this node runs, and the value that decided it.

    Returns `state` of "on", "off" or "conditional" - the last where the
    gate is derived or compound and cannot honestly be resolved from the
    options alone. Reporting that as "on" would be the more comfortable
    answer and the wrong one.
    """
    gate = node.get("gate", "always")
    # An explicit selection wins over whatever the stage's own flag says:
    # it is the one place a reader can turn a stage off by name.
    chosen = None
    if isinstance(options, dict):
        # `options` reaches here in three shapes: the decoder's options
        # object, the raw `rf_options` dict (whose "stages" is the flag's
        # TEXT), and an already-parsed selection. Only a parsed one can
        # answer this, and reading a string as one is how the render first
        # failed with "'str' object has no attribute 'get'".
        selection = options.get("stages")
        if isinstance(selection, str):
            try:
                selection = parse_selection(selection, {"by_name": {
                    n["name"]: n for n in _CURRENT.get("node", [])}})
            except Exception:                                # noqa: BLE001
                selection = None
        if isinstance(selection, dict) and "stages" in selection:
            chosen = selection["stages"].get(node["name"])
        elif isinstance(selection, dict):
            chosen = selection.get(node["name"])
    if chosen is not None:
        return {"state": "on" if chosen else "off",
                "label": "--stages %s%s" % ("+" if chosen else "-", node["name"]),
                "value": chosen}
    if gate == "always":
        return {"state": "on", "label": "always", "value": None}

    option = gate.get("option")
    predicate = gate.get("predicate", "!= 0")
    # TOML cannot express null, so an absent default is written as the
    # string "none". Left unnormalised it is a NON-empty string and every
    # `is not None` gate reads as enabled - which is how `--notch` and
    # `--residual_channels` first drew as on in a decode that set neither.
    default = gate.get("default")
    if isinstance(default, str) and default.strip().lower() == "none":
        default = None
    value = default
    if options is not None:
        # Options reach this from two places and arrive in two shapes: the
        # decoder's Options namedtuple, and the plain `rf_options` dict the
        # command line builds before that namedtuple exists.
        if hasattr(options, "get") and not hasattr(options, option):
            value = options.get(option, default)
        else:
            value = getattr(options, option, default)

    test = PREDICATES.get(predicate)
    if test is None:
        return {"state": "conditional", "label": "--%s %s" % (option, predicate),
                "value": value}
    try:
        state = "on" if test(value) else "off"
    except TypeError:
        state = "conditional"
    shown = "default" if options is None else "%s" % (value,)
    return {"state": state, "label": "--%s %s" % (option, shown), "value": value}


def _identifier(name: str) -> str:
    return "n_" + name.replace("-", "_")


def _escape(text: str) -> str:
    """Mermaid labels are delimited by quotes and brackets."""
    return (str(text).replace('"', "'").replace("[", "(").replace("]", ")")
            .replace("\n", " ").strip())


def _label(node: Dict[str, Any], gate: Dict[str, Any]) -> str:
    axis = node.get("axis") or []
    parts = [node["name"]]
    if axis:
        parts.append(" · ".join(axis))
    parts.append(gate["label"])
    return _escape("<br/>".join(parts))


def render_mermaid(declared: Dict[str, Any], options: Any = None,
                   title: Optional[str] = None) -> str:
    """The graph, as mermaid. Text, so it diffs in git and a reviewer can
    see a change of shape in a pull request."""
    nodes = declared["node"]
    groups = declared["collapse_by_name"]
    gates = {n["name"]: resolve_gate(n, options) for n in nodes}

    out: List[str] = []
    if title:
        out.append("---")
        out.append("title: %s" % _escape(title))
        out.append("---")
    out.append("flowchart TD")

    for scope, heading in (("rf_block", "RF block scope — N worker threads"),
                           ("field", "Field scope — one thread, serial")):
        in_scope = [n for n in nodes if n.get("scope") == scope]
        if not in_scope:
            continue
        out.append('  subgraph %s["%s"]' % (scope, _escape(heading)))
        out.append("    direction TB")
        placed = set()
        # collapse groups first, so the folding is visible as a box
        for group_name, group in groups.items():
            members = [n for n in in_scope if n.get("collapse") == group_name]
            if not members:
                continue
            out.append('    subgraph %s_%s["%s"]' % (
                scope, group_name, _escape(group["title"])))
            out.append("      direction TB")
            for node in members:
                out.append('      %s["%s"]' % (_identifier(node["name"]),
                                               _label(node, gates[node["name"]])))
                placed.add(node["name"])
            out.append("    end")
        for node in in_scope:
            if node["name"] not in placed:
                out.append('    %s["%s"]' % (_identifier(node["name"]),
                                             _label(node, gates[node["name"]])))
        out.append("  end")

    out.append("")
    for line in _edges(declared, gates):
        out.append("  " + line)

    out.append("")
    out.extend("  " + line for line in _styles(declared, gates))
    return "\n".join(out)


def _edges(declared: Dict[str, Any],
           gates: Optional[Dict[str, Any]] = None) -> List[str]:
    """Every edge, by kind.

    Data edges are inferred - the most recent writer of a channel, to its
    next reader - and threaded over the nodes that are actually ENABLED.
    A disabled stage does not pass the signal on, it is absent from the
    chain, so routing an edge through one would draw a pipeline this decode
    does not run. Disabled nodes are still drawn; they simply stand outside
    the chain, which is what "available but not enabled" looks like.

    Measurement and feed-forward edges are declared rather than inferred,
    because they cannot be inferred and must never be reordered across.
    Those are drawn whatever the gate says: a measurement edge records
    where a quantity is TAKEN, and that contract holds even when the
    consumer is off.
    """
    nodes = declared["node"]
    lines: List[str] = []
    seen = set()

    def live(name: str) -> bool:
        if gates is None:
            return True
        return gates[name]["state"] in ("on", "conditional")

    last_writer: Dict[str, str] = {}
    for node in nodes:
        if not live(node["name"]):
            continue
        for channel in node.get("reads", []):
            producer = last_writer.get(channel)
            if producer and producer != node["name"]:
                key = (producer, node["name"], "data")
                if key not in seen:
                    seen.add(key)
                    lines.append("%s %s|%s| %s" % (
                        _identifier(producer), DATA_EDGE, _escape(channel),
                        _identifier(node["name"])))
        for channel in node.get("writes", []):
            last_writer[channel] = node["name"]

    for node in nodes:
        for measured in node.get("measures", []):
            lines.append("%s %s|%s| %s" % (
                _identifier(measured["node"]), MEASUREMENT_EDGE,
                _escape("measured at " + measured["node"]),
                _identifier(node["name"])))
        for forward in node.get("feeds_forward", []):
            lines.append("%s %s|%s| %s" % (
                _identifier(node["name"]), FEED_FORWARD_EDGE,
                _escape(forward.get("delay", "next field")),
                _identifier(forward["target"])))
    return lines


def _styles(declared: Dict[str, Any], gates: Dict[str, Any]) -> List[str]:
    lines = [
        "classDef on fill:#1f6f43,stroke:#8fe3b0,stroke-width:2px,color:#f2fff8;",
        "classDef off fill:#20242b,stroke:#4a5262,stroke-width:1px,color:#8b94a5;",
        "classDef cond fill:#4a3a12,stroke:#d8b25a,stroke-width:2px,color:#fff6e0;",
        "classDef meas fill:#123a52,stroke:#6fc3e8,stroke-width:2px,color:#eaf8ff;",
    ]
    buckets: Dict[str, List[str]] = {"on": [], "off": [], "cond": [], "meas": []}
    for node in declared["node"]:
        name = node["name"]
        if node.get("kind") == "measurement":
            buckets["meas"].append(_identifier(name))
            continue
        state = gates[name]["state"]
        buckets[{"on": "on", "off": "off", "conditional": "cond"}[state]].append(
            _identifier(name))
    for style, members in buckets.items():
        if members:
            lines.append("class %s %s;" % (",".join(members), style))
    return lines


def summarise(declared: Dict[str, Any], options: Any = None) -> Dict[str, Any]:
    """What the graph says, as numbers - for a report or a log line."""
    nodes = declared["node"]
    gates = {n["name"]: resolve_gate(n, options) for n in nodes}
    enabled = [n["name"] for n in nodes
               if gates[n["name"]]["state"] == "on"
               and n.get("kind") != "measurement"]
    collapse: Dict[str, List[str]] = {}
    for node in nodes:
        group = node.get("collapse")
        if group:
            collapse.setdefault(group, []).append(node["name"])
    return {
        "nodes": len(nodes),
        "enabled": enabled,
        "measurements": [n["name"] for n in nodes
                         if n.get("kind") == "measurement"],
        "collapse": collapse,
        "measurement_edges": sum(len(n.get("measures", [])) for n in nodes),
        "feed_forward_edges": sum(len(n.get("feeds_forward", [])) for n in nodes),
        "races": [(n["name"], a["key"], a.get("race"))
                  for n in nodes for a in n.get("accumulates", [])],
    }


def write(path: str, options: Any = None, title: Optional[str] = None) -> str:
    """Render to a file and return what was written."""
    declared = load()
    text = render_mermaid(declared, options, title)
    with open(path, "w") as handle:
        handle.write(text + "\n")
    return text
