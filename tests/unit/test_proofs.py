"""The proof suite, run as a test.

`tools/ringing_measure/proofs.py` regenerates every number that appears in
`docs/THE_ALGORITHM.md`, `docs/ELLIPTICAL_COLLAPSE.md`,
`docs/COMPONENT_MAPPINGS.md` and `docs/MATHEMATICS.md`. Running it here makes
the documents and the code fail together rather than drift apart: if a claim
stops reproducing, this test says so.

It includes the proofs that establish the FAILURES - the nested matrix's
statistics, the literal zero, the wrong scatter, the blindness to a shared
departure. Those are expected to demonstrate their defect, and they fail if
the defect stops reproducing, because the withdrawal recorded in
ELLIPTICAL_COLLAPSE.md section 6c would then no longer match the code.
"""

import pytest

from tools.ringing_measure import proofs


def test_every_proof_in_the_documents_still_reproduces(capsys):
    failures = proofs.run()
    captured = capsys.readouterr().out
    assert failures == 0, (
        "a claim in the documents no longer reproduces:\n" + captured[-4000:])


def test_the_suite_covers_every_document():
    """Each proof names the document that cites it, so a document with no
    proof behind it is visible."""
    cited = " ".join(cites for _, cites, _ in proofs.PROOFS)
    for document in ("THE_ALGORITHM.md", "ELLIPTICAL_COLLAPSE.md",
                     "COMPONENT_MAPPINGS.md"):
        assert document in cited, f"nothing proves anything in {document}"
    assert len(proofs.PROOFS) >= 20
