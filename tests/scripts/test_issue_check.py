import issue_check
import pytest
from issue_check import SECTIONS, check, context, current_branch, findings

RESOLUTION = "## Resolution\n" + "".join(f"### {s}\ntext\n" for s in SECTIONS)


def issue(
    n, labels=("data",), body="", parent=None, blocked_by=(), subs=(), comments=(), open=True
):
    return {
        "number": n,
        "title": f"title {n}",
        "open": open,
        "body": body,
        "labels": list(labels),
        "comments": list(comments),
        "parent": parent,
        "blocked_by": list(blocked_by),
        "sub_issues": list(subs),
    }


def repo(*items):
    return {i["number"]: i for i in items}


def test_finished_workstream_is_ok():
    issues = repo(
        issue(1, body="- [x] done", subs=[3]),
        issue(3, parent=1, comments=["thanks", RESOLUTION]),
        issue(2, labels=["backtest"], blocked_by=[3]),
    )
    assert findings(issues, 1, closes={1, 3}) == []


def test_unticked_steps_and_open_blockers():
    issues = repo(
        issue(
            2, labels=["backtest"], body="- [x] a\n- [ ] add BIL\n  * [ ] nested", blocked_by=[3]
        ),
        issue(3, parent=1),
        issue(1, subs=[3]),
    )
    assert findings(issues, 2, closes=None) == [
        "#2 has an unticked step: add BIL",
        "#2 has an unticked step: nested",
        "#2 is blocked by open #3",
    ]


def test_closed_blocker_and_closed_subissue_with_resolution_are_ok():
    issues = repo(
        issue(2, labels=["backtest"], blocked_by=[3], subs=[5]),
        issue(3, open=False),
        issue(5, labels=["backtest"], parent=2, open=False, comments=[RESOLUTION]),
    )
    assert findings(issues, 2, closes={2}) == []


@pytest.mark.parametrize(
    ("comments", "problem"),
    [
        ([], "no '## Resolution' comment"),
        (["see below\n" + RESOLUTION], "no '## Resolution' comment"),
        ([RESOLUTION.replace("### Where\ntext\n", "")], "Resolution missing or empty: Where"),
        (
            [RESOLUTION.replace("### Done when\ntext\n", "### Done when\n\n")],
            "missing or empty: Done when",
        ),
        (
            [RESOLUTION, "## Resolution\n### What changed\nx"],
            "missing or empty: Where, How to use it",
        ),
    ],
)
def test_resolution_comment(comments, problem):
    issues = repo(issue(1, subs=[3]), issue(3, parent=1, comments=comments))
    assert any(problem in f for f in findings(issues, 1, closes={1, 3}))


def test_pr_must_close_parent_and_open_subissues():
    issues = repo(issue(1, subs=[3]), issue(3, parent=1, comments=[RESOLUTION]))
    assert findings(issues, 1, closes=set()) == [
        "the PR does not close #1",
        "the PR does not close #3",
    ]
    assert findings(issues, 1, closes=None) == []  # no PR yet


def test_pipeline_order_and_missing_area():
    issues = repo(
        issue(1, subs=[3], blocked_by=[6]),
        issue(3, parent=1, comments=[RESOLUTION], blocked_by=[7]),
        issue(6, labels=["eval"], open=False),
        issue(7, labels=[], open=False),
    )
    assert findings(issues, 1, closes={1, 3}) == [
        "#1 (data) is blocked by #6 (eval): against pipeline order",
        "#7 has no area label",
    ]


def test_cycle_names_the_subissue_to_ship_first():
    issues = repo(
        issue(1, labels=["data"], subs=[3]),
        issue(3, labels=["data", "backtest"], parent=1, blocked_by=[5]),
        issue(2, labels=["backtest"], subs=[5], blocked_by=[3]),
        issue(5, labels=["train"], parent=2),
    )
    assert findings(issues, 2, closes=None)[-1] == (
        "cycle: #2 waits on #1 waits on #2. Ship #3 first on its own branch from main; "
        "label it ship-first and comment the order on #2, #1"
    )


def test_ship_first_branch_is_checked_as_its_own_handoff():
    issues = repo(
        issue(1, subs=[3], body="- [ ] later"),
        issue(3, labels=["data", "ship-first"], parent=1),
    )
    assert findings(issues, 3, closes={3}) == ["#3: no '## Resolution' comment"]


def test_not_an_open_issue():
    assert findings(repo(issue(4, open=False)), 4, None) == ["#4 is not an open issue"]
    assert findings({}, 9, None) == ["#9 is not an open issue"]


def test_branches_without_an_issue_number(monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_HEAD_REF", "research/macro-sensitivity-baseline")
    check(ci=True)
    assert "issue check skipped" in capsys.readouterr().out
    monkeypatch.setenv("GITHUB_HEAD_REF", "backtester")
    check(ci=False)
    assert "has no issue number; issue check skipped" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="<N>-<slug>"):
        check(ci=True)


def test_merge_queue_checks_the_queued_prs_branch(monkeypatch):
    monkeypatch.delenv("GITHUB_HEAD_REF", raising=False)
    monkeypatch.setenv("GITHUB_REF_NAME", "gh-readonly-queue/main/pr-12-0a1b2c")
    prs = {"12": "2-backtester"}
    monkeypatch.setattr(issue_check, "gh", lambda *args: prs[args[2]])
    assert current_branch() == "2-backtester"


def test_session_context_shows_handoffs_in_full_and_blocker_resolutions():
    issues = repo(
        issue(1, body="- [ ] build it", subs=[3, 8], blocked_by=[6]),
        issue(3, parent=1, body="## Need\ndelisting returns"),
        issue(8, parent=1, open=False),
        issue(6, labels=["infra"], open=False, comments=["## Resolution\n### Where\nx.py"]),
        issue(2, labels=["backtest"], blocked_by=[3]),
    )
    assert context(issues, 1).splitlines() == [
        "Workstream #1: title 1",
        "",
        "- [ ] build it",
        "",
        "Open sub-issues (handoffs you own):",
        "--- #3: title 3",
        "## Need",
        "delisting returns",
        "",
        "Blockers:",
        "  #6 (closed) title 6",
        "## Resolution",
        "### Where",
        "x.py",
    ]
    assert context(issues, 2).splitlines()[-4:] == [
        "Open sub-issues (handoffs you own): none",
        "Blockers:",
        "  #3 (open) title 3",
        '     Finish independent work, then stop and report "blocked on #3".',
    ]
