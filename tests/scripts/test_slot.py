from slot import status


def blocker(n, open=False, commit=None):
    return {"number": n, "open": open, "commit": commit}


def test_status():
    on_branch = {"abc"}.__contains__
    assert status([], on_branch) == "working"
    assert status([blocker(3, open=True), blocker(5, commit="def")], on_branch) == "blocked on #3"
    assert status([blocker(3, commit="def")], on_branch) == (
        "resume: rebase on main, then read the Resolution of #3"
    )
    assert status([blocker(3, commit="abc")], on_branch) == "working"  # already rebased
    assert status([blocker(3)], on_branch) == "working"  # closed by hand, no PR
