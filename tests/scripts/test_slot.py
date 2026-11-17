import subprocess
from pathlib import Path

import slot
from slot import in_merged_pr, status


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


def git(cwd: Path, *args: str) -> str:
    out = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return out.stdout.strip()


def commit(cwd: Path, message: str) -> str:
    who = ["-c", "user.name=t", "-c", "user.email=t@example.com"]
    git(cwd, *who, "commit", "-q", "--allow-empty", "-m", message)
    return git(cwd, "rev-parse", "HEAD")


def test_branch_behind_its_merged_pr_counts_as_merged(tmp_path, monkeypatch):
    """The PR got a commit on GitHub after this clone last fetched it (e.g. "Update branch"):
    the local branch is behind the PR head, which is only on refs/pull/<N>/head."""
    origin, home = tmp_path / "origin", tmp_path / "home"
    origin.mkdir()
    git(origin, "init", "-q", "-b", "main")
    commit(origin, "a")
    git(origin, "switch", "-q", "-c", "7-feature")
    pushed = commit(origin, "b")
    git(tmp_path, "clone", "-q", str(origin), str(home))
    head = commit(origin, "c: merge main, pushed from elsewhere")
    git(origin, "update-ref", "refs/pull/7/head", head)
    git(home, "switch", "-q", "-c", "local", pushed)
    local_only = commit(home, "d: never pushed")
    monkeypatch.setattr(slot, "home", lambda: home)
    pr = {"number": 7, "headRefOid": head}

    assert in_merged_pr(pushed, pr)  # fetches the PR head, which home lacked
    assert in_merged_pr(head, pr)
    assert not in_merged_pr(local_only, pr)
