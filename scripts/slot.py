"""Reusable git worktrees ("slots") for working on several branches at once.

Each slot is a sibling folder of the home repo (<repo>-slot<N>) with its own conda env
(gnns-slot<N>, cloned from gnns once) and the package installed from that folder, so the
slot always runs its own code. Datasets are shared with the home repo via machine.yaml.
Stdlib only; run it from any env that has `conda` available (`sync` also needs `gh`).

    python scripts/slot.py setup 1 2 3         # create slots (slow only the first time)
    python scripts/slot.py start 1 backtester  # new branch from main in slot 1 (seconds)
    python scripts/slot.py list
    python scripts/slot.py sync                # after a merge: update main, free slots
    python scripts/slot.py remove 1
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from issue_check import gh

BASE_ENV = "gnns"
MARKER = "slot-env-sha"  # environment.yml hash the slot env was last synced to


def run(*cmd: str | Path, cwd: Path | None = None, capture: bool = False) -> str:
    out = subprocess.run(
        [str(c) for c in cmd], cwd=cwd, check=True, text=True, capture_output=capture
    )
    return out.stdout.strip() if capture else ""


def home() -> Path:
    """The main worktree, even when this runs from inside a slot."""
    common = run(
        "git",
        "rev-parse",
        "--path-format=absolute",
        "--git-common-dir",
        cwd=Path(__file__).parent,
        capture=True,
    )
    return Path(common).parent


def slot_path(n: int) -> Path:
    h = home()
    return h.parent / f"{h.name}-slot{n}"


def slot_paths() -> list[Path]:
    return sorted(home().parent.glob(f"{home().name}-slot*"))


def env_name(n: int) -> str:
    return f"{BASE_ENV}-slot{n}"


def conda(*args: str) -> str:
    return run(os.environ.get("CONDA_EXE", "conda"), *args, capture=True)


def env_exists(name: str) -> bool:
    envs = json.loads(conda("env", "list", "--json"))["envs"]
    return any(Path(e).name == name for e in envs)


def env_sha(path: Path) -> str:
    return hashlib.sha256((path / "environment.yml").read_bytes()).hexdigest()


def marker(path: Path) -> Path:
    """Inside the worktree's own git dir, so it never shows up as a working-tree change."""
    git_dir = run("git", "rev-parse", "--path-format=absolute", "--git-dir", cwd=path, capture=True)
    return Path(git_dir) / MARKER


def dirty(path: Path) -> bool:
    return bool(run("git", "status", "--porcelain", cwd=path, capture=True))


def setup(n: int) -> None:
    """Create slot n if it does not exist: worktree, .env, machine.yaml, env, install."""
    h, p, env = home(), slot_path(n), env_name(n)
    if p.exists():
        print(f"slot{n}: exists at {p}")
        return
    print(f"slot{n}: creating worktree {p}")
    run("git", "worktree", "add", "--detach", p, "main", cwd=h)
    if (h / ".env").exists():
        shutil.copy2(h / ".env", p / ".env")
    lines = []
    if (h / "config" / "machine.yaml").exists():
        lines = [
            line
            for line in (h / "config" / "machine.yaml").read_text().splitlines()
            if not line.startswith("data_root:")
        ]
    lines.insert(0, f"data_root: {(h / 'data').as_posix()}  # shared with the home repo")
    (p / "config" / "machine.yaml").write_text("\n".join(lines) + "\n")
    if not env_exists(env):
        print(f"slot{n}: cloning env {BASE_ENV} -> {env} (a few minutes, once)")
        conda("create", "-y", "-q", "-n", env, "--clone", BASE_ENV)
    print(f"slot{n}: installing the package from {p} into {env}")
    conda(
        "run",
        "-n",
        env,
        "python",
        "-m",
        "pip",
        "install",
        "-q",
        "--no-deps",
        "--no-build-isolation",
        "-e",
        str(p),
    )
    marker(p).write_text(env_sha(p))


def start(n: int, task: str) -> None:
    """Put slot n on branch `task` (new from main, or the existing branch), sync its env."""
    p, env = slot_path(n), env_name(n)
    setup(n)
    if dirty(p):
        sys.exit(f"slot{n} has uncommitted changes; commit or stash them first")
    branches = run("git", "branch", "--list", task, cwd=p, capture=True)
    run("git", "switch", task, cwd=p) if branches else run(
        "git", "switch", "-c", task, "main", cwd=p
    )
    m = marker(p)
    if not m.exists() or m.read_text() != env_sha(p):
        print(f"slot{n}: environment.yml changed; updating {env}")
        conda("env", "update", "-q", "-n", env, "-f", str(p / "environment.yml"))
        m.write_text(env_sha(p))
    print(
        f"\nslot{n} is on branch {task!r}. Start a session there with:\n"
        f"  conda activate {env}\n  cd {p}\n  claude      (or: code .)"
    )


def show() -> None:
    for p in slot_paths():
        n = p.name.rsplit("slot", 1)[1]
        branch = run("git", "branch", "--show-current", cwd=p, capture=True) or "(detached)"
        print(f"slot{n}: {branch:24s} {'uncommitted changes' if dirty(p) else 'clean'}  {p}")


BLOCKERS = """blockedBy(first: 20) { nodes { number state timelineItems(itemTypes: [CLOSED_EVENT],
last: 1) { nodes { ... on ClosedEvent { closer { ... on PullRequest { mergeCommit { oid } } } } } }
} }"""


def blockers(numbers: list[int]) -> dict[int, list[dict[str, Any]]]:
    """Each issue's blockers: number, open, and the merge commit of the PR that closed it."""
    issues = " ".join(f"i{n}: issue(number: {n}) {{ {BLOCKERS} }}" for n in numbers)
    query = f"query($o: String!, $r: String!) {{ repository(owner: $o, name: $r) {{ {issues} }} }}"
    args = ["-F", "o={owner}", "-F", "r={repo}", "-f", f"query={query}"]
    repo = json.loads(gh("api", "graphql", *args))["data"]["repository"]
    out = {}
    for n in numbers:
        out[n] = []
        for b in repo[f"i{n}"]["blockedBy"]["nodes"]:
            closer = (b["timelineItems"]["nodes"] or [{}])[0].get("closer") or {}
            commit = (closer.get("mergeCommit") or {}).get("oid")
            out[n].append({"number": b["number"], "open": b["state"] == "OPEN", "commit": commit})
    return out


def has_commit(branch: str, commit: str) -> bool:
    git = ["git", "merge-base", "--is-ancestor", commit, branch]
    return subprocess.run(git, cwd=home(), capture_output=True).returncode == 0


def status(blocked_by: list[dict[str, Any]], contains: Callable[[str], bool]) -> str:
    """Blocked while a blocker is open; resume once a blocker's closing PR is on main but not
    yet in the branch; otherwise working."""
    if found := [b["number"] for b in blocked_by if b["open"]]:
        return "blocked on " + ", ".join(f"#{b}" for b in found)
    if found := [b["number"] for b in blocked_by if b["commit"] and not contains(b["commit"])]:
        reads = ", ".join(f"#{b}" for b in found)
        return f"resume: rebase on main, then read the Resolution of {reads}"
    return "working"


def sync() -> None:
    """After a merge: update main, free the slots of merged branches, show each thread."""
    h = home()
    os.chdir(h)  # gh finds the GitHub repo from the current folder
    if run("git", "branch", "--show-current", cwd=h, capture=True) != "main" or dirty(h):
        sys.exit(f"home ({h}) must be on main with no uncommitted changes")
    before = run("git", "rev-parse", "--short", "main", cwd=h, capture=True)
    run("git", "pull", "-q", "--ff-only", "--prune", cwd=h)
    print(f"home: main {before}..{run('git', 'rev-parse', '--short', 'main', cwd=h, capture=True)}")
    slots = {p: run("git", "branch", "--show-current", cwd=p, capture=True) for p in slot_paths()}
    try:
        refs = run(
            "git", "for-each-ref", "--format=%(refname:short)", "refs/heads", cwd=h, capture=True
        )
        for b in refs.split():
            if b == "main":
                continue
            merged = gh("pr", "list", "--state", "merged", "--head", b, "--json", "headRefOid",
                        "-q", ".[].headRefOid").split()  # fmt: skip
            if not merged:
                continue
            if run("git", "rev-parse", b, cwd=h, capture=True) not in merged:
                print(f"{b}: PR merged, but the branch has newer commits; kept")
                continue
            held = [p for p, sb in slots.items() if sb == b]
            if busy := [p.name for p in held if dirty(p)]:
                print(f"{b}: merged, but {', '.join(busy)} has uncommitted changes; kept")
                continue
            for p in held:
                run("git", "switch", "-q", "--detach", "main", cwd=p)
                slots[p] = ""
            run("git", "branch", "-q", "-D", b, cwd=h)
            print(f"{b}: merged; branch deleted" + "".join(f", {p.name} at main" for p in held))
        work = {p: int(m[1]) for p, b in slots.items() if (m := re.match(r"(\d+)-", b))}
        found = blockers(sorted(set(work.values()))) if work else {}
    except (OSError, subprocess.CalledProcessError) as e:
        msg = e.stderr.strip() if isinstance(e, subprocess.CalledProcessError) else str(e)
        sys.exit(f"stopped after updating main (merged branches and threads need gh): {msg}")
    for p, b in slots.items():
        if p in work:
            state = status(found[work[p]], lambda c, b=b: has_commit(b, c))
        else:
            state = "" if b else "idle"
        print(f"{p.name}: {b or '(detached)'}" + (f": {state}" if state else ""))


def remove(n: int, force: bool) -> None:
    p, env = slot_path(n), env_name(n)
    if p.exists():
        if dirty(p) and not force:
            sys.exit(f"slot{n} has uncommitted changes; commit them or pass --force")
        run("git", "worktree", "remove", *(["--force"] if force else []), p, cwd=home())
    if env_exists(env):
        conda("env", "remove", "-y", "-q", "-n", env)
    print(f"slot{n}: removed (branches are kept)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reusable git worktrees with their own envs.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup", help="create slots").add_argument("slots", type=int, nargs="+")
    s = sub.add_parser("start", help="put a slot on a task branch")
    s.add_argument("slot", type=int)
    s.add_argument("task", help="branch name, e.g. backtester")
    sub.add_parser("list", help="show slots, branches, uncommitted work")
    sub.add_parser("sync", help="after a merge: update main, free merged slots, show threads")
    r = sub.add_parser("remove", help="delete a slot's worktree and env")
    r.add_argument("slot", type=int)
    r.add_argument("--force", action="store_true", help="discard uncommitted changes")
    args = parser.parse_args()
    if args.cmd == "setup":
        for n in args.slots:
            setup(n)
    elif args.cmd == "start":
        start(args.slot, args.task)
    elif args.cmd == "list":
        show()
    elif args.cmd == "sync":
        sync()
    else:
        remove(args.slot, args.force)


if __name__ == "__main__":
    main()
