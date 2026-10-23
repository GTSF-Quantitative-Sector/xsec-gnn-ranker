"""Check a workstream's GitHub issues before its PR merges.

Branch <N>-<slug> works on issue #N. `check` lists what still blocks its PR: unticked steps,
open blockers, sub-issues (handoffs) without a complete Resolution comment or not closed by
the PR, dependencies against pipeline order, and cycles. Stdlib plus the `gh` CLI.

    python scripts/issue_check.py check        # print findings, exit 0
    python scripts/issue_check.py check --ci   # exit 1 on any finding
"""

import argparse
import io
import json
import os
import re
import subprocess
import sys
from collections import deque
from typing import Any

AREAS = ["infra", "data", "graphs", "train", "eval", "backtest"]  # pipeline order, upstream first
SECTIONS = ["What changed", "Where", "How to use it", "Done when", "Differs from the request"]
RESEARCH = "research/"  # branches you merge by hand, without the issue check

Issue = dict[str, Any]

QUERY = """
query($owner: String!, $name: String!, $branch: String!) {
  repository(owner: $owner, name: $name) {
    issues(states: OPEN, first: 100) {
      totalCount
      nodes {
        ...Ref
        body
        comments(last: 50) { nodes { body } }
        parent { number }
        blockedBy(first: 20) { nodes { ...Ref } }
        subIssues(first: 20) { nodes { ...Ref comments(last: 50) { nodes { body } } } }
      }
    }
    pullRequests(headRefName: $branch, states: OPEN, first: 1) {
      nodes { closingIssuesReferences(first: 50) { nodes { number } } }
    }
  }
}
fragment Ref on Issue { number state labels(first: 10) { nodes { name } } }
"""


def gh(*args: str) -> str:
    out = subprocess.run(["gh", *args], check=True, capture_output=True, encoding="utf-8")
    return out.stdout.strip()


def fetch(branch: str) -> tuple[dict[int, Issue], set[int] | None]:
    """Open issues plus the closed ones they reference, and the issues the branch's open PR
    closes (None if it has no open PR)."""
    repo = os.environ.get("GITHUB_REPOSITORY") or gh(
        "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"
    )
    owner, name = repo.split("/")
    args = ["-f", f"query={QUERY}", "-f", f"owner={owner}", "-f", f"name={name}"]
    data = json.loads(gh("api", "graphql", *args, "-f", f"branch={branch}"))["data"]["repository"]
    if data["issues"]["totalCount"] > 100:
        raise OSError("more than 100 open issues; issue_check.py needs pagination")
    issues: dict[int, Issue] = {}

    def add(node: dict[str, Any]) -> Issue:
        issue = issues.setdefault(
            node["number"],
            {
                "number": node["number"],
                "open": node["state"] == "OPEN",
                "labels": [lab["name"] for lab in node["labels"]["nodes"]],
                "body": "",
                "comments": [],
                "parent": None,
                "blocked_by": [],
                "sub_issues": [],
            },
        )
        if "body" in node:
            issue["body"] = node["body"]
        if "comments" in node:
            issue["comments"] = [c["body"] for c in node["comments"]["nodes"]]
        return issue

    for node in data["issues"]["nodes"]:
        add(node).update(
            parent=(node["parent"] or {}).get("number"),
            blocked_by=[add(b)["number"] for b in node["blockedBy"]["nodes"]],
            sub_issues=[add(s)["number"] for s in node["subIssues"]["nodes"]],
        )
    prs = data["pullRequests"]["nodes"]
    closes = {i["number"] for i in prs[0]["closingIssuesReferences"]["nodes"]} if prs else None
    return issues, closes


def area(issue: Issue) -> int | None:
    found = [AREAS.index(lab) for lab in issue["labels"] if lab in AREAS]
    return min(found) if found else None


def unticked(body: str) -> list[str]:
    return re.findall(r"^\s*[-*] \[ \] (.+)$", body, re.MULTILINE)


def resolution_problem(issue: Issue) -> str | None:
    """None if the issue's latest Resolution comment has every section filled in."""
    found = [c for c in issue["comments"] if c.strip().lower().startswith("## resolution")]
    if not found:
        return "no '## Resolution' comment"
    filled: dict[str, bool] = {}
    section = None
    for line in found[-1].strip().splitlines()[1:]:
        if line.strip().startswith("### "):
            section = line.strip()[4:].strip().lower()
            filled[section] = False
        elif section and line.strip():
            filled[section] = True
    missing = [s for s in SECTIONS if not filled.get(s.lower())]
    return f"Resolution missing or empty: {', '.join(missing)}" if missing else None


def order_problems(issues: dict[int, Issue], work: set[int]) -> list[str]:
    """Blocked-by edges touching the workstream must point upstream (or within an area)."""
    out = []
    for i in sorted(issues):
        for b in issues[i]["blocked_by"]:
            if i not in work and b not in work:
                continue
            ai, ab = area(issues[i]), area(issues[b])
            if ai is None or ab is None:
                out += [f"#{x} has no area label" for x, a in ((i, ai), (b, ab)) if a is None]
            elif ab > ai:
                out.append(
                    f"#{i} ({AREAS[ai]}) is blocked by #{b} ({AREAS[ab]}): against pipeline order"
                )
    return out


def cycle(issues: dict[int, Issue], n: int) -> str | None:
    """A cycle through workstream n among open issues, and the sub-issue to ship first:
    the most upstream area, ties lowest number."""

    def ws(i: int) -> int:
        return issues[i]["parent"] or i

    waits: dict[int, dict[int, list[int]]] = {}  # workstream -> blocking workstream -> issues
    for i, issue in issues.items():
        for b in issue["blocked_by"]:
            if issue["open"] and issues[b]["open"] and ws(i) != ws(b):
                waits.setdefault(ws(i), {}).setdefault(ws(b), []).append(b)
    prev: dict[int, int] = {}
    queue = deque([n])
    while queue:
        u = queue.popleft()
        for v in waits.get(u, {}):
            if v == n:
                path = [u]
                while path[-1] != n:
                    path.append(prev[path[-1]])
                path = [*reversed(path), n]
                blockers = [b for a, c in zip(path, path[1:], strict=False) for b in waits[a][c]]
                subs = [b for b in blockers if issues[b]["parent"]] or blockers
                first = min(subs, key=lambda b: (area(issues[b]) or len(AREAS), b))
                chain = " waits on ".join(f"#{w}" for w in path)
                out = f"cycle: {chain}. Ship #{first} first on its own branch from main"
                if "ship-first" not in issues[first]["labels"]:
                    parents = ", ".join(f"#{w}" for w in path[:-1])
                    out += f"; label it ship-first and comment the order on {parents}"
                return out
            if v not in prev:
                prev[v] = u
                queue.append(v)
    return None


def findings(issues: dict[int, Issue], n: int, closes: set[int] | None) -> list[str]:
    """What blocks the PR of workstream n. A sub-issue on its own branch (ship-first) is
    checked as its own handoff: its Resolution, its blockers, and that the PR closes it."""
    me = issues.get(n)
    if not me or not me["open"]:
        return [f"#{n} is not an open issue"]
    handoff = me["parent"] is not None
    subs = [n] if handoff else me["sub_issues"]
    work = {n, *subs}
    out = [f"#{n} has an unticked step: {s}" for s in unticked(me["body"])]
    for i in sorted(work):
        out += [
            f"#{i} is blocked by open #{b}" for b in issues[i]["blocked_by"] if issues[b]["open"]
        ]
    for s in subs:
        if problem := resolution_problem(issues[s]):
            out.append(f"#{s}: {problem}")
    if closes is not None:
        out += [
            f"the PR does not close #{i}"
            for i in sorted(work)
            if issues[i]["open"] and i not in closes
        ]
    out += order_problems(issues, work)
    if not handoff and (c := cycle(issues, n)):
        out.append(c)
    return list(dict.fromkeys(out))


def check(ci: bool) -> None:
    branch = (
        os.environ.get("GITHUB_HEAD_REF")
        or subprocess.run(
            ["git", "branch", "--show-current"], capture_output=True, encoding="utf-8"
        ).stdout.strip()
    )
    if branch.startswith(RESEARCH):
        print(f"{branch}: research branch, issue check skipped")
        return
    if not (m := re.match(r"(\d+)-", branch)):
        msg = f"branch {branch or '(detached)'} has no issue number"
        if ci:
            sys.exit(f"{msg}: PRs come from a <N>-<slug> branch for an open issue")
        print(f"{msg}; issue check skipped")
        return
    n = int(m[1])
    try:
        issues, closes = fetch(branch)
    except (OSError, subprocess.CalledProcessError) as e:
        msg = e.stderr.strip() if isinstance(e, subprocess.CalledProcessError) else str(e)
        if ci:
            sys.exit(f"issue check failed: {msg}")
        print(f"issue check skipped: {msg}")
        return
    out = findings(issues, n, closes)
    print(f"issue check for #{n}: " + (f"{len(out)} to resolve before the PR" if out else "ok"))
    for f in out:
        print(f"  - {f}")
    if ci and out:
        sys.exit(1)


def main() -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")  # issue text is UTF-8; Windows pipes are not
    parser = argparse.ArgumentParser(description="Check a workstream's GitHub issues.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="what still blocks this branch's PR")
    c.add_argument("--ci", action="store_true", help="exit 1 on any finding")
    args = parser.parse_args()
    check(args.ci)


if __name__ == "__main__":
    main()
