"""Reject obvious credentials, local paths, and runtime data in publishable files.

This heuristic complements human review; it is not a complete secret detector.
Only file names and rule names are printed, never matching secret values.
"""

import argparse
import ipaddress
import json
import re
import subprocess
import sys
from pathlib import Path

PATTERNS = {
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github-token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b"),
    "api-key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b"),
    "aws-access-key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\b"),
    "local-user-path": re.compile(r"/(?:home|Users)/[^\s/]+/"),
    "url-credentials": re.compile(r"https?://[^\s/]+:[^\s/]+@"),
}
IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
PRIVATE_NETWORKS = [
    ipaddress.ip_network(value)
    for value in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10")
]


def scan_text(name, text, deny_markers=()):
    findings = [rule for rule, pattern in PATTERNS.items() if pattern.search(text)]
    # Network constants in this scanner are policy definitions, not deployment data.
    if name != "scripts/check_public_tree.py":
        for candidate in IPV4.findall(text):
            try:
                address = ipaddress.ip_address(candidate)
            except ValueError:
                continue
            if any(address in network for network in PRIVATE_NETWORKS):
                findings.append("private-network-address")
                break
    if any(marker.lower() in text.lower() for marker in deny_markers):
        findings.append("private-marker")
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deny-file", type=Path, help="Optional private JSON marker list; keep outside Git"
    )
    args = parser.parse_args()
    markers = json.loads(args.deny_file.read_text()) if args.deny_file else []
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    names = sorted(set(result.stdout.decode().split("\0")) - {""})
    if not names:
        raise SystemExit("No publishable files found; refusing an empty scan")
    failures = []
    for name in names:
        path = root / name
        if path.is_symlink():
            failures.append((name, "symlink"))
            continue
        if not path.is_file():
            continue
        if (
            (path.name.startswith(".env") and path.name != ".env.example")
            or path.suffix in {".db", ".key", ".pem", ".log", ".pyc"}
            or ".sqlite" in path.name
            or "state" in path.parts
        ):
            failures.append((name, "runtime-or-secret-file"))
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            failures.append((name, "unexpected-binary"))
            continue
        failures.extend((name, rule) for rule in scan_text(name, text, markers))
    for name, rule in failures:
        print(f"{name}: {rule}")
    if failures:
        raise SystemExit(1)
    print(f"Public-tree check passed: {len(names)} text files; no matching sensitive markers")


if __name__ == "__main__":
    sys.exit(main())
