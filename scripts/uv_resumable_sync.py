"""Resumable `uv sync` for unstable networks.

uv cannot resume a partially downloaded wheel, so large wheels (CUDA, cuDNN,
NCCL, ...) restart from zero whenever the connection drops. This script:

1. asks `uv sync --locked --dry-run` which packages still need installing;
2. downloads the matching locked wheels that are not in the uv cache with
   `curl -C -` (resumable, retried until complete) and checks their sha256;
3. serves them on 127.0.0.1, temporarily points their `uv.lock` URLs there,
   and runs `uv sync --frozen`; `uv.lock` is always restored afterwards;
4. runs `uv sync --locked` to finish and verify the environment.

Usage (proxy variables are honored by both curl and uv):

    uv run --no-project --python 3.11 --with packaging \\
        scripts/uv_resumable_sync.py -- --python 3.11 --no-install-package torch

Arguments after `--` are passed to `uv sync`. Rerun after any interruption;
completed and partial downloads in `.uv-wheels/` are reused.
"""

import argparse
import functools
import hashlib
import http.server
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import tomllib

from packaging.tags import sys_tags
from packaging.utils import canonicalize_name
from packaging.utils import parse_wheel_filename

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOCK = ROOT / "uv.lock"
LOCK_BACKUP = ROOT / "uv.lock.resumable-sync.bak"
LOCAL_HOSTS = "localhost,127.0.0.1,::1"


def uv_sync(extra: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["uv", "sync", *extra], cwd=ROOT, check=check, text=True, capture_output=capture, env=_env())


def _env() -> dict[str, str]:
    env = dict(os.environ)
    # `uv run` points VIRTUAL_ENV at its throwaway env; uv sync must target the project's .venv.
    env.pop("VIRTUAL_ENV", None)
    for key in ("NO_PROXY", "no_proxy"):
        env[key] = ",".join(filter(None, [env.get(key, ""), LOCAL_HOSTS]))
    env.setdefault("GIT_LFS_SKIP_SMUDGE", "1")
    return env


def pending_packages(sync_args: list[str]) -> list[tuple[str, str]]:
    result = uv_sync(["--locked", "--dry-run", *sync_args], capture=True, check=False)
    output = result.stdout + result.stderr
    if result.returncode != 0:
        sys.exit(f"`uv sync --locked --dry-run` failed:\n{output}")
    # Lines like " + nvidia-nccl-cu12==2.26.2"; workspace/git entries carry "(from ...)" and are left to uv.
    return re.findall(r"^ [+~] ([A-Za-z0-9._-]+)==(\S+)\s*$", output, re.MULTILINE)


def select_wheel(package: dict) -> dict | None:
    priority = {tag: i for i, tag in enumerate(sys_tags())}
    best = None
    for wheel in package.get("wheels", []):
        filename = wheel["url"].rsplit("/", 1)[1]
        ranks = [priority[t] for t in parse_wheel_filename(filename)[3] if t in priority]
        if ranks and (best is None or min(ranks) < best[0]):
            best = (min(ranks), wheel)
    return best[1] if best else None


def is_cached(cache_dir: pathlib.Path, name: str, filename: str) -> bool:
    # uv stores registry wheels as wheels-v*/pypi/<name>/<version>-<tags> (long tag sets are hashed,
    # which this misses; those are re-fetched, which is harmless).
    entry = filename.removesuffix(".whl").split("-", 1)[1]
    return any((d / "pypi" / name / entry).exists() for d in cache_dir.glob("wheels-v*"))


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: pathlib.Path, size: int | None, expected_hash: str, max_attempts: int) -> None:
    algo, _, expected = expected_hash.partition(":")
    if algo != "sha256":
        sys.exit(f"Unsupported hash {expected_hash} for {dest.name}")
    if dest.exists() and sha256(dest) == expected:
        return
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, max_attempts + 1):
        done = part.stat().st_size if part.exists() else 0
        print(f"[{attempt}/{max_attempts}] {dest.name}: {done / 2**20:.0f}/{(size or 0) / 2**20:.0f} MiB", flush=True)
        if size is None or done < size:
            subprocess.run(
                [
                    "curl",
                    "-L",
                    "--fail",
                    "-C",
                    "-",
                    "-o",
                    str(part),
                    "--connect-timeout",
                    "30",
                    "--speed-limit",
                    "10240",
                    "--speed-time",
                    "60",
                    url,
                ],
                check=False,
            )
        if part.exists() and (size is None or part.stat().st_size >= size):
            if sha256(part) == expected:
                part.rename(dest)
                return
            print(f"sha256 mismatch for {dest.name}; restarting this file.", flush=True)
            part.unlink()
        time.sleep(3)
    sys.exit(f"Giving up on {dest.name} after {max_attempts} attempts; rerun to continue.")


def restore_lock() -> None:
    if LOCK_BACKUP.exists():
        os.replace(LOCK_BACKUP, LOCK)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wheel-dir", type=pathlib.Path, default=ROOT / ".uv-wheels")
    parser.add_argument("--max-attempts", type=int, default=1000, help="curl attempts per wheel")
    parser.add_argument(
        "--min-size-mib", type=float, default=20, help="only prefetch larger wheels; uv downloads the rest"
    )
    parser.add_argument("sync_args", nargs=argparse.REMAINDER, help="arguments for `uv sync` (after --)")
    args = parser.parse_args()
    sync_args = [a for a in args.sync_args if a != "--"] or ["--python", "3.11"]

    # A previous run killed with SIGKILL may have left the rewritten lock behind.
    restore_lock()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    lock = tomllib.loads(LOCK.read_text())
    by_key = {(canonicalize_name(p["name"]), p["version"]): p for p in lock["package"] if "version" in p}
    cache_dir = pathlib.Path(
        subprocess.run(["uv", "cache", "dir"], capture_output=True, text=True, check=True).stdout.strip()
    )

    wanted = []
    for name, version in pending_packages(sync_args):
        package = by_key.get((canonicalize_name(name), version))
        if not package or "registry" not in package.get("source", {}):
            continue
        wheel = select_wheel(package)
        if not wheel or wheel.get("size", 0) < args.min_size_mib * 2**20:
            continue
        if not is_cached(cache_dir, package["name"], wheel["url"].rsplit("/", 1)[1]):
            wanted.append(wheel)

    total = sum(w.get("size", 0) for w in wanted) / 2**20
    print(f"{len(wanted)} wheel(s) to prefetch ({total:.0f} MiB).", flush=True)
    args.wheel_dir.mkdir(parents=True, exist_ok=True)
    for wheel in wanted:
        filename = wheel["url"].rsplit("/", 1)[1]
        download(wheel["url"], args.wheel_dir / filename, wheel.get("size"), wheel["hash"], args.max_attempts)

    if wanted:
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(args.wheel_dir))
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_address[1]}/"
        text = LOCK.read_text()
        for wheel in wanted:
            text = text.replace(f'url = "{wheel["url"]}"', f'url = "{base}{wheel["url"].rsplit("/", 1)[1]}"')
        shutil.copy2(LOCK, LOCK_BACKUP)
        try:
            LOCK.write_text(text)
            uv_sync(["--frozen", *sync_args])
        finally:
            restore_lock()
            server.shutdown()

    uv_sync(["--locked", *sync_args])
    print("Environment is in sync with uv.lock.")


if __name__ == "__main__":
    main()
