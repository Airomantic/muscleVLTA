"""Upload a local LeRobot dataset to Hugging Face, including native tactile files.

Run from the repository root:
    .venv/bin/python muscleForceWare/upload_dataset_to_hf.py --dry-run
    .venv/bin/python muscleForceWare/upload_dataset_to_hf.py --login

Authentication uses HF_TOKEN or the cached Hugging Face login. --login prompts
for a token without adding it to Git credentials. Do not put tokens in this file.
Use the official HF_ENDPOINT=https://huggingface.co for uploads, not a mirror.

Files go directly into the dataset repository root. Matching remote paths are
updated; unrelated remote files are not deleted. Hidden files and Python caches
are excluded. New repositories are private unless --public is given; uploading
to an existing public repository also requires --public.

Large uploads create multiple commits and save progress in .cache/huggingface.
To resume, run the same command with the same folder and repository. An
interrupted individual file may be uploaded again. Do not modify the dataset or
run concurrent upload processes while uploading. The target marker prevents
accidentally reusing committed-file cache entries for a different repository.
"""

import argparse
import json
import logging
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub import get_token
from huggingface_hub import login
from huggingface_hub.constants import ENDPOINT
from huggingface_hub.utils import filter_repo_objects
from huggingface_hub.utils import validate_repo_id

logger = logging.getLogger(__name__)

DEFAULT_FOLDER = Path("/data/jiang/utilData/ar3_daimon_Plug")
DEFAULT_REPO_ID = "airomantic88/force_aware_visuotactile"
OFFICIAL_ENDPOINT = "https://huggingface.co"
IGNORE_PATTERNS = (".*", "**/.*", "__pycache__/*", "**/__pycache__/*", "*.pyc")


def collect_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    if (folder / ".conversion_incomplete").exists():
        raise ValueError("Dataset conversion is incomplete; finish conversion before uploading.")
    if not (folder / "meta/info.json").is_file() or not (folder / "data").is_dir():
        raise ValueError(f"Expected a LeRobot dataset root containing meta/info.json and data/: {folder}")

    files = []
    for path in sorted(folder.rglob("*")):
        relative = path.relative_to(folder).as_posix()
        if not list(filter_repo_objects([relative], ignore_patterns=list(IGNORE_PATTERNS))):
            continue
        if path.is_symlink():
            raise ValueError(f"Refusing to upload a symlink; use files inside the dataset: {path}")
        if path.is_file():
            files.append(path)
    if not any(path.is_relative_to(folder / "data") and path.suffix == ".parquet" for path in files):
        raise ValueError("No data/*.parquet files found in the dataset.")
    return files


def check_upload_cache(folder: Path, repo_id: str) -> tuple[Path, dict[str, str]]:
    cache = folder / ".cache/huggingface"
    marker = cache / "muscleforce_upload_target.json"
    if any(path.is_symlink() for path in (folder / ".cache", cache, cache / "upload", marker)):
        raise ValueError("The Hugging Face upload cache must not use symlinks.")
    if not cache.resolve().is_relative_to(folder):
        raise ValueError("The Hugging Face upload cache must remain inside the dataset directory.")
    target = {"repo_id": repo_id, "repo_type": "dataset", "endpoint": OFFICIAL_ENDPOINT}
    if marker.exists():
        if json.loads(marker.read_text()) != target:
            raise ValueError(
                "This folder's upload cache belongs to a different repository. "
                "Use a fresh copy without .cache for another target; do not reuse this upload cache."
            )
    elif (cache / "upload").exists() and any((cache / "upload").rglob("*.metadata")):
        raise ValueError(
            "Found an existing upload cache with an unknown target. "
            "Use a fresh dataset copy without .cache to avoid skipping files committed to another repository."
        )
    return marker, target


def upload_dataset(
    folder_path: Path = DEFAULT_FOLDER,
    repo_id: str = DEFAULT_REPO_ID,
    *,
    login_first: bool = False,
    public: bool = False,
    num_workers: int = 4,
    dry_run: bool = False,
) -> None:
    validate_repo_id(repo_id)
    if num_workers <= 0:
        raise ValueError("--num-workers must be positive")
    folder = folder_path.expanduser().resolve()
    files = collect_files(folder)
    marker, target = check_upload_cache(folder, repo_id)
    total_bytes = sum(path.stat().st_size for path in files)
    logger.info("Source: %s", folder)
    logger.info("Destination: %s/datasets/%s (repository root)", OFFICIAL_ENDPOINT, repo_id)
    logger.info("Selected %d files, %.2f GiB; hidden files and Python caches excluded", len(files), total_bytes / 2**30)
    logger.info("Public upload allowed: %s", public)
    if dry_run:
        logger.info("Dry run only: no login, network requests, uploads, or cache writes.")
        return

    if ENDPOINT.rstrip("/") != OFFICIAL_ENDPOINT:
        raise ValueError(
            "Set HF_ENDPOINT=https://huggingface.co before starting this script; do not log in to a mirror."
        )
    if login_first:
        login(add_to_git_credential=False)
    token = get_token()
    if not token:
        raise ValueError(
            "No Hugging Face token found. Rerun with --login or set HF_TOKEN with dataset write permission."
        )
    api = HfApi(endpoint=OFFICIAL_ENDPOINT, token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=not public)
    info = api.repo_info(repo_id=repo_id, repo_type="dataset")
    if not info.private and not public:
        raise ValueError("The target dataset is public. Rerun with --public only if these data may be made public.")
    logger.info("Remote visibility: %s", "private" if info.private else "public")

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(target, indent=2) + "\n")
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=folder,
        private=not public,
        ignore_patterns=list(IGNORE_PATTERNS),
        num_workers=num_workers,
        print_report=True,
        print_report_every=30,
    )
    logger.info("Upload complete: %s/datasets/%s", OFFICIAL_ENDPOINT, repo_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--folder-path", type=Path, default=DEFAULT_FOLDER, help=f"Dataset root (default: {DEFAULT_FOLDER})"
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help=f"Dataset repository (default: {DEFAULT_REPO_ID})")
    parser.add_argument(
        "--login", action="store_true", help="Interactively log in before uploading; never used in dry runs"
    )
    parser.add_argument(
        "--public", action="store_true", help="Allow public uploads; create a missing repository publicly"
    )
    parser.add_argument("--num-workers", type=int, default=4, help="Upload worker count (default: 4)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show local upload summary without authentication or network"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    upload_dataset(
        folder_path=args.folder_path,
        repo_id=args.repo_id,
        login_first=args.login,
        public=args.public,
        num_workers=args.num_workers,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
