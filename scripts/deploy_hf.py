"""Deploy SEAF to a Hugging Face Space (Docker SDK, free CPU basic).

    export HF_TOKEN=...            # write-scoped token (never commit it)
    export HF_SPACE=user/seaf      # or pass --space
    python scripts/deploy_hf.py [--space user/seaf] [--private] [--dry-run]

Creates the Space if it does not exist (sdk=docker, cpu-basic) and uploads the
repository contents the image needs.  Runtime state, tests and caches are
excluded.  The README's YAML header configures the Space (app_port 7860).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INCLUDE = ["Dockerfile", ".dockerignore", "README.md", "requirements.txt", "app.py", ".streamlit/**",
           "assets/**", "app_pages/**", "ui/**", "seaf/**", "scripts/**", "artifacts/**", "results/**",
           "data/**", "docs/**"]
EXCLUDE = ["**/__pycache__/**", "**/*.pyc", "runtime/**", "**/*.db", ".git/**", "tests/**"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default=os.environ.get("HF_SPACE"), help="owner/name of the Space")
    ap.add_argument("--private", action="store_true", help="create the Space as private")
    ap.add_argument("--dry-run", action="store_true", help="list what would be uploaded and exit")
    args = ap.parse_args()

    if args.dry_run:
        from fnmatch import fnmatch
        files = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()]
        keep = [f for f in files if any(fnmatch(f, g) for g in INCLUDE) and not any(fnmatch(f, g) for g in EXCLUDE)]
        size = sum((ROOT / f).stat().st_size for f in keep)
        print(f"{len(keep)} files, {size / 1e6:.1f} MB")
        big = [f for f in keep if (ROOT / f).stat().st_size > 10e6]
        print("files > 10 MB (would need LFS):", big or "none")
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set (needs a write-scoped Hugging Face token).", file=sys.stderr)
        return 2
    if not args.space or "/" not in args.space:
        print("Give the Space as owner/name via --space or HF_SPACE.", file=sys.stderr)
        return 2

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(ROOT), repo_id=args.space, repo_type="space",
                      allow_patterns=INCLUDE, ignore_patterns=EXCLUDE,
                      commit_message="Deploy SEAF dashboard")
    print(f"Uploaded. Build logs: https://huggingface.co/spaces/{args.space}?logs=build")
    print(f"App URL once built:   https://huggingface.co/spaces/{args.space}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
