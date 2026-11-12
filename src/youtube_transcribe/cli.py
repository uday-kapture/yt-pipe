"""Command line: ``discover`` episodes, ``download`` episodes to keep locally, ``run`` the batch upload loop."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tomllib
from pathlib import Path

from .youtube import Filters, discover, fetch_episode

log = logging.getLogger("youtube_transcribe")


def _download(catalog: list[dict], args: argparse.Namespace, cookies: str | None) -> int:
    if args.only:
        wanted = set(args.only.read_text().split())
        catalog = [row for row in catalog if row["episode_id"] in wanted]
    failures = {}
    for row in catalog:
        try:
            fetch_episode(row, args.out / "episodes", args.out / "asr_input", cookies, video=args.video)
            log.info("done %s", row["episode_id"])
        except Exception as error:  # one bad video must not stop the rest
            failures[row["episode_id"]] = repr(error)
            log.error("failed %s: %r", row["episode_id"], error)
    if failures:
        (args.out / "download_failures.json").write_text(json.dumps(failures, indent=1))
        log.warning("%d downloads failed; see %s", len(failures), args.out / "download_failures.json")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="youtube-transcribe", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {
        "discover": sub.add_parser("discover", help="list episodes from the sources into catalog.jsonl"),
        "download": sub.add_parser("download", help="fetch catalog episodes and keep them locally"),
        "run": sub.add_parser("run", help="batches: fetch, transcribe, upload segments to HF, delete local files"),
    }
    for p in commands.values():
        p.add_argument("--config", type=Path, required=True, help="sources TOML (see sources.example.toml)")
        p.add_argument("--out", type=Path, required=True, help="data directory")
    commands["download"].add_argument("--video", action="store_true", help="also fetch ≤360p video (cut detection)")
    commands["download"].add_argument("--only", type=Path, help="file with episode ids to fetch, one per line")
    commands["run"].add_argument("--server", required=True, help="Nemotron 3.5 Transcribe vLLM server URL")
    commands["run"].add_argument("--hf-repo", required=True, help="private HF dataset repo, e.g. org/name")
    commands["run"].add_argument("--batch-hours", type=float, default=10.0, help="audio hours per batch")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    config = tomllib.loads(args.config.read_text())
    cookies = config.get("download", {}).get("cookies_file")

    if args.command == "discover":
        discover(config["sources"], Filters(**config.get("filters", {})), args.out)
        return 0

    catalog_path = args.out / "catalog.jsonl"
    if not catalog_path.exists():
        log.error("no %s; run `discover` first", catalog_path)
        return 1
    catalog = [json.loads(line) for line in catalog_path.read_text().splitlines()]
    if args.command == "download":
        return _download(catalog, args, cookies)

    from .batch import run  # heavy imports (datasets, nemo) only for this command

    return run(catalog, args.out, args.server, args.hf_repo, args.batch_hours, cookies)


if __name__ == "__main__":
    sys.exit(main())
