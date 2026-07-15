"""CLI for the local Shazam-style matcher: build index, run single query, or evaluate dataset."""

import argparse
import csv
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from local_fingerprint import LocalShazamMatcher, MatchResult  # noqa: E402
from _normalize import canon_title  # noqa: E402

# Repo root (parent of ``shazam/``): lets ``--query`` / ``--dataset-dir`` work when cwd is not the project.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_existing_path(path_str: str, *, expect_dir: bool) -> Path:
    """Resolve ``path_str`` relative to cwd first, then relative to repo root."""
    p = Path(path_str).expanduser()
    ok = p.is_dir() if expect_dir else p.is_file()
    if ok:
        return p.resolve()
    under = _REPO_ROOT / path_str
    ok = under.is_dir() if expect_dir else under.is_file()
    if ok:
        return under.resolve()
    label = "Directory" if expect_dir else "File"
    msg = (
        f"{label} not found: {path_str}\n"
        f"  cwd={os.getcwd()}\n"
        f"  Tried: {p.resolve()} and {under}"
    )
    if not expect_dir and "butter" in path_str.lower() and (_REPO_ROOT / "butter_eval").is_dir():
        examples = sorted((_REPO_ROOT / "butter_eval").glob("*/*.mp3"))[:3]
        if examples:
            rels = ", ".join(str(e.relative_to(_REPO_ROOT)) for e in examples)
            msg += f"\n  Example chunk paths in this repo: {rels}"
    raise FileNotFoundError(msg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Shazam-like matching experiments.")
    parser.add_argument("--songs-dir", default="mp3_songs", help="Directory with full source songs.")
    parser.add_argument("--dataset-dir", default="dataset", help="Directory with chunked folders.")
    parser.add_argument(
        "--index-path",
        default="shazam/index.pkl",
        help="Pickle index path to build/load.",
    )
    parser.add_argument("--build-index", action="store_true", help="Build index from --songs-dir.")
    parser.add_argument("--query", default=None, help="Single clip path to query.")
    parser.add_argument("--run-dataset", action="store_true", help="Evaluate all dataset chunks to CSV.")
    parser.add_argument(
        "--output-csv",
        default="dataset/shazam_chunk_results.csv",
        help="CSV output path when --run-dataset is used.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=10.0,
        help="Chunk duration (seconds) used to compute expected offset; must match dataset build.",
    )
    parser.add_argument(
        "--time-tolerance-sec",
        type=float,
        default=2.5,
        help="Tolerance for is_correct_place.",
    )
    # CLAP fallback (when fingerprint has no match in DB)
    parser.add_argument(
        "--clap-fallback",
        action="store_true",
        help="If fingerprint returns no song, rank by CLAP embedding similarity (needs --clap-index).",
    )
    parser.add_argument(
        "--clap-index",
        default="shazam/clap_index.json",
        help="Path to CLAP index JSON (sibling .npy with embeddings).",
    )
    parser.add_argument(
        "--build-clap-index",
        action="store_true",
        help="Build CLAP embedding index from --songs-dir (plus --clap-extra-audio) and write --clap-index.",
    )
    parser.add_argument(
        "--clap-extra-audio",
        action="append",
        default=[],
        metavar="PATH",
        help="Extra audio file for CLAP index only (repeatable). Skipped if the same file is already under --songs-dir.",
    )
    parser.add_argument(
        "--clap-model",
        default="laion/larger_clap_general",
        help="HuggingFace model id for CLAP (e.g. laion/larger_clap_general, laion/larger_clap_music).",
    )
    parser.add_argument(
        "--clap-min-sim",
        type=float,
        default=0.35,
        help="Min cosine similarity to accept a CLAP neighbor (rough, tune ~0.35–0.5).",
    )
    parser.add_argument(
        "--clap-fallback-if-fp-votes-below",
        type=int,
        default=None,
        metavar="N",
        help="Also try CLAP when fingerprint returns a song but votes < N (weak hit). If CLAP clears --clap-min-sim, its result replaces the FP match.",
    )
    parser.add_argument(
        "--clap-embed-seconds",
        type=float,
        default=10.0,
        help="Max seconds of audio embedded for query; index uses same window length.",
    )
    parser.add_argument(
        "--clap-hop-seconds",
        type=float,
        default=None,
        help=(
            "If set at build time, index each song as sliding windows of --clap-embed-seconds "
            "every --clap-hop-seconds (e.g. 5). The fallback will then report not only the "
            "most similar song but also the most similar part (approximate offset in sec). "
            "If omitted, a single middle-of-song embedding is used (legacy; no offset)."
        ),
    )
    parser.add_argument(
        "--clap-device",
        default=None,
        help="Force device: cuda, cpu, mps, or omit for auto.",
    )
    parser.add_argument(
        "--clap-use-offset",
        action="store_true",
        help=(
            "Allow CLAP fallback to emit a window offset. Default OFF because CLAP offsets "
            "can look good under overlap leakage but are not reliable for true localization."
        ),
    )
    return parser.parse_args()


def load_or_build_matcher(args: argparse.Namespace) -> LocalShazamMatcher:
    if args.build_index:
        matcher = LocalShazamMatcher()
        songs_dir = str(_resolve_existing_path(args.songs_dir, expect_dir=True))
        print(f"Indexing songs from: {songs_dir}")
        num_songs = matcher.build_from_directory(songs_dir)
        index_out = Path(args.index_path).expanduser()
        if not index_out.is_absolute():
            index_out = (_REPO_ROOT / args.index_path).resolve()
        os.makedirs(str(index_out.parent) or ".", exist_ok=True)
        matcher.save_index(str(index_out))
        print(f"Indexed {num_songs} songs and saved to {index_out}")
        return matcher
    index_path = Path(args.index_path).expanduser()
    if not index_path.is_file():
        alt = _REPO_ROOT / args.index_path
        if alt.is_file():
            index_path = alt
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Index not found: {args.index_path}. Run with --build-index first.\n"
            f"  cwd={os.getcwd()}\n"
            f"  Tried: {Path(args.index_path).expanduser()} and {_REPO_ROOT / args.index_path}"
        )
    return LocalShazamMatcher.load_index(str(index_path.resolve()))


def maybe_clap_fallback(
    r0: MatchResult,
    clip_path: str,
    clap_retriever,  # CLAPRetriever | None
    clap_paths: list[str] | None,
    clap_matrix,
    min_sim: float,
    embed_sec: float,
    *,
    fp_votes_ceiling: int | None = None,
    clap_windows: list[dict] | None = None,
    use_offset: bool = False,
) -> tuple[MatchResult, str]:
    """Try CLAP when fingerprint missed the DB, or when FP hit is weak (votes < ceiling).

    If the index was built with ``--clap-hop-seconds`` (sliding windows), the result also
    carries the most similar *part* of the song as ``offset_sec``.

    Returns (result, clap_cosine or '').
    """
    try_clap = r0.song_id is None
    if (
        fp_votes_ceiling is not None
        and r0.song_id is not None
        and r0.votes < fp_votes_ceiling
    ):
        try_clap = True
    if not try_clap:
        return r0, ""
    if clap_retriever is None or not clap_paths or clap_matrix is None or getattr(clap_matrix, "size", 0) == 0:
        return r0, ""
    near = clap_retriever.nearest(
        clip_path, clap_paths, clap_matrix, max_seconds=embed_sec, windows=clap_windows
    )
    if not near or near.similarity < min_sim:
        return r0, ""
    near_offset = near.offset_sec if use_offset else None
    if near_offset is not None:
        strategy = f"clap_fallback(cos={near.similarity:.3f}, t={near.offset_sec:.1f}s)"
    else:
        strategy = f"clap_fallback(cos={near.similarity:.3f})"
    return (
        MatchResult(
            song_id=near.song_id,
            song_path=near.song_path,
            confidence=round(near.similarity, 4),
            votes=0,
            total_hits=r0.total_hits,
            offset_sec=near_offset,
            strategy=strategy,
        ),
        f"{near.similarity:.4f}",
    )


def load_clap_or_none(args: argparse.Namespace):
    if not args.clap_fallback:
        return None, None, None, None
    clap_json = Path(args.clap_index).expanduser()
    if not clap_json.is_file():
        alt = _REPO_ROOT / args.clap_index
        if alt.is_file():
            clap_json = alt
    if not clap_json.is_file():
        print(
            f"Warning: --clap-fallback but index missing: {args.clap_index}\n"
            f"  cwd={os.getcwd()}\n"
            f"  Tried: {Path(args.clap_index).expanduser()} and {_REPO_ROOT / args.clap_index}\n"
            f"  Build with: .venv/bin/python shazam/run_experiment.py --build-clap-index --clap-index {args.clap_index}"
        )
        return None, None, None, None
    from clap_fallback import CLAPRetriever, load_clap_index  # noqa: E402

    paths, matrix, meta = load_clap_index(str(clap_json.resolve()))
    model_name = meta.get("model_name", args.clap_model)
    retriever = CLAPRetriever(model_name=model_name, device=args.clap_device)
    windows = meta.get("windows")
    return retriever, paths, matrix, windows


def chunk_index_from_name(chunk_path: str) -> int | None:
    m = re.search(r"_(\d+)\.[^.]+$", os.path.basename(chunk_path))
    return int(m.group(1)) if m else None


def run_dataset(
    matcher: LocalShazamMatcher,
    args: argparse.Namespace,
    clap_retriever,
    clap_paths: list[str] | None,
    clap_matrix,
    clap_windows: list[dict] | None = None,
) -> None:
    rows: list[dict] = []
    dataset_dir = _resolve_existing_path(args.dataset_dir, expect_dir=True)
    chunks = sorted(dataset_dir.glob("*/*.mp3"))
    for chunk in chunks:
        r0 = matcher.match(str(chunk))
        fp_strat = r0.strategy
        result = r0
        clap_cos = ""
        if args.clap_fallback:
            result, clap_cos = maybe_clap_fallback(
                r0,
                str(chunk),
                clap_retriever,
                clap_paths,
                clap_matrix,
                args.clap_min_sim,
                args.clap_embed_seconds,
                fp_votes_ceiling=args.clap_fallback_if_fp_votes_below,
                clap_windows=clap_windows,
                use_offset=bool(args.clap_use_offset),
            )

        expected_title = chunk.parent.name
        predicted_title = Path(result.song_path).stem if result.song_path else None
        is_correct_song = canon_title(predicted_title) == canon_title(expected_title)

        idx = chunk_index_from_name(str(chunk))
        expected_offset_sec = idx * args.chunk_seconds if idx is not None else None
        offset_error_sec = None
        is_correct_place = False
        if expected_offset_sec is not None and result.offset_sec is not None:
            offset_error_sec = round(abs(result.offset_sec - expected_offset_sec), 2)
            is_correct_place = offset_error_sec <= args.time_tolerance_sec

        rows.append(
            {
                "chunk_path": str(chunk),
                "expected_title": expected_title,
                "predicted_title": predicted_title,
                "song_id": result.song_id,
                "confidence": result.confidence,
                "votes": result.votes,
                "total_hits": result.total_hits,
                "estimated_offset_sec": result.offset_sec,
                "expected_offset_sec": expected_offset_sec,
                "offset_error_sec": offset_error_sec,
                "chunk_index": idx,
                "fingerprint_strategy": fp_strat,
                "recognition_strategy": result.strategy,
                "clap_cosine": clap_cos,
                "is_correct_song": is_correct_song,
                "is_correct_place": is_correct_place,
                "is_correct_song_and_place": bool(is_correct_song and is_correct_place),
            }
        )

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    correct_song = sum(1 for r in rows if r["is_correct_song"])
    correct_place = sum(1 for r in rows if r["is_correct_place"])
    correct_both = sum(1 for r in rows if r["is_correct_song_and_place"])
    pct = (lambda n: (100.0 * n / total) if total else 0.0)
    print(f"Dataset evaluated: {total} chunks")
    print(f"Song accuracy:         {correct_song}/{total} ({pct(correct_song):.2f}%)")
    print(f"Place accuracy:        {correct_place}/{total} ({pct(correct_place):.2f}%)")
    print(f"Song+Place accuracy:   {correct_both}/{total} ({pct(correct_both):.2f}%)")
    print(f"CSV written to:        {args.output_csv}")


def run_single_query(
    matcher: LocalShazamMatcher,
    query_path: str,
    args: argparse.Namespace,
    clap_retriever,
    clap_paths: list[str] | None,
    clap_matrix,
    clap_windows: list[dict] | None = None,
) -> None:
    r0 = matcher.match(query_path)
    fp_strat = r0.strategy
    result = r0
    clap_cos = ""
    if args.clap_fallback:
        result, clap_cos = maybe_clap_fallback(
            r0,
            query_path,
            clap_retriever,
            clap_paths,
            clap_matrix,
            args.clap_min_sim,
            args.clap_embed_seconds,
            fp_votes_ceiling=args.clap_fallback_if_fp_votes_below,
            clap_windows=clap_windows,
            use_offset=bool(args.clap_use_offset),
        )
    print(f"query={query_path}")
    print(f"fingerprint_strategy={fp_strat}")
    print(f"song_id={result.song_id}")
    print(f"song_path={result.song_path}")
    print(f"confidence={result.confidence}")
    print(f"votes={result.votes}")
    print(f"total_hits={result.total_hits}")
    print(f"offset_sec={result.offset_sec}")
    print(f"recognition_strategy={result.strategy}")
    if args.clap_fallback:
        print(f"clap_cosine={clap_cos or '(none)'}")


if __name__ == "__main__":
    import sys

    args = parse_args()

    if args.build_clap_index:
        from clap_fallback import build_clap_index  # noqa: E402

        songs_dir = str(_resolve_existing_path(args.songs_dir, expect_dir=True))
        clap_out = Path(args.clap_index).expanduser()
        if not clap_out.is_absolute():
            clap_out = (_REPO_ROOT / args.clap_index).resolve()
        extra: list[str] | None = None
        if args.clap_extra_audio:
            extra = []
            for raw in args.clap_extra_audio:
                try:
                    extra.append(str(_resolve_existing_path(raw, expect_dir=False)))
                except FileNotFoundError as exc:
                    sys.exit(str(exc))
        n = build_clap_index(
            songs_dir,
            str(clap_out),
            model_name=args.clap_model,
            embed_seconds=args.clap_embed_seconds,
            device=args.clap_device,
            extra_audio_paths=extra,
            hop_seconds=args.clap_hop_seconds,
        )
        mode = (
            f"sliding windows ({args.clap_embed_seconds}s every {args.clap_hop_seconds}s)"
            if args.clap_hop_seconds is not None
            else "1 embedding per song (middle)"
        )
        print(f"CLAP index: {n} songs, {mode} -> {clap_out}")
        if not (args.build_index or args.query or args.run_dataset):
            sys.exit(0)

    matcher: LocalShazamMatcher | None = None
    if args.build_index or args.query or args.run_dataset:
        matcher = load_or_build_matcher(args)

    clap_retriever, clap_paths, clap_matrix, clap_windows = load_clap_or_none(args)

    if args.query:
        if matcher is None:
            sys.exit("Use --build-index first or pass --query together with an existing fingerprint index.")
        try:
            query_path = str(_resolve_existing_path(args.query, expect_dir=False))
        except FileNotFoundError as exc:
            sys.exit(str(exc))
        run_single_query(
            matcher, query_path, args, clap_retriever, clap_paths, clap_matrix, clap_windows
        )
    if args.run_dataset:
        if matcher is None:
            sys.exit("Use --build-index first or pass --run-dataset with an existing fingerprint index.")
        run_dataset(matcher, args, clap_retriever, clap_paths, clap_matrix, clap_windows)

    if not args.query and not args.run_dataset:
        if args.build_index:
            print("Fingerprint index saved.")
        elif not args.build_clap_index:
            print("Nothing to run. Use --query, --run-dataset, --build-index, or --build-clap-index.")
