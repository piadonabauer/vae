#!/usr/bin/env python3
"""
Specialized NeRSemble preprocessing for one expression at 256px.

This script is intentionally narrow:
- Process ALL participants (from extracted folders OR from *.tar archives)
- Process ONLY sequence/expression: EMO-1-shout+laugh
- Use 2 middle upper cameras (same logic as preprocess_nersemble.py)
- Save merged tensor as:
    /datasets/lindell-proj/neumayr/nersemble_v2/processed/256-res/pXXX/EMO-1-shout+laugh/EMO-1-shout+laugh.pt

Note: On many clusters the raw dataset is only available as participant .tar files (no p018/ folders).
In that case use --from-tars, or rely on auto-detect (no folders + .tars present -> tar mode).
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import torch

from preprocess_nersemble import (
    ARRAY_OUTPUT_ROOT,
    DEFAULT_RVM_CHECKPOINT,
    Converter,
    TAR_IGNORE_SEQUENCE_NAMES,
    build_nersemble_managers,
    list_participant_tars,
    list_sequences_in_tar,
    prepare_temp_nersemble_from_tar,
    process_sequence,
)


TARGET_EXPRESSION = "EMO-1-shout+laugh"
TARGET_IMAGE_SIZE = 256
TARGET_FRAMES = 13
TARGET_OUTPUT_ROOT = ARRAY_OUTPUT_ROOT / "256-res"


def _slurm_array_chunk(items: list, label: str) -> list:
    """Chunk only when Slurm provides an array task count (avoids empty slices from stale env)."""
    tc_raw = os.environ.get("SLURM_ARRAY_TASK_COUNT")
    tid_raw = os.environ.get("SLURM_ARRAY_TASK_ID")
    if tc_raw is None or tid_raw is None:
        return items
    total = int(tc_raw)
    task_id = int(tid_raw)
    if total <= 0:
        return items
    n = len(items)
    chunk = (n + total - 1) // total
    start, end = task_id * chunk, (task_id + 1) * chunk
    out = items[start:end]
    print(
        f"[preprocess-emo1-256] Slurm array: task {task_id}/{total}, "
        f"{label}[{start}:{end}] -> {len(out)} item(s)"
    )
    return out


def _rename_frames_pt(output_root: Path, participant_id: int) -> None:
    seq_dir = output_root / f"p{participant_id:03d}" / TARGET_EXPRESSION
    legacy_pt = seq_dir / "frames.pt"
    target_pt = seq_dir / f"{TARGET_EXPRESSION}.pt"
    if legacy_pt.exists():
        legacy_pt.replace(target_pt)
        print(f"[preprocess-emo1-256] Saved {target_pt}")
    elif target_pt.exists():
        print(f"[preprocess-emo1-256] Saved {target_pt}")
    else:
        print(
            f"[preprocess-emo1-256] WARNING: expected output missing for "
            f"p{participant_id:03d} {TARGET_EXPRESSION}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Preprocess only EMO-1-shout+laugh to 256px merged .pt files."
    )
    p.add_argument(
        "--nersemble-root",
        type=Path,
        default=Path("/datasets/lindell-proj/neumayr/nersemble_v2/"),
        help="NeRSemble root: extracted participant folders, or a directory of *.tar (--from-tars).",
    )
    p.add_argument(
        "--from-tars",
        action="store_true",
        help="Read participant *.tar archives (same layout as preprocess_nersemble.py --from-tars).",
    )
    p.add_argument(
        "--no-auto-tar",
        action="store_true",
        help="Do not auto-switch to tar mode when no participant folders exist.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=TARGET_OUTPUT_ROOT,
        help="Output root. Default is fixed to .../processed/256-res.",
    )
    p.add_argument(
        "--rvm-checkpoint",
        type=Path,
        default=DEFAULT_RVM_CHECKPOINT,
        help="Path to rvm_mobilenetv3.pth checkpoint.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if target .pt already exists.",
    )
    p.add_argument(
        "--num-participants",
        type=int,
        default=None,
        help="Optional limit (first N participants or first N .tar files).",
    )
    p.add_argument(
        "--temp-dir",
        type=Path,
        default=Path("/home/piado/scratch"),
        help="Parent directory for temporary files.",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.temp_dir is not None:
        td = args.temp_dir.expanduser().resolve()
        td.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(td)
        print(f"[preprocess-emo1-256] Temp parent (TMPDIR): {td}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    nersemble_root = Path(args.nersemble_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[preprocess-emo1-256] Input root:  {nersemble_root}")
    print(f"[preprocess-emo1-256] Output root: {output_root}")
    print(f"[preprocess-emo1-256] Expression:  {TARGET_EXPRESSION}")
    print(f"[preprocess-emo1-256] Image size:  {TARGET_IMAGE_SIZE}")

    converter = Converter("mobilenetv3", str(args.rvm_checkpoint), device=device)

    data_folder, ParticipantManager = build_nersemble_managers(nersemble_root)
    participants = sorted(data_folder.list_participants())
    tar_pairs = list_participant_tars(nersemble_root)

    use_tar = bool(args.from_tars)
    if not use_tar and not args.no_auto_tar and not participants and tar_pairs:
        use_tar = True
        print(
            "[preprocess-emo1-256] No participant directories under root; "
            f"found {len(tar_pairs)} .tar file(s) -> using tar mode (like --from-tars)."
        )

    if use_tar:
        if not tar_pairs:
            print(
                "[preprocess-emo1-256] ERROR: --from-tars (or auto-tar) set but no *.tar under "
                f"{nersemble_root}"
            )
            raise SystemExit(1)
        tar_items = tar_pairs
        if args.num_participants:
            tar_items = tar_items[: args.num_participants]
        tar_items = _slurm_array_chunk(tar_items, "archives")
        print(f"[preprocess-emo1-256] Tar mode: {len(tar_items)} archive(s) to scan")

        for tar_path, pid in tar_items:
            try:
                sequences = list_sequences_in_tar(tar_path, pid)
            except OSError as e:
                print(f"[preprocess-emo1-256] Skip unreadable {tar_path.name}: {e}")
                continue

            if TARGET_EXPRESSION not in sequences:
                print(
                    f"[preprocess-emo1-256] Skip p{pid:03d}: {TARGET_EXPRESSION} "
                    f"not in {tar_path.name}"
                )
                continue

            seq_dir = output_root / f"p{pid:03d}" / TARGET_EXPRESSION
            target_pt = seq_dir / f"{TARGET_EXPRESSION}.pt"
            if args.skip_existing and target_pt.exists():
                print(f"[preprocess-emo1-256] Skip existing: {target_pt}")
                continue

            print(f"[preprocess-emo1-256] Processing p{pid:03d} {TARGET_EXPRESSION} from {tar_path.name}")
            try:
                with tempfile.TemporaryDirectory(
                    dir=os.environ.get("TMPDIR") or None
                ) as tmp:
                    tmp_path = Path(tmp)
                    nersemble_local = prepare_temp_nersemble_from_tar(
                        tar_path,
                        tmp_path,
                        pid,
                        TARGET_EXPRESSION,
                        2,
                    )
                    process_sequence(
                        nersemble_root=nersemble_local,
                        participant_id=pid,
                        sequence_name=TARGET_EXPRESSION,
                        converter=converter,
                        output_root=output_root,
                        image_size=TARGET_IMAGE_SIZE,
                        upper_views=2,
                        skip_existing=False,
                        target_frames=TARGET_FRAMES,
                        save_merged_pt=True,
                        write_mp4_per_camera=False,
                        test_dump_dir=None,
                    )
            except Exception as e:
                print(f"[preprocess-emo1-256] ERROR p{pid:03d} {TARGET_EXPRESSION}: {e}")
                continue
            _rename_frames_pt(output_root, pid)
        return

    if not participants:
        subs = sorted(
            p.name
            for p in nersemble_root.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )[:25]
        print(
            f"[preprocess-emo1-256] ERROR: no participant folders parsed under {nersemble_root}.\n"
            f"  Sample subdirs: {subs if subs else '(none)'}\n"
            f"  If your data is *.tar only, run with: --from-tars\n"
            f"  (or omit --no-auto-tar to auto-detect.)"
        )
        raise SystemExit(1)

    if args.num_participants:
        participants = participants[: args.num_participants]
    participants = _slurm_array_chunk(participants, "participants")

    print(f"[preprocess-emo1-256] Extracted mode: {len(participants)} participant(s)")

    for pid in participants:
        pm = ParticipantManager(str(nersemble_root), pid)
        sequences = pm.list_sequences()
        if TARGET_EXPRESSION not in sequences:
            print(
                f"[preprocess-emo1-256] Skip p{pid:03d}: "
                f"{TARGET_EXPRESSION} not found"
            )
            continue
        if TARGET_EXPRESSION in TAR_IGNORE_SEQUENCE_NAMES:
            continue

        seq_dir = output_root / f"p{pid:03d}" / TARGET_EXPRESSION
        target_pt = seq_dir / f"{TARGET_EXPRESSION}.pt"

        if args.skip_existing and target_pt.exists():
            print(f"[preprocess-emo1-256] Skip existing: {target_pt}")
            continue

        print(f"[preprocess-emo1-256] Processing p{pid:03d} {TARGET_EXPRESSION}")
        try:
            process_sequence(
                nersemble_root=nersemble_root,
                participant_id=pid,
                sequence_name=TARGET_EXPRESSION,
                converter=converter,
                output_root=output_root,
                image_size=TARGET_IMAGE_SIZE,
                upper_views=2,
                skip_existing=False,
                target_frames=TARGET_FRAMES,
                save_merged_pt=True,
                write_mp4_per_camera=False,
                test_dump_dir=None,
            )
        except Exception as e:
            print(f"[preprocess-emo1-256] ERROR p{pid:03d} {TARGET_EXPRESSION}: {e}")
            continue

        _rename_frames_pt(output_root, pid)


if __name__ == "__main__":
    main()
