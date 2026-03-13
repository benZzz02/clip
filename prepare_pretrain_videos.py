import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm


DEFAULT_MAIN_CSV = "/mnt/mydisk/CLIP/summary_csv/all_videos.csv"
DEFAULT_SRC_ROOT = "/mnt/mydisk/download_youtube_video/downloaded_video"
DEFAULT_ANNOTATIONS_FOLDER = "/mnt/mydisk/CLIP/csv_outputs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch transcode and resize pretraining videos to a target folder."
    )
    parser.add_argument("--main-csv", default=DEFAULT_MAIN_CSV)
    parser.add_argument("--src-root", default=DEFAULT_SRC_ROOT)
    parser.add_argument("--dst-root", default=None)
    parser.add_argument("--annotations-folder", default=DEFAULT_ANNOTATIONS_FOLDER)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--ffmpeg-threads", type=int, default=1)
    parser.add_argument("--codec", default="libx264")
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--profile", default="high")
    parser.add_argument("--level", default="4.1")
    parser.add_argument("--keyframe-seconds", type=float, default=1.0)
    parser.add_argument("--movflags", default="+faststart")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-videos", type=int, default=None)
    return parser.parse_args()


def resolve_dst_root(src_root, dst_root, size):
    if dst_root:
        return dst_root
    clean_src_root = src_root.rstrip("/\\")
    return f"{clean_src_root}_{size}"


def load_video_jobs(main_csv_path, src_root, dst_root, annotations_folder, max_videos):
    jobs = []
    seen_filenames = set()

    with open(main_csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            relative_video_path = (row.get("video_path") or "").strip()
            if not relative_video_path:
                continue

            video_filename = os.path.basename(relative_video_path)
            if video_filename in seen_filenames:
                continue

            if annotations_folder:
                annotation_csv = os.path.join(
                    annotations_folder,
                    f"{os.path.splitext(video_filename)[0]}.csv",
                )
                if not os.path.exists(annotation_csv):
                    continue

            src_path = os.path.join(src_root, video_filename)
            if not os.path.exists(src_path):
                print(f"[missing source] {src_path}", file=sys.stderr)
                continue

            dst_path = os.path.join(dst_root, video_filename)
            jobs.append((src_path, dst_path))
            seen_filenames.add(video_filename)

            if max_videos is not None and len(jobs) >= max_videos:
                break

    return jobs


def build_ffmpeg_cmd(
    src_path,
    dst_path,
    size,
    codec,
    preset,
    crf,
    profile,
    level,
    keyframe_seconds,
    movflags,
    ffmpeg_threads,
    overwrite,
):
    force_key_frames = f"expr:gte(t,n_forced*{keyframe_seconds})"
    return [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-v",
        "error",
        "-threads",
        str(ffmpeg_threads),
        "-i",
        src_path,
        "-vf",
        f"scale={size}:{size}:flags=lanczos,setsar=1",
        "-an",
        "-sn",
        "-dn",
        "-c:v",
        codec,
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-profile:v",
        profile,
        "-level:v",
        level,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        movflags,
        "-force_key_frames",
        force_key_frames,
        dst_path,
    ]


def transcode_one(job, args):
    src_path, dst_path = job

    if os.path.exists(dst_path) and not args.overwrite:
        return "skipped", dst_path, ""

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cmd = build_ffmpeg_cmd(
        src_path=src_path,
        dst_path=dst_path,
        size=args.size,
        codec=args.codec,
        preset=args.preset,
        crf=args.crf,
        profile=args.profile,
        level=args.level,
        keyframe_seconds=args.keyframe_seconds,
        movflags=args.movflags,
        ffmpeg_threads=args.ffmpeg_threads,
        overwrite=args.overwrite,
    )
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return "failed", src_path, proc.stderr.strip()[:400]
    return "done", dst_path, ""


def main():
    args = parse_args()
    args.dst_root = resolve_dst_root(args.src_root, args.dst_root, args.size)

    os.makedirs(args.dst_root, exist_ok=True)

    jobs = load_video_jobs(
        main_csv_path=args.main_csv,
        src_root=args.src_root,
        dst_root=args.dst_root,
        annotations_folder=args.annotations_folder,
        max_videos=args.max_videos,
    )

    print(f"main_csv={args.main_csv}")
    print(f"src_root={args.src_root}")
    print(f"dst_root={args.dst_root}")
    print(f"annotations_folder={args.annotations_folder}")
    print(f"target_size={args.size}")
    print(f"workers={args.workers}")
    print(f"codec={args.codec}")
    print(f"profile={args.profile}")
    print(f"level={args.level}")
    print(f"videos_to_process={len(jobs)}")

    if not jobs:
        return 0

    done = 0
    skipped = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(transcode_one, job, args) for job in jobs]
        with tqdm(total=len(futures), desc="Transcoding videos", unit="video") as pbar:
            for future in as_completed(futures):
                status, path, message = future.result()
                if status == "done":
                    done += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    failed += 1
                    print(f"[failed] {path}")
                    if message:
                        print(message, file=sys.stderr)

                pbar.set_postfix(
                    done=done,
                    skipped=skipped,
                    failed=failed,
                )
                pbar.update(1)

    print(
        f"finished | done={done} skipped={skipped} failed={failed} "
        f"dst_root={args.dst_root}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
