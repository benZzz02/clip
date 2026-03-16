import argparse
import glob
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm


DEFAULT_SRC_ROOT = "/mnt/mydisk/download_youtube_video/downloaded_video"
DEFAULT_ANNOTATIONS_FOLDER = "/mnt/mydisk/VATS_audio/transcripted_audio_peskavlp_style_valid"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch transcode and resize videos based on annotation CSV filenames."
    )
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


def find_video_by_stem(src_root, stem):
    """
    根据 csv 文件名（不带扩展名）查找对应视频。
    优先查找常见视频扩展名。
    """
    exts = [".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI", ".mkv", ".MKV"]
    for ext in exts:
        candidate = os.path.join(src_root, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def load_video_jobs_from_annotation_folder(src_root, dst_root, annotations_folder, max_videos):
    jobs = []

    if not os.path.isdir(annotations_folder):
        raise FileNotFoundError(f"annotations folder not found: {annotations_folder}")

    csv_files = sorted(glob.glob(os.path.join(annotations_folder, "*.csv")))
    if not csv_files:
        print(f"[warning] no csv files found in: {annotations_folder}", file=sys.stderr)
        return jobs

    for csv_path in csv_files:
        try:
            stem = os.path.splitext(os.path.basename(csv_path))[0]
            src_path = find_video_by_stem(src_root, stem)

            if src_path is None:
                print(f"[skip] no matching video for csv: {csv_path}", file=sys.stderr)
                continue

            video_filename = os.path.basename(src_path)
            dst_path = os.path.join(dst_root, video_filename)
            jobs.append((src_path, dst_path))

            if max_videos is not None and len(jobs) >= max_videos:
                break

        except Exception as e:
            print(f"[skip] failed to parse csv filename: {csv_path} | {e}", file=sys.stderr)
            continue

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
        "-v", "error",
        "-threads", str(ffmpeg_threads),
        "-i", src_path,
        "-vf", f"scale={size}:{size}:flags=lanczos,setsar=1",
        "-an",
        "-sn",
        "-dn",
        "-c:v", codec,
        "-preset", preset,
        "-crf", str(crf),
        "-profile:v", profile,
        "-level:v", level,
        "-pix_fmt", "yuv420p",
        "-movflags", movflags,
        "-force_key_frames", force_key_frames,
        dst_path,
    ]


def transcode_one(job, args):
    try:
        src_path, dst_path = job

        if not os.path.exists(src_path):
            return "skipped", src_path, "source video disappeared before transcoding"

        if os.path.exists(dst_path) and not args.overwrite:
            return "skipped", dst_path, "destination already exists"

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

    except Exception as e:
        return "failed", job[0] if isinstance(job, tuple) and len(job) > 0 else "unknown", str(e)


def main():
    args = parse_args()
    args.dst_root = resolve_dst_root(args.src_root, args.dst_root, args.size)

    os.makedirs(args.dst_root, exist_ok=True)

    jobs = load_video_jobs_from_annotation_folder(
        src_root=args.src_root,
        dst_root=args.dst_root,
        annotations_folder=args.annotations_folder,
        max_videos=args.max_videos,
    )

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
                try:
                    status, path, message = future.result()
                except Exception as e:
                    status, path, message = "failed", "unknown", str(e)

                if status == "done":
                    done += 1
                elif status == "skipped":
                    skipped += 1
                    if message:
                        print(f"[skipped] {path} | {message}", file=sys.stderr)
                else:
                    failed += 1
                    print(f"[failed] {path}", file=sys.stderr)
                    if message:
                        print(message, file=sys.stderr)

                pbar.set_postfix(done=done, skipped=skipped, failed=failed)
                pbar.update(1)

    print(
        f"finished | done={done} skipped={skipped} failed={failed} "
        f"dst_root={args.dst_root}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())