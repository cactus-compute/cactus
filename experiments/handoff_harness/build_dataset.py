"""Build the (image, audio) evaluation manifest.

Pairing rules:
    Desc obj   -> IMG_3128, IMG_3136          (object-description audio on objects)
    Desc pers  -> IMG_3129                    (person-description audio on person)
    Gen rel,
    Solve eq   -> all images + video frames   (non-desc audios test against everything)

For each (image, clean_audio) we also produce noisy variants mixing in each of
the crowd-noise mp3s at several SNRs.

Usage:
    python build_dataset.py
"""
from __future__ import annotations

import json
from pathlib import Path

from io_utils import (
    TARGET_SR,
    decode_to_mono16k_wav,
    extract_random_frames,
    mix_at_snr,
    read_wav_int16,
    write_wav_int16,
)


ROOT = Path(__file__).resolve().parents[2]
HARNESS = Path(__file__).resolve().parent
VISUAL = ROOT / "visual_prompts"
AUDIO_SRC = ROOT / "audio_prompts"
NOISE_SRC = [
    ROOT / "31736081-crowd-noise-sports-375087.mp3",
    ROOT / "freesound_community-audience-77867.mp3",
    ROOT / "u_vshmonplts-crowd-noise-390947.mp3",
]

AUDIO_CLEAN_DIR = HARNESS / "dataset" / "audio_clean"
AUDIO_NOISY_DIR = HARNESS / "dataset" / "audio_noisy"
FRAMES_DIR = HARNESS / "dataset" / "frames"
NOISE_CACHE_DIR = HARNESS / "dataset" / "noise_cache"
MANIFEST = HARNESS / "manifest.jsonl"

SNR_DB_LEVELS = [40.0, 20.0, 0.0]  # plus "clean"
FRAMES_PER_VIDEO = 2

# Existing audio prompts. Pairing is driven by `audio_pairs_for_image` below.
AUDIO_TAGS = {
    "Desc obj.m4a":   "desc_obj",
    "Desc pers.m4a":  "desc_pers",
    "Gen rel.m4a":    "gen_rel",
    "Solve eq.m4a":   "solve_eq",
}

# Additional general-knowledge / Q&A audio prompts. These have no relation to
# the image, so we pair them only with the audience-video frames to stress-test
# mismatched multimodal inputs.
EXTRA_AUDIO_TAGS = {
    "2_plus_2.mp3":        "qa_2plus2",
    "capital_france.mp3":  "qa_capital_france",
    "gpu_tracer.mp3":      "qa_gpu_tracer",
    "how_older.mp3":       "qa_how_older",
    "jit.mp3":             "qa_jit",
    "p_np.mp3":            "qa_p_np",
    "sky_color.mp3":       "qa_sky_color",
    "spanish+hello.mp3":   "qa_spanish_hello",
    "type_inference.mp3":  "qa_type_inference",
}


def prep_clean_audios(tag_map: dict[str, str]) -> dict[str, Path]:
    out = {}
    for src_name, tag in tag_map.items():
        src = AUDIO_SRC / src_name
        dst = AUDIO_CLEAN_DIR / f"{tag}.wav"
        if not dst.exists():
            decode_to_mono16k_wav(src, dst)
        out[tag] = dst
    return out


def prep_noise_wavs() -> list[tuple[str, Path]]:
    out = []
    for src in NOISE_SRC:
        dst = NOISE_CACHE_DIR / (src.stem + ".wav")
        if not dst.exists():
            decode_to_mono16k_wav(src, dst)
        # short tag for manifest
        if "sports" in src.stem:       tag = "noise_sports"
        elif "freesound" in src.stem:  tag = "noise_audience_fs"
        else:                          tag = "noise_crowd"
        out.append((tag, dst))
    return out


def prep_video_frames() -> list[Path]:
    """Gather frame jpgs for each audience video.

    Prefers extracting from .mov sources when present; otherwise falls back to
    any already-committed frames in FRAMES_DIR so the harness stays runnable
    on a fresh clone where the large .mov files are gitignored."""
    videos = sorted(VISUAL.glob("*.mov"))
    frames: list[Path] = []
    if videos:
        for v in videos:
            frames.extend(extract_random_frames(v, FRAMES_PER_VIDEO, FRAMES_DIR, seed=42))
        return frames
    # No source videos on disk — use pre-extracted frames.
    if FRAMES_DIR.exists():
        frames = sorted(FRAMES_DIR.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(
            f"No .mov files under {VISUAL} and no pre-extracted frames in "
            f"{FRAMES_DIR}. Either add audience_*.mov back or restore the "
            f"frame jpgs."
        )
    return frames


def audio_pairs_for_image(tag: str, is_object: bool, is_person: bool) -> list[str]:
    """Return list of audio tags that pair with this image."""
    pairs = []
    if is_object:
        pairs.append("desc_obj")
    if is_person:
        pairs.append("desc_pers")
    pairs.extend(["gen_rel", "solve_eq"])
    return pairs


def _emit_rows(img_path: Path, audio_tags: list[str], clean: dict[str, Path],
               noise_samples: list[tuple[str, list[int]]], is_obj: bool, is_pers: bool,
               rows: list[dict]) -> None:
    for atag in audio_tags:
        clean_wav = clean[atag]
        rows.append({
            "image": str(img_path),
            "audio": str(clean_wav),
            "audio_tag": atag,
            "noise_tag": "clean",
            "snr_db": None,
            "is_object": is_obj,
            "is_person": is_pers,
            "is_frame": img_path.parent == FRAMES_DIR,
        })
        clean_arr = read_wav_int16(clean_wav)
        for ntag, narr in noise_samples:
            for snr in SNR_DB_LEVELS:
                mixed = mix_at_snr(clean_arr, narr, snr)
                out_wav = AUDIO_NOISY_DIR / f"{atag}__{ntag}__snr{int(snr)}.wav"
                if not out_wav.exists():
                    write_wav_int16(out_wav, mixed, sr=TARGET_SR)
                rows.append({
                    "image": str(img_path),
                    "audio": str(out_wav),
                    "audio_tag": atag,
                    "noise_tag": ntag,
                    "snr_db": snr,
                    "is_object": is_obj,
                    "is_person": is_pers,
                    "is_frame": img_path.parent == FRAMES_DIR,
                })


def main() -> None:
    clean = prep_clean_audios(AUDIO_TAGS)
    extra_clean = prep_clean_audios(EXTRA_AUDIO_TAGS)
    noises = prep_noise_wavs()
    frames = prep_video_frames()

    # Primary visuals:
    visuals: list[tuple[Path, bool, bool]] = [
        (VISUAL / "IMG_3128.jpg", True,  False),
        (VISUAL / "IMG_3136.jpg", True,  False),
        (VISUAL / "IMG_3129.jpg", False, True),
    ]
    if not (VISUAL / "IMG_3136.jpg").exists() and (ROOT / "IMG_3136.jpg").exists():
        visuals[1] = (ROOT / "IMG_3136.jpg", True, False)
    for fp in frames:
        visuals.append((fp, False, False))

    noise_samples = [(tag, read_wav_int16(p)) for tag, p in noises]

    rows: list[dict] = []
    # Original audio prompts paired by audio_pairs_for_image
    for img_path, is_obj, is_pers in visuals:
        tags = audio_pairs_for_image(img_path.stem, is_obj, is_pers)
        _emit_rows(img_path, tags, clean, noise_samples, is_obj, is_pers, rows)

    # New Q&A prompts paired only with audience (video) frames
    extra_tags = list(EXTRA_AUDIO_TAGS.values())
    for fp in frames:
        _emit_rows(fp, extra_tags, extra_clean, noise_samples,
                   is_obj=False, is_pers=False, rows=rows)

    MANIFEST.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {len(rows)} rows to {MANIFEST}")
    print(f"  primary audios : {len(clean)}")
    print(f"  extra Q&A audios: {len(extra_clean)}")
    print(f"  noise sources  : {len(noises)}")
    print(f"  video frames   : {len(frames)}")
    print(f"  primary visuals: {len(visuals)}")


if __name__ == "__main__":
    main()
