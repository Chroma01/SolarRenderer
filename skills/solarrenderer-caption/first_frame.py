"""first_frame.py: the image prompt for a SolarRenderer first frame, and the crop that fits it to the canvas.

    python first_frame.py frame0 structure.mp4 frame0.png        # frame 0 of the structure clip, the reference image
    python first_frame.py prompt prompt.txt [--stylised]          # the image prompt, from a finished six-part prompt
    python first_frame.py fit generated.png first.png [768x1344]  # crop and scale to the canvas (default 1344x768)

The first frame is made by an image model that edits frame 0 of the structure clip: same camera, same layout, same
characters in the same pose and facing, with the look the prompt describes. Any image model that takes a reference
image works (Codex's image generation, gpt-image, Nano Banana, ...).
"""
import re
import subprocess
import sys

KEEP = ("Re-render the reference image with IDENTICAL camera, framing, perspective and composition: every object, "
        "building, prop, terrain shape and character stays exactly where it is, at the same size and with the same "
        "outline and pose, pixel-aligned with the reference. Every character keeps the same facing direction as in "
        "the reference (if we see their back, we still see their back), the same pose and the same size. Only replace "
        "the untextured grey or flat-coloured surfaces with final materials, lighting and detail. ")
QUALITY = (" Masterful cinematic frame from a next-generation AAA game or film: physically based materials, rich "
           "micro-detail, volumetric light and atmosphere, deliberate colour grading, sharp focus on the subject. "
           "No text, no logos, no UI, no watermark, no borders.")
# Stylised looks (pixel art, toon, painterly, clay, ...) must not be pulled back to photoreal by the line above.
QUALITY_STYLISED = (" A polished, faithful frame rendered strictly in the art style described above - do not make it "
                    "photorealistic, keep every hallmark of that style (its geometry, texture resolution, palette, "
                    "shading and rendering artefacts). No text, no logos, no UI, no watermark, no borders.")


def image_prompt(prompt: str, stylised: bool = False) -> str:
    """KEEP + the style sentences of the detailed description + the first three sentences of [Shot 1] + quality."""

    detail = prompt.split("detailed_description:", 1)[-1].split("overall_soundscape:", 1)[0]
    style, _, shot = detail.partition("[Shot 1]")
    shot = re.sub(r"^\s*The shot begins from <Video \d>\.\s*", "", shot.strip())
    opening = " ".join(re.split(r"(?<=[.!?])\s+", shot)[:3])
    return KEEP + style.strip() + " The first moment of the shot: " + opening + (QUALITY_STYLISED if stylised else QUALITY)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    cmd, *rest = sys.argv[1:]
    if cmd == "frame0":
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", rest[0], "-frames:v", "1", rest[1]], check=True)
    elif cmd == "prompt":
        print(image_prompt(open(rest[0]).read(), stylised="--stylised" in rest))
    elif cmd == "fit":
        w, h = (rest[2] if len(rest) > 2 else "1344x768").split("x")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", rest[0], "-vf",
                        f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}", rest[1]], check=True)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
