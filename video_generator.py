from __future__ import annotations

import re
import shutil
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


class ExplainerVideoGenerator:
    """Generate a simple MP4 explainer slideshow from summary text."""

    def __init__(self, output_dir: str = "generated_videos") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _ffmpeg_available(self) -> bool:
        return shutil.which("ffmpeg") is not None

    def _extract_scenes(self, summary: str, max_scenes: int = 6) -> list[str]:
        lines = [line.strip(" -\t*") for line in summary.splitlines() if line.strip()]
        bullet_like = [line for line in lines if len(line) > 10]

        if bullet_like:
            scenes = bullet_like[:max_scenes]
        else:
            parts = re.split(r"(?<=[.!?])\s+", summary.strip())
            scenes = [p.strip() for p in parts if p.strip()][:max_scenes]

        if not scenes:
            scenes = ["No summary content available to render."]

        return [scene[:220] for scene in scenes]

    def _load_font(self, size: int) -> ImageFont.ImageFont:
        for font_name in ("arial.ttf", "segoeui.ttf", "calibri.ttf"):
            try:
                return ImageFont.truetype(font_name, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _create_slide(self, title: str, body: str, path: Path) -> None:
        width, height = 1280, 720
        image = Image.new("RGB", (width, height), color=(18, 24, 38))
        draw = ImageDraw.Draw(image)

        title_font = self._load_font(48)
        body_font = self._load_font(34)

        draw.rectangle([(0, 0), (width, 120)], fill=(34, 64, 118))
        draw.text((48, 32), title, fill=(240, 245, 255), font=title_font)

        wrapped = textwrap.wrap(body, width=52)
        y = 180
        for line in wrapped:
            draw.text((72, y), line, fill=(235, 238, 245), font=body_font)
            y += 52

        image.save(path, format="PNG")

    def generate_video_from_summary(
        self,
        summary: str,
        title: str = "Document Explainer",
        seconds_per_scene: int = 4,
        max_scenes: int = 6,
    ) -> str:
        if not self._ffmpeg_available():
            raise RuntimeError(
                "ffmpeg is not installed or not in PATH. Install ffmpeg to enable video generation."
            )

        scenes = self._extract_scenes(summary, max_scenes=max_scenes)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        slide_paths: list[Path] = []
        for idx, scene in enumerate(scenes, start=1):
            slide_path = run_dir / f"slide_{idx:02d}.png"
            self._create_slide(title=title, body=scene, path=slide_path)
            slide_paths.append(slide_path)

        concat_file = run_dir / "slides.txt"
        with concat_file.open("w", encoding="utf-8") as handle:
            for slide_path in slide_paths:
                handle.write(f"file '{slide_path.as_posix()}'\n")
                handle.write(f"duration {seconds_per_scene}\n")
            handle.write(f"file '{slide_paths[-1].as_posix()}'\n")

        output_file = self.output_dir / f"explainer_{timestamp}.mp4"
        command = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-vf",
            "format=yuv420p",
            "-movflags",
            "+faststart",
            str(output_file),
        ]

        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()}")

        return str(output_file)
