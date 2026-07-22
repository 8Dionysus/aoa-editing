"""Synthetic fixtures that share no content with the sealed reference."""

from __future__ import annotations

import subprocess
from pathlib import Path
from random import Random

from PIL import Image, ImageDraw


class FixtureError(RuntimeError):
    """A synthetic fixture could not be produced."""


def create_fixtures(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    still = root / "abstract-orbit.png"
    speech = root / "synthetic-speech.mp4"
    memory = root / "color-memory.mp4"
    _still(still)
    _speech(speech)
    _memory(memory)
    return {"still.motion": still, "speech.clean": speech, "memory.montage": memory}


def create_transfer_fixture(path: Path) -> Path:
    """Create a feature-rich landscape source unrelated to the reference case."""

    path.parent.mkdir(parents=True, exist_ok=True)
    random = Random(24071984)
    image = Image.new("RGB", (1024, 768), "#10263b")
    draw = ImageDraw.Draw(image)
    for y in range(768):
        blend = y / 767
        draw.line(
            (0, y, 1024, y),
            fill=(
                round(18 + 44 * blend),
                round(44 + 38 * blend),
                round(72 + 26 * blend),
            ),
        )
    for _ in range(1400):
        x = random.randrange(8, 1016)
        y = random.randrange(8, 760)
        radius = random.randrange(1, 5)
        color = random.choice(("#8ed6c9", "#f4c36a", "#cf6f75", "#dbe8ef"))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    draw.polygon([(40, 690), (255, 260), (440, 690)], fill="#2d6674")
    draw.polygon([(350, 690), (620, 170), (910, 690)], fill="#98545e")
    draw.ellipse((705, 65, 910, 270), fill="#efc45e", outline="#f7e4a3", width=12)
    draw.line((55, 620, 965, 540), fill="#d5e8e6", width=18)
    image.save(path, optimize=True)
    return path


def create_transfer_corpus_fixtures(root: Path) -> dict[str, Path]:
    """Create heterogeneous deterministic sources unrelated to the sealed case."""

    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "square_illustration": root / "square-illustration.png",
        "portrait_source": root / "portrait-source.png",
        "alpha_source": root / "alpha-source.png",
        "low_resolution": root / "low-resolution.png",
        "complex_texture": root / "complex-texture.png",
        "inapplicable_composition": root / "independent-panels.png",
    }
    _square_illustration(paths["square_illustration"])
    _portrait_source(paths["portrait_source"])
    _alpha_source(paths["alpha_source"])
    _low_resolution(paths["low_resolution"])
    _complex_texture(paths["complex_texture"])
    _independent_panels(paths["inapplicable_composition"])
    return paths


def _square_illustration(path: Path) -> None:
    random = Random(0xA0A13)
    image = Image.new("RGB", (768, 768), "#10182d")
    draw = ImageDraw.Draw(image)
    for radius in range(350, 20, -18):
        hue = radius / 350
        ring_color = (
            round(35 + 155 * hue),
            round(55 + 70 * (1 - hue)),
            round(95 + 95 * hue),
        )
        draw.ellipse(
            (384 - radius, 384 - radius, 384 + radius, 384 + radius),
            outline=ring_color,
            width=9,
        )
    for index in range(180):
        x = random.randrange(24, 744)
        y = random.randrange(24, 744)
        size = random.randrange(3, 16)
        accent_color = ("#f4c95d", "#68d8d6", "#ee6c9b", "#f2f4f3")[index % 4]
        draw.regular_polygon((x, y, size), n_sides=3 + index % 5, fill=accent_color)
    draw.polygon([(95, 620), (382, 88), (690, 620)], outline="#ffffff", width=15)
    image.save(path, optimize=True)


def _portrait_source(path: Path) -> None:
    random = Random(0xA0A14)
    image = Image.new("RGB", (640, 960), "#241b35")
    draw = ImageDraw.Draw(image)
    for y in range(960):
        draw.line(
            (0, y, 640, y),
            fill=(36 + y // 24, 27 + y // 32, 53 + y // 18),
        )
    draw.rounded_rectangle((118, 90, 522, 870), radius=170, fill="#d48372")
    draw.ellipse((182, 185, 458, 540), fill="#f3c8a8", outline="#5b2a3c", width=13)
    draw.arc((210, 292, 430, 510), start=15, end=165, fill="#6a3145", width=17)
    for _ in range(650):
        x = random.randrange(26, 614)
        y = random.randrange(36, 924)
        radius = random.randrange(1, 4)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=random.choice(("#f6d365", "#83e8ba", "#b5a4e3", "#f3f0e8")),
        )
    image.save(path, optimize=True)


def _alpha_source(path: Path) -> None:
    random = Random(0xA0A15)
    image = Image.new("RGBA", (768, 768), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (65, 80, 703, 688),
        radius=120,
        fill=(30, 74, 104, 218),
        outline=(244, 204, 96, 255),
        width=18,
    )
    for index in range(420):
        x = random.randrange(90, 678)
        y = random.randrange(105, 663)
        radius = random.randrange(2, 9)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(
                80 + index % 150,
                110 + index % 100,
                170 + index % 80,
                100 + index % 156,
            ),
        )
    draw.polygon(
        [(120, 560), (315, 155), (650, 590)],
        fill=(232, 104, 135, 225),
        outline=(255, 255, 255, 255),
    )
    image.save(path, optimize=True)


def _low_resolution(path: Path) -> None:
    image = Image.new("RGB", (48, 48), "#15253c")
    draw = ImageDraw.Draw(image)
    for index in range(0, 48, 4):
        draw.line((0, index, 47, 47 - index), fill="#e8bc4f", width=2)
        draw.line((index, 0, 47 - index, 47), fill="#70d6c5", width=1)
    draw.ellipse((11, 11, 37, 37), outline="#f5f3ef", width=3)
    image.save(path, optimize=True)


def _complex_texture(path: Path) -> None:
    random = Random(0xA0A16)
    width, height = 1024, 768
    image = Image.new("RGB", (width, height), "#17253a")
    draw = ImageDraw.Draw(image, "RGBA")
    for y in range(height):
        draw.line(
            (0, y, width, y),
            fill=(20 + y // 20, 38 + y // 18, 62 + y // 14, 255),
        )
    for _ in range(2600):
        x = random.randrange(8, width - 8)
        y = random.randrange(8, height - 8)
        radius = random.randrange(2, 8)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=random.choice(
                (
                    (240, 188, 75, 230),
                    (77, 206, 190, 230),
                    (222, 96, 142, 230),
                    (226, 234, 240, 230),
                )
            ),
        )
    for index in range(90):
        inset = 8 + index * 4
        draw.rectangle(
            (inset, inset, width - inset, height - inset),
            outline=(255, 255 - index * 2, 80 + index, 105),
            width=3,
        )
    for x in range(0, width, 32):
        draw.line((x, 0, width - x, height), fill=(40, 210, 190, 145), width=4)
    image.save(path, optimize=True)


def _independent_panels(path: Path) -> None:
    image = Image.new("RGB", (960, 640), "#0e1526")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((35, 45, 455, 595), radius=42, fill="#bb4967")
    draw.rounded_rectangle((505, 45, 925, 595), radius=42, fill="#397d84")
    for index in range(18):
        draw.ellipse(
            (65 + index * 18, 95 + index * 20, 115 + index * 18, 145 + index * 20),
            fill="#f5d36c",
        )
        draw.rectangle(
            (535 + index * 17, 510 - index * 20, 580 + index * 17, 555 - index * 20),
            fill="#d9f0ee",
        )
    draw.text((155, 305), "LAYER A", fill="#ffffff")
    draw.text((625, 305), "LAYER B", fill="#ffffff")
    draw.line((455, 320, 505, 320), fill="#ffffff", width=8)
    image.save(path, optimize=True)


def _still(path: Path) -> None:
    image = Image.new("RGB", (480, 270), "#081a2e")
    draw = ImageDraw.Draw(image)
    draw.ellipse((80, 25, 350, 260), fill="#2d6a8a")
    draw.ellipse((150, 55, 410, 240), fill="#f3a712")
    draw.polygon([(235, 25), (120, 240), (400, 225)], fill="#f4f1de")
    draw.ellipse((205, 90, 315, 200), fill="#7b2cbf")
    image.save(path, optimize=True)


def _speech(path: Path) -> None:
    audio = (
        "aevalsrc=if(between(t\\,1.1\\,2.0)\\,0\\,"
        "0.16*sin(2*PI*(220+40*sin(2*PI*t))*t)):s=48000:d=3.2"
    )
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=480x270:rate=30:duration=3.2",
        "-f",
        "lavfi",
        "-i",
        audio,
        "-shortest",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(path),
    ]
    _run(command)


def _memory(path: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=0x203a43:s=480x270:r=30:d=1.1",
        "-f",
        "lavfi",
        "-i",
        "color=c=0xf2a65a:s=480x270:r=30:d=1.2",
        "-f",
        "lavfi",
        "-i",
        "color=c=0x5b3758:s=480x270:r=30:d=1.3",
        "-filter_complex",
        "[0:v][1:v][2:v]concat=n=3:v=1:a=0,format=yuv420p[v]",
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        str(path),
    ]
    _run(command)


def _run(command: list[str]) -> None:
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise FixtureError(result.stderr[-3000:])
