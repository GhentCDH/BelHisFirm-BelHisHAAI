import argparse
import logging
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path

from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

TIFF_EXTENSIONS = {".tif", ".tiff"}
JPEG_QUALITY = 85

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _ns_tag(namespace: str, tag: str) -> str:
    return f"{{{namespace}}}{tag}" if namespace else tag


def parse_alto(alto_path: Path) -> tuple[float, float, list[dict]]:
    tree = ET.parse(alto_path)
    root = tree.getroot()

    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag[1 : root.tag.index("}")]

    page_el = root.find(f".//{_ns_tag(ns, 'Page')}")
    if page_el is None:
        logger.warning(f"No <Page> element found in {alto_path}")
        return 0.0, 0.0, []

    page_w = float(page_el.get("WIDTH", 0))
    page_h = float(page_el.get("HEIGHT", 0))

    strings = []
    for string_el in root.iter(_ns_tag(ns, "String")):
        content = string_el.get("CONTENT", "")
        if not content.strip():
            continue
        strings.append(
            {
                "content": content,
                "hpos": float(string_el.get("HPOS", 0)),
                "vpos": float(string_el.get("VPOS", 0)),
                "width": float(string_el.get("WIDTH", 0)),
                "height": float(string_el.get("HEIGHT", 0)),
            }
        )

    return page_w, page_h, strings


def tiff_to_jpeg_bytes(tiff_path: Path) -> tuple[bytes, int, int, float]:
    with Image.open(tiff_path) as img:
        dpi_info = img.info.get("dpi", (300.0, 300.0))
        dpi = float(dpi_info[0]) if isinstance(dpi_info, (tuple, list)) else float(dpi_info)
        if dpi <= 0:
            dpi = 300.0

        if img.mode in {"RGBA", "LA"}:
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img.convert("RGB"), mask=img.split()[-1])
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size
        buf = BytesIO()
        img.save(buf, "JPEG", quality=JPEG_QUALITY)
        return buf.getvalue(), w, h, dpi


def add_page(c: canvas.Canvas, tif_path: Path, alto_path: Path | None) -> None:
    logger.info(f"  Processing page: {tif_path.name}")
    jpeg_bytes, img_w, img_h, dpi = tiff_to_jpeg_bytes(tif_path)

    page_w_pts = img_w * 72.0 / dpi
    page_h_pts = img_h * 72.0 / dpi

    c.setPageSize((page_w_pts, page_h_pts))
    c.drawImage(ImageReader(BytesIO(jpeg_bytes)), 0, 0, width=page_w_pts, height=page_h_pts)

    if alto_path and alto_path.is_file():
        alto_w, alto_h, strings = parse_alto(alto_path)

        # Fall back to image dimensions if ALTO has no page size
        if alto_w == 0:
            alto_w = img_w
        if alto_h == 0:
            alto_h = img_h

        scale_x = page_w_pts / alto_w
        scale_y = page_h_pts / alto_h

        for s in strings:
            font_size = max(s["height"] * scale_y, 1.0)
            x = s["hpos"] * scale_x
            # PDF Y-axis is bottom-up; ALTO is top-down
            y = page_h_pts - (s["vpos"] + s["height"]) * scale_y

            text_obj = c.beginText(x, y)
            text_obj.setFont("Helvetica", font_size)
            text_obj.setTextRenderMode(3)  # invisible

            text_width = c.stringWidth(s["content"], "Helvetica", font_size)
            if text_width > 0 and s["width"] > 0:
                h_scale = (s["width"] * scale_x / text_width) * 100.0
                text_obj.setHorizScale(h_scale)

            text_obj.textOut(s["content"])
            c.drawText(text_obj)
    elif alto_path:
        logger.warning(f"  ALTO file not found: {alto_path}")

    c.showPage()


def build_pdf(tif_dir: Path, alto_dir: Path, output_path: Path) -> bool:
    tif_files = sorted(
        f for f in tif_dir.iterdir() if f.suffix.lower() in TIFF_EXTENSIONS and f.is_file()
    )

    if not tif_files:
        logger.warning(f"No TIFF files found in {tif_dir}, skipping")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_path))

    for tif_path in tif_files:
        alto_path = alto_dir / tif_path.with_suffix(".xml").name
        add_page(c, tif_path, alto_path)

    c.save()
    logger.info(f"Saved: {output_path}")
    return True


def process_year(tif_year_dir: Path, alto_year_dir: Path, pdf_year_dir: Path) -> None:
    # Collect all directories that directly contain TIFF files
    tiff_dirs: set[Path] = set()
    for f in tif_year_dir.rglob("*"):
        if f.suffix.lower() in TIFF_EXTENSIONS and f.is_file():
            tiff_dirs.add(f.parent)

    if not tiff_dirs:
        logger.warning(f"No TIFF files found anywhere under {tif_year_dir}")
        return

    for tif_dir in sorted(tiff_dirs):
        rel = tif_dir.relative_to(tif_year_dir)
        alto_dir = alto_year_dir / rel

        if rel == Path("."):
            # TIFFs sit directly in the year folder
            pdf_path = pdf_year_dir.parent / f"{pdf_year_dir.name}.pdf"
        else:
            pdf_path = pdf_year_dir / rel.parent / f"{rel.name}.pdf"

        logger.info(f"Processing folder: {tif_dir}")
        build_pdf(tif_dir, alto_dir, pdf_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Combine ALTO XML and TIFF files into searchable PDFs. "
            "One PDF is created per leaf folder under TIF/<year>/."
        ),
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Root directory containing TIF and ALTO subfolders (e.g. .../EHC_B665_O/20251228)",
    )
    parser.add_argument(
        "year",
        help="Year subfolder to process (e.g. 1920)",
    )
    parser.add_argument(
        "--tif-subdir",
        default="TIF",
        help="Name of the TIF folder under root (default: TIF)",
    )
    parser.add_argument(
        "--alto-subdir",
        default="ALTO",
        help="Name of the ALTO folder under root (default: ALTO)",
    )
    parser.add_argument(
        "--pdf-subdir",
        default="PDF",
        help="Name of the output PDF folder under root (default: PDF)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        metavar="1-95",
        help="JPEG compression quality for embedded images (default: 85)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    global JPEG_QUALITY
    JPEG_QUALITY = args.jpeg_quality

    root = args.root.resolve()
    tif_year_dir = root / args.tif_subdir / args.year
    alto_year_dir = root / args.alto_subdir / args.year
    pdf_year_dir = root / args.pdf_subdir / args.year

    if not tif_year_dir.is_dir():
        logger.error(f"TIF directory not found: {tif_year_dir}")
        return 1

    if not alto_year_dir.is_dir():
        logger.error(f"ALTO directory not found: {alto_year_dir}")
        return 1

    process_year(tif_year_dir, alto_year_dir, pdf_year_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
