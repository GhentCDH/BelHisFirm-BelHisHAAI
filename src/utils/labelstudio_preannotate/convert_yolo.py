"""Convert YOLO layout regions on one page into a Label Studio prediction,
importable as a pre-annotation for a RectangleLabels-based labeling config.
"""

from __future__ import annotations


def regions_to_prediction(
    regions: list[dict],
    width: int,
    height: int,
    model_version: str,
    from_name: str = "label",
    to_name: str = "image",
) -> dict:
    """Build a single Label Studio prediction dict for the REGIONS of one page.

    REGIONS are dicts with 'bbox' ([x0, y0, x1, y1] in pixels), 'label' and
    'confidence' keys, as produced by VisionAnalyzer.detect_layout. Label
    Studio wants rectangles as percentages of the image size.
    """
    result = []
    for i, region in enumerate(regions):
        x0, y0, x1, y1 = region["bbox"]
        result.append(
            {
                "id": f"r{i}",
                "type": "rectanglelabels",
                "from_name": from_name,
                "to_name": to_name,
                "original_width": width,
                "original_height": height,
                "image_rotation": 0,
                "score": region["confidence"],
                "value": {
                    "x": x0 / width * 100,
                    "y": y0 / height * 100,
                    "width": (x1 - x0) / width * 100,
                    "height": (y1 - y0) / height * 100,
                    "rotation": 0,
                    "rectanglelabels": [region["label"]],
                },
            }
        )

    score = sum(region["confidence"] for region in regions) / len(regions) if regions else 0.0
    return {"model_version": model_version, "score": score, "result": result}
