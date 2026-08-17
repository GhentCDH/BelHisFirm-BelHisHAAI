import re
import difflib
import logging
import unicodedata
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from surya.detection import DetectionPredictor
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.common.surya.schema import TaskNames

_log = logging.getLogger("index_parser")

_TOP_BAND_RATIO = 0.20

# Tried in order: the first tier with any match in the search window wins (its earliest
# matching page becomes the start). Some volumes bundle quarterly tables followed by one
# comprehensive annual table ("table annuelle") — the annual one is the "yearly" index we
# actually want, so its markers are tried first. Volumes with only a single table per
# volume (no quarterly/annual split) fall through to the generic tier.
#
# "table methodique des matieres des tomes" (1912-era wording for the annual/cumulative
# index, e.g. "TABLE MÉTHODIQUE DES MATIÈRES DES TOMES XCII à XCV") is deliberately in
# tier 1, not the plain "table methodique des matieres" fallback below — that fallback
# matches the SAME divider page too, but only as a lower-priority tier, so if a distinct
# tier-1 anchor (e.g. the running "recueil...societes commerciales" header) also matches
# a page or two later in the same volume, tier 1 would win and skip past the true divider.
# Keeping the more specific "...des tomes" phrasing in tier 1 avoids that.
_DEFAULT_ANCHOR_TIERS = [
    [
        "table annuelle",
        "table methodique des matieres des tomes",
        "recueil special des actes et documents relatifs aux societes commerciales",
        "receuil special des actes et documents relatifs aux societes commerciales",
    ],
    [
        "table du recueil special des actes et documents relatifs aux societes",
        "table methodique des matieres",
        "table alfabetique des matieres",
    ],
]
_FUZZY_THRESHOLD = 0.90
_MIN_TEXT_BOXES = 30

# A page whose "table annuelle"-style header also mentions a quarter/trimester (e.g.
# "ANNEXE AU MONITEUR BELGE. — TABLE ANNUELLE. — 1er TRIM. 1896.") is a running header
# WITHIN the annual table showing which quarter's pages follow, not the true start of
# the comprehensive index — that start is a clean title/divider page with no quarter
# mentioned at all. Excluded from yearly-tier matches so locate() keeps searching past
# these instead of anchoring on the first one it sees.
_QUARTER_EXCLUDE_TERMS = ["trimestre", "trim"]


def _page_num(image_path):
    match = re.search(r'_(\d+)\.tif$', image_path.name)
    return int(match.group(1)) if match else None


def _normalize(text):
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _matches_anchor(lines, anchors, exclude_words=None):
    """Matches anchor phrases against individual OCR lines rather than the whole
    header-band blob — comparing a short anchor against a long multi-line blob makes
    difflib's ratio() unreliable (unrelated running headers can score deceptively high).

    Requires every word of the anchor to literally appear in the line before falling
    back to fuzzy matching. These anchors are often near-identical to a much more
    common plain running header, differing only by one distinguishing word — e.g.
    "TABLE DU <title>" vs just "<title>", or "<title> COMMERCIALES" vs "<title>" — so a
    plain similarity ratio over the whole phrase would treat them as a near-match and
    miss the one word that actually matters.

    exclude_words, if given, disqualifies a line from matching at all if it contains
    any of those words literally — used to reject a quarterly sub-header that would
    otherwise look like a match (see _QUARTER_EXCLUDE_TERMS).
    """
    anchors_norm = [_normalize(a) for a in anchors]
    anchors_norm = [a for a in anchors_norm if a]
    exclude_norm = {_normalize(w) for w in exclude_words} if exclude_words else set()
    for line in lines:
        norm = _normalize(line)
        if len(norm) < 15:
            continue
        words = set(norm.split())
        if exclude_norm and words & exclude_norm:
            continue
        for anchor_norm in anchors_norm:
            anchor_words = anchor_norm.split()
            if not anchor_words or not all(w in words for w in anchor_words):
                continue
            if anchor_norm in norm:
                return True
            if difflib.SequenceMatcher(None, norm, anchor_norm).ratio() >= _FUZZY_THRESHOLD:
                return True
    return False


class IndexLocator:
    """Guesses the page range of the index/table section in a scanned volume with a
    quick Surya OCR pass, instead of requiring a human to page through the scans."""

    def __init__(self, device="cuda"):
        self.det_predictor = DetectionPredictor(device=device)
        self.foundation_predictor = FoundationPredictor(device=device)
        self.rec_predictor = RecognitionPredictor(self.foundation_predictor)

    def _top_bands(self, image_paths):
        bands = []
        for path in image_paths:
            with Image.open(path) as img:
                w, h = img.size
                band = img.crop((0, 0, w, int(h * _TOP_BAND_RATIO))).convert("RGB")
                bands.append(band.copy())
        return bands

    def _headers_lines(self, image_paths):
        if not image_paths:
            return []
        bands = self._top_bands(image_paths)
        task_names = [TaskNames.ocr_with_boxes] * len(bands)
        results = self.rec_predictor(bands, task_names=task_names, det_predictor=self.det_predictor)
        return [[line.text for line in res.text_lines] for res in results]

    def _page_has_text(self, image_path):
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
        preds = self.det_predictor([rgb])
        return bool(preds) and len(preds[0].bboxes) >= _MIN_TEXT_BOXES

    def locate(self, folder, anchors=None, search_window=100, from_front=False):
        """Returns (start_page, end_page, kind) of the detected index section within
        the last/first search_window pages, or None if no anchor match was found
        there. kind is "yearly" (tier 1: a comprehensive annual/"table annuelle"-style
        index), "regular" (tier 2: the generic single-table fallback), or "custom"
        (a caller-supplied anchors list, which has no tiering).

        anchors, if given, is used as a single flat tier (all OR'd together, no
        fallback). Otherwise the built-in priority tiers are tried in order within
        that same window — the first tier with any match wins, using its earliest
        match (yearly-specific markers are tried before the generic single-table
        fallback)."""
        if anchors:
            anchor_tiers = [(anchors, "custom")]
        else:
            anchor_tiers = [(_DEFAULT_ANCHOR_TIERS[0], "yearly"), (_DEFAULT_ANCHOR_TIERS[1], "regular")]
        tif_paths = sorted(Path(folder).rglob("*.tif"), key=lambda p: _page_num(p) or 0)
        if not tif_paths:
            return None

        window = tif_paths[:search_window] if from_front else tif_paths[-search_window:]

        batch_size = 20
        all_headers = []
        for i in tqdm(range(0, len(window), batch_size), desc="Scannen naar index-header", unit="batch"):
            all_headers.extend(self._headers_lines(window[i:i + batch_size]))

        start_pos_in_window = None
        kind = None
        for tier, tier_kind in anchor_tiers:
            exclude = _QUARTER_EXCLUDE_TERMS if tier_kind == "yearly" else None
            start_pos_in_window = next(
                (i for i, lines in enumerate(all_headers) if _matches_anchor(lines, tier, exclude_words=exclude)), None
            )
            if start_pos_in_window is not None:
                kind = tier_kind
                break
        if start_pos_in_window is None:
            return None

        start_pos = tif_paths.index(window[start_pos_in_window])

        end_pos = start_pos
        for pos in tqdm(range(start_pos, len(tif_paths)), desc="Einde van index zoeken", unit="pagina"):
            if self._page_has_text(tif_paths[pos]):
                end_pos = pos
            else:
                break

        return _page_num(tif_paths[start_pos]), _page_num(tif_paths[end_pos]), kind
