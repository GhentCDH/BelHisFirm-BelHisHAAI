""" Table processing as five sub-stages that each read and write a record's "tables" folder,
so any of them can be skipped or run again on its own (same split as BelHisFirm-TaPro):

    crop        tables/_crops/<stem>.png + crops.json     table (+ caption) crops from the page layout
    transcribe  tables/_transcriptions/<stem>.txt         GLM-OCR plain text per crop
    structure   tables/<stem>.json (+ <stem>.png)         schema-constrained JSON via vLLM
    parse       tables/<stem>.json, in place              gender, address and share count rules
    excel       tables/<stem>.xlsx + <stem>.csv           flattened workbook and CSV per table

A table that could not be structured ends up under tables/FAILED. The transcribe and structure
sub-stages skip tables they already have an output for.
"""

import json
import shutil
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from logging import getLogger
from pathlib import Path
from PIL import Image

from src.processors.record_processor.data import Record, TableConfig

from .table_cropper import TableCropper
from .vllm_client import make_client, structure_transcription

logger = getLogger(__name__)

TABLE_OCR_PROMPT = "Transcribe all text in this table image, preserving line breaks between rows."


class TablePipeline:

    def __init__(self, table_config: TableConfig, glm_ocr):
        """ Args: glm_ocr (GLMOCREngine): GLM-OCR engine used to transcribe the table crops. """

        self.table_config = table_config
        self.glm_ocr = glm_ocr
        self.table_cropper = TableCropper(table_config)
        self._client = None

    # -- folders and files of one record --

    @staticmethod
    def tables_dir(record_folder: Path) -> Path:
        return record_folder / "tables"

    @staticmethod
    def crops_dir(record_folder: Path) -> Path:
        return record_folder / "tables" / "_crops"

    @staticmethod
    def transcriptions_dir(record_folder: Path) -> Path:
        return record_folder / "tables" / "_transcriptions"

    @staticmethod
    def failed_dir(record_folder: Path) -> Path:
        return record_folder / "tables" / "FAILED"

    @classmethod
    def load_crops(cls, record_folder: Path) -> list[dict]:
        """ The crops written by the crop sub-stage: 'stem', 'page' (1-based) and 'bbox' per table. """

        crops_path = cls.crops_dir(record_folder) / "crops.json"
        if not crops_path.exists():
            return []
        return json.loads(crops_path.read_text(encoding="utf-8"))

    @classmethod
    def table_json_paths(cls, record_folder: Path) -> list[Path]:
        return sorted(cls.tables_dir(record_folder).glob("*.json"))

    # -- sub-stages --

    def crop(self, record: Record, record_folder: Path, layout: list[list[dict]]) -> int:
        """ Crops every table (extended with a caption directly above it) from the pages of a record.

            Returns: Number of tables found.
        """

        crops, images = [], {}
        for idx, image in enumerate(record.images):
            regions = layout[idx]
            table_regions = [r for r in regions if r["label"] == self.table_config.table_class_name]
            caption_regions = [r for r in regions if r["label"] == self.table_config.caption_class_name]

            for n, crop in enumerate(self.table_cropper.build_crops(image, table_regions, caption_regions)):
                stem = f"page_{idx + 1:03d}_table_{n:02d}"
                crops.append({"stem": stem, "page": idx + 1, "bbox": [int(c) for c in crop["bbox"]]})
                images[stem] = crop["image"]

        # Other tables than last time means the pages changed, so nothing in the folder applies anymore
        crops_path = self.crops_dir(record_folder) / "crops.json"
        if crops_path.exists() and crops != self.load_crops(record_folder):
            shutil.rmtree(self.tables_dir(record_folder), ignore_errors=True)

        if crops:
            crops_dir = self.crops_dir(record_folder)
            crops_dir.mkdir(parents=True, exist_ok=True)
            for stem, image in images.items():
                image.save(crops_dir / f"{stem}.png")
            (crops_dir / "crops.json").write_text(json.dumps(crops, indent=2), encoding="utf-8")

        self.save_index(record_folder)
        return len(crops)

    def transcribe(self, record_folders: list[Path]) -> int:
        """ Transcribes every crop without a transcription yet via GLM-OCR.

            Returns: Number of crops transcribed.
        """

        todo, total = [], 0
        for record_folder in record_folders:
            for crop in self.load_crops(record_folder):
                total += 1
                text_path = self.transcriptions_dir(record_folder) / f"{crop['stem']}.txt"
                if not text_path.exists():
                    todo.append((record_folder, crop, text_path))

        logger.info(f"{len(todo)} table(s) to transcribe, {total - len(todo)} already done.")

        for number, (record_folder, crop, text_path) in enumerate(todo, start=1):
            start = time.monotonic()
            image = Image.open(self.crops_dir(record_folder) / f"{crop['stem']}.png").convert("RGB")
            text = self.glm_ocr.ocr(image, prompt=TABLE_OCR_PROMPT)

            text_path.parent.mkdir(parents=True, exist_ok=True)
            text_path.write_text(text + "\n", encoding="utf-8")
            logger.info(
                f"[{number}/{len(todo)}] Transcribed {self._name(record_folder, crop)} "
                f"({len(text)} characters, {time.monotonic() - start:.1f}s)"
            )

        return len(todo)

    def structure(self, record_folders: list[Path]) -> tuple[int, int]:
        """ Structures every transcription without a JSON yet via vLLM, several at once.

            Returns: Number of tables structured, and number of tables that failed.
        """

        todo = []
        failed = 0
        already_done = 0
        for record_folder in record_folders:
            for crop in self.load_crops(record_folder):
                json_path = self.tables_dir(record_folder) / f"{crop['stem']}.json"
                if json_path.exists():
                    structured = json.loads(json_path.read_text(encoding="utf-8")).get("structured") or {}
                    self._place_image_copy(record_folder, crop, structured)
                    already_done += 1
                    continue

                text_path = self.transcriptions_dir(record_folder) / f"{crop['stem']}.txt"
                if not text_path.exists():
                    self._mark_failed(record_folder, crop, "No transcription found - run the OCR TRANSCRIPTION sub-stage first.")
                    logger.error(f"No transcription found for {self._name(record_folder, crop)}, run the OCR TRANSCRIPTION sub-stage first.")
                    failed += 1
                    continue

                todo.append((record_folder, crop, text_path.read_text(encoding="utf-8").strip()))

        if todo and self._client is None:
            self._client = make_client(self.table_config.vllm_base_url)

        logger.info(
            f"{len(todo)} table(s) to structure, {already_done} already done. "
            f"(Up to {max(1, self.table_config.structuring_concurrency)} at once)"
        )

        structured_count = 0
        finished = 0
        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=max(1, self.table_config.structuring_concurrency)) as pool:
            futures = {
                pool.submit(
                    structure_transcription, self._client, self.table_config.vllm_model, plain_text,
                    f"{record_folder.name}/{crop['stem']}",
                ): (record_folder, crop, plain_text)
                for record_folder, crop, plain_text in todo
            }
            for future in as_completed(futures):
                record_folder, crop, plain_text = futures[future]
                finished += 1
                try:
                    structured = future.result()
                except Exception as e:
                    raw_content = getattr(e, "raw_content", None)
                    logger.error(
                        f"[{finished}/{len(todo)}] Structuring failed for table {crop['stem']} of record {record_folder} "
                        f"(model {self.table_config.vllm_model} at {self.table_config.vllm_base_url}): {type(e).__name__}: {e}"
                        + (f"\nRaw model output:\n{raw_content}" if raw_content else ""),
                        exc_info=e,
                    )
                    self._mark_failed(record_folder, crop, f"{type(e).__name__}: {e}", getattr(e, "raw_content", None))
                    failed += 1
                    continue

                structured.pop("plain_text", None)  # stored next to it instead
                json_path = self.tables_dir(record_folder) / f"{crop['stem']}.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "page": crop["page"],
                        "bbox": crop["bbox"],
                        "plain_text": plain_text,
                        "structured": structured,
                    }, f, ensure_ascii=False, indent=2)

                self._place_image_copy(record_folder, crop, structured)
                self._clear_failed(record_folder, crop)
                structured_count += 1
                kind = "shareholder register" if structured.get("is_shareholder_register") else "other table"
                logger.info(
                    f"[{finished}/{len(todo)}] Structured {self._name(record_folder, crop)} "
                    f"({kind}, {time.monotonic() - start:.0f}s since start)"
                )

        for record_folder in record_folders:
            self.save_index(record_folder)

        return structured_count, failed

    def parse(self, record_folders: list[Path]) -> int:
        """ Applies the text and numerical rules to every structured shareholder register, in place.

            Returns: Number of tables the rules were applied to.
        """

        # Imported here, the rules pull in spaCy and text2num which the other sub-stages do not need
        from .parsing import numerical_rules, text_rules

        json_paths = [path for record_folder in record_folders for path in self.table_json_paths(record_folder)]
        logger.info(f"{len(json_paths)} structured table(s) to check for rule parsing.")

        parsed = 0
        for number, json_path in enumerate(json_paths, start=1):
            name = f"{json_path.parent.parent.name}/{json_path.stem}"
            data = json.loads(json_path.read_text(encoding="utf-8"))
            table = data.get("structured") or {}
            if not table.get("is_shareholder_register"):
                logger.info(f"[{number}/{len(json_paths)}] Skipped {name} (not a shareholder register)")
                continue

            start = time.monotonic()
            try:
                text_rules.apply_rules(table)
                numerical_rules.apply_rules(table)
            except Exception as e:
                logger.error(
                    f"[{number}/{len(json_paths)}] Rule parsing failed for {json_path}, the table is left unparsed: "
                    f"{type(e).__name__}: {e}", exc_info=True,
                )
                continue

            json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            entries = len((table.get("shareholder_register") or {}).get("entries") or [])
            logger.info(f"[{number}/{len(json_paths)}] Parsed {name} ({entries} entries, {time.monotonic() - start:.1f}s)")
            parsed += 1

        return parsed

    def excel(self, record_folders: list[Path]) -> int:
        """ Writes an .xlsx and a flat .csv next to every structured table JSON.

            Returns: Number of tables exported.
        """

        from .parsing.excel_export import convert_table

        json_paths = [path for record_folder in record_folders for path in self.table_json_paths(record_folder)]
        logger.info(f"{len(json_paths)} structured table(s) to export.")

        exported = 0
        for number, json_path in enumerate(json_paths, start=1):
            table = json.loads(json_path.read_text(encoding="utf-8")).get("structured") or {}
            try:
                convert_table(table, json_path)
            except Exception as e:
                logger.error(f"[{number}/{len(json_paths)}] Excel export failed for {json_path}: {type(e).__name__}: {e}", exc_info=True)
                continue
            logger.info(f"[{number}/{len(json_paths)}] Exported {json_path.parent.parent.name}/{json_path.stem} (.xlsx + .csv)")
            exported += 1

        return exported

    # -- helpers --

    @staticmethod
    def _name(record_folder: Path, crop: dict) -> str:
        return f"{record_folder.name}/{crop['stem']}"

    def _place_image_copy(self, record_folder: Path, crop: dict, structured: dict) -> None:
        """ Puts the crop next to its JSON, with focus_shareholders only for shareholder registers. """

        image_path = self.tables_dir(record_folder) / f"{crop['stem']}.png"
        if self.table_config.focus_shareholders and not structured.get("is_shareholder_register"):
            image_path.unlink(missing_ok=True)
        elif not image_path.exists():
            shutil.copy2(self.crops_dir(record_folder) / f"{crop['stem']}.png", image_path)

    def _mark_failed(self, record_folder: Path, crop: dict, error: str, raw_content: str | None = None) -> None:
        failed_dir = self.failed_dir(record_folder)
        failed_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.crops_dir(record_folder) / f"{crop['stem']}.png", failed_dir / f"{crop['stem']}.png")
        (failed_dir / f"{crop['stem']}.txt").write_text(error, encoding="utf-8")
        if raw_content:
            (failed_dir / f"{crop['stem']}.raw.txt").write_text(raw_content, encoding="utf-8")

    def _clear_failed(self, record_folder: Path, crop: dict) -> None:
        for suffix in (".png", ".txt", ".raw.txt"):
            (self.failed_dir(record_folder) / f"{crop['stem']}{suffix}").unlink(missing_ok=True)

    @classmethod
    def save_index(cls, record_folder: Path) -> None:
        """ Writes tables_index.json: one pointer entry per table, as stored under "tables" in
            ocr_data.json and used for the CSV counts. Built from what is on disk, so it is
            right whichever sub-stages ran. """

        index = []
        for crop in cls.load_crops(record_folder):
            stem = crop["stem"]
            json_path = cls.tables_dir(record_folder) / f"{stem}.json"
            error_path = cls.failed_dir(record_folder) / f"{stem}.txt"

            error = None
            crop_file = f"tables/_crops/{stem}.png"
            if json_path.exists():
                if (cls.tables_dir(record_folder) / f"{stem}.png").exists():
                    crop_file = f"tables/{stem}.png"
            elif error_path.exists():
                error = error_path.read_text(encoding="utf-8")
                crop_file = f"tables/FAILED/{stem}.png"

            index.append({
                "page": crop["page"],
                "bbox": crop["bbox"],
                "crop_file": crop_file,
                "json_file": f"tables/{stem}.json" if json_path.exists() else None,
                "error": error,
            })

        with open(record_folder / "tables_index.json", 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
