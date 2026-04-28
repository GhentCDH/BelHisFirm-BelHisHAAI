import sys
import json
import re
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

_INDEX_PARSER_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_INDEX_PARSER_DIR))

from processing.text_extraction2 import TextExtractor2
from processing.OCR2 import OCR
from CRF.utils.convert_to_features import Convert_To_Features

_DELIMITERS = {".", ",", ";", ":", "!", "?", "(", ")", "°", "/", "&", '"', "—", "-"}
_VALID_LABELS = {"ID", "N", "AD", "EV", "EX"}
_DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

_SYSTEM_PROMPT = """\
You are an expert annotator for a historical Belgian firms index (19th–early 20th century).
Each entry records a firm's record number, name, location, and a business event.

Label definitions:
- ID: Record identifier (always a number at the very start of an entry)
- N: Firm or company name. "Id" or "Id." means ibidem (same firm as the previous entry) and must be labeled N
- AD: Address or geographic location
- EV: Business event or action (e.g. Constitution, Dissolution, Stichting, Ontbinding)
- EX: Extra information that does not fit the other categories

You receive the full entry text as context and a JSON list of tokens to label.
Delimiters and punctuation are already handled — label only the provided tokens.

Reply ONLY with valid JSON on a single line: {"labels": ["LABEL1", "LABEL2", ...]}
One label per token, same order, using only: ID, N, AD, EV, EX\
"""


def _build_user_prompt(tokens_to_label: list[str], all_tokens: list[str]) -> str:
    context = " ".join(t for t in all_tokens if t not in ("START", "END"))
    numbered = "\n".join(f"{i + 1}: {t}" for i, t in enumerate(tokens_to_label))
    n = len(tokens_to_label)
    return (
        f"Entry: {context}\n\n"
        f"Tokens to label ({n} tokens, one label per line number):\n{numbered}\n\n"
        f"Respond with exactly {n} labels in order."
    )


def _parse_label_response(text: str, expected: int) -> list[str] | None:
    matches = list(re.finditer(r'\{[^{}]*"labels"\s*:\s*(\[[^\]]*\])[^{}]*\}', text, re.DOTALL))
    if not matches:
        return None
    try:
        labels = json.loads(matches[-1].group(1))
    except json.JSONDecodeError:
        return None
    if len(labels) != expected or not all(lbl in _VALID_LABELS for lbl in labels):
        return None
    return labels


_BATCH_SIZE = 8


_BATCH_SIZE = 8


def annotate_with_llm(
    df: pd.DataFrame,
    model_id: str = _DEFAULT_MODEL,
    log_path: Path | None = None,
) -> pd.DataFrame:
    """Fill None keys in df by labeling tokens with a locally loaded Qwen model."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto",
            attn_implementation="flash_attention_2",
        )
        print("  Using Flash Attention 2")
    except (ImportError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto",
        )
    model.eval()

    # Collect all entries that need labeling
    entries = []
    for sid in df["id"].unique():
        sent = df[df["id"] == sid]
        unlabeled = sent["key"].isna()
        if not unlabeled.any():
            continue
        entries.append((sid, sent, unlabeled))

    df = df.copy()
    failed_entries: list[dict] = []

    for batch_start in tqdm(range(0, len(entries), _BATCH_SIZE), desc="LLM annotation"):
        batch = entries[batch_start : batch_start + _BATCH_SIZE]

        prompts = []
        for sid, sent, unlabeled in batch:
            tokens_to_label = sent.loc[unlabeled, "value"].tolist()
            all_tokens = sent["value"].tolist()
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(tokens_to_label, all_tokens)},
            ]
            prompts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            ))

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        for j, (sid, sent, unlabeled) in enumerate(batch):
            response = tokenizer.decode(out[j][input_len:], skip_special_tokens=True)
            tokens_to_label = sent.loc[unlabeled, "value"].tolist()
            all_tokens = sent["value"].tolist()
            line = " ".join(t for t in all_tokens if t not in ("START", "END"))

            labels = _parse_label_response(response, len(tokens_to_label))
            if labels is None:
                failed_entries.append({"id": sid, "line": line, "response": response})
                continue

            for idx, label in zip(sent[unlabeled].index, labels):
                df.at[idx, "key"] = label

    if failed_entries:
        print(f"LLM annotation complete. {len(failed_entries)} entries could not be parsed and remain unlabeled.")
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as f:
                f.write(f"Failed entries: {len(failed_entries)}\n")
                f.write("=" * 60 + "\n\n")
                for entry in failed_entries:
                    f.write(f"[{entry['id']}] {entry['line']}\n")
                    f.write(f"Response: {entry['response']}\n\n")
            print(f"  Log written to: {log_path}")

    return df


def make_ground_truth(
    image_path: str | Path,
    output_path: str | Path | None = None,
    exclude_strings: list[str] | None = None,
    exclude_whole: list[str] | None = None,
    annotate: bool = False,
    model_id: str = _DEFAULT_MODEL,
) -> Path:
    image_path = Path(image_path)
    exclude_strings = exclude_strings or []
    exclude_whole = exclude_whole or []

    text_extractor = TextExtractor2()
    ocr_system = OCR()

    # Extract line images from the page
    line_crops = text_extractor.extract_text_lines(str(image_path))

    # OCR each line, skip lines matching any exclude pattern
    ocr_results = []
    for crop, is_outlier in tqdm(line_crops, desc="Running OCR"):
        text = ocr_system.run(crop)
        if not text:
            continue
        if any(s in text for s in exclude_strings) or text in exclude_whole:
            continue
        ocr_results.append((text, is_outlier))

    # Combine continuation lines (outlier = indented, belongs to previous entry)
    combined_lines = []
    for text, is_outlier in ocr_results:
        if is_outlier and combined_lines:
            combined_lines[-1] += " " + text
        else:
            combined_lines.append(text)

    # Tokenize, auto-assign D for delimiters and START/END for sentinels
    ids, tokens, keys = [], [], []
    for sentence_id, line in enumerate(tqdm(combined_lines, desc="Tokenizing"), start=1):
        for token in Convert_To_Features.tokenize_string(line):
            ids.append(sentence_id)
            tokens.append(token)
            if token in ("START", "END"):
                keys.append(token)
            elif token in _DELIMITERS:
                keys.append("D")
            else:
                keys.append(None)

    df = pd.DataFrame({"id": ids, "value": tokens, "key": keys})

    if output_path is None:
        output_path = Path(__file__).parent.parent / "BelHisFirm-GT" / f"{image_path.stem}_GT.csv"
    output_path = Path(output_path)

    if annotate:
        log_path = output_path.with_name(output_path.stem + "_annotation_failures.log")
        df = annotate_with_llm(df, model_id=model_id, log_path=log_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Ground truth saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a ground truth template from a page image.")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (optional)")
    parser.add_argument(
        "--exclude", type=str, action="append", default=None, metavar="STRING",
        help="Exclude OCR lines containing this string (can be repeated)",
    )
    parser.add_argument(
        "--exclude-whole", type=str, action="append", default=None, metavar="STRING",
        help="Exclude OCR lines whose full text matches this string exactly (can be repeated)",
    )
    parser.add_argument(
        "--annotate", action="store_true",
        help="Auto-annotate tokens using an LLM (requires GPU, ~18 GB VRAM for the default 9B model)",
    )
    parser.add_argument(
        "--model", type=str, default=_DEFAULT_MODEL,
        help=f"HuggingFace model ID used for annotation (default: {_DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    make_ground_truth(
        args.image_path, args.output,
        args.exclude, args.exclude_whole,
        args.annotate, args.model,
    )

# Example usage:
# uv run src/index_parser/CRF/train/ground_truth_maker.py \
#   "/home/bas/Documents/Visual Code Repos/BelHisFirm-BelHisHAAI/src/index_parser/testdata/EHC_B665_O_2025_1892_III-IV_0926.tif" \
#   --exclude "TABLE DU RECUEIL" --exclude "N° d'ordre" --exclude-whole "d'acte" \
#   --annotate
