# Index Parser Refactor Plan

## Priority 1 — Runtime Crashes / Correctness Bugs

These will cause failures during normal execution and should be fixed first.

### 1.1 Wrong kernel type in `text_extraction.py`
**File:** `processing/text_extraction.py`, `__sentence_detector`, line 148  
**Issue:** `cv.dilate(image, (7,7), iterations=2)` passes a plain tuple as the kernel. OpenCV requires a `np.ndarray`.  
**Fix:** Replace with `cv.dilate(image, np.ones((7, 7), np.uint8), iterations=2)`.

### 1.2 `NameError` on empty input in `predict_crf.py`
**File:** `CRF/predict_crf.py`, `predict()`, line 38  
**Issue:** `tuple_list` is only assigned inside the `for item in source` loop. If `source` is empty, the reference to `tuple_list` on line 38 raises a `NameError`.  
**Fix:** Initialize `tuple_list = []` before the loop, and guard the `string_out` call:
```python
tuple_list = []
for item in source:
    ...
if not sentences_to_be_tranformed:
    return None
output_string = self.output.string_out(tuple_list=tuple_list)
```

### 1.3 `IndexError` on empty bounding boxes in `text_extraction.py`
**File:** `processing/text_extraction.py`, `combine_overlapping_boxes_with_iou()`, line 116  
**Issue:** `current_box = bounding_boxes[0]` raises `IndexError` when YOLO detects no boxes.  
**Fix:** Add an early return at the start of the method:
```python
if not bounding_boxes:
    return []
```

---

## Priority 2 — Performance Issues

### 2.1 `events.json` read on every token
**File:** `CRF/convert_to_features.py`, `__contains_event()`, line 134  
**Issue:** The JSON file is opened and parsed for every single token during feature extraction — potentially thousands of times per document.  
**Fix:** Load the event map once as a class-level variable:
```python
class Convert_To_Features:
    with open("index_parser/CRF/features/events.json", 'r') as f:
        _event_map = json.load(f)

    @staticmethod
    def __contains_event(token):
        return token in Convert_To_Features._event_map.values()
```

### 2.2 YOLO model reloaded per image split
**File:** `processing/text_extraction.py`, `__yolo_detector()`, line 69  
**Issue:** `model = YOLO('index_parser/model/best.pt')` is called inside `__yolo_detector`, which runs for every split of every image. The model is re-instantiated each time.  
**Fix:** Load the model once in `__init__`:
```python
def __init__(self):
    ...
    self.yolo_model = YOLO('index_parser/model/best.pt')
```
Then replace the line in `__yolo_detector` with `self.yolo_model`.

---

## Priority 3 — Incorrect Behavior / Silent Failures

### 3.1 `tqdm.disable = True` at module level in `OCR.py`
**File:** `processing/OCR.py`, line 2  
**Issue:** `tqdm.disable = True` disables tqdm globally for the entire process as soon as `OCR.py` is imported, breaking all other progress bars in `workflow.py`.  
**Fix:** Remove the line entirely. If tqdm output from the transformers library is unwanted, suppress it locally using `transformers.logging.set_verbosity_error()` instead.

### 3.2 Hardcoded `"cuda"` in `OCR.py`
**File:** `processing/OCR.py`, `__qwen_call()`, line 71  
**Issue:** `inputs.to("cuda")` crashes on CPU-only machines, even though the model uses `device_map="auto"`.  
**Fix:** Derive the device from the model:
```python
device = next(self.model.parameters()).device
inputs = inputs.to(device)
```

### 3.3 Bare `except` in `predict_crf.py`
**File:** `CRF/predict_crf.py`, `choose_model()`, line 16  
**Issue:** The bare `except` swallows all exceptions, including programming errors.  
**Fix:** Catch specific exceptions:
```python
except (FileNotFoundError, EOFError) as e:
    print(f"The model could not be loaded: {e}")
```

### 3.4 Absolute import in `train_crf.py`
**File:** `CRF/train_crf.py`, line 1  
**Issue:** `from convert_to_features import Convert_To_Features` is an absolute import that fails when the CRF module is used as part of the package.  
**Fix:** Change to a relative import: `from .convert_to_features import Convert_To_Features`.

### 3.5 `i` double-incremented in `text_extraction.py`
**File:** `processing/text_extraction.py`, `__yolo_detector()`, lines 73, 86, 91  
**Issue:** `i = 0` is set, then `for i, ... in enumerate(combined_boxes)` re-assigns it, then `i += 1` increments again — the saved filenames will have wrong indices.  
**Fix:** Remove the `i = 0` initialization and the `i += 1` line. Let `enumerate` manage the index.

---

## Priority 4 — Design / Architecture

### 4.1 Side effects in `Workflow.__init__`
**File:** `workflow.py`, `__init__` and `load()`  
**Issue:** The constructor calls `load()`, which calls `process_images()` at the end, triggering the full pipeline during object construction. This makes the class untestable and tightly couples initialization with execution.  
**Fix:** Remove the `self.process_images()` call from `load()`. Let the caller explicitly start processing:
```python
wf = Workflow(input_path, model_path)
wf.process_images()
```
Update `cli.py` accordingly.

### 4.2 Artificial `time.sleep(1)` during loading
**File:** `workflow.py`, `load()`, line 55  
**Issue:** Adds 4+ seconds of fake delay with no functional purpose.  
**Fix:** Remove `time.sleep(1)`.

### 4.3 Train/test split is computed but training uses full dataset
**File:** `CRF/train_crf.py`, `train()`, lines 55–59  
**Issue:** `create_test_set()` splits the data into `X_Train`/`Y_Train` and `X_Test`/`Y_Test`, but `model.fit` is called on `self.training_data_token`/`self.training_data_label` (the full set). The split is pointless.  
**Fix:** Either train on `X_Train`/`Y_Train`, or remove the unused split entirely:
```python
self.model.fit(self.X_Train, self.Y_Train)
```

---

## Priority 5 — Code Cleanup

### 5.1 Typo: `proces_text_list` → `process_text_list`
**File:** `workflow.py`, lines 114, 198, 201  
**Fix:** Rename consistently.

### 5.2 Typo: `verify_intendation` → `verify_indentation`
**Files:** `processing/VIN.py` line 11, `workflow.py` line 11 (import)  
**Fix:** Rename in both `VIN.py` and the import in `workflow.py`.

### 5.3 Unused imports
| File | Unused imports |
|------|---------------|
| `workflow.py` | `stdev`, `mean` |
| `processing/VIN.py` | `os`, `time`, `shutil`, `Path`, `pd`, `stdev`, `mean` |

**Fix:** Remove them.

### 5.4 Unused / dead variables
| File | Variable | Location |
|------|----------|----------|
| `workflow.py` | `self.is_indentation_beginning_of_sentence` | `__init__` |
| `workflow.py` | `result` | `load()` |
| `text_extraction.py` | `previous_intendation`, `valid_intendation`, `last_id` | `text_line_extractor()` |

**Fix:** Remove them.

### 5.5 `self.clean()` called twice in `output_crf.py`
**File:** `CRF/output_crf.py`, `create_csv()`, lines 74–75  
**Fix:** Remove the duplicate call.

### 5.6 Redundant `json_file.close()` in `output_crf.py`
**File:** `CRF/output_crf.py`, `collect_labels()`, line 26  
**Fix:** Remove — the `with` block already closes the file.

### 5.7 `cv.imread` with `Path` object in `OCR.py`
**File:** `processing/OCR.py`, `__image_processing()`, line 97  
**Fix:** Cast to string: `cv.imread(str(image_path))`.

---

## Suggested Order of Work

1. **P1** fixes first — get the code running without crashes.
2. **P2** performance fixes — especially `events.json` and YOLO reload.
3. **P3** silent failures — `tqdm` global disable, hardcoded CUDA.
4. **P4** design — decouple constructor from execution.
5. **P5** cleanup — can be done incrementally alongside the above.
BelHisFirm-BelHisHAAI/src/