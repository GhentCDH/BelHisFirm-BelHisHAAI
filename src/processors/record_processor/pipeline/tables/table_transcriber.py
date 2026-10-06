from .vllm_client import make_client, structure_transcription, TranscriptionError

TABLE_OCR_PROMPT = "Transcribe all text in this table image, preserving line breaks between rows."


class TableTranscriber:
    """Transcribes a table crop via the shared GLM-OCR engine, then structures
    that plain text into schema-constrained JSON via a vLLM-hosted model."""

    def __init__(self, glm_ocr, table_config):
        self.glm_ocr = glm_ocr
        self.table_config = table_config
        self._client = make_client(table_config.vllm_base_url)

    def transcribe(self, crop: dict, page_index: int):
        from src.processors.record_processor.data import TableResult

        plain_text = self.glm_ocr.ocr(crop["image"], prompt=TABLE_OCR_PROMPT)
        try:
            structured = structure_transcription(
                self._client, self.table_config.vllm_model, plain_text, source_name=f"page{page_index + 1}_table"
            )
            structured.pop("plain_text", None)  # already stored separately on TableResult
            return TableResult(page_index, crop["bbox"], crop["image"], plain_text, structured, error=None)
        except TranscriptionError as e:
            return TableResult(page_index, crop["bbox"], crop["image"], plain_text, structured=None, error=str(e))
