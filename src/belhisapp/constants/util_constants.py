from src.belhisapp.widgets.common import UtilButton
from src.recordprocessing.pipeline import IOManager
from src.recordprocessing import RecordProcessor

class UtilConstants:

    # Define RecordProcessor here for now, not great but no time to think of a cleaner alternative
    record_processor = RecordProcessor()

    # Buttons for the util form, each has a label and callable method attached for that button to work with
    UtilButtons: list[UtilButton] = [
        UtilButton("Run Pipeline", record_processor.run),
        UtilButton("Generate Folder Name", IOManager.generate_folder_name)
    ]