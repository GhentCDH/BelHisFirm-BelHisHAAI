from collections.abc import Callable
from textual.widgets import Button

from src.recordprocessing.pipeline import IOManager
from src.recordprocessing import RecordProcessor

class UtilConstants:

    # Define RecordProcessor here for now, not great but no time to think of a cleaner alternative
    record_processor = RecordProcessor()

    # Buttons for the util form, each has a callable method attached for that button to run
    UtilButtons: dict[Button, Callable] = {
        Button("Run Pipeline", classes="FormButton") : record_processor.run,
        Button("Generate Folder Name", classes="FormButton") : IOManager.generate_folder_name
    }