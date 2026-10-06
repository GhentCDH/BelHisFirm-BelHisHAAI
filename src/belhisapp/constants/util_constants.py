from src.belhisapp.widgets.common import UtilButton
from src.processors.record_processor.pipeline import IOManager
from src.processors.record_processor import RecordProcessor

class UtilConstants:
    """ Class used for managing the utilities window. """

    # Define RecordProcessor here for now, not great but no time to think of a cleaner alternative
    record_processor = RecordProcessor()

    UtilButtons = [
        UtilButton("Run Pipeline", record_processor.run),
        UtilButton("Generate Folder Name", IOManager.generate_folder_name)
    ]