from textual.containers import Vertical, Horizontal, Center
from textual.widgets import Static
from textual.widgets import Input

from src.belhisapp.widgets.window.window import Window
from src.belhisapp.constants import ConfigWindowConstants

class ConfigWindow(Window):

    _textboxes: list[Input]

    def __init__(self):

        rows: list[Horizontal] = []

        # Textboxes used for JSON parsing
        _textboxes: list[Input] = []

        header: Static = Static(ConfigWindowConstants.CONFIG_LOGO, classes="FormHeader")

        # Build a row for every config field with an input box and a label
        for name in ConfigWindowConstants.CONFIG_FIELDS:
            label = Static(f"{name}:", classes="FormLabel")
            textbox = Input(placeholder=name, classes="FormTextbox")

            # Save textbox separately so we can easily access its value later
            _textboxes.append(textbox)

            rows.append(Horizontal(label, textbox, classes="FormRow"))

        form = Center(Vertical(*rows))

        super().__init__([header, form])

        """
        self.padding = 50
        self.sus_table_confidence_threshold = 0.85
        self.sus_table_area_threshold = 0.4
        self.spine_vertical_margin = 200
        self.spine_margin = 300
        self.skip_ocr = skip_ocr

        self.record = None
        
        # Labels to exclude from OCR output (add more labels here as needed)
        self.ocr_excluded_labels = {"Table", "Picture", "Figure", "Form", "Handwriting", "Formula"}
        """