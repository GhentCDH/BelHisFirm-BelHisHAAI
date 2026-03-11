from textual.containers import Vertical
from textual.widgets import Static

from src.belhisapp.widgets.window.window import Window
from src.belhisapp.constants import ConfigWindowConstants

class ConfigWindow(Window):

    def __init__(self):

        labels: list[Static] = ConfigWindow._get_labels(gap=2, css_class="")

        labels_container: Vertical = Vertical(*labels)

        super().__init__([labels_container])

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

    @staticmethod
    def _get_labels(gap: int = 0, css_class: str = "") -> list[Static]:
        """ Builds a list of static widgets to act as labels in the config window

            Args: gap (int): The number of lines inbetween each label
            Args: css_class (str): The CSS class of all generated labels

            Returns: A new list of static widgets
        """

        label_names: list[str] = ConfigWindowConstants.CONFIG_FIELDS
        labels: list[Static] = []

        # Build labels array
        for i in range(len(label_names)):
            labels.append(Static(label_names[i], classes=css_class))

            # Add gap
            for j in range(gap):
                labels.append(Static(""))

        return labels