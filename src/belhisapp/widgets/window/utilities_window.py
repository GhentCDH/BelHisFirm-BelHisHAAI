from src.belhisapp.config import ConfigField, BuildFormResult, FormBuilder
from src.belhisapp.widgets.window import Window
from src.belhisapp.type_helper import FunctionInspector

from src.recordprocessing.processor import RecordProcessor

from textual.containers import Center

class UtilitiesWindow(Window):

    def __init__(self):
        processor = RecordProcessor()

        params: list[dict] = FunctionInspector.get_function_params(processor.process_record)
        config_fields: list[ConfigField] = FunctionInspector.parse_config_fields_from_function_inspection(params)
        form_build_result: BuildFormResult = FormBuilder.build_form(config_fields)

        super().__init__([Center(form_build_result.form)])