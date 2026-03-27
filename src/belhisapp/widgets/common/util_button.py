from typing import Callable

class UtilButton:

    def __init__(self, label: str, method: Callable) -> None:
        self.label = label
        self.method = method