from abc import ABC, abstractmethod
from textual.widget import Widget

class Window(ABC):

    widgets: list[Widget]

    def __init__(self, widgets: list[Widget]):
        self.widgets = widgets