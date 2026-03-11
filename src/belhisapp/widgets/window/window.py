from abc import ABC, abstractmethod
from textual.widget import Widget

class Window(ABC):
    """ Abstract base class for window widgets, these can be rendered in a window container. """

    widgets: list[Widget]

    @abstractmethod
    def __init__(self, widgets: list[Widget]):
        self.widgets = widgets