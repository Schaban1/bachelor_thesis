from abc import abstractmethod, ABC

class UIComponent(ABC):
    def __init__(self, webUI):
        self.webUI = webUI
        self.build_userinterface()

    @abstractmethod
    def build_userinterface(self):
        pass