from abc import abstractmethod, ABC


class UIComponent(ABC):
    """
    A UI Component contains the code to build a specific part of the WebUI.
    A UI Component should not necessarily work on its own, but its rather used to outsource code from the main WebUI class.
    """

    def __init__(self, webUI):
        """
        Initializes the UIComponent instance and automatically builds the userinterface.

        Args:
            webUI: The parent WebUI instance that uses this component.
        """
        self.webUI = webUI
        self.build_userinterface()
    
    @abstractmethod
    def build_userinterface(self):
        pass
