from nicegui import ui as ngUI

from prototype.webuserinterface.components.ui_component import UIComponent
from prototype.constants import WebUIState


class PlotUI(UIComponent):
    """
    Contains the code for the interactive plot UI.
    """

    def build_userinterface(self):
        """
        Builds the UI for the interactive plot state.
        """
        with ngUI.column().classes('mx-auto items-center').bind_visibility_from(self.webUI, 'is_interactive_plot', value=True):
            ngUI.button('Back', on_click=self.on_back_to_main_loop_button_click)
            # TODO: Plot UI

    def on_back_to_main_loop_button_click(self):
        """
        Returns to the 'Main Loop' screen.
        """
        self.webUI.change_state(WebUIState.MAIN_STATE)
