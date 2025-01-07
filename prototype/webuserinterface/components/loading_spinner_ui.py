from nicegui import ui as ngUI

from prototype.webuserinterface.components.ui_component import UIComponent


class LoadingSpinnerUI(UIComponent):

    def build_userinterface(self):
        """
        Builds the UI for the generating state.
        """
        with ngUI.column().classes('mx-auto items-center').bind_visibility_from(self.webUI, 'is_generating', value=True):
            ngUI.label('Generating images...').style('font-size: 200%;')
            ngUI.spinner(size='128px')
