from nicegui import ui as ngUI

from prototype.webuserinterface.components.ui_component import UIComponent


class LoadingSpinnerUI(UIComponent):
    """
    Contains the code for the loading spinner UI.
    """

    def build_userinterface(self):
        """
        Builds the UI for the generating state.
        """
        with ngUI.column().classes('mx-auto items-center').bind_visibility_from(self.webUI, 'is_generating', value=True):
            ngUI.label('Generating images...').style('font-size: 200%;')
            ngUI.space().classes('m-4')
            self.loading_progess = ngUI.linear_progress(value=0)
            ngUI.spinner(size='10em', color='#323232')
        self.webUI.generator.callback = self.update_progess
    
    def update_progess(self, step, timestep, callback_kwargs):
        self.loading_progess.set_value(step/self.webUI.args.generator.num_inference_steps)
