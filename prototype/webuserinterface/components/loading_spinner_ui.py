from nicegui import ui as ngUI

from prototype.webuserinterface.components.ui_component import UIComponent

num_inference_steps = 30
loading_progress = None


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
            global loading_progress
            loading_progress = ngUI.linear_progress(value=0)
            ngUI.spinner(size='10em', color='#323232')
        global num_inference_steps
        num_inference_steps = self.webUI.args.generator.num_inference_steps
        self.webUI.generator.callback = update_progess
    
def update_progess(pipe, step, timestep, callback_kwargs):
    global num_inference_steps
    global loading_progress
    loading_progress.set_value(step/num_inference_steps)
    return callback_kwargs
