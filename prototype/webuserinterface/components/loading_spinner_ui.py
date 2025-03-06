from nicegui import ui as ngUI

from prototype.webuserinterface.components.ui_component import UIComponent

generator_pipeline = None
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
            loading_progress = ngUI.linear_progress(value=0, show_value=False, color='#323232')
        global generator_pipeline
        generator_pipeline = self.webUI.generator.pipe
        self.webUI.generator.callback = update_progess
    
def update_progess(pipe, step_index, timestep, callback_kwargs):
    global generator_pipeline
    global loading_progress
    loading_progress.set_value(step_index/generator_pipeline.num_timesteps)
    return callback_kwargs
