from nicegui import ui as ngUI
from ui_component import UIComponent

class LoadingUI(UIComponent):
    def build_userinterface(self):
        with ngUI.column().classes('mx-auto items-center').bind_visibility_from(self.webUI, 'is_generating', value=True):
            ngUI.label('Generating images...').style('font-size: 200%;')
            ngUI.space().classes('m-4')
            self.loading_progress = ngUI.linear_progress(value=0, show_value=False, color='#323232').props('size=md;')
        self.webUI.generator.callback = update_progress

    def reset_progress_bar(self):
        self.loading_progress.set_value(0)

def update_progress(pipe, step_index, timestep, callback_kwargs, current_step, num_embeddings, loading_progress, batch_size, num_steps):
    loading_progress.set_value((current_step + (step_index / num_steps * batch_size)) / num_embeddings)
    return callback_kwargs