import asyncio
from nicegui import ui as ngUI
from constants import WebUIState
from prototype_lean.webuserinterface.components.ui_component import UIComponent

class InitialIterationUI(UIComponent):
    def build_userinterface(self):
        with ngUI.column().classes('mx-auto items-center').bind_visibility_from(self.webUI, 'is_initial_iteration', value=True):
            ngUI.markdown("""
**Interactive Demo**

Enter a prompt to generate 4 images. Adjust sliders to edit each image's style.
Example prompts:
- `a fox firefighter`
- `an avocado chair`
- `an illustration of love`
            """)
            ngUI.label('Start by generating images.').classes('mt-8').style('font-size: 200%; font-weight: bold;')
            with ngUI.card().classes('self-center no-box-shadow bg-grey-3 p-0 m-0 mt-4 gap-0').style('border-radius: 30px;').tight():
                with ngUI.column().classes('items-stretch p-0 gap-0'):
                    self.prompt_field = ngUI.input(placeholder='Enter your prompt').props("size=80 autofocus borderless dense item-aligned").style('font-size: 16px;').bind_value(self.webUI, 'user_prompt').on('keypress.enter', self.on_generate_images_button_click)
                    with self.prompt_field.add_slot("append"):
                        ngUI.button(icon='start', on_click=self.on_generate_images_button_click).props('flat fab color=black').tooltip('Generate images')
            ngUI.space().classes('w-full h-[2vh]')
            self.generate_button = ngUI.button('Generate images', on_click=self.on_generate_images_button_click).style('font-weight: bold;').props('icon-right="start" color=grey-8 unelevated rounded')

    async def on_generate_images_button_click(self):
        if not self.webUI.user_prompt:
            ngUI.notify('Please enter a prompt!')
            return
        self.webUI.change_state(WebUIState.GENERATING_STATE)
        self.webUI.iteration += 1
        ngUI.notify('Generating images...')
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.webUI.generate_images)
        self.webUI.update_image_displays()
        self.webUI.change_state(WebUIState.MAIN_STATE)
        self.webUI.loading_ui.reset_progress_bar()