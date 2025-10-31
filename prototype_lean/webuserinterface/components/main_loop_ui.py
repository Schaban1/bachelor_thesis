from nicegui import ui as ngUI
from .ui_component import UIComponent


class MainLoopUI(UIComponent):
    def __init__(self, webUI):
        super().__init__(webUI)
        self.webUI = webUI
        self.slider_containers = []
        self.build_userinterface()

    async def build_userinterface(self):
        with ngUI.column().classes('mx-auto items-center pl-24 pr-24') \
                .bind_visibility_from(self.webUI, 'is_main_loop_iteration', value=True):
            ngUI.label('Edit images by adjusting concepts.').style('font-size: 200%;')

            with ngUI.row().classes('w-full items-center justify-start'):
                ngUI.icon('subject', size='2rem').classes('mr-2')
                ngUI.label().bind_text_from(self.webUI, 'user_prompt').style('font-size: 120%;')

            with ngUI.row().classes('mx-auto items-start mt-4 gap-8'):
                for i in range(self.webUI.num_images_to_generate):
                    with ngUI.column().classes('items-center'):
                        self.webUI.images_display[i] = ngUI.interactive_image() \
                            .style(f'width: {self.webUI.image_display_width}px; '
                                   f'height: {self.webUI.image_display_height}px; '
                                   f'object-fit: scale-down; border: 2px solid #ccc;')

                        # One container per image
                        container = ngUI.column().classes('w-full mt-2 space-y-1')
                        self.slider_containers.append(container)

            ngUI.space()
            ngUI.notify("[DEBUG mainloop webuserinterface builduserinterface: was async def build_userinterface() called?")

    def refresh_sliders(self, concepts_per_image):
        print("[DEBUG] refresh_sliders called with {len(concepts_per_image)} images")
        for idx, container in enumerate(self.slider_containers):
            container.clear()
            print("[DEBUG] Clearing container {idx}")
            with container:
                for concept, concept_idx in concepts_per_image[idx]:
                    with ngUI.row().classes('items-center justify-between w-full'):
                        ngUI.label("Less").classes('text-xs text-gray-600')
                        slider = ngUI.slider(
                            min=-0.3, max=0.3, step=0.15, value=0
                        ).props('label').classes('flex-grow mx-2')
                        ngUI.label().bind_text_from(slider, 'value', lambda v: f"{v:+.2f}")
                        ngUI.label("More").classes('text-xs text-gray-600')
                        slider.on('update:model-value',
                                  lambda e, img=idx, c=concept:
                                  self.webUI.slider_controller.on_slider_change(img, c, e['args']))