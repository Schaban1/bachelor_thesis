from nicegui import ui as ngUI
from .ui_component import UIComponent


class MainLoopUI(UIComponent):
    def create_label_updater(self, name, base_value, is_relative=False):
        return lambda v: f"{name}: {max(0.0, base_value + v):.2f}"

    def _sae_label_text(self, i):
        phase = self.webUI.study_phase
        if phase == 'prompt_engineering':
            sel = self.webUI.study_selected_image_idx
        elif phase == 'sliders':
            sel = self.webUI.study_selected_sae_idx
        else:
            sel = None
        return '✓ Selected' if sel == i else ''

    def _splice_label_text(self, i):
        sel = self.webUI.study_selected_splice_idx
        return '✓ Selected' if sel == i else ''

    def build_userinterface(self):
        print("[DEBUG] build_userinterface: START", flush=True)
        with ngUI.column().classes('mx-auto items-center pl-24 pr-24') \
                .bind_visibility_from(self.webUI, 'is_main_loop_iteration', value=True):

            ngUI.label('Edit images by adjusting concept sliders.').style('font-size: 200%;') \
                .bind_visibility_from(self.webUI, 'show_sliders')
            ngUI.label('Review the generated images below.').style('font-size: 200%;') \
                .bind_visibility_from(
                    self.webUI, 'study_phase', value='prompt_engineering'
                )

            with ngUI.row().classes('w-full items-center justify-start'):
                ngUI.icon('subject', size='2rem').classes('mr-2')
                ngUI.label().bind_text_from(self.webUI, 'user_prompt').style('font-size: 120%;')

            # ── FIRST ROW: SAE extractor ──────────────────────────────────
            with ngUI.row().classes('mx-auto items-start mt-4 gap-8 justify-center'):

                with ngUI.column().classes('w-full items-center') \
                        .bind_visibility_from(self.webUI, 'show_sliders'):
                    ngUI.label('SAE Extractor').classes('text-center font-bold')
                    ngUI.label(
                        "SAE sliders adjust concept strength in absolute 0.5 steps (min./max.: ±1.0)"
                    ).classes('text-sm text-gray-600 italic mb-2 text-center')

                for i in range(self.webUI.num_images_to_generate):
                    with ngUI.column().classes('items-center'):
                        self.webUI.images_display[i] = ngUI.interactive_image() \
                            .style(
                                f'width: {self.webUI.image_display_width}px; '
                                f'height: {self.webUI.image_display_height}px; '
                                f'object-fit: scale-down; border: 2px solid #ccc;'
                            )

                        # Slider container hidden during PE round
                        container = ngUI.column() \
                            .classes('w-full mt-2 space-y-1') \
                            .bind_visibility_from(self.webUI, 'show_sliders')
                        self.webUI.slider_containers.append(container)

                        # "Select as final image" shown during any study task
                        with ngUI.column().classes('w-full items-center mt-2 gap-1') \
                                .bind_visibility_from(self.webUI, 'study_phase_is_task'):
                            ngUI.button(
                                'Select as final image',
                                on_click=lambda i=i: self.webUI.select_study_image(i, row='sae')
                            ).props('outline size=sm color=primary')
                            ngUI.label() \
                                .bind_text_from(
                                    self.webUI, 'study_selection_tick',
                                    backward=lambda _, i=i: self._sae_label_text(i)
                                ).classes('text-xs text-green-600 font-bold')

            # SECOND ROW: Splice extractor
            with ngUI.row().classes('mx-auto items-start mt-4 gap-8 justify-center') \
                    .bind_visibility_from(self.webUI, 'show_sliders'):

                with ngUI.column().classes('w-full items-center'):
                    ngUI.label('Splice Extractor').classes('text-center font-bold')
                    ngUI.label(
                        "Splice sliders adjust concept strength in absolute 0.2 steps (min./max.: ±0.4)"
                    ).classes('text-sm text-gray-600 italic mb-2 text-center')

                for i in range(self.webUI.num_images_to_generate):
                    with ngUI.column().classes('items-center'):
                        self.webUI.images_display_splice[i] = ngUI.interactive_image() \
                            .style(
                                f'width: {self.webUI.image_display_width}px; '
                                f'height: {self.webUI.image_display_height}px; '
                                f'object-fit: scale-down; border: 2px solid #ccc;'
                            )
                        container = ngUI.column().classes('w-full mt-2 space-y-1')
                        self.webUI.slider_containers_splice.append(container)

                        with ngUI.column().classes('w-full items-center mt-2 gap-1') \
                                .bind_visibility_from(self.webUI, 'study_phase', value='sliders'):
                            ngUI.button(
                                'Select as final image',
                                on_click=lambda i=i: self.webUI.select_study_image(i, row='splice')
                            ).props('outline size=sm color=primary')
                            ngUI.label() \
                                .bind_text_from(
                                    self.webUI, 'study_selection_tick',
                                    backward=lambda _, i=i: self._splice_label_text(i)
                                ).classes('text-xs text-green-600 font-bold')

            ngUI.space()
            print("[DEBUG] mainloop build_userinterface: done", flush=True)

    def refresh_sliders(self, prompt, concepts_per_image, splice_concepts_per_image):
        # SAE (first row)
        print(f"[DEBUG] refresh_sliders called with {len(concepts_per_image)} images", flush=True)
        for idx, container in enumerate(self.webUI.slider_containers):
            container.clear()
            print(f"[DEBUG] Clearing container {idx}", flush=True)
            with container:
                for concept_name, concept_value in concepts_per_image[idx]:
                    ngUI.label(concept_name).classes('text-center font-bold text-sm mb-1 text-blue-600')
                    with ngUI.row().classes('w-full items-center gap-2'):
                        ngUI.label("Less").classes('text-xs text-gray-500 w-12 text-left')
                        with ngUI.row().classes('flex-grow items-center'):
                            slider = ngUI.slider(min=-1.0, max=1.0, step=0.5, value=0) \
                                .props('label-always') \
                                .classes('flex-grow')
                            ngUI.label().bind_text_from(
                                slider, 'value',
                                backward=self.create_label_updater(concept_name, concept_value, is_relative=True)
                            ).classes('text-xs text-gray-500 w-32 text-center')
                        ngUI.label("More").classes('text-xs text-gray-500 w-12 text-right')
                        slider.on(
                            'update:model-value',
                            lambda e, i=idx, c=concept_name:
                                self.webUI.slider_controller.on_slider_change(prompt, i, c, e.args, is_sae=True)
                        )

        # Splice (second row)
        for idx, container in enumerate(self.webUI.slider_containers_splice):
            container.clear()
            with container:
                for concept_name, value in splice_concepts_per_image[idx]:
                    ngUI.label(concept_name).classes('text-center font-bold text-sm mb-1 text-blue-600')
                    with ngUI.row().classes('w-full items-center gap-2'):
                        ngUI.label("Less").classes('text-xs text-gray-500 w-12 text-left')
                        with ngUI.row().classes('flex-grow items-center'):
                            slider = ngUI.slider(min=-0.4, max=0.4, step=0.2, value=0) \
                                .props('label-always') \
                                .classes('flex-grow')
                            ngUI.label().bind_text_from(
                                slider, 'value',
                                backward=self.create_label_updater(concept_name, value)
                            ).classes('text-xs text-gray-500 w-32 text-center')
                        ngUI.label("More").classes('text-xs text-gray-500 w-12 text-right')
                        slider.on(
                            'update:model-value',
                            lambda e, i=idx, c=concept_name:
                                self.webUI.slider_controller.on_slider_change(prompt, i, c, e.args, is_sae=False)
                        )

    def on_image_cached(self, was_cached):
        if was_cached:
            ngUI.notify("↺ Cached image loaded!", type='positive')
        else:
            ngUI.notify("✨ New image generated!", type='positive')