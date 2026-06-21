import json
import datetime
from pathlib import Path
from nicegui import ui as ngUI
from constants import WebUIState

SUS_FORM_PE_URL      = "https://forms.gle/M8wmRJYx9YAdkov7A"
SUS_FORM_SLIDERS_URL = "https://forms.gle/Uq2RTyYjkQeg8e6a8"

STUDY_TASKS = [
    {
        "prompt":   "a cozy living room",
        "scenario": "You work in marketing and need an image for a winter campaign.",
        "goal":     "Make the scene feel warmer and cozier.",
    },
    {
        "prompt":   "a modern workspace",
        "scenario": "You are a designer at a tech startup.",
        "goal":     "Make it look more futuristic and minimalist.",
    },
    {
        "prompt":   "a lush forest",
        "scenario": "You need a nature image for a sustainable brand.",
        "goal":     "Make it feel greener and more vibrant.",
    },
]
NUM_TASKS = len(STUDY_TASKS)

PE_INSTRUCTIONS = (
    "You can generate as many times as you like. Just edit the prompt above and "
    "press Generate again. Once you are happy with one of the resulting images, "
    "click \"Select as final image\" underneath it, then click \"Task Done\" to continue."
)
SLIDER_INSTRUCTIONS = (
    "Use the sliders below to edit the generated images. "
    "Once you are happy with one SAE-edited image and one Splice-edited image, "
    "click \"Select as final image\" underneath each of them, then click \"Task Done\" to continue."
)


class StudyUI:
    """Builds three NiceGUI islands into the current page:
       1. Intro screen   (study_phase == 'intro')
       2. Task bar       (study_phase_is_task == True)
       3. Done screen    (study_phase == 'done')
    """

    def __init__(self, webUI):
        self.webUI = webUI
        self._build_intro()
        self._build_task_bar()
        self._build_done()

    # Screen builders

    def _build_intro(self):
        with ngUI.column() \
                .classes('mx-auto items-center gap-6 p-12 max-w-2xl') \
                .bind_visibility_from(self.webUI, 'study_phase', value='intro'):

            ngUI.label('Welcome to the User Study') \
                .style('font-size: 200%; font-weight: bold;')

            ngUI.markdown("""
This short study compares **two methods** for editing AI-generated images:

**Round 1 — Prompt Engineering:** You edit images by modifying the text prompt only.

**Round 2 — Concept Sliders:** You edit images using two types of concept sliders (SAE and Splice).

You will complete the same **3 tasks** in both rounds.  
After each round, please fill in a short questionnaire.

            """).classes('text-base text-gray-700')

            with ngUI.row().classes('gap-4 mt-4'):
                ngUI.button('Start Study', on_click=self._start_study) \
                    .props('color=grey-8 unelevated rounded') \
                    .style('font-weight: bold;')
                ngUI.button('Back to Demo Mode', on_click=self._go_to_demo) \
                    .props('flat rounded')

    def _build_task_bar(self):
        with ngUI.column() \
                .classes('w-full max-w-5xl mx-auto px-8 pt-4 pb-2 gap-2') \
                .bind_visibility_from(self.webUI, 'study_phase_is_task'):

            with ngUI.row().classes('w-full items-center justify-between'):
                ngUI.label() \
                    .bind_text_from(
                        self.webUI, 'study_phase',
                        backward=lambda p:
                            'Round 1 — Prompt Engineering' if p == 'prompt_engineering'
                            else 'Round 2 — Concept Sliders'
                    ) \
                    .style('font-size: 120%; font-weight: bold; color: #444;')

                with ngUI.row().classes('items-center gap-4'):
                    ngUI.label() \
                        .bind_text_from(
                            self.webUI, 'study_task_idx',
                            backward=lambda i: f'Task {i + 1} of {NUM_TASKS}'
                        ) \
                        .style('color: #888;')
                    ngUI.button('Back to Demo Mode', on_click=self._go_to_demo) \
                        .props('flat rounded size=sm')

            # Scenario card
            with ngUI.card() \
                    .classes('w-full p-4') \
                    .style('background: #f0f4f8; border-radius: 12px;'):
                ngUI.label() \
                    .bind_text_from(
                        self.webUI, 'study_task_idx',
                        backward=lambda i: f"Scenario: {STUDY_TASKS[i]['scenario']}"
                    ).style('font-weight: bold;')
                ngUI.label() \
                    .bind_text_from(
                        self.webUI, 'study_task_idx',
                        backward=lambda i: f"Goal: {STUDY_TASKS[i]['goal']}"
                    ).classes('mt-1 text-gray-700')
                ngUI.label() \
                    .bind_text_from(
                        self.webUI, 'study_task_idx',
                        backward=lambda i:
                            f'Suggested starting prompt: "{STUDY_TASKS[i]["prompt"]}"'
                    ).classes('mt-1 text-sm text-gray-500 italic')
                # Phase-specific instructions
                ngUI.label() \
                    .bind_text_from(
                        self.webUI, 'study_phase',
                        backward=lambda p: PE_INSTRUCTIONS if p == 'prompt_engineering' else SLIDER_INSTRUCTIONS
                    ).classes('mt-2 text-sm text-blue-700')

            with ngUI.row().classes('w-full justify-between items-center mt-1') \
                    .bind_visibility_from(self.webUI, 'is_main_loop_iteration'):

                # "Generate New Images" only in PE phase
                ngUI.button('↻ Generate New Images', on_click=self._on_regenerate) \
                    .props('outline rounded') \
                    .bind_visibility_from(
                        self.webUI, 'study_phase', value='prompt_engineering'
                    )

                ngUI.space()

                ngUI.button('Task Done ✓', on_click=self._on_task_done) \
                    .props('color=green unelevated rounded') \
                    .style('font-weight: bold;')

    def _build_done(self):
        with ngUI.column() \
                .classes('mx-auto items-center gap-6 p-12 max-w-2xl') \
                .bind_visibility_from(self.webUI, 'study_phase', value='done'):

            ngUI.label('Study complete — thank you! 🎉') \
                .style('font-size: 200%; font-weight: bold;')
            ngUI.label(
                'Your results have been saved. You can now close this window.'
            ).classes('text-base text-gray-600')
            ngUI.button('Back to Demo Mode', on_click=self._go_to_demo) \
                .props('flat rounded')

    # Actions

    def _start_study(self):
        self.webUI.study_phase              = 'prompt_engineering'
        self.webUI.study_phase_is_task      = True
        self.webUI.show_demo_ui             = True
        self.webUI.show_sliders             = False   # PE round: no sliders
        self.webUI.study_task_idx           = 0
        self.webUI.study_selected_image_idx  = None
        self.webUI.study_selected_sae_idx    = None
        self.webUI.study_selected_splice_idx = None
        self._prefill_prompt()
        self.webUI.change_state(WebUIState.INIT_STATE)

    def _go_to_demo(self):
        self.webUI.study_phase              = 'demo'
        self.webUI.study_phase_is_task      = False
        self.webUI.show_demo_ui             = True
        self.webUI.show_sliders             = True
        self.webUI.study_selected_image_idx  = None
        self.webUI.study_selected_sae_idx    = None
        self.webUI.study_selected_splice_idx = None
        self.webUI.change_state(WebUIState.INIT_STATE)

    def _prefill_prompt(self):
        self.webUI.user_prompt = STUDY_TASKS[self.webUI.study_task_idx]['prompt']

    def _on_regenerate(self):
        """PE phase only: goes back to the prompt field, keeps current prompt text,
        lets the user tweak it and generates again."""
        self.webUI.study_selected_image_idx = None
        self.webUI.change_state(WebUIState.INIT_STATE)

    def _on_task_done(self):
        if self.webUI.study_phase == 'prompt_engineering' \
                and self.webUI.study_selected_image_idx is None:
            ngUI.notify('Please select one image as your final image first.', type='warning')
            return

        if self.webUI.study_phase == 'sliders' and (
                self.webUI.study_selected_sae_idx is None
                or self.webUI.study_selected_splice_idx is None
        ):
            ngUI.notify('Please select a final image for both SAE and Splice.', type='warning')
            return

        self._save_task_log()
        next_idx = self.webUI.study_task_idx + 1
        self.webUI.study_selected_image_idx  = None
        self.webUI.study_selected_sae_idx    = None
        self.webUI.study_selected_splice_idx = None

        if next_idx < NUM_TASKS:
            self.webUI.study_task_idx = next_idx
            self._prefill_prompt()
            self.webUI.change_state(WebUIState.INIT_STATE)
        else:
            if self.webUI.study_phase == 'prompt_engineering':
                self._show_sus_popup(SUS_FORM_PE_URL, then=self._start_sliders_round)
            else:
                self._show_sus_popup(SUS_FORM_SLIDERS_URL, then=self._finish_study)

    def _start_sliders_round(self):
        self.webUI.study_phase           = 'sliders'
        self.webUI.study_task_idx        = 0
        self.webUI.show_sliders          = True
        self.webUI.study_selected_sae_idx    = None
        self.webUI.study_selected_splice_idx = None
        self._prefill_prompt()
        self.webUI.change_state(WebUIState.INIT_STATE)

    def _finish_study(self):
        self.webUI.study_phase         = 'done'
        self.webUI.study_phase_is_task = False
        self.webUI.show_demo_ui        = False

    # Popups

    def _show_sus_popup(self, url, then):
        with ngUI.dialog() as dialog, ngUI.card().classes('p-6 items-center gap-4'):
            ngUI.label('Round complete!') \
                .style('font-size: 150%; font-weight: bold;')
            ngUI.label('Please fill in the short usability questionnaire before continuing.')
            ngUI.link('Open questionnaire ↗', url, new_tab=True) \
                .classes('text-blue-600 font-bold text-lg')
            ngUI.label('Return here once you have submitted the form.') \
                .classes('text-sm text-gray-500')

            def on_continue():
                dialog.close()
                then()

            ngUI.button('Continue →', on_click=on_continue) \
                .props('color=grey-8 unelevated rounded') \
                .style('font-weight: bold;')
        dialog.open()

    # Logging

    def _save_task_log(self):
        out_dir = Path("study_results") / self.webUI.study_session_id
        out_dir.mkdir(parents=True, exist_ok=True)

        log_entry = {
            "timestamp":   datetime.datetime.now().isoformat(),
            "study_phase": self.webUI.study_phase,
            "task_idx":    self.webUI.study_task_idx,
            "prompt_used": self.webUI.user_prompt,
        }

        if self.webUI.study_phase == 'prompt_engineering':
            idx = self.webUI.study_selected_image_idx
            img = self.webUI.images_sae[idx]
            p = out_dir / f"task{self.webUI.study_task_idx}_prompt_engineering_final.jpg"
            img.save(str(p))
            log_entry["final_image"]  = str(p)
            log_entry["selected_idx"] = idx

        elif self.webUI.study_phase == 'sliders':
            sae_idx    = self.webUI.study_selected_sae_idx
            splice_idx = self.webUI.study_selected_splice_idx

            img_sae = self.webUI.images_sae[sae_idx]
            p_sae = out_dir / f"task{self.webUI.study_task_idx}_sliders_sae_final.jpg"
            img_sae.save(str(p_sae))
            log_entry["final_sae_image"]  = str(p_sae)
            log_entry["selected_sae_idx"] = sae_idx

            img_splice = self.webUI.images_splice[splice_idx]
            p_splice = out_dir / f"task{self.webUI.study_task_idx}_sliders_splice_final.jpg"
            img_splice.save(str(p_splice))
            log_entry["final_splice_image"]  = str(p_splice)
            log_entry["selected_splice_idx"] = splice_idx

        with open(out_dir / "log.jsonl", 'a') as f:
            f.write(json.dumps(log_entry) + "\n")