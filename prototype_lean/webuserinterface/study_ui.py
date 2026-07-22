import json
import datetime
from pathlib import Path
from nicegui import ui as ngUI
from constants import WebUIState

SUS_FORM_PE_URL      = "https://forms.gle/M8wmRJYx9YAdkov7A"
SUS_FORM_SLIDERS_URL = "https://forms.gle/Uq2RTyYjkQeg8e6a8"

STUDY_TASKS = [
    {
        "prompt":   "a small workshop",
        "scenario": "A craft business wants an image for a page about patience and skill.",
        "goal":     "The image should show that the work takes time and attention. It should not look like a factory or a messy storage room.",
    },
    {
        "prompt":   "a park bench",
        "scenario": "A counseling service needs a calm image for a page about taking a pause.",
        "goal":     "The bench should feel like a place where stopping for a moment is acceptable and helpful, rather than a sign of loneliness.",
    },
    {
        "prompt":   "a classroom",
        "scenario": "A school website needs an image for a page about curiosity.",
        "goal":     "The picture should make learning feel active and open, not only strict or exam-focused.",
    },
    {
        "prompt":   "a mountain cabin",
        "scenario": "A travel page needs an image for a quiet weekend retreat.",
        "goal":     "The target audience is tired from work and wants a place that feels removed from everyday life. Suggest rest and distance, without looking lonely or abandoned.",
    },
    {
        "prompt":   "a city street in the evening",
        "scenario": "A local community group needs an image for a post about feeling safe in the neighborhood after dark.",
        "goal":     "The image should still look like evening, but not feel threatening. It should suggest that people could walk there comfortably.",
    },
    {
        "prompt":   "study desk",
        "scenario": "A student is building the front page for a study blog.",
        "goal":     "The desk should give the impression that someone could sit down and start working without feeling pressured.",
    },
]
NUM_TASKS = len(STUDY_TASKS)  # 6

# Group A: tasks 0-2 with PE, tasks 3-5 with Sliders
# Group B: tasks 0-2 with Sliders, tasks 3-5 with PE
GROUP_A_PE_TASKS     = {0, 1, 2}
GROUP_A_SLIDER_TASKS = {3, 4, 5}
GROUP_B_PE_TASKS     = {3, 4, 5}
GROUP_B_SLIDER_TASKS = {0, 1, 2}

PE_INSTRUCTIONS = (
    "Edit the starting prompt above and press Generate as many times as you like. "
    "Once you are satisfied with one of the images, click 'Select as final image' "
    "underneath it, then click 'Task Done' to continue."
)
SLIDER_INSTRUCTIONS = (
    "Use the SAE and Splice sliders below to edit the images. "
    "Select one final image from each slider row (SAE and Splice separately), "
    "then click 'Task Done' to continue."
)


def _phase_for(group: str, task_idx: int) -> str:
    if group == 'a':
        return 'prompt_engineering' if task_idx in GROUP_A_PE_TASKS else 'sliders'
    return 'prompt_engineering' if task_idx in GROUP_B_PE_TASKS else 'sliders'


class StudyUI:
    def __init__(self, webUI):
        self.webUI = webUI
        self._build_intro()
        self._build_task_bar()
        self._build_done()

    def _bump_tick(self):
        self.webUI.study_tick = (self.webUI.study_tick or 0) + 1

    def _current_phase(self):
        return _phase_for(self.webUI.study_group, self.webUI.study_task_idx)

    def _build_intro(self):
        with ngUI.column() \
                .classes('mx-auto items-center gap-6 p-12 max-w-2xl') \
                .bind_visibility_from(self.webUI, 'study_phase', value='intro'):

            ngUI.label('Welcome to the User Study') \
                .style('font-size: 200%; font-weight: bold;')

            ngUI.markdown("""
In this study you will complete **6 image editing tasks**.

For each task you will use one of two methods:

- **Prompt Engineering:** Edit images by rewriting the text prompt
- **Concept Sliders:** Edit images using SAE and Splice concept sliders

After every three tasks, please fill in a short questionnaire.

The study takes approximately **20 minutes**.
There are no right or wrong answers.
            """).classes('text-base text-gray-700')

            ngUI.label('Select your assigned group:') \
                .style('font-weight: bold; margin-top: 8px;')

            with ngUI.row().classes('gap-4'):
                ngUI.button('Group A', on_click=lambda: self._start_study('a')) \
                    .props('color=grey-8 unelevated rounded') \
                    .style('font-weight: bold;')
                ngUI.button('Group B', on_click=lambda: self._start_study('b')) \
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
                    self.webUI, 'study_tick',
                    backward=lambda _: (
                        'Prompt Engineering'
                        if self._current_phase() == 'prompt_engineering'
                        else 'Concept Sliders'
                    )
                ).style('font-size: 120%; font-weight: bold; color: #444;')

                with ngUI.row().classes('items-center gap-4'):
                    ngUI.label() \
                        .bind_text_from(
                        self.webUI, 'study_tick',
                        backward=lambda _: f'Task {self.webUI.study_task_idx + 1} of {NUM_TASKS}'
                    ).style('color: #888;')
                    ngUI.button('Back to Demo Mode', on_click=self._go_to_demo) \
                        .props('flat rounded size=sm')

            with ngUI.card() \
                    .classes('w-full p-4') \
                    .style('background: #f0f4f8; border-radius: 12px;'):
                ngUI.label() \
                    .bind_text_from(
                    self.webUI, 'study_tick',
                    backward=lambda _: f"Context: {STUDY_TASKS[self.webUI.study_task_idx]['scenario']}"
                ).style('font-weight: bold;')
                ngUI.label() \
                    .bind_text_from(
                    self.webUI, 'study_tick',
                    backward=lambda _: f"Goal: {STUDY_TASKS[self.webUI.study_task_idx]['goal']}"
                ).classes('mt-1 text-gray-700')
                ngUI.label() \
                    .bind_text_from(
                    self.webUI, 'study_tick',
                    backward=lambda _: f'Starting prompt: "{STUDY_TASKS[self.webUI.study_task_idx]["prompt"]}"'
                ).classes('mt-1 text-sm text-gray-500 italic')
                ngUI.label() \
                    .bind_text_from(
                    self.webUI, 'study_tick',
                    backward=lambda _: (
                        PE_INSTRUCTIONS
                        if self._current_phase() == 'prompt_engineering'
                        else SLIDER_INSTRUCTIONS
                    )
                ).classes('mt-2 text-sm text-blue-700')

            with ngUI.row().classes('w-full justify-between items-center mt-1') \
                    .bind_visibility_from(self.webUI, 'is_main_loop_iteration'):
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

            ngUI.label('Study complete, thank you!') \
                .style('font-size: 200%; font-weight: bold;')
            ngUI.label(
                'Your results have been saved. You can now close this window.'
            ).classes('text-base text-gray-600')
            ngUI.button('Back to Demo Mode', on_click=self._go_to_demo) \
                .props('flat rounded')

    # Actions

    def _start_study(self, group: str):
        self.webUI.study_group           = group
        self.webUI.study_task_idx        = 0
        self.webUI.study_phase_is_task   = True
        self.webUI.show_demo_ui          = True
        self.webUI.study_selected_image_idx  = None
        self.webUI.study_selected_sae_idx    = None
        self.webUI.study_selected_splice_idx = None
        self.webUI.study_selected_prompt = ""
        self.webUI.study_original_images = []
        self.webUI.study_task_has_original = False
        self._apply_phase()
        print(
            f"[STUDY DEBUG] group={self.webUI.study_group}, "
            f"task={self.webUI.study_task_idx}, "
            f"phase={self.webUI.study_phase}, "
            f"show_sliders={self.webUI.show_sliders}",
            flush=True
        )
        self._prefill_prompt()
        self._bump_tick()
        self.webUI.change_state(WebUIState.INIT_STATE)

    def _apply_phase(self):
        phase = self._current_phase()
        self.webUI.study_phase  = phase
        self.webUI.show_sliders = (phase == 'sliders')

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
        self.webUI.study_selected_image_idx = None
        self.webUI.change_state(WebUIState.INIT_STATE)

    def _on_task_done(self):
        phase = self._current_phase()

        if phase == 'prompt_engineering' \
                and self.webUI.study_selected_image_idx is None:
            ngUI.notify('Please select one image as your final image first.', type='warning')
            return

        if phase == 'sliders' and (
                self.webUI.study_selected_sae_idx is None
                or self.webUI.study_selected_splice_idx is None
        ):
            ngUI.notify('Please select a final image for both SAE and Splice.', type='warning')
            return

        self._save_task_log()
        self.webUI.study_selected_image_idx  = None
        self.webUI.study_selected_sae_idx    = None
        self.webUI.study_selected_splice_idx = None

        self.webUI.study_selected_prompt = ""

        idx = self.webUI.study_task_idx

        if idx == 2:
            sus_url = SUS_FORM_PE_URL if phase == 'prompt_engineering' else SUS_FORM_SLIDERS_URL
            self._show_sus_popup(
                title='Halfway done, please fill in the first questionnaire.',
                url=sus_url,
                then=self._advance_task
            )
            return

        if idx == 5:
            sus_url = SUS_FORM_SLIDERS_URL if phase == 'sliders' else SUS_FORM_PE_URL
            self._show_sus_popup(
                title='Study complete, please fill in the final questionnaire.',
                url=sus_url,
                then=self._finish_study
            )
            return

        self._advance_task()

    def _advance_task(self):
        self.webUI.study_task_idx += 1
        self.webUI.study_original_images = []
        self.webUI.study_task_has_original = False
        self._apply_phase()
        self._prefill_prompt()
        self._bump_tick()
        self.webUI.change_state(WebUIState.INIT_STATE)

    def _finish_study(self):
        self.webUI.study_phase         = 'done'
        self.webUI.study_phase_is_task = False
        self.webUI.show_demo_ui        = False

    def _show_sus_popup(self, title, url, then):
        with ngUI.dialog() as dialog, ngUI.card().classes('p-6 items-center gap-4'):
            ngUI.label(title).style('font-size: 130%; font-weight: bold;')
            ngUI.label('Please submit the form before continuing.')
            ngUI.link('Open questionnaire ↗', url, new_tab=True) \
                .classes('text-blue-600 font-bold text-lg')
            ngUI.label('Return here once you have submitted.') \
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
        task_idx = self.webUI.study_task_idx
        phase = self._current_phase()

        out_dir = (
            Path.home() / "study_results"
            / self.webUI.study_session_id
            / f"task_{task_idx + 1:02d}_{phase}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        log_entry = {
            "timestamp":    datetime.datetime.now().isoformat(),
            "group":        self.webUI.study_group,
            "phase":        phase,
            "task_idx":     task_idx,
            "task_prompt":  STUDY_TASKS[task_idx]['prompt'],
            "prompt_used":  self.webUI.user_prompt,
        }

        # Save original images from images_sae
        for i, img in enumerate(self.webUI.study_original_images):
            p = out_dir / f"original_{i:02d}.jpg"
            img.save(str(p))
        log_entry["originals"] = [
            str(out_dir / f"original_{i:02d}.jpg")
            for i in range(len(self.webUI.study_original_images))
        ]

        if phase == 'prompt_engineering':
            idx = self.webUI.study_selected_image_idx
            p   = out_dir / "pe_final.jpg"
            self.webUI.images_sae[idx].save(str(p))
            log_entry["pe_final"]     = str(p)
            log_entry["selected_idx"] = idx
            log_entry["prompt_for_final"] = self.webUI.study_selected_prompt

        elif phase == 'sliders':
            sae_idx    = self.webUI.study_selected_sae_idx
            splice_idx = self.webUI.study_selected_splice_idx

            p_sae = out_dir / "sae_final.jpg"
            self.webUI.images_sae[sae_idx].save(str(p_sae))
            log_entry["sae_final"] = str(p_sae)
            log_entry["sae_idx"]   = sae_idx

            p_splice = out_dir / "splice_final.jpg"
            self.webUI.images_splice[splice_idx].save(str(p_splice))
            log_entry["splice_final"] = str(p_splice)
            log_entry["splice_idx"]   = splice_idx

        log_path = Path.home() / "study_results" / self.webUI.study_session_id / "log.jsonl"
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")