from nicegui import ui as ngUI
from nicegui import binding
from PIL import Image
from constants import WebUIState, ScoreMode
from .components import InitialIterationUI, LoadingUI, MainLoopUI, SliderController
import torch
import os

class WebUI:
    session_id = binding.BindableProperty()
    state = binding.BindableProperty()
    is_initial_iteration = binding.BindableProperty()
    is_main_loop_iteration = binding.BindableProperty()
    is_generating = binding.BindableProperty()
    user_prompt = binding.BindableProperty()
    num_images_to_generate = binding.BindableProperty()
    score_mode = binding.BindableProperty()
    image_display_width = binding.BindableProperty()
    image_display_height = binding.BindableProperty()

    @classmethod
    async def create(cls, args, pipe, generator, queue_lock):
        self = cls()
        loading_label = ngUI.label("Starting session...")
        await ngUI.context.client.connected()
        self.args = args
        self.pipe = pipe
        self.generator = generator
        self.queue_lock = queue_lock
        self.session_id = "demo"
        self.state = None
        self.is_initial_iteration = False
        self.is_generating = False
        #self.iteration = 0
        self.user_prompt = ""
        self.num_images_to_generate = self.args.num_recommendations
        self.score_mode = self.args.score_mode
        self.image_display_width, self.image_display_height = tuple(self.args.image_display_size)
        self.images = [Image.new('RGB', (self.image_display_width, self.image_display_height)) for _ in range(self.num_images_to_generate)]
        self.images_display = [None for _ in range(self.num_images_to_generate)]
        self.slider_containers = []
        self.slider_controller = SliderController(self, generator.splice, generator)
        self.setup_root()
        loading_label.delete()
        return self

    def run(self):
        print("Start running the Web UI.")
        self.change_state(WebUIState.INIT_STATE)
        self.root.clear()
        self.build_userinterface()

    def change_state(self, new_state: WebUIState):
        self.state = new_state
        self.update_state_variables()

    def update_state_variables(self):
        self.is_initial_iteration = self.state == WebUIState.INIT_STATE
        self.is_main_loop_iteration = self.state == WebUIState.MAIN_STATE
        self.is_generating = self.state == WebUIState.GENERATING_STATE

    def build_userinterface(self):
        print("Building User Interface.")
        webis_template_top, webis_template_bottom = self.get_webis_demo_template_html()
        with self.root:
            ngUI.html(webis_template_top).classes('w-full')
            ngUI.space().classes('w-full h-full')
            InitialIterationUI(self)
            self.main_loop_ui = MainLoopUI(self)
            self.loading_ui = LoadingUI(self)
            ngUI.space().classes('w-full h-full')
            ngUI.html(webis_template_bottom).classes('w-full')

    def setup_root(self):
        self.root = ngUI.column().classes('w-full h-full').style('font-family:"Product Sans","Noto Sans","Verdana", sans-serif;')
        ngUI.add_head_html('<style>.nicegui-content { padding: 0; }</style>')
        ngUI.query('.nicegui-content').classes('w-full')
        ngUI.query('.q-page').classes('flex')

    def generate_images(self):
        print("Generate new Images.")
        embeddings = self.pipe.encode_prompt(
            self.user_prompt, device=self.pipe.device, num_images_per_prompt=self.num_images_to_generate, do_classifier_free_guidance=False
        )[0]
        latents = torch.randn(
            (self.num_images_to_generate, self.pipe.unet.config.in_channels, 512 // 8, 512 // 8),
            device=self.pipe.device
        ) * self.pipe.scheduler.init_noise_sigma
        self.images = self.generator.generate_image(embeddings, latents, self.loading_ui.loading_progress, self.queue_lock)
        print("[DEBUG webuserinterface: were the images generated?",flush=True)
        self.slider_controller.on_images_generated(self.images)


    def get_webis_demo_template_html(self):
        script_dir = os.path.dirname(__file__)
        top_path = os.path.join(script_dir, "webis_template_top.html")
        bottom_path = os.path.join(script_dir, "webis_template_bottom.html")
        with open(top_path) as f:
            top = f.read()
        with open(bottom_path) as f:
            bottom = f.read()
        return top, bottom