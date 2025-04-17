from nicegui import ui as ngUI
import torch
from diffusers import StableDiffusionPipeline

from prototype.webuserinterface import WebUI

global_args = None

@ngUI.page('/demo')
async def start_demo_instance():
    """
    Creates a new instance of the WebUI and runs it.
    This instance is private with the user and not shared.
    """
    global global_args
    ui = await WebUI.create(global_args)
    ui.run()

@ngUI.page('/')
def start():
    """
    Just redirects to '/demo', because '/' is the auto-index page.
    """
    ngUI.navigate.to('/demo')

class App:
    """
    The entry point into the application.
    """
    def __init__(self, args):
        global global_args
        global_args = args
        self.device = torch.device("cuda") if (global_args.device == "cuda" and torch.cuda.is_available()) else torch.device("cpu")

        global_args.generator.pipe = StableDiffusionPipeline.from_pretrained(
            global_args.hf_model_name,
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir=global_args.path.cache_dir,
            torch_dtype=torch.bfloat16,
        ).to(device=self.device)
    
    def start(self):
        """
        Start the application.
        """
        global global_args
        ngUI.run(title='Image Generation System Demo', port=global_args.port, reconnect_timeout=global_args.reconnect_timeout)
        start()
    