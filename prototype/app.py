from nicegui import ui as ngUI
import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL

from prototype.webuserinterface import WebUI

global_args = None
pipe = None

@ngUI.page('/demo')
async def start_demo_instance():
    """
    Creates a new instance of the WebUI and runs it.
    This instance is private with the user and not shared.
    """
    global global_args
    global pipe
    ui = await WebUI.create(global_args, pipe)
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

        global pipe
        pipe = StableDiffusionPipeline.from_pretrained(
            global_args.hf_model_name,
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir=global_args.path.cache_dir,
            torch_dtype=torch.bfloat16,
        ).to(device=self.device)

        pipe.unet = torch.compile(self.pipe.unet, backend="cudagraphs")

        pipe.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device=self.pipe.device, dtype=self.pipe.dtype)
        pipe.vae = torch.compile(self.pipe.vae, backend="cudagraphs")
    
    def start(self):
        """
        Start the application.
        """
        global global_args
        ngUI.run(title='Image Generation System Demo', port=global_args.port, reconnect_timeout=global_args.reconnect_timeout)
        start()
    