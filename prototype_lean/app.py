import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL
from nicegui import ui as ngUI
from splice_custom import get_splice_model, VLMBackbone
from generator import Generator
from webuserinterface.webuserinterface import WebUI
from utils import ProducerConsumer

'''
class ProducerConsumer:
    def __enter__(self): pass
    def __exit__(self, exc_type, exc_val, exc_tb): pass
'''

global_args = None
pipe = None
queue_lock = ProducerConsumer()

@ngUI.page('/demo')
async def start_demo_instance():
    global global_args, pipe, generator
    ui = await WebUI.create(global_args, pipe, generator, queue_lock)
    ui.run()

@ngUI.page('/')
def start():
    ngUI.navigate.to('/demo')

class App:
    def __init__(self, args):
        global global_args
        global_args = args
        self.device = torch.device("cuda") if (
                global_args.device == "cuda" and torch.cuda.is_available()) else torch.device("cpu")

        global pipe
        pipe = StableDiffusionPipeline.from_pretrained(
            args.hf_model_name,
            requires_safety_checker=True,
            cache_dir=args.path.cache_dir,
            torch_dtype=torch.bfloat16,
        ).to(device=self.device)
        pipe.unet = torch.compile(pipe.unet, backend="cudagraphs")
        pipe.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device=pipe.device, dtype=pipe.dtype)
        pipe.vae = torch.compile(pipe.vae, backend="cudagraphs")
        global generator
        generator = Generator(
            cache_dir=args.path.cache_dir,
            device=args.device,
            hf_model_name=args.hf_model_name,
            pipe=pipe,
            **args.generator,
        )
        generator.splice = get_splice_model()
        generator.vlm_backbone = VLMBackbone()

    def start(self):
        global global_args
        ngUI.run(title='Image Generation System Demo', port=global_args.port, reconnect_timeout=global_args.reconnect_timeout, reload=False)
        start()