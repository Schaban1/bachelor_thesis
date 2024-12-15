from prototype.app import App
from prototype.utils import seed_everything
import torch
from omegaconf import DictConfig

#TODO (Discuss) How to combine hydra and multi processing?
#import hydra
#@hydra.main(version_base=None, config_path="/home/phahn/repositories/project-multimodal-machine-learning-lab-wise24/config/", config_name="config")
#def main(args):

device = "cuda" if torch.cuda.is_available() else "cpu"
args = DictConfig({
    'path' : {'cache_dir' : './cache/', 'images_save_dir': './saved_images/'}, 
    'num_recommendations' : 5, 
    'port': 2048,
    'device': device,
    'random_seed' : 42, 
    'generator' : {
        'num_inference_steps' : 25, 
        'guidance_scale':8, 
        'random_latents' : False
    },
    'recommender' : {
        'extend_original_prompt' : True,
        'embedding_bounds' : (0., 1.),
        'n_embedding_axis' : 8,
        'latent_bounds' : (0., 3.),
        'n_latent_axis' : 2,
        'use_latent_center' : True
    }
})
seed_everything(args.random_seed)
app = App(args=args)
app.start()


#if __name__ in {"__main__", "__mp_main__"}:
#    main()
