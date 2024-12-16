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
        'num_inference_steps' : 30, 
        'guidance_scale': 7.5, 
        'random_latents' : False,
        'use_negative_prompt' : False
    },
    'recommender' : {
        'extend_original_prompt' : True,
        'n_embedding_axis' : 10,
        'embedding_bounds' : (-1., 1.), #(0., 1.),
        'use_embedding_center' : True,
        'n_latent_axis' : 3,
        'latent_bounds' : (0., 1.),
        'use_latent_center' : True
    }
})
seed_everything(args.random_seed)
app = App(args=args)
app.start()


#if __name__ in {"__main__", "__mp_main__"}:
#    main()
