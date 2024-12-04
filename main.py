from prototype.webuserinterface import WebUI
from prototype.utils import seed_everything


from omegaconf import DictConfig

#TODO (Discuss) How to combine hydra and multi processing?
#import hydra
#@hydra.main(version_base=None, config_path="/home/phahn/repositories/project-multimodal-machine-learning-lab-wise24/config/", config_name="config")
#def main(args):

seed_everything(42)
args = DictConfig({'path' : {'cache_dir' : './cache/'}, 'num_recommendations' : 5})
ui = WebUI(args=args)
ui.run()


#if __name__ in {"__main__", "__mp_main__"}:
#    main()