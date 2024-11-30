from prototype.webuserinterface import WebUI

#import hydra
from omegaconf import DictConfig


#@hydra.main(version_base=None, config_path="/home/phahn/repositories/project-multimodal-machine-learning-lab-wise24/config/", config_name="config")
#def main(args):
args = DictConfig({'path' : {'cache_dir' : './cache/'}, 'num_recommendations' : 4})
ui = WebUI(args=args)
ui.run()


#if __name__ in {"__main__", "__mp_main__"}:
#    main()