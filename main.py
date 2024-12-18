from prototype.app import App
from prototype.utils import seed_everything
import hydra

@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(args):
    seed_everything(args.random_seed)
    app = App(args=args)
    app.start()

if __name__ in {"__main__", "__mp_main__"}:
    main()
