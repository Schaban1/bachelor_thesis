import hydra
from app import App

@hydra.main(config_path="configs", config_name="config")
def main(cfg):
    print("Initialize Application.")
    app = App(cfg)
    print("Start Application.")
    app.start()

if __name__ in {"__main__", "__mp_main__"}:
    main()