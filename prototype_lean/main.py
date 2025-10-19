import hydra
from prototype_lean.app import App

@hydra.main(config_path="configs", config_name="config")
def main(cfg):
    print("Initialize Application.")
    app = App(cfg)
    print("Start Application.")
    app.start()

if __name__ == "__main__":
    main()