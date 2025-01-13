from prototype.app import App
import hydra

@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(args):
    app = App(args=args)
    app.start()

if __name__ in {"__main__", "__mp_main__"}:
    main()