from prototype.app import App
import hydra
import logging

# Press F9 in the app to access the debug menu!
@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(args):
    logging.info("Initialize Application.")
    app = App(args=args)
    logging.info("Start Application.")
    app.start()

if __name__ in {"__main__", "__mp_main__"}:
    main()
