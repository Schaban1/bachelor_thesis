# Concept Sliders System

## Guide for running the system

- **EDIT**: Use the following image when running the container:
```
--container-image=schaban1/bachelor:latest
```

- After you have cloned the repo, use this command, which enables navigation to the 
encapsulated `prototype_lean` folder, to deploy the WebUI.
```
bash -c "cd ~/bachelor_thesis/prototype_lean && python3 main.py"
```

## Overview

- `main.py` starts the Hydra-configured application
- `app.py` initializes the Stable Diffusion pipeline, generator, and NiceGUI server
- `generator.py` loads Stable Diffusion, LCM-LoRA, the trained SAE checkpoint, and the Splice model and performs initial and edited image generation
- `splice_custom.py` constructs the Splice model in the Stable Diffusion text-conditioning space
- `concept_extractor.py` extracts prompt-level Splice concepts and image-specific SAE concepts
- `image_editor.py` applies Splice coefficient edits and SAE-selected text-direction edits and regenerates individual images
- `webuserinterface` contains the NiceGUI demo interface, user-study workflow, image displays, slider controls, and questionnaire handling
- `configs/config.yaml` contains model, generation, device, display, and output settings
- `resources` contains the trained SAE checkpoint and the mapping from SAE unit indices to concept names

The application provides a demo mode for exploration and a study mode comparing iterative prompt revision with the concept-slider workflow.