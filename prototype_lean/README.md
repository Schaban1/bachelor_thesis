# Concept Sliders System - Splice

## Guide for running the system

- `image_mean` file: There needs to be a correct path to the 
`image_mean`file so that the custom backbone of Splice is 
correctly set up. Adjust the path to the `image_mean` 
file in **line 28** of `splice_custom.py` if necessary.

- Use the following image when running the container:
```
--container-image=schaban1/concept-slider:latest
```

- After you have cloned the repo, use this command, which enables navigation to the 
encapsulated `prototype_lean` folder, to deploy the WebUI.
```
bash -c "cd ~/bachelor_thesis/prototype_lean && python3 main.py"
```

## Overview

- `webuserinterface`: Contains the WebUI, leaner version of 
prototype, where `slider.py` handles slider logic of sliders in 
`main_loop_ui.py`.
- `generator.py`: Contains two different image generation pipes
one standard pipe for initial image generation and one ip_pipe
which is used to edit the images when adjusting the
sliders.
- `splice_custom.py`: Custom backbone for Splice because we use
a CLIP model that needs compatibility with **ip_pipe**.
- `concept_extractor.py`: Uses the method `encode_image(image)` from Splice repo
to extract top k=3 concepts.
- `image_editor.py`: Uses the methods `encode_image(image)` 
and from Splice repo `recompose_image(weights)`
to output edited image.