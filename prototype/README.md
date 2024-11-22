# Prototype for a Probability-Based System

Here, the code for the prototype of a probability-based image generation system is stored.
This prototype aims to model the probability space defined by user surprise and use this to show the user images that are adapted to this individual user.

## Components

- `webuserinterface`: Main Loop and interactive web UI.
- `recommender`: Derives recommended samples for the next iteration.
- `generator`: Generate images quickly.
- `userprofilehost`: Mapping from CLIP to user profile space. Optimizer integrated, to adapt the user profile based on feedback.
