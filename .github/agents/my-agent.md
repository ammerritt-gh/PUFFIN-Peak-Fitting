---
name: BigFit Agent
description: Gives explanations to copilot agents for BigFit structure.
---

# My Agent

Describe what your agent does here...

main_window.py will handle user-presented GUI work only, without calculations or logic.
view_box.py controls the custom viewbox for the main plot.
fitter_vm.py handles the intermediate logic, interfacing between the GUI and the models/workers.
model_specs.py keeps the models and their information.
model_state.py holds the state of the model while it is being used.
fit_worker.py handles the fitting logic.
The dataio folder holds the logic for IO operations, such as saving and loading data or configuration files.
