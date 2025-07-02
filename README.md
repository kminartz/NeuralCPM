# NeuralCPM: Deep Neural Cellular Potts Models
This is the repository accompanying the paper [Deep Neural Cellular Potts Models](https://openreview.net/forum?id=3xznpzabYQ).


# Setup
To download the code and install the dependencies, run:

```
git clone https://github.com/kminartz/NeuralCPM
pip install -r requirements.txt
pip install jax==0.4.30
```

For GPU acceleration, you may need to adapt the jax installation above, see [these instructions](https://docs.jax.dev/en/latest/installation.html).

You can run the following commands to quickly check if your install seems to work correctly:

```
python -m unittest models/test_models.py
python -m unittest sampling/test_samplers.py
```

# Data

Datasets can be generated using the bash scripts in the 
data_generation directory.
The synthetic data generation for the bi-polar axial organization 
experiment requires the open source software Morpheus to be installed 
from https://morpheus.gitlab.io and then the model file 
main/data_generation/Morpheus_model_bipolar.xml can be run from command line, 
script or GUI.


# Training

To train a model, run a command like below:

```
python train_ebm.py <config_name> --<change_parameter>=<new_value>
```

where <config_name> is a python module name from the configs directory, and <change_parameter> and <new_value> optionally specify parameters to be changed to their new value. For example:

```
python train_ebm.py experiment_1 --model_name=nh --num_mcs=0.25
```

Please consult the config files for more details on parameters.


# Evaluation

To generate simulations for evaluation, run 

```
cd experiments
python generate_samples.py <config> <model_weight_path> --model_name=<model_name_value>
```

where <model_weight_path> is the location of the trained model weights relative to the experiments directory and <model_name_value> is the corresponding name of the model to load the weights into (see configs for details). Generation may take some time depending on model and hardware. Analysis of the results is done in experiments/experiment_*.ipynb.

---

If you found this work interesting, consider citing our paper:

```
@inproceedings{
minartz2025neuralcpm,
title={Deep Neural Cellular Potts Models},
author={Koen Minartz and Tim d'Hondt and Leon Hillmann and J{\"o}rn Starru{\ss} and Lutz Brusch and Vlado Menkovski},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=3xznpzabYQ}
}
```

