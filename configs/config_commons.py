import numpy as np
from ml_collections import ConfigDict
from sampling.samplers import MCMCSampler
from sampling import transition_kernels
from sampling import initializers
import utils
import os
import models.models as models
import optax
import jax
import jax.numpy as jnp
import equinox.nn as nn
import equinox as eqx

def d(**kwargs):
    return ConfigDict(initial_dictionary=kwargs)


def common_exp0(experiment_name: str):  # fitting cellsort parameters
    cfg = ConfigDict()
    cfg.experiment_name = experiment_name
    cfg.seed = 42
    cfg.num_cell_ids = 101  # 101 -- 100 cells + 1 background cell
    cfg.num_cell_types = 3  # two cell types + 1 background type
    cfg.grid_size = (200, 200)  # resolution

    cfg.model = models.CellsortHamiltonian
    cfg.model_config = d(
        v_pref=200.,# NOTE: putting things here as float instead of jax array -> will be fixed and ignored by optimizer!
        gamma_J=jnp.array([jnp.log(jnp.e - 1)]),  # contact energies is softplus(gamma_J) * J + bias_J
        bias_J=jnp.array([2.])
    )

    cfg.training_config = d(
        model_weight_path=os.path.join(
            utils.make_dir_safe(os.path.join('model_weights', experiment_name)
                                ),'experiment_0_model.eqx'),
        optimizer=optax.adam,
        optimizer_config=d(learning_rate=1e-3, ),
        training_data_bs=16,
        generated_data_bs=16,
        energy_l2_reg_lambda=0.0,
    )

    cfg.log_frequency = 100
    cfg.model_metrics_to_log = 'all'
    cfg.sampler_metrics_to_log = 'all'
    cfg.data_metrics_to_log = 'all'
    cfg.log_to_wandb = True  # whether to log to wandb

    return cfg

def common_exp1(experiment_name:str):
    cfg = ConfigDict()
    cfg.experiment_name = experiment_name

    cfg.seed = 43

    cfg.num_cell_ids = 51  # 51 -- 50 cells + 1 background cell
    cfg.num_cell_types = 3  # two cell types + 1 background type
    cfg.grid_size = (100,100)  # resolution

    optimizer_schedule = optax.schedules.constant_schedule(1e-3)

    cfg.training_config = d(
        training_data_dir=os.path.join('data', 'MNIST_jaxCPM_data'),
        model_weight_path=os.path.join(
            utils.make_dir_safe(os.path.join('model_weights', experiment_name)), 'experiment_1_model_MNIST.eqx'
        ),
        optimizer=optax.adam,
        optimizer_config=d(learning_rate=optimizer_schedule,),
        training_data_bs=16,
        generated_data_bs=16,
        energy_l2_reg_lambda=0.0005,
        ema_model_weight=0.99,
        replay_buffer_size=None
    )

    cfg.log_frequency = 50
    cfg.model_metrics_to_log = 'all'
    cfg.sampler_metrics_to_log = 'all'
    cfg.data_metrics_to_log = 'all'
    cfg.log_to_wandb = True  # set to True to enable wandb logging

    return cfg

def common_exp2(experiment_name:str):
    cfg = ConfigDict()
    cfg.experiment_name = experiment_name

    cfg.seed = 43

    cfg.num_cell_ids = 41  # 41 -- 40 cells + 1 background cell
    cfg.num_cell_types = 3  # two cell types + 1 background type
    cfg.grid_size = (125,125)

    cfg.training_config = d(
        training_data_dir=os.path.join('data', 'Exp_2_toda_padded'),
        model_weight_path=os.path.join(
            utils.make_dir_safe(os.path.join('model_weights', experiment_name)), 'experiment_2_model_toda.eqx'
        ),
        optimizer=optax.adam,
        optimizer_config=d(learning_rate=1e-3,),
        training_data_bs=16,
        generated_data_bs=16,
        energy_l2_reg_lambda=0.0005,
        rotate_augm=True,
        ema_model_weight=0.99
    )

    cfg.log_frequency = 50
    cfg.model_metrics_to_log = 'all'
    cfg.sampler_metrics_to_log = 'all'
    cfg.data_metrics_to_log = 'all'
    cfg.log_to_wandb = True  # set to True to enable wandb logging

    return cfg








