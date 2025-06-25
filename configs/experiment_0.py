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
from configs.config_commons import common_exp0, d


experiment_name = 'Exp0/scenario/parameters'


def get_config(overriding_args: str = None):
    # load the common configuration for experiment 0:
    cfg = ConfigDict(common_exp0(experiment_name))  # load all common parameters for exp 0

    # default arguments for experiment 0, which will possibly be overwritten by the overriding_args
    cfg.sampler_name = 'cpm'
    cfg.num_mcs = 1.
    cfg.training_config.training_data_dir=os.path.join('data', 'Exp_0_jaxcpm', 'scenario_a')  #or scenario_b, etc.
    cfg.num_training_iterations_for_samplers = [10000]  # how many training step for each sampler in the samplers list
    # (most often, this is jsut a list of one sampler)

    # override the default arguments with the arguments in the overriding_args string if provided:
    if overriding_args is not None:
        cfg = utils.overwrite_config(cfg, overriding_args)

    # initialize sampler and model objects based on the possibly overwritten arguments:
    cfg.samplers = [MCMCSampler]
    cfg.sampler_configs = [
        sampler_config_exp0(cfg.sampler_name, cfg.num_mcs)
    ]


    return cfg


def sampler_config_exp0(sampler_config_key, num_mcs):
    data_shape_exp0 = (2, 200, 200)
    num_steps = int(data_shape_exp0[1] * data_shape_exp0[2] / 100 * num_mcs)  # num steps defined as number of 'monte carlo steps' from cpm literature
    temperature_schedule = optax.schedules.constant_schedule(1.0)
    dict_with_configs = dict(
        cpm=d(
            initializer=initializers.DataInitializer(), # initialize the chain from a datapoint
            transition_kernel=transition_kernels.ParallelizedCPMKernel(  # approximate cpm kernel with parallelized spin flips
                num_flip_attempts=100
            ),
            num_steps=num_steps, # num attempts to flip num_parallel_flips pixels. so total amount of spin flip attempts is this multiplied by num_flip_attempts
            temp_schedule=temperature_schedule
        ),
    )

    return dict_with_configs[sampler_config_key]
