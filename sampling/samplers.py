import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import os
import optax as tx
import subprocess
import time
import signal
from ml_collections import ConfigDict
import numpy as np
import utils
from typing import *
import sys
import matplotlib.pyplot as plt
from matplotlib import colors
import functools
from sampling import transition_kernels, initializers

jax.numpy.set_printoptions(threshold=sys.maxsize)


class MCMCSampler(eqx.Module):
    initializer: eqx.Module
    transition_kernel: eqx.Module
    num_steps: int
    temp_schedule: tx.schedules.Schedule

    def __init__(self, initializer: eqx.Module,
            transition_kernel: eqx.Module,
            num_steps: int,
            temp_schedule: tx.schedules.Schedule = tx.schedules.constant_schedule(1.0)):
        """
        MCMC Sampler class that runs a Markov Chain Monte Carlo sampler with a given initializer and transition kernel.
        :param initializer: initializer object to initialize the sampler state.
        :param transition_kernel: transition_kernel object that proposes a state, calculates acceptance probabilities,
         and performs updates accordingly
        :param num_steps: the amount of calls to the transition_kernel
        :param temp_schedule: callable, temperature schedule for the sampler; defaults to constant schedule of 1.0.
        """
        self.initializer = initializer
        self.transition_kernel = transition_kernel
        self.num_steps = num_steps
        self.temp_schedule = temp_schedule

    def init(self, key: jax.Array, **kwargs) -> Tuple[jax.Array, jax.Array, jax.Array]:
        init_state = self.initializer(key, **kwargs)
        return init_state


    def sample(self, key: jax.Array,
            energy_fn: eqx.Module, init_state: Tuple[jax.Array, jax.Array, jax.Array]) -> Tuple[jax.Array, jax.Array]:

        # possibly prepare some transition-kernel dependent quantities that need to be passed down the carry:
        init_state = self.transition_kernel.prepare_first_init(*init_state, energy_fn=energy_fn,
                                                               temperature=self.temp_schedule(0))
        run_mcmc_jit = eqx.filter_jit(
            self._run_mcmc)
        # run the sampler:
        state, metrics = run_mcmc_jit(key, init_state, self.transition_kernel, self.num_steps, energy_fn, self.temp_schedule)
        metrics['num steps'] = self.num_steps
        return state, metrics

    @staticmethod
    def _run_mcmc(sampling_key, init_state, transition_kernel, num_steps, model, temp_schedule):
        transition_keys = jax.random.split(sampling_key, num_steps)
        temperatures = jax.vmap(temp_schedule)(jnp.arange(num_steps)) # unroll temperatures beforehand
        # define the transition kernel with a specific energy function:
        tk_energy_fn = functools.partial(transition_kernel, energy_fn=model)
        # run the sampler:
        state, metrics = jax.lax.scan(tk_energy_fn, init_state, (transition_keys, temperatures))
        return state, metrics



################################### STANDARD UTILITY SAMPLERS #################################################################


def cellular_potts_sampler(x_shape, num_steps, temp_schedule=tx.schedules.constant_schedule(1.0)):
    return MCMCSampler(
        initializer=initializers.DataInitializer(),
        transition_kernel=transition_kernels.CPMKernel(
        ),
        num_steps=num_steps,
        temp_schedule=temp_schedule
    )

def parallelized_cellular_potts_sampler(x_shape, num_steps, num_parallel_flips, temp_schedule=tx.schedules.constant_schedule(1.0)):
    return MCMCSampler(
        initializer=initializers.DataInitializer(),
        transition_kernel=transition_kernels.ParallelizedCPMKernel(
            num_flip_attempts=num_parallel_flips
        ),
        num_steps=num_steps,
        temp_schedule=temp_schedule
    )

