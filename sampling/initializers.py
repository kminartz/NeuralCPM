import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Tuple

import numpy as np

import utils


class InitializerBaseClass(eqx.Module):

    def __call__(self, key, x, **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """

		:param key: prng key
		:param x: the initial state of the sampler, e.g. a datapoint, or something else.
		:param kwargs: any keyword arguments required for the initializer. These will depend on the initialization technique.
		should return cpm, energy (possibly inf), boundary_mask
		"""
        raise NotImplementedError


class DataInitializer(InitializerBaseClass):
    '''
        Initialize the sampler from a provided datapoint x
    '''

    def __call__(self, key, x, **kwargs):
        # x is the datapoint to serve as initial state for the sampler
        x = jnp.asarray(x)  # ensure x is a jnp array
        out = x, jnp.inf, utils.create_boundary_mask(x)
        return out

class PersistentInitializer(InitializerBaseClass):
    '''
        Keep a running persistent chains across training steps, aka 'persistent sampler' from Tielemans and Hinton
        re-initialize the chain with small probability using a different initializer object.

    '''
    initializer: InitializerBaseClass = DataInitializer()
    p_reset: float = 0.025

    def __call__(self, key, x, previous_state, force_init=False, **kwargs):
        x = jnp.asarray(x)  # ensure x is a jnp array
        previous_state = jnp.asarray(previous_state)  # ensure previous_state is a jnp array
        key, use_key = jax.random.split(key)
        if force_init:
            reset = True
        else:
            reset = jax.random.bernoulli(use_key, self.p_reset)
        x = jax.lax.cond(reset,
                         lambda: self.initializer(key, x=x, previous_state=previous_state, **kwargs)[0],
                         lambda: previous_state)
        out = x, jnp.inf, utils.create_boundary_mask(x)
        return out

class PermuteTypeInitializer(InitializerBaseClass):
    '''
        Permute the types of the cells randomly
    '''

    def __call__(self, key, x, x_cell_type_vec, **kwargs):
        x = jnp.asarray(x)  # ensure x is a jnp array
        current_cell_types = jnp.asarray(x_cell_type_vec)  # vector of length num_cell_ids which indicates the type of each cell

        # create a random id_to_type dict
        key, use_key = jax.random.split(key)
        new_types_excl_medium = jax.random.permutation(use_key, current_cell_types[1:])  # first element is always medium
        new_types = jnp.concatenate([
            jnp.array([0]), new_types_excl_medium
        ], axis=0)  # add medium back in

        new_type_channel = jnp.zeros_like(x[1])
        ii, jj = jnp.meshgrid(jnp.arange(x.shape[1]), jnp.arange(x.shape[2]))
        i, j = ii.flatten(), jj.flatten()
        new_type_channel = new_type_channel.at[i, j].set(new_types[x[0][i,j]])

        x = x.at[1].set(new_type_channel)

        out = x, jnp.inf, utils.create_boundary_mask(x)
        return out


