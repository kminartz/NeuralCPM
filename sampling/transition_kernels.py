import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple
from functools import partial
import models.models
import utils

class CPMKernelBaseClass(eqx.Module):
    """
    Base class for CPM-like transition kernels that operate by copying spin values into their neighbours.
    """

    def __call__(self, *args, **kwargs):
        raise NotImplementedError('To be implemented by subclasses')

    @staticmethod
    def sample_neighbor(key, sampled_x, sampled_y, cpm, min_cell_volume=0):
        return utils.sample_neighbor(key, sampled_x, sampled_y, cpm, min_cell_volume=min_cell_volume)

    def prepare_first_init(self, cpm, e_current, boundary_mask, **kwargs):
        assert 'energy_fn' in kwargs
        energy_fn = kwargs['energy_fn']
        e_current = 1 / kwargs['temperature'] * energy_fn(cpm)[0]
        return cpm, e_current, boundary_mask

class CPMKernel(CPMKernelBaseClass):
    """
	Transition kernel for the flip of one lattice site at a time, as usual for the cpm
	"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, carry, iterate, energy_fn):
        print('compiling kernel!')
        cpm, original_energy, boundary_mask = carry
        key, temperature = iterate

        # choose a random lattice site i that lies on the boundary
        key, use_key = jax.random.split(key)
        p = boundary_mask / boundary_mask.sum()
        p_flat = p.ravel()
        sampled_x, sampled_y = jnp.unravel_index(
            jax.random.choice(key=use_key, a=jnp.arange(len(p_flat)), p=p_flat),
            p.shape)

        # sample a neighbour that is not the same value
        key, use_key = jax.random.split(key)
        neighbour_x, neighbour_y, prob_sample_neigh = self.sample_neighbor(use_key, sampled_x, sampled_y, cpm)

        # determine the proposed new state:
        cpm_proposed = cpm.at[:, neighbour_x, neighbour_y].set(cpm[:, sampled_x, sampled_y])

        # calculate the energy increment:
        delta = 1 / temperature * energy_fn.delta_energy(cpm, neighbour_x, neighbour_y, cpm[0, sampled_x, sampled_y],
                                                cpm[1, sampled_x, sampled_y], original_energy)[0]
        new_energy = original_energy + delta

        # metropolis step
        cpm, energy, accepted = jax.lax.switch(
            (delta >= 0).astype(int),
            [lambda _: (cpm_proposed, new_energy, True),  # if we have a decrease, accept
             lambda k: jax.lax.switch( # if we have an increase
                 jax.random.bernoulli(k, p=jnp.exp(-delta)).astype(int),
                 [lambda: (cpm, original_energy, False), # reject with prob 1-exp(-delta)
                  lambda: (cpm_proposed, new_energy, True)  # accept with prob exp(-delta)
                  ])
             ],key
        )

        boundary_mask = utils.update_boundary_mask(cpm, boundary_mask, neighbour_x, neighbour_y)

        state = cpm, energy, boundary_mask
        accept = accepted.astype(int)
        metrics = {'energy': energy,
                'delta': delta,
                'accept': accept,
                'flips': accept}
        return state, metrics


class ParallelizedCPMKernel(CPMKernelBaseClass):
    """
	Parallelized CPM kernel, proposing multiple spin flips, calculating the delta energy for all of them, and accepting/
	rejecting each spin flip independently.

	"""

    num_flip_attempts: int

    def __init__(self, num_flip_attempts=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_flip_attempts = num_flip_attempts

    def __call__(self, state, iterate, energy_fn):
        print('compiling kernel!')
        cpm, original_energy, boundary_mask = state
        key, temperature = iterate

        # choose a random lattice site i that lies on the boundary
        key, use_key = jax.random.split(key)
        p = boundary_mask / boundary_mask.sum()
        p_flat = p.ravel()
        # NOTE: in principle boundary_mask.sum() needs to be >= self.num_flip_attempts.
        # However, I think that if this is not true, jax.random.choice will simply sample pixels with zero probability.
        # Since these are not a boundary pixel, this is not a problem, since it does not affect the state change.
        sampled_x, sampled_y = jnp.unravel_index(
            jax.random.choice(key=use_key, a=jnp.arange(len(p_flat)), p=p_flat,
                              shape=(self.num_flip_attempts,), replace=False
                              ),
            p.shape)

        # for each of the selected sites, select a neighbor to copy into:
        key, key_neighbour_selection = jax.random.split(key, 2)
        key_neighbour_selection = jax.random.split(key_neighbour_selection, self.num_flip_attempts)
        neighbours_x, neighbours_y, prob_sample_neigh = jax.vmap(
            self.sample_neighbor, in_axes=(0,0,0,None)
        )(key_neighbour_selection, sampled_x, sampled_y, cpm)
        values_to_set = cpm[:, sampled_x, sampled_y].T

        ## calculate the energy in a more efficient way:
        delta_batched = jax.vmap(energy_fn.delta_energy, in_axes=(None, 0,0,0,0, None))
        deltas = 1 / temperature * delta_batched(cpm, neighbours_x, neighbours_y, cpm[0, sampled_x, sampled_y],
                                                cpm[1, sampled_x, sampled_y], original_energy)
        key, use_key = jax.random.split(key, 2)
        accepts = jax.random.uniform(use_key, shape=deltas.shape, minval=0., maxval=1.) < jnp.exp(-deltas)
        accepts = accepts.astype(int)

        # for those flips that were accepted, update the cpm:
        updated_spin_values = jnp.where(jnp.reshape(accepts, (-1,1)), values_to_set, cpm[:, neighbours_x, neighbours_y].T)
        cpm = cpm.at[:, neighbours_x, neighbours_y].set(updated_spin_values.T)

        #update boundary mask:
        boundary_mask = utils.update_boundary_mask_multiple_sites(cpm, boundary_mask, neighbours_x, neighbours_y)

        # calculate new energy and delta for logging:
        energy = 1 / temperature * energy_fn(cpm)[0]
        delta_true = energy - original_energy
        state = cpm, energy, boundary_mask

        metrics = {'energy': energy,
            'delta': deltas.mean(),
            'delta_only_accept': (deltas * accepts).sum() / deltas.shape[0],
            'delta_true': delta_true / deltas.shape[0],  # true delta avgd per spin flip
            'accept': accepts.mean(),
            'flips': accepts.sum()}
        # jax.debug.breakpoint()
        return state, metrics


