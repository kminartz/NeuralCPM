import os, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
import models.models
import jax
import numpy as np
from sampling.samplers import cellular_potts_sampler, parallelized_cellular_potts_sampler
import time
import jax.numpy as jnp
import utils
import matplotlib.pyplot as plt
import equinox as eqx
from ml_collections import ConfigDict
import argparse

def get_energy_fn(key, scenario='b', v_pref=200.):
    """
    Get the energy function for the specified scenario.
    :param key: jax PRNG key
    :param scenario: scenario to use, one of 'a', 'b', 'd', 'f' (see https://morpheus.gitlab.io/model/m2007/ for details)
    :param v_pref: preferred volume of each cell (V^*)
    :return: CellsortHamiltonian energy function for the specified scenario
    """
    cfg = ConfigDict({'num_cell_ids': 101, 'num_cell_types': 3})
    #note: morpheus uses 3.074 for Int. act. energy normalization for neighborhood 2 (https://gitlab.com/morpheus.lab/dev/morpheus/-/blob/develop/morpheus/core/cpm_shape.cpp?ref_type=heads)
    if scenario.lower() == 'b':
        energy_fn = models.models.CellsortHamiltonian(key, cfg, interaction_params={
            # keys celltypeA-celltypeB, values the interaction energy (e.g. J11, J12, ..)
            '0-0': jnp.array([0.]),
            '0-1': jnp.array([2.5]),
            '0-2': jnp.array([1.]),
            '1-1': jnp.array([1.]),
            '1-2': jnp.array([4.5]),
            '2-2': jnp.array([1.]),
        },
                                           v_pref=jnp.array([v_pref]), lamb=jnp.array([jnp.log(jnp.e**0.5 - 1)]),  # softplus(...) =0.5
                                            gamma_J=jnp.array([jnp.log(jnp.e - 1)]),  # (softplus(...) = 1
                                            bias_J=jnp.array([0.])
                                           )
    elif scenario.lower() == 'a':
        energy_fn = models.models.CellsortHamiltonian(key, cfg, interaction_params={
            # keys celltypeA-celltypeB, values the interaction energy (e.g. J11, J12, ..)
            '0-0': jnp.array([0.]),
            '0-1': jnp.array([0.5]),
            '0-2': jnp.array([0.5]),
            '1-1': jnp.array([0.333333]),
            '1-2': jnp.array([0.2]),
            '2-2': jnp.array([0.266667]), # note the higher temperature in https://morpheus.gitlab.io/model/m2007/ scenario a
        },
                                           v_pref=jnp.array([v_pref]), lamb=jnp.array([jnp.log(jnp.e**0.1 - 1)]),  # softplus(...) =0.1
                                            gamma_J=jnp.array([jnp.log(jnp.e - 1)]),  # (softplus(...) = 1
                                            bias_J=jnp.array([0.])
                                           )
    elif scenario.lower() == 'd':
        energy_fn = models.models.CellsortHamiltonian(key, cfg, interaction_params={
            # keys celltypeA-celltypeB, values the interaction energy (e.g. J11, J12, ..)
            '0-0': jnp.array([0.]),
            '0-1': jnp.array([7.5]),
            '0-2': jnp.array([4.0]),
            '1-1': jnp.array([3.5]),
            '1-2': jnp.array([2.25]),
            '2-2': jnp.array([0.5]),
        },
                                           v_pref=jnp.array([v_pref]), lamb=jnp.array([jnp.log(jnp.e**0.1 - 1)]),  # softplus(...) =0.1
                                            gamma_J=jnp.array([jnp.log(jnp.e - 1)]),  # (softplus(...) = 1
                                            bias_J=jnp.array([0.])
                                           )
    elif scenario.lower() == 'f':
        energy_fn = models.models.CellsortHamiltonian(key, cfg, interaction_params={
            # keys celltypeA-celltypeB, values the interaction energy (e.g. J11, J12, ..)
            '0-0': jnp.array([0.]),
            '0-1': jnp.array([0.1]),
            '0-2': jnp.array([0.8]),
            '1-1': jnp.array([0.7]),
            '1-2': jnp.array([0.55]),
            '2-2': jnp.array([0.2]),
            # note the higher temperature in https://morpheus.gitlab.io/model/m2007/ scenario a
        },
                                                      v_pref=jnp.array([v_pref]),
                                                      lamb=jnp.array([jnp.log(jnp.e ** 0.04 - 1)]),  # softplus(...) =0.04
                                                      gamma_J=jnp.array([jnp.log(jnp.e - 1)]),  # (softplus(...) = 1
                                                      bias_J=jnp.array([0.])
                                                      )
    else:
        raise ValueError('scenario not recognized')
    return energy_fn


def main(run_id, init_state, key, num_outer_steps, num_inner_steps, energy_fn, parallel=1):
    """
    Run a Cellular Potts Model (CPM) simulation from an initial state.
    :param run_id: run id (int)
    :param init_state: initial state of the CPM
    :param key: jax PRNG key for random number generation
    :param num_outer_steps: number of steps to save
    :param num_inner_steps: number of intermediate steps to run per outer step
    :param energy_fn: energy function to use
    :param parallel: whether to run in parallel (>0) or standard cpm sampler (0)
    :return: all_states: all states of the CPM, shape (num_outer_steps, grid_size, grid_size)
             all_energies: all energies of the CPM, shape (num_outer_steps,)
    """
    print('starting simulation', run_id, flush=True)

    key, use_key = jax.random.split(key)

    if not parallel:

        sampler = cellular_potts_sampler(**dict(
            x_shape=init_state.shape,
            # init_state=init_cpm,
            num_steps=num_inner_steps  # num spin flips attempts
            )
        )
    else:
        sampler = parallelized_cellular_potts_sampler(
            x_shape=init_state.shape,
            num_steps=int(num_inner_steps / 100),
            num_parallel_flips=100
        )

    all_states, all_energies = eqx.filter_jit(
        utils.run_cpm_from_init_state
    )(init_state, sampler, energy_fn, num_outer_steps, key)

    return all_states.astype(int), all_energies  # shape (bs, num_outer_steps, 2, grid_size, grid_size), (bs, num_outer_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run_id')
    parser.add_argument('run_id', type=int, help='run_id')
    parser.add_argument('scenario', type=str, help='scenario to run, used to select energy function')
    parser.add_argument('--parallel', type=int, help='do not use (0) or use (1) parallel approximate cpm sampler for faster generation',
                        default=1, required=False)
    args = parser.parse_args()
    run_id = args.run_id
    scenario = args.scenario.lower()
    parallel = args.parallel

    # total number of spin flip attempts is num_outer_steps * num_inner_steps
    # we save the state every num_outer_steps
    num_outer_steps = 20
    num_inner_steps = 1_000_000
    num_cells=100  # do not count medium here!
    init_radius = 60.
    v_pref=200.

    save_path = os.path.join('Cellsort_jaxCPM_data', f'scenario_{scenario}_new')
    key = jax.random.PRNGKey(run_id)
    key, use_key = jax.random.split(key)

    # get the initial state:
    init_state = utils.get_points_init_state_numpy(use_key, num_cells=num_cells,
                                                   id_to_type_dict={i:(i-1) // (num_cells / 2) + 1 for i in range(1,num_cells+1)},
                                      grid_shape=(200,200), init_radius=init_radius)
    # get energy function for the specified scenario:
    energy_fn = get_energy_fn(use_key, scenario, v_pref=v_pref)

    # run the simulation:
    start = time.time()
    all_states, all_energies = main(run_id, init_state, use_key, num_outer_steps, num_inner_steps, energy_fn,
                                    parallel=parallel)
    stop = time.time()
    print('Took ', stop-start, ' seconds for ', num_outer_steps, ' outer steps and ', num_inner_steps, ' inner steps (total spin flip attempts: ', num_outer_steps*num_inner_steps, ')')
    # save the data to disk:
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    p = os.path.join(save_path, f'all_cpms_{run_id}.npz')
    np.savez_compressed(p, data=all_states)

    print('done -- data saved at', save_path, flush=True)

