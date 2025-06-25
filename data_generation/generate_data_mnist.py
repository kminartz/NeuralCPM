import os
import sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, ".."))

import jax
import numpy as np
from sampling.samplers import (
    cellular_potts_sampler,
)
import time
import jax.numpy as jnp
import utils
import equinox as eqx
from ml_collections import ConfigDict
import argparse
import mnist
from scipy import ndimage
from models.models import ExternalFieldHamiltonian



def main(run_id, init_state, key, num_outer_steps, num_inner_steps, energy_fn):
    """
    Run a Cellular Potts Model (CPM) simulation from an initial state.
    :param run_id: run id (int)
    :param init_state: initial state of the CPM
    :param key: jax PRNG key for random number generation
    :param num_outer_steps: number of steps to save
    :param num_inner_steps: number of intermediate steps to run per outer step
    :param energy_fn: energy function to use
    :return: all_states: all states of the CPM, shape (num_outer_steps, grid_size, grid_size)
             all_energies: all energies of the CPM, shape (num_outer_steps,)
    """
    print("starting simulation", run_id, flush=True)
    # code:
    key, _ = jax.random.split(key)

    sampler = cellular_potts_sampler(
        **dict(
            x_shape=init_state.shape,
            num_steps=num_inner_steps,  # num spin flips attempts
        )
    )

    all_states, all_energies = eqx.filter_jit(utils.run_cpm_from_init_state)(
        init_state, sampler, energy_fn, num_outer_steps, key
    )
    return (
        all_states.astype(int),
        all_energies,
    )


def load_mnist_field(size: int, digit: int) -> jnp.ndarray:
    """
    Download the MNIST dataset and create distance transform fields for a given digit.

    This will take ~20 seconds.
    """
    mnist.datasets_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"

    labels, images = (mnist.test_labels(), mnist.test_images())
    digit_images = images[labels == digit]

    im_idx = np.random.randint(0, digit_images.shape[0])
    image = digit_images[im_idx]

    field = jnp.array(
        ndimage.zoom(
            ndimage.distance_transform_edt(image < 128), size / 28
        )
    ).astype(float)
    return field


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run_id")
    parser.add_argument("run_id", type=int, help="run_id")
    args = parser.parse_args()
    run_id = args.run_id

    # total number of spin flip attempts is num_outer_steps * num_inner_steps
    # we save the state every num_outer_steps
    num_outer_steps = 20
    num_inner_steps = 8_000 #20_000  # 1_000_000
    grid_size = 100
    num_cells = 50
    init_radius = 25.0
    digit = np.random.randint(0, 10)
    field = load_mnist_field(grid_size, digit)

    save_path = os.path.join("MNIST_jaxCPM_data")
    key = jax.random.PRNGKey(run_id)
    key, use_key = jax.random.split(key)

    # get initial state for the simulation
    init_state = utils.get_points_init_state_numpy(
        use_key,
        num_cells=num_cells,
        id_to_type_dict={
            i: (i - 1) // (num_cells / 2) + 1 for i in range(1, num_cells + 1)
        },
        grid_shape=(100, 100),
        init_radius=init_radius,
    )
    cfg = ConfigDict({'num_cell_ids': 51, 'num_cell_types': 3})

    # get the energy function - imposing an external field following the shape of the MNIST digit
    energy_fn = ExternalFieldHamiltonian(
        key,
        cfg,
        field=field,
        field_coupling=jnp.array([0.0, 0.0, 10.0]),
        interaction_params={
            # keys celltypeA-celltypeB, values the interaction energy (e.g. J11, J12, ..)
            "1-1": jnp.array([3.0]),
            "1-2": jnp.array([8.0]),
            "0-0": jnp.array([0.0]),
            "0-1": jnp.array([6.0]),
            "0-2": jnp.array([6.0]),
            "2-2": jnp.array([3.0]),
        },
        v_pref=jnp.array([100.0]),
        lamb=jnp.array([0.5]),
    )

    # start the simulation
    start = time.time()
    all_states, all_energies = main(
        run_id, init_state, use_key, num_outer_steps, num_inner_steps, energy_fn
    )
    stop = time.time()
    print(
        "Took ",
        stop - start,
        " seconds for ",
        num_outer_steps,
        " outer steps and ",
        num_inner_steps,
        " inner steps (total spin flip attempts: ",
        num_outer_steps * num_inner_steps,
        ")",
    )
    # save the data to disk:
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    p = os.path.join(save_path, f"all_cpms_{run_id}.npz")
    np.savez_compressed(p, data=all_states, label=digit)

    print("done -- data saved at", save_path, flush=True)
