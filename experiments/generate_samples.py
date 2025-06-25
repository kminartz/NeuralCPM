import sys
sys.path.append('..')
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import time
import equinox as eqx
import os
import sampling.samplers as samplers
import matplotlib.pyplot as plt
import sampling.initializers as initializers

import utils
KEY = jax.random.PRNGKey(43)

def generate_samples(config, weight_path, data_dir, initializer='PermuteTypeInitializer'):

    run_model_save_trajectory(config, weight_path, data_dir,
                              num_inner_steps=134,
                              num_outer_steps=100,
                              num_parallel_flips=50,
                              num_trajectories_batched=10,
                              num_batches=5,
                              initializer=initializer
                              )


def run_model_save_trajectory(config, weight_path, data_dir, num_inner_steps, num_outer_steps, num_parallel_flips,
                              num_trajectories_batched=1, num_batches=1, initializer='PermuteTypeInitializer'):
    """
    Run the model and save the trajectory samples. total number of spin flips is num_inner_steps * num_outer_steps * num_parallel_flips.
    :param config: ConfigDict specificying the expimeriment configuration.
    :param weight_path: model weight path
    :param data_dir: directory of training data which we randomize from to initialize the sampler
    :param num_inner_steps: number of steps per outer step in the sampler
    :param num_outer_steps: number of steps to save per trajectory -- num_inner steps are ran in between each outer step
    :param num_parallel_flips: number of parallel flips to perform in the sampler
    :param num_trajectories_batched: batch size of sim
    :param num_batches: number of batches to run.
    :param initializer: initializer string, should be PermuteTypeInitializer unless you have a specific reason to change it.
    :return: None, saves the samples and energies to disk.
    """
    initializer_str = initializer
    key, use_key = jax.random.split(KEY)

    # load the model:
    model = utils.initialize_model(config, use_key, **config.model_config)
    model = utils.load_model_weights(config, model, path=weight_path)
    all_states, all_energies = [], []
    try:
        print('model loaded -- model weight basis/neural', model.weight_basis, model.weight_neural, flush=True)
    except:
        print('model loaded', flush=True)
    # do sampler initialization and run the sampler from the initial states, and save intermediate outputs
    run_fn = eqx.filter_jit(
        jax.vmap(
            utils.run_cpm_from_init_state, in_axes=(0, None, None, None, 0)
        )
    )

    # load some data for the initialization of the sampler
    for batch in range(num_batches):
        key, use_key = jax.random.split(key)
        some_data = []
        files = os.listdir(data_dir)
        selected_files = np.random.choice(files, num_trajectories_batched)
        for f in selected_files:
            some_data.append(
                utils.load_data_from_file(os.path.join(data_dir, f))[-1][None, ...].astype(int)
            ) # now shape (1, 2, grid_size, grid_size)
        some_data = np.concatenate(some_data, axis=0)

        # PRNG key logistics:
        key, init_key, sampling_key = jax.random.split(use_key, 3)
        init_keys = jax.random.split(init_key, num_trajectories_batched)
        sampling_keys = jax.random.split(sampling_key, num_trajectories_batched)

        start = time.time()
        # initialize the sampler:
        sampler = samplers.parallelized_cellular_potts_sampler(some_data[0].shape, num_inner_steps, num_parallel_flips)

        id_to_type_vecs = jax.vmap(utils.get_id_to_type_vec, in_axes=(0, None))(some_data, config.num_cell_ids)

        # get the initial state:
        initializer = getattr(initializers, initializer_str)()  # PermuteTypeInitializer creates a randomized input as initial state
        init_state, *_ = jax.vmap(initializer)(init_keys, x=some_data,
                                                                             x_cell_type_vec=id_to_type_vecs)

        states_this, energies_this = run_fn(init_state, sampler, model, num_outer_steps, sampling_keys)  # bs, t, c, h, w
        states_this = np.concatenate([init_state[:, None, ...], states_this], axis=1)
        all_states.append(states_this)
        all_energies.append(energies_this)
        print('batch', batch, 'took:', time.time()-start, 'seconds', flush=True)
        # shape (bs, num_outer_steps, 2, grid_size, grid_size), (bs, num_outer_steps)

    all_states = np.concatenate(all_states, axis=0)
    all_energies = np.concatenate(all_energies, axis=0)

    spath = os.path.join('trajectory_samples', config.experiment_name, os.path.basename(weight_path))
    if not os.path.exists(spath):
        os.makedirs(spath)

    # save the samples as npz file:
    np.savez_compressed(
        os.path.join(spath, f'samples_{initializer_str}_{num_inner_steps}_{num_outer_steps}_{num_parallel_flips}.npz'),
                        all_samples=all_states, all_energies=all_energies
    )

    scale_figsize = 1.5
    fig, axes = plt.subplots(nrows=num_trajectories_batched, ncols=num_outer_steps,
                             figsize=(num_outer_steps*scale_figsize, num_trajectories_batched*scale_figsize), squeeze=False)

    # plot one batch:
    for i in range(num_trajectories_batched):
        for j in range(num_outer_steps):
            ax = axes[i, j]
            sample = all_states[i, j, 1]
            ax.imshow(sample, cmap='plasma')
            ax.axis('off')

    plt.savefig(
        os.path.join(spath, f'samples_{initializer_str}_{num_inner_steps}_{num_outer_steps}_{num_parallel_flips}.png')
    )
    plt.tight_layout()
    plt.show()

    # now plot the energies:
    fig, axs = plt.subplots(ncols=num_trajectories_batched, squeeze=False)
    for i in range(num_trajectories_batched):
        ax = axs[0, i]
        ax.plot(all_energies[i], label=f'trajectory {i}')
    plt.savefig(
        os.path.join(spath, f'energies_{initializer_str}_{num_inner_steps}_{num_outer_steps}_{num_parallel_flips}.png')
    )
    plt.show()



    return









if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Config and hyperparameters')
    parser.add_argument('cfg_str', type=str, help='config string')
    parser.add_argument('weight_path', type=str, help='path to file weights')
    args, remaining_args = parser.parse_known_args()

    cfg_str = args.cfg_str
    weight_path = args.weight_path
    initializer = args.initializer

    cfg = utils.load_config(cfg_str, remaining_args=remaining_args)


    data_path = os.path.join('..', cfg.training_config.training_data_dir)


    generate_samples(cfg, weight_path, data_dir=data_path)











