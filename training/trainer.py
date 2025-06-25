import os
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import utils
import time
from functools import partial
import optax
import matplotlib.pyplot as plt
import datetime
from scipy.ndimage import rotate as rotation_fn

class Trainer:

    def __init__(self, config):
        self.cfg = config
        self.training_cfg = self.cfg.training_config
        assert self.training_cfg.training_data_dir
        self.key = jax.random.key(self.cfg.seed)
        self.key, use_key = jax.random.split(self.key)


        # initialize model
        model = utils.initialize_model(self.cfg, key=use_key, **self.cfg.model_config)
        self.model = model

        # initialize model optimizer
        self.optimizer = self.training_cfg.optimizer(**self.training_cfg.optimizer_config)
        # we optimize all leafs that return true for eqx.is_array, i.e. all parameters of the model that are np or jnp arrays
        self.opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))
        self.energy_l2_reg_lambda = self.training_cfg.energy_l2_reg_lambda if hasattr(
            self.training_cfg, 'energy_l2_reg_lambda'
        ) else 0.

        self.ema_model_weight = self.training_cfg.ema_model_weight if hasattr(
            self.training_cfg, 'ema_model_weight'
        ) else 0.99  # ema is implemented, but normally we used the normal model weights

        # initialize logger
        self.logger = utils.Logger(
            project='NeuralCPM',
            wandb_config=self.cfg.to_dict(),
            run_name=self.cfg.experiment_name,
            model_metrics_to_log=self.cfg.model_metrics_to_log,
            sampler_metrics_to_log=self.cfg.sampler_metrics_to_log,
            data_metrics_to_log=self.cfg.data_metrics_to_log)

        return

    def run_train_loop(self):
        """
        Train the model

        """

        model = self.model
        ema_model = eqx.filter(self.model, True)  # copy of original model for intiialization
        opt_state = self.opt_state
        sampler_states = None

        # run the samplers sequentially according to the specified schedule -- typically, just one sampler
        for sampler_def, sampler_cfg, num_training_iter in zip(self.cfg.samplers, self.cfg.sampler_configs,
                                                           self.cfg.num_training_iterations_for_samplers):
            self.key, use_key = jax.random.split(self.key)
            # init sampler object
            sampler = sampler_def(**sampler_cfg)

            for i, k in enumerate(jax.random.split(use_key, num_training_iter)):

                # run a training iter, which updates the model weights, optimizer state, and sampler states
                model, opt_state, sampler_states, ema_model = self.train_iter(
                    k, i, model, sampler, opt_state, sampler_states, ema_model)

        self.logger.run.finish()


    def train_iter(self, key, iteration, model, sampler, opt_state, sampler_states,
                   ema_model):
        """
        Sample a batch of real data and a batch of synthetic data, compute the
        gradient of the MLE objective approximation, and perform a model update step

        """
        tic = time.time()
        # load training and generated data:
        real_data = self.load_data_batch(
            max(self.training_cfg.training_data_bs, self.training_cfg.generated_data_bs),
            self.training_cfg.training_data_dir,
            t=self.training_cfg.load_timestep if 'load_timestep' in self.training_cfg.keys() else -1,
            rotate=self.training_cfg.rotate_augm if 'rotate_augm' in self.training_cfg.keys() else False
        )

        if sampler_states is None:
            sampler_states = real_data  # if we dont have any sampler states, we initialize them with the real data


        self.key, init_key, sampling_key = jax.random.split(self.key, 3)
        init_keys = jax.random.split(init_key, self.training_cfg.generated_data_bs)
        sampling_keys = jax.random.split(sampling_key, self.training_cfg.generated_data_bs)

        # get the id_to_type mapping from all cells in the REAL data since this might be used in an initializer that
        # starts from a real datapoint (e.g. persistent initializer with permute init)
        id_to_type_vecs = jax.vmap(utils.get_id_to_type_vec, in_axes=(0, None))(real_data, self.cfg.num_cell_ids)

        # return initial sampler state based on what initializer we have chosen:
        sampler_states = jax.vmap(sampler.init)(init_keys,
                                                x=real_data[:self.training_cfg.generated_data_bs],
                                                previous_state=sampler_states,
                                                x_cell_type_vec=id_to_type_vecs
        )

        # run the sampling chains:
        (generated_data, *_), sampling_metrics = self.generate_samples(sampling_keys, sampler_states, sampler, model)

        # update model weights:
        model, opt_state, train_metrics = self.train_step(
            model=model, optimizer=self.optimizer, opt_state=opt_state, loss_func=self.compute_loss,
            real_data_batch=real_data[:self.training_cfg.training_data_bs], generated_data_batch=generated_data,
            energy_reg_lambda=self.energy_l2_reg_lambda)

        # also update ema model:
        ema_model = self.ema(model, ema_model, self.ema_model_weight)
        toc = time.time()

        # log metrics, model weights every log_frequency iterations, and at 1st and second call:
        if (iteration % self.cfg.log_frequency == 0) or (iteration == 1):
            print(f'{datetime.datetime.now()} -- Iteration {iteration} took {toc - tic} seconds', flush=True)
            path_model = utils.save_model_weights(self.cfg, model, suffix=str(iteration))
            path_ema = utils.save_model_weights(self.cfg, model, suffix=str(iteration) + '_ema')
            data_metrics = {'real_data': real_data, 'generated_data': generated_data}
            loss_ema, (energy_real_ema, energy_generated_ema, loss_reg_ema) = self.compute_loss(
                ema_model, real_data, generated_data, self.energy_l2_reg_lambda)
            ema_metrics = {
                'Loss EMA': loss_ema,
                'Energy Training Data EMA': energy_real_ema,
                'Energy Generated Data EMA': energy_generated_ema,
                'Energy Reg Loss EMA': loss_reg_ema
            }
            ema_model_metrics = ema_model.get_metrics()
            # append 'EMA' to the metrics to distinguish them from the regular metrics:
            ema_model_metrics = {k + ' EMA': v for k, v in ema_model_metrics.items()}

            self.logger.log(
                iteration=iteration,
                model_metrics=train_metrics | ema_metrics | model.get_metrics() | ema_model_metrics,
                sampler_metrics=sampling_metrics,
                data_metrics=data_metrics,
                time_per_training_step=(toc - tic),
                model_weights_path=path_model,
                model_weights_path_ema=path_ema)


        return model, opt_state, generated_data, ema_model


    @staticmethod
    @eqx.filter_jit
    @partial(jax.vmap, in_axes=(0, 0, None, None))
    def generate_samples(sampling_key, sampler_init, sampler, model):
        sampled, *other = sampler.sample(sampling_key, model, sampler_init)
        return sampled, *other

    @staticmethod
    @eqx.filter_jit
    def train_step(model: eqx.Module,
               optimizer: optax.GradientTransformation,
               opt_state: optax.OptState,
               loss_func,
               real_data_batch, generated_data_batch, energy_reg_lambda):

        # calculate loss and gradients:
        (loss_value, (energy_real, energy_generated, loss_reg)), grads = eqx.filter_value_and_grad(
            loss_func, has_aux=True
        )(model, real_data_batch, generated_data_batch,  energy_reg_lambda)

        # update model weights:
        updates, opt_state = optimizer.update(grads, opt_state, params=eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)

        metrics = {
            'Negative Log Likelihood': loss_value,
            'Energy Training Data': energy_real,
            'Energy Generated Data': energy_generated,
            'Gradients': grads,
            'Energy Reg Loss': loss_reg,
        }
        return model, opt_state, metrics

    # TODO: allow for other losses?
    @staticmethod
    @eqx.filter_jit
    def compute_loss(model, real_data_batch, generated_data_batch, energy_reg_lambda):
        print('compiling loss')

        energy_real = jax.vmap(model)(real_data_batch)
        energy_generated = jax.vmap(model)(generated_data_batch)
        loss_cd = energy_real.mean() - energy_generated.mean()
        loss_reg = jnp.square(energy_real).mean() + jnp.square(energy_generated).mean()

        return (loss_cd
                + energy_reg_lambda * loss_reg), (energy_real.mean(), energy_generated.mean(), loss_reg)

    def load_data_batch(self, bs, data_dir, t=-1, rotate=False):

        files = os.listdir(data_dir)
        # select some random files to load:
        selected_files = np.random.choice(files, size=bs, replace=True)  # list of .npz files
        data = []
        for f in selected_files:
            # load the final state of each simulation. shape is (t, 2, grid_size, grid_size)
            arr = utils.load_data_from_file(os.path.join(data_dir, f))
            if rotate:
                # randomly rotate all all channels with a random number of degrees using built-in function from library:
                angle = np.random.uniform(0, 360)
                rotated_arr = rotation_fn(arr, angle, axes=(2, 3), reshape=False, order=0)
                arr = rotated_arr
            t_int = t
            if not isinstance(t, int):
                if t == 'random_uniform':
                    t_int = np.random.randint(0, arr.shape[0], size=1)[0]
                else:
                    raise NotImplementedError(f'load_timestep {t} not implemented')
            data.append(
                arr[t_int][None, ...]
            )  # now shape (1, 2, grid_size, grid_size)
        return np.concatenate(data).astype(int)

    def ema(self, model, ema_model, beta):
        if beta == 0:
            return model
        ema_model_new = self._ema(model, ema_model, jnp.array(beta))
        return ema_model_new

    @staticmethod
    @eqx.filter_jit
    def _ema(model, ema_model, beta):
        print('compiling ema')
        model_params = eqx.filter(model, eqx.is_array)
        ema_params = eqx.filter(ema_model, eqx.is_array)
        ema_params_new = optax.incremental_update(model_params, ema_params,
                                                  1. - beta)  # beta and step size are switched in the formulation of optax
        ema_model_new = jax.tree_util.tree_map(lambda old, new: new if new is not None else old,
                                               ema_model,
                                               ema_params_new,
                                               is_leaf=lambda x: x is None)
        return ema_model_new

