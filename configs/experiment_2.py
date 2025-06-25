import numpy as np
from ml_collections import ConfigDict
import utils
import os
import models.models as models
import optax
import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
from sampling import initializers
from sampling import transition_kernels
from sampling.samplers import MCMCSampler
from configs.config_commons import d, common_exp2
from models.modules import VectorEmb

experiment_name = 'Exp2/neuralcpm_base'

def get_config(overriding_args=None):
    cfg = common_exp2(experiment_name)

    # default arguments for experiment 2, which will possibly be overwritten by the overriding_args
    cfg.sampler_name = 'cpm_permuteinit' # string that is key for the function sampler_config_exp1
    cfg.model_name = 'nch' # nch -> neural closure hamiltonian. see model_and_model_config_exp1 function for more options
    cfg.num_mcs = 1.0  # number of monte carlo sweeps per training step. num_spin_flips = num_mcs * grid_size
    cfg.num_training_iterations_for_samplers = [10000] # max number of training step, for each sampler in the samplers list.
    # (normally, this is just a list of one sampler)

    # override the default arguments with the arguments in the overriding_args
    if overriding_args is not None:
        cfg = utils.overwrite_config(cfg, overriding_args)

    # initialize sampler and model objects based on the possibly overwritten arguments:
    model, model_config = model_and_model_config_exp2(cfg.model_name)
    cfg.model = model
    cfg.model_config = model_config

    cfg.samplers = [MCMCSampler]
    cfg.sampler_configs = [
        sampler_config_exp2(cfg.sampler_name, cfg.num_mcs)
    ]

    # override the default arguments with the arguments in the overriding_args -- useful if we need to override something in model or sampler config:
    if overriding_args is not None:
        cfg = utils.overwrite_config(cfg, overriding_args)


    cfg.training_config.model_weight_path=os.path.join(
            utils.make_dir_safe(os.path.join('model_weights', cfg.experiment_name)), 'experiment_2_model_TODA.eqx'
        )

    return cfg




def sampler_config_exp2(sampler_config_key, num_mcs):
    data_shape_exp0 = (2, 125, 125)  #NOTE: actual shape is (149,149) but this is due to padding, so we use the unpadded size to define the mcs
    num_steps = int(data_shape_exp0[1] * data_shape_exp0[2] / 50 * num_mcs)
    temperature_schedule = optax.schedules.constant_schedule(1.0)
    dict_with_configs = dict(
        cpm_permuteinit=d(  # parallelized cpm sampler -- our sampler with permute-type initializer
            initializer=initializers.PersistentInitializer(
                initializers.PermuteTypeInitializer(),
                p_reset=0.025 * num_mcs
            ),
            transition_kernel=transition_kernels.ParallelizedCPMKernel(
                num_flip_attempts=50,
    ),
            num_steps=num_steps,
            temp_schedule=temperature_schedule
            # num attempts to flip num_parallel_flips pixels. so total amount of spin flip attempts is this multiplied by num_flip_attempts
        ),
    )

    return dict_with_configs[sampler_config_key]


def model_and_model_config_exp2(model_config_key):
    dict_with_models_and_configs = dict(
        cellsort_baseline=(models.CellsortHamiltonian,  # specifiying the parameters as jnp.array([float]) -> picked up by the optimizer. normal floats will stay constant
                  d(bias_J=jnp.array([2.]),
                    v_pref=jnp.array([150.]),
                    lamb=jnp.array([jnp.log(jnp.e ** 0.05 - 1)]), # lambda of volume constraint is softplus of this number
                    offset=jnp.array([0.]), # energy = cellsort_energy + offset*offset_scale -- offset can be used to shift the energy landscape, e.g. when using regularization.
                    offset_scale=jnp.array([1000.])
                    )),
        external_field_baseline=(models.ExternalFieldHamiltonian,  # cellsort baseline + external potential on each pixel
                                 d(field=jnp.zeros(shape=(149, 149)),  # 149x149 grid - jnp.array(float) so picked up by the optimizer
                                   field_coupling=jnp.ones(shape=(3)),  # three cell types -- array so picked up by optimizer
                                   v_pref=jnp.array([150.]),  # target volume of all cells
                                   lamb=jnp.array([jnp.log(jnp.e ** 0.05 - 1)]), # lambda of volume constraint is softplus of this number
                                   bias_J=jnp.array([2.]), # contact energy is softplus(gamma_J) * J + bias_J
                                   offset=jnp.array([0.]),  # energy = cellsort_energy + offset*offset_scale -- offset can be used to shift the energy landscape, e.g. when using regularization.
                                   offset_scale=jnp.array([1000.])
                                   )),
        nch=(models.NeuralClosureHamiltonian,
                        d(  # see NeuralHamiltonian class in models.py for more details on params
                            basis_model=models.DifferentiableCellsortHamiltonian,
                            weight_basis=jnp.array([0.25]),
                            weight_neural=jnp.array([1.0]),
                            basis_model_kwargs=dict(
                                bias_J=jnp.array([2.0]),
                                gamma_J=jnp.array([(jnp.log(jnp.e - 1.))]),  # softplus of this is 1.0
                                interaction_params=dict({
                                    "1-1": jnp.array([1.56]),
                                    "1-2": jnp.array([1.21]),
                                    "0-0": jnp.array([0.0]),
                                    "0-1": jnp.array([2.26]),
                                    "0-2": jnp.array([2.79]),
                                    "2-2": jnp.array([2.71]),
                                }),
                                v_pref=jnp.array([150]),
                                lamb=jnp.array([jnp.log(jnp.e ** 0.05 - 1)]),  # softplus of this is 0.02
                                offset=0.0,
                                offset_scale=1.0,
                            ),
                            num_layers_per_block=1,
                            spatial_downsampling_per_block=(2, 1, 2, 1, 2, 1),
                            node_emb_dims=(16, 32, 32, 64, 64, 64),  # (8, 16, 32)
                            edge_emb_dims=(16, 32, 32, 64, 64, 64),
                            emb_dim_before_agg=32,
                            emb_dims_conv=tuple(),
                            spatial_downsampling_per_block_conv=tuple(),
                            num_layers_per_block_conv=0,
                            emb_dims_mlp=(32, 32),
                            activation=jax.nn.silu,
                            use_residual=True,
                            embedding_module=nn.Conv2d,
                            embedding_module_kwargs=dict(
                                kernel_size=(5, 5),
                                stride=(5, 5),
                            )
                        )
                        ),

        nch_no_pooling=(models.NeuralClosureHamiltonian, #nch without pooling between nh blocks
              d(  # see NeuralHamiltonian class in models.py for more details on params
                  basis_model=models.DifferentiableCellsortHamiltonian,
                  weight_basis=jnp.array([0.25]),
                  weight_neural=jnp.array([1.0]),
                  basis_model_kwargs=dict(
                      bias_J=jnp.array([2.]),
                      v_pref=jnp.array([150.]),
                      lamb=jnp.array([jnp.log(jnp.e ** 0.05 - 1)]),  # softplus of this is 0.5
                      offset=jnp.array([0.]),
                      offset_scale=jnp.array([1000.])
                  ),
                  num_layers_per_block=1,  # 2
                  spatial_downsampling_per_block=(1, 1, 1, 1, 1, 1),
                  node_emb_dims=(16, 32, 32, 64, 64, 64),  # (8, 16, 32)
                  edge_emb_dims=(16, 32, 32, 64, 64, 64),
                  # (8, 16, 32),  # putting an edge_emb_dim to 0 means no interaction for that layer!
                  emb_dim_before_agg=32,
                  emb_dims_conv=tuple(),
                  spatial_downsampling_per_block_conv=tuple(),
                  num_layers_per_block_conv=0,
                  emb_dims_mlp=(32, 32),
                  activation=jax.nn.silu,
                  use_residual=True,
                  embedding_module=nn.Conv2d,
                  embedding_module_kwargs=dict(
                      kernel_size=(5, 5),
                      stride=(5, 5),
                  )  # put here all kwargs for the embedding module except in_channels, out_channels and key
                  # pass here also any kwargs to be passed to the griddeepset module
              )
              ),

        nch_no_interactions=(models.NeuralClosureHamiltonian,  # nch without interactions in nh layers, type info out of cell is masked out
              d(  # see NeuralHamiltonian class in models.py for more details on params
                  basis_model=models.DifferentiableCellsortHamiltonian,
                  weight_basis=jnp.array([0.25]),
                  weight_neural=jnp.array([1.0]),
                  basis_model_kwargs=dict(
                      bias_J=jnp.array([2.]),
                      v_pref=jnp.array([150.]),
                      lamb=jnp.array([jnp.log(jnp.e ** 0.05 - 1)]),  # softplus of this is 0.5
                      offset=jnp.array([0.]),
                      offset_scale=jnp.array([1000.])
                  ),
                  num_layers_per_block=1,  # 2
                  spatial_downsampling_per_block=(2, 1, 2, 1, 2, 1),
                  node_emb_dims=(16, 32, 32, 64, 64, 64),  # (8, 16, 32)
                  edge_emb_dims=(0,0,0,0,0,0),
                  # (8, 16, 32),  # putting an edge_emb_dim to 0 means no interaction for that layer!
                  emb_dim_before_agg=32,
                  emb_dims_conv=tuple(),
                  spatial_downsampling_per_block_conv=tuple(),
                  num_layers_per_block_conv=0,
                  emb_dims_mlp=(32, 32),
                  activation=jax.nn.silu,
                  use_residual=True,
                  mask_interactions=True,
                  embedding_module=nn.Conv2d,
                  embedding_module_kwargs=dict(
                      kernel_size=(5, 5),
                      stride=(5, 5),
                  )  # put here all kwargs for the embedding module except in_channels, out_channels and key
                  # pass here also any kwargs to be passed to the griddeepset module
              )
              ),



        nh=(models.NeuralHamiltonian,  # vanilla NH model
            d(  # see NeuralHamiltonian class in models.py for more details on params
                num_layers_per_block=1,  # 2
                spatial_downsampling_per_block=(2, 1, 2, 1, 2, 1),
                node_emb_dims=(16, 32, 32, 64, 64, 64),  # (8, 16, 32)
                edge_emb_dims=(16, 32, 32, 64, 64, 64),
                # (8, 16, 32),  # putting an edge_emb_dim to 0 means no interaction for that layer!
                emb_dim_before_agg=32,
                emb_dims_conv=tuple(),
                spatial_downsampling_per_block_conv=tuple(),
                num_layers_per_block_conv=0,
                emb_dims_mlp=(32, 32),
                activation=jax.nn.silu,
                use_residual=True,
                embedding_module=nn.Conv2d,
                embedding_module_kwargs=dict(
                    kernel_size=(5, 5),
                    stride=(5, 5),
                )  # put here all kwargs for the embedding module except in_channels, out_channels and key
                # pass here also any kwargs to be passed to the griddeepset module
            )
            ),

        shallow_nh=(models.NeuralHamiltonian,  # 1 NH layer -> CNN -> MLP
            d(  # see NeuralHamiltonian class in models.py for more details on params
                num_layers_per_block=1,  # 2
                spatial_downsampling_per_block=(2,),
                node_emb_dims=(16,),  # (8, 16, 32)
                edge_emb_dims=(16,),
                # (8, 16, 32),  # putting an edge_emb_dim to 0 means no interaction for that layer!
                emb_dim_before_agg=32,
                emb_dims_conv=(32, 64, 64, 128),
                spatial_downsampling_per_block_conv=(1, 2, 1, 2),
                num_layers_per_block_conv=2,
                emb_dims_mlp=(32, 32),
                activation=jax.nn.silu,
                use_residual=True,
                embedding_module=nn.Conv2d,
                embedding_module_kwargs=dict(
                    kernel_size=(5, 5),
                    stride=(5, 5),
                )  # put here all kwargs for the embedding module except in_channels, out_channels and key
                # pass here also any kwargs to be passed to the griddeepset module
            )
            ),

        conv_ham=(  # simple CNN
        models.SimpleConvNeuralHamiltonian, d(  # see NeuralHamiltonian class in models.py for more details on params
            num_layers_per_block=2,
            spatial_downsampling_per_block=(1, 2, 1, 2, 1, 2, 1),
            node_emb_dims=(16, 16, 32, 32, 64, 64, 128),
            emb_dim_before_agg=64,
            emb_dims_mlp=(32, 32),
            activation=jax.nn.silu,
            use_residual=True,
            embedding_module=nn.Conv2d,
            embedding_module_kwargs=dict(
                kernel_size=(3, 3),
                stride=(3, 3),
            )  # put here all kwargs for the embedding module except in_channels, out_channels and key
            # pass here also any kwargs to be passed to the griddeepset module
        )),

        closure_gnn=(models.NeuralClosureHamiltonian,  # GNN as closure model
         # At the implementation level, data is flattened to (num_cells, num_channels*grid_size, 1, 1) by VectorEmb
         # and then passed to the Neural Hamiltonian, equivalent to standard message passing GNN.
              d(  # see NeuralHamiltonian class in models.py for more details on params
                  basis_model=models.DifferentiableCellsortHamiltonian,
                  weight_basis=jnp.array([0.25]),
                  weight_neural=jnp.array([1.0]),
                  basis_model_kwargs=dict(
                      bias_J=jnp.array([2.]),
                      v_pref=jnp.array([150.]),
                      lamb=jnp.array([jnp.log(jnp.e ** 0.05 - 1)]),  # softplus of this is 0.5
                      offset=jnp.array([0.]),
                      offset_scale=jnp.array([1000.])
                  ),
                  num_layers_per_block=1,  # 2
                  spatial_downsampling_per_block=(1, 1, 1, 1, 1, 1),
                  node_emb_dims=(64, 64, 64, 32, 32, 16),  # (8, 16, 32)
                  edge_emb_dims=(64, 64, 64, 32, 32, 16),
                  # (8, 16, 32),  # putting an edge_emb_dim to 0 means no interaction for that layer!
                  emb_dim_before_agg=32,
                  kernel_size=1,
                  edge_network_downsampling_factor=1,
                  emb_dims_conv=tuple(),
                  spatial_downsampling_per_block_conv=tuple(),
                  num_layers_per_block_conv=0,
                  emb_dims_mlp=(32, 32),
                  activation=jax.nn.silu,
                  use_residual=True,
                  embedding_module=VectorEmb,
                  embedding_module_kwargs=dict(
                      flattened_size=149*149*4,
                  )  # put here all kwargs for the embedding module except in_channels, out_channels and key
                  # pass here also any kwargs to be passed to the griddeepset module
              )
              ),

        gnn=(models.NeuralHamiltonian,  # vanilla gnn
             # At the implementation level, data is flattened to (num_cells, num_channels*grid_size, 1, 1) by VectorEmb
             # and then passed to the Neural Hamiltonian, equivalent to standard message passing GNN.
                     d(  # see NeuralHamiltonian class in models.py for more details on params
                         num_layers_per_block=1,  # 2
                         spatial_downsampling_per_block=(1, 1, 1, 1, 1, 1),
                         node_emb_dims=(64, 64, 64, 32, 32, 16),  # (8, 16, 32)
                         edge_emb_dims=(64, 64, 64, 32, 32, 16),
                         # (8, 16, 32),  # putting an edge_emb_dim to 0 means no interaction for that layer!
                         emb_dim_before_agg=32,
                         kernel_size=1,
                         edge_network_downsampling_factor=1,
                         emb_dims_conv=tuple(),
                         spatial_downsampling_per_block_conv=tuple(),
                         num_layers_per_block_conv=0,
                         emb_dims_mlp=(32, 32),
                         activation=jax.nn.silu,
                         use_residual=True,
                         embedding_module=VectorEmb,
                         embedding_module_kwargs=dict(
                             flattened_size=149 * 149 * 4,
                         )  # put here all kwargs for the embedding module except in_channels, out_channels and key
                         # pass here also any kwargs to be passed to the griddeepset module
                     )
                     ),

    )
    return dict_with_models_and_configs[model_config_key.lower()]
