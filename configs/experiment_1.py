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
from configs.config_commons import d, common_exp1
from models.modules import VectorEmb

experiment_name = 'Exp1/neuralcpm_base'

def get_config(overriding_args: str = None):
    cfg = common_exp1(experiment_name)

    # default arguments for experiment 1, which will possibly be overwritten by the overriding_args
    cfg.sampler_name = 'cpm_permuteinit'  # string that is key for the function sampler_config_exp1
    cfg.model_name = 'nch'  # nch -> neural closure hamiltonian. see model_and_model_config_exp1 function for more options
    cfg.num_mcs = 0.5  # number of monte carlo sweeps per training step. num_spin_flips = num_mcs * grid_size
    cfg.num_training_iterations_for_samplers = [10000]  # max number of training step, for each sampler in the samplers list.
    # (normally, this is just a list of one sampler)

    # override the default arguments with the arguments in the overriding_args if specified:
    if overriding_args is not None:
        cfg = utils.overwrite_config(cfg, overriding_args)

    # initialize sampler and model objects based on the possibly overwritten arguments:
    model, model_config = model_and_model_config_exp1(cfg.model_name)
    cfg.model = model
    cfg.model_config = model_config

    cfg.samplers = [MCMCSampler]
    cfg.sampler_configs = [
        sampler_config_exp1(cfg.sampler_name, cfg.num_mcs)
    ]


    # override the default arguments with the arguments in the overriding_args -- useful if we need to override something in model or sampler config:
    if overriding_args is not None:
        cfg = utils.overwrite_config(cfg, overriding_args)

    # where to store the weights of the model:
    cfg.training_config.model_weight_path=os.path.join(
            utils.make_dir_safe(os.path.join('model_weights', cfg.experiment_name)), 'experiment_1_model_MNIST.eqx'
        )

    return cfg




def sampler_config_exp1(sampler_config_key, num_mcs):
    data_shape_exp0 = (2, 100, 100)
    num_steps = int(data_shape_exp0[1] * data_shape_exp0[2] / 50 * num_mcs)
    temperature_schedule = optax.schedules.constant_schedule(1.0)
    dict_with_configs = dict(
        cpm_permuteinit=d(
            initializer=initializers.PersistentInitializer(  # persistent initializer: keeps current state of mcmc chain with prob (1 - p_reset), initializes with below specified initializer otherwis
                initializers.PermuteTypeInitializer(),  # permute-type initializer: randomly assigns a type > 0 to each cell_id > 0
                p_reset=0.025 * num_mcs
            ),
            transition_kernel=transition_kernels.ParallelizedCPMKernel(  # approximate cpm kernel with parallelized spin flips
                num_flip_attempts=50, # number of attempts to flip num_parallel_flips pixels. so total amount of spin flip attempts per step is this multiplied by num_flip_attempts
            ),
            num_steps=num_steps,
            temp_schedule=temperature_schedule
            # num attempts to flip num_parallel_flips pixels. so total amount of spin flip attempts is this multiplied by num_flip_attempts
        )
    )

    return dict_with_configs[sampler_config_key]


def model_and_model_config_exp1(model_config_key):
    dict_with_models_and_configs = dict(
        cellsort_baseline=(models.CellsortHamiltonian,
                d(bias_J=jnp.array([2.]),  # specifiying the parameters as jnp.array([float]) -> picked up by the optimizer. normal floats will stay constant
                  v_pref=jnp.array([97.]),
                  lamb=jnp.array([jnp.log(jnp.e**0.05 - 1)]),  # lambda of volume constraint is softplus of this number
                  offset=jnp.array([0.]),  # energy = cellsort_energy + offset*offset_scale -- offset can be used to shift the energy landscape, e.g. when using regularization.
                  offset_scale=jnp.array([1000.])
                )),
        external_field_baseline=(models.ExternalFieldHamiltonian,  # cellsort baseline + external potential on each pixel
                  d(field=jnp.zeros(shape=(100, 100)),  # 100x100 grid -- jnp.array(float) so picked up by the optimizer
                    field_coupling=jnp.ones(shape=(3)),  # three cell types -- array so picked up by optimizer
                    v_pref=jnp.array([97.]),  # target volume of all cells
                    lamb=jnp.array([jnp.log(jnp.e ** 0.05 - 1)]),  # lambda of volume constraint is softplus of this number
                    bias_J=jnp.array([2.]),  # contact energy is softplus(gamma_J) * J + bias_J
                    offset=jnp.array([0.]),  # energy = cellsort_energy + offset*offset_scale -- offset can be used to shift the energy landscape, e.g. when using regularization.
                    offset_scale=jnp.array([1000.])
                    )),
        shallow_nh=(models.NeuralHamiltonian,  # 1 NH layer -> convnet -> MLP
                d(  # see NeuralHamiltonian class in models.py for more details on params
                num_layers_per_block=1,  # 2
                spatial_downsampling_per_block=(3,),
                node_emb_dims=(8,),  # (8, 16, 32)
                edge_emb_dims=(8,),  # (8, 16, 32),  # putting an edge_emb_dim to 0 means no interaction for that layer!
                emb_dim_before_agg=32,
                emb_dims_conv=(32, 64, 128),
                spatial_downsampling_per_block_conv=(1, 2, 2),
                num_layers_per_block_conv=2,
                emb_dims_mlp=(32, 32),
                activation=jax.nn.silu,
                use_residual=True,
                embedding_module=nn.Conv2d,
                embedding_module_kwargs=dict(
                    kernel_size=(3, 3),
                    stride=(3, 3),
                )  # put here all kwargs for the embedding module except in_channels, out_channels and key
                # pass here also any kwargs to be passed to the griddeepset module
            )
        ),
        nh=(  # standard Neural Hamiltonian
            models.NeuralHamiltonian,
                d(  # see NeuralHamiltonian class in models.py for more details on params
                    num_layers_per_block=1,  # 2
                    spatial_downsampling_per_block=(3, 2, 1, 1),
                    node_emb_dims=(8, 16, 32, 32),  # (8, 16, 32)
                    edge_emb_dims=(8, 16, 32, 32),
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
                        kernel_size=(3, 3),
                        stride=(3, 3),
                    )  # put here all kwargs for the embedding module except in_channels, out_channels and key
                    # pass here also any kwargs to be passed to the griddeepset module
                )
             ),


    nch=(# Neural Closure Hamiltonian -- NH + biology-informed term expressed by the specified basis_model
        models.NeuralClosureHamiltonian,
             d(  # see NeuralHamiltonian class in models.py for more details on params
                 basis_model=models.DifferentiableCellsortHamiltonian,
                 weight_basis=jnp.array([0.25]),
                 weight_neural=jnp.array([1.0]),
                 basis_model_kwargs=d(
                     bias_J=jnp.array([2.]),
                     v_pref=jnp.array([97.]),
                     lamb=jnp.array([jnp.log(jnp.e ** 0.05 - 1)]),  # softplus of this is 0.05
                     offset=jnp.array([0.]),
                     offset_scale=jnp.array([1000.])
                 ),
                 num_layers_per_block=1,  # 2
                 spatial_downsampling_per_block=(3,2,1,1),
                 node_emb_dims=(8, 16, 32, 32),  # (8, 16, 32)
                 edge_emb_dims=(8, 16, 32, 32),
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
                     kernel_size=(3, 3),
                     stride=(3, 3),
                 )  # put here all kwargs for the embedding module except in_channels, out_channels and key
                 # pass here also any kwargs to be passed to the griddeepset module
        )
        ),

    conv_ham=(  # CNN neural Hamiltonian -- no permutation symmetry
        models.SimpleConvNeuralHamiltonian, d(  # see NeuralHamiltonian class in models.py for more details on params
        num_layers_per_block=2,
        spatial_downsampling_per_block=(1,2,1,2,1,2,1),
        node_emb_dims=(16, 16, 32, 32, 64, 64, 128),
        emb_dim_before_agg=64,
        emb_dims_mlp=(32, 32),
        activation=jax.nn.silu,
        use_residual=True,
        embedding_module=nn.Conv2d,
        embedding_module_kwargs=dict(
            kernel_size=(3,3),
            stride=(3,3),
        ) # put here all kwargs for the embedding module except in_channels, out_channels and key
        # pass here also any kwargs to be passed to the griddeepset module
    )),

    nch_no_pooling=( # NCH but without pooling between NH layers
        models.NeuralClosureHamiltonian,
             d(  # see NeuralHamiltonian class in models.py for more details on params
                 basis_model=models.DifferentiableCellsortHamiltonian,
                 weight_basis=jnp.array([0.25]),
                 weight_neural=jnp.array([1.0]),
                 basis_model_kwargs=d(
                     bias_J=jnp.array([2.]),
                     v_pref=jnp.array([97.]),
                     lamb=jnp.array([jnp.log(jnp.e ** 0.05 - 1)]),  # softplus of this is 0.05
                     offset=jnp.array([0.]),
                     offset_scale=jnp.array([1000.])
                 ),
                 num_layers_per_block=1,  # 2
                 spatial_downsampling_per_block=(1,1,1,1),
                 node_emb_dims=(8, 16, 32, 32),  # (8, 16, 32)
                 edge_emb_dims=(8, 16, 32, 32),
                 # (8, 16, 32),  # putting an edge_emb_dim to 0 means no interaction for that layer!
                 emb_dim_before_agg=32,
                 emb_dims_conv=tuple(),
                 spatial_downsampling_per_block_conv=tuple(),
                 num_layers_per_block_conv=0,
                 emb_dims_mlp=(32, 32),
                 activation=jax.nn.silu,
                 use_residual=True,
                 mask_interactions=False,
                 embedding_module=nn.Conv2d,
                 embedding_module_kwargs=dict(
                     kernel_size=(3, 3),
                     stride=(3, 3),
                 )  # put here all kwargs for the embedding module except in_channels, out_channels and key
                 # pass here also any kwargs to be passed to the griddeepset module
        )
        ),


        nch_no_interactions=(  # NCH, but without interactions between cells. also type channel outside own cell is masked to prevent any interactions
            models.NeuralClosureHamiltonian,
              d(  # see NeuralHamiltonian class in models.py for more details on params
                  basis_model=models.DifferentiableCellsortHamiltonian,
                  weight_basis=jnp.array([0.25]),
                  weight_neural=jnp.array([1.0]),
                  basis_model_kwargs=d(
                      bias_J=jnp.array([2.]),
                      v_pref=jnp.array([97.]),
                      lamb=jnp.array([jnp.log(jnp.e ** 0.05 - 1)]),  # softplus of this is 0.05
                      offset=jnp.array([0.]),
                      offset_scale=jnp.array([1000.])
                  ),
                  num_layers_per_block=1,  # 2
                  spatial_downsampling_per_block=(3, 2, 1, 1),
                  node_emb_dims=(8, 16, 32, 32),  # (8, 16, 32)
                  edge_emb_dims=(0,0,0,0),
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
                      kernel_size=(3, 3),
                      stride=(3, 3),
                  )  # put here all kwargs for the embedding module except in_channels, out_channels and key
                  # pass here also any kwargs to be passed to the griddeepset module
              )
              ),


        gnn=(  # GNN model.
            # At the implementation level, data is flattened to (num_cells, num_channels*grid_size, 1, 1) by VectorEmb
            # and then passed to the Neural Hamiltonian, equivalent to standard message passing GNN.
            models.NeuralHamiltonian,
                     d(  # see NeuralHamiltonian class in models.py for more details on params
                         num_layers_per_block=1,  # 2
                         spatial_downsampling_per_block=(1, 1, 1, 1),
                         node_emb_dims=(32, 32, 16, 8),  # (8, 16, 32)
                         edge_emb_dims=(32, 32, 16, 8),
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
                             flattened_size=100*100*4,
                         )  # put here all kwargs for the embedding module except in_channels, out_channels and key
                         # pass here also any kwargs to be passed to the griddeepset module
                     )
                     ),

        closure_gnn=(  # GNN, but as closure
            models.NeuralClosureHamiltonian,
              d(  # see NeuralHamiltonian class in models.py for more details on params
                  basis_model=models.DifferentiableCellsortHamiltonian,
                  weight_basis=jnp.array([0.25]),
                  weight_neural=jnp.array([1.0]),
                  basis_model_kwargs=d(
                      bias_J=jnp.array([2.]),
                      v_pref=jnp.array([97.]),
                      lamb=jnp.array([jnp.log(jnp.e ** 0.05 - 1)]),  # softplus of this is 0.05
                      offset=jnp.array([0.]),
                      offset_scale=jnp.array([1000.])
                  ),
                  num_layers_per_block=1,  # 2
                  spatial_downsampling_per_block=(1, 1, 1, 1),
                  node_emb_dims=(32, 32, 16, 8),  # (8, 16, 32)
                  edge_emb_dims=(32, 32, 16, 8),
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
                  mask_interactions=None,
                  embedding_module=VectorEmb,
                  embedding_module_kwargs=dict(
                      flattened_size=100 * 100 * 4,
                  )  # put here all kwargs for the embedding module except in_channels, out_channels and key
                  # pass here also any kwargs to be passed to the griddeepset module
              )
              ),

    )
    return dict_with_models_and_configs[model_config_key.lower()]
