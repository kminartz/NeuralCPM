# define jax modules here, to be used by themselves or to build jax models
import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
from typing import *
import utils


# types
Array = jax.Array
PRNGKey = jax.Array

class NHLayer(eqx.Module):

    residual: bool
    edge_emb_dim: int
    edge_network_layers: list
    node_network_layers: list
    downsampling: eqx.Module
    upsampling: eqx.Module
    pre_resid: eqx.Module

    def __init__(self,
                 key: PRNGKey,
                 in_channels: int,
                 node_emb_dim: int = 4,
                 use_residual: bool = False,
                 kernel_size: Union[Tuple[int, int], int] = 3,
                 activation: Callable[[Array], Array] = jax.nn.silu,
                 edge_network_downsampling_factor: int = 1,
                 edge_emb_dim: int = 4,
                 num_edge_network_layers: int = 2,
                 num_node_network_layers: int = 2,
                 **kwargs):

        """
        A single NH layer as illustrated in the paper.
        The implementation contains several minor details that were not covered in the paper text, e.g. up/downsampling
        before cell-interaction CNN, ...
        :param key: jax PRNGKey for random initialization of the layer
        :param in_channels: amount of input channels per cell
        :param node_emb_dim: internal node embedding dimension, i.e. number of channels
        :param use_residual: whether to use residual connections in the node update
        :param kernel_size: kernel size of conv layers
        :param activation: activation function to use in the layers
        :param edge_network_downsampling_factor: downsampling rate before applying cell-interaction CNN
        :param edge_emb_dim: edge embedding dimension, i.e. number of channels in the aggregate embedding A
        :param num_edge_network_layers: number of layers in the cellwise CNN phi
        :param num_node_network_layers: number of layers in the cell-interaction CNN psi
        :param kwargs: any other kwargs to pass to the nn.Conv2d layers
        """

        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size)

        self.residual = use_residual
        self.edge_emb_dim = edge_emb_dim

        return_0_channels = nn.Lambda(
                lambda x: jnp.zeros(shape=(0, *x.shape[1:]))  # tensor with zero channels
        )

        # construct nn edge function
        key, use_key = jax.random.split(key)
        keys = jax.random.split(use_key, num_edge_network_layers)
        self.edge_network_layers = []
        if edge_emb_dim > 0:
            for l in range(num_edge_network_layers):
                self.edge_network_layers.append(nn.Conv2d(in_channels=edge_emb_dim,
                                            out_channels=edge_emb_dim,
                                            kernel_size=kernel_size,
                                            padding='SAME', key=keys[l], **kwargs))
                self.edge_network_layers.append(activation)
        else:  # no interactions!
            self.edge_network_layers.append(return_0_channels)

        # construct nn node function
        l = 0
        key, use_key = jax.random.split(key)
        keys = jax.random.split(use_key, num_node_network_layers)
        self.node_network_layers = [nn.Conv2d(in_channels=edge_emb_dim + in_channels,
                                    out_channels=node_emb_dim,
                                    kernel_size=kernel_size,
                                    padding='SAME', key=keys[0], **kwargs)]
        for l in range(num_node_network_layers - 2):
            self.node_network_layers.append(activation)
            self.node_network_layers.append(nn.Conv2d(in_channels=node_emb_dim,
                                    out_channels=node_emb_dim,
                                    kernel_size=kernel_size,
                                    padding='SAME', key=keys[l + 1], **kwargs))
        self.node_network_layers.append(activation)
        self.node_network_layers.append(nn.Conv2d(in_channels=node_emb_dim,
                                out_channels=node_emb_dim,
                                kernel_size=kernel_size,
                                padding='SAME', key=keys[l + 1], **kwargs))

        # up and downsampling + changing channels of aggregate embedding (A) to optimize computation and memory usage
        kye, use_key = jax.random.split(key)
        key1, key2 = jax.random.split(use_key)
        self.downsampling = nn.Conv2d(in_channels=in_channels,
                                    out_channels=edge_emb_dim,
                                    kernel_size=edge_network_downsampling_factor,
                                    stride=edge_network_downsampling_factor, key=key1) if edge_emb_dim > 0 else return_0_channels
        self.upsampling = nn.ConvTranspose2d(in_channels=edge_emb_dim,
                                    out_channels=edge_emb_dim,
                                    kernel_size=edge_network_downsampling_factor,
                                    stride=edge_network_downsampling_factor, key=key2) if edge_emb_dim > 0 else return_0_channels

        self.pre_resid = nn.Identity()
        if self.residual and in_channels != node_emb_dim:
            # we need to map the input to the output shape by changing the amount of channels using a 1x1 conv:
            self.pre_resid = nn.Conv2d(in_channels=in_channels,
                                    out_channels=node_emb_dim,
                                    kernel_size=1, key=key)

    def node_update_before_agg(self, set_elements):
        ''' Compute edge message for a node -- the cell-wise CNN phi in the paper
         set_elements: set of embeddings to process over, shape ch, h, w
         '''
        x = set_elements
        for layer in self.edge_network_layers:
            x = layer(x)
        return x

    def node_update(self, x, A):
        ''' Update embedding for each node -- cell interaction CNN psi in the paper
        x: node features, shape (c, h, w)
        A: edge features, shape (c, h, w
        '''
        y = jnp.concatenate([x, A], axis=0)
        for layer in self.node_network_layers:
            y = layer(y)
        if self.residual:
            x_right_num_channels = self.pre_resid(x)
            y = y + x_right_num_channels
        return y

    def __call__(self, x: Array) -> Array:
        # x, edge_index, batch_idx = x

        # shape of x: (cells, channels, height, width)

        # downsample the grid
        z = jax.vmap(self.downsampling)(x)

        # compute edge message for each node and sum up
        A = jax.vmap(self.node_update_before_agg)(z).sum(axis=0)

        # upsample again
        A = self.upsampling(A)
        # pad to be the same shape as x in the last 2 dims:
        A = utils.pad_array(A, x.shape[-2:], num_spatial_dims=2)

        # for each node, compute the node features
        x = jax.vmap(self.node_update, in_axes=(0, None))(x, A)
        assert self.edge_emb_dim == A.shape[0]
        return x


class NHBlock(eqx.Module):

    layers: list

    def __init__(self,
            key: PRNGKey,
            in_channels: int,
            num_layers: int,
            node_emb_dim: int = 4,
                 **kwargs
                 ):
        """
        block of NH layers to do before any spatial pooling in the NH architecture.
        :param key: jax PRNGKey
        :param in_channels: number of input channels per cell for this block
        :param num_layers: number of NH layers
        :param node_emb_dim: dimension of the node embedding, i.e. number of channels, to keep in this NH block
        :param kwargs: other kwargs for NH layer
        """


        keys = jax.random.split(key, num_layers)

        # in_out shapes of layers
        in_out_key = [(in_channels, in_channels, keys[0])]
        for l in range(num_layers - 1):
            in_out_key.append((node_emb_dim, in_channels, keys[l + 1]))

        self.layers = [NHLayer(
                key=k,
                in_channels=in_channels,
                node_emb_dim=node_emb_dim,
                **kwargs
        )

            for l_num, (in_channels, _, k) in enumerate(in_out_key)]


    def __call__(self, x: Array) -> Array:

        for layer in self.layers:
            x = layer(x)
        return x


class MLP(eqx.Module):

    layers: list[eqx.Module]
    in_features: int
    out_features: int
    pre_resid: eqx.Module
    use_residual: bool

    def __init__(self, layers: list, use_residual=False, *, key):
        """
        Simple MLP module
        :param layers: list of linear layers + activations that make up the MLP
        :param use_residual: whether to use residual connections
        :param key: jax prngkey
        """
        self.layers = layers
        self.in_features = layers[0].in_features
        self.out_features = layers[0].out_features
        self.use_residual = use_residual
        self.pre_resid = nn.Identity()
        key, use_key = jax.random.split(key)
        if self.in_features != self.out_features and use_residual:
            self.pre_resid = nn.Linear(self.in_features, self.out_features, key=use_key)



    def __call__(self, x):
        x_in = x
        for layer in self.layers:
            x = layer(x)
        if self.use_residual:
            x = x + self.pre_resid(x_in)
        return x

class ConvNet(eqx.Module):

    layers: list[eqx.Module]
    in_features: int
    out_features: int
    pre_resid: eqx.Module
    use_residual: bool

    def __init__(self, layers: list, use_residual=False, *, key):
        """
        Simple convnet module
        :param layers: list of conv layers and activations that make up the convnet
        :param use_residual: whether to use residual connections
        :param key: jax prngkey
        """
        self.layers = layers
        self.in_features = layers[0].in_channels
        self.out_features = layers[0].out_channels
        self.use_residual = use_residual
        self.pre_resid = nn.Identity()
        key, use_key = jax.random.split(key)
        if self.in_features != self.out_features and use_residual:
            self.pre_resid = nn.Conv2d(self.in_features, self.out_features, kernel_size=1, key=use_key)

    def __call__(self, x):
        x_in = x
        for layer in self.layers:
            x = layer(x)
        if self.use_residual:
            x = x + self.pre_resid(x_in)
        return x



class VectorEmb(eqx.Module):
    """
    Embeds grid as a vector. We return a tensor of shape (channels, 1, 1) to remain compatibility with further
    code assuming grid-like inputs for conv layers -- esssentially these conv layers then act as a fully-connected layer.
    """
    linear: eqx.Module

    def __init__(self, key, in_channels, out_channels, flattened_size, **kwargs):
        self.linear = nn.Linear(flattened_size, out_channels, key=key)

    def __call__(self, x):
        x = x.reshape(-1)
        x = self.linear(x)
        x = x.reshape(-1, 1, 1) # to play nice with the rest of the code where we use kernel_size=1 so basically MLPs
        return x




















