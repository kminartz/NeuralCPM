# define jax models here. These should be at the level that they are easily initialized and used in training scripts
# as well as the morpheus py_script extension
from functools import partial
from typing import Callable, Union, Tuple
import warnings
warnings.simplefilter("once")
import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import numpy as np
from equinox.nn import Linear

import utils
from ml_collections import ConfigDict

from models.modules import NHBlock, MLP, ConvNet

class HamiltonianBaseClass(eqx.Module):
    """
    Base class for all Hamiltonian models.

    Note: Every parameter that is an initialized as np or jnp array, will be picked up by the optimizer during training.
    Parameters that are regular python floats etc will not be optimized, and instead remain constant.
    """

    def __call__(self, cpm: jnp.ndarray):
        """
        forward pass of the Hamiltonian to calculate the energy
        cpm: the cpm lattice (shape [2, h, w]), where channel 0 is the cell id channel, and channel 1 is the cell type channel
        :return: the energy of the system
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def delta_energy(self, cpm, i_flip, j_flip, new_cell_id, new_cell_type, old_energy=None):
        """
        calculate the energy difference of flipping the cell at position i,j to the new cell id and corresponding cell type
        :param cpm: the cpm lattice
        :param i_flip, j_flip: index of to-be-flipped pixel in the cpm grid
        :param new_cell_id: the new cell id to be placed at i,j
        :param new_cell_type: the new cell type to be placed at i,j, matching the new_cell_id
        :param old_energy: the old energy of the system, if known, to avoid recalculating it, which could otherwise be required for some models
        :return: the energy difference
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_metrics(self):
        """
        return a dict of metrics that can be used to monitor the model during training
        :return: dict of metrics
        """
        return {}


class CellsortHamiltonian(HamiltonianBaseClass):
    interaction_params: dict
    J: jnp.ndarray
    v_pref: jnp.ndarray
    lamb: jnp.ndarray
    num_cell_ids: int
    num_cell_types: int
    gamma_J: jnp.ndarray
    bias_J: jnp.ndarray
    differentiable: bool
    offset: float
    offset_scale: jnp.array

    def __init__(
        self,
        key: jax.random.PRNGKey,
        cfg: ConfigDict,
        interaction_params=None,
        v_pref=None,
        lamb=None,
        gamma_J=jnp.array(
            [jnp.log(jnp.e - 1)]
        ),  # initialize s.t. softplus of this value equals 1
        bias_J=jnp.array([0.0]),
        offset=jnp.array([0.0]),
        offset_scale=jnp.array([1.0])
    ):
        """
        Classic Cellsort Hamiltonian model -- see our paper and Graner, F. and Glazier, J. A. (1992).
        :param key: jax random key for initializing the parameters
        :param cfg: configuration dict containing at least num_cell_ids and num_cell_types
        :param interaction_params: a dict of interaction parameters, where keys are strings of the form "celltypeA-celltypeB" and values are the interaction energies (e.g. J11, J12, ..). If None, the parameters are initialized randomly.
        :param v_pref: preferred volume (V^*) of the cells, if None, initialized randomly
        :param lamb: volume constraint strength (lambda), if None, initialized randomly
        :param gamma_J: a trainable parameter that scales the interaction energy matrix J
        :param bias_J: a trainable parameter that shifts the interaction energy matrix J
        :param offset: a learnable or constant offset to be added to the energy, can be used for regularization or centering the energy around 0, if None, initialized to 0.0
        :param offset_scale: a learnable or constant scale to be applied to the offset, can be used for regularization or centering the energy around 0, if None, initialized to 1.0
        """
        super().__init__()

        # random init the parameters randomly if not provided:

        ## NOTE: if parameters are np or jnp arrays, they will be picked up by the optimizer during training.
        # if they are regular python floats, they will not be optimized, and instead remain constant.
        keys = jax.random.split(key, 8)
        self.interaction_params = (
            {  # keys celltypeA-celltypeB, values the interaction energy (e.g. J11, J12, ..)
                "1-1": jax.random.uniform(keys[0], shape=(1,), minval=-1.0, maxval=1.0),
                "1-2": jax.random.uniform(keys[1], shape=(1,), minval=-1.0, maxval=1.0),
                "0-0": jnp.array([0.0]),
                "0-1": jax.random.uniform(keys[3], shape=(1,), minval=-1.0, maxval=1.0),
                "0-2": jax.random.uniform(keys[4], shape=(1,), minval=-1.0, maxval=1.0),
                "2-2": jax.random.uniform(keys[5], shape=(1,), minval=-1.0, maxval=1.0),
            }
            if interaction_params is None
            else dict(interaction_params)
        )
        self.v_pref = (
            jax.random.uniform(keys[6], shape=(1,), minval=100, maxval=300)
            if v_pref is None
            else v_pref
        )  # preferred volume of the cells
        self.lamb = (
            jax.random.uniform(keys[7], shape=(1,), minval=0.1, maxval=2)
            if lamb is None
            else lamb
        )  # volume constraint strength
        self.num_cell_ids = cfg.num_cell_ids
        self.num_cell_types = cfg.num_cell_types
        self.J = jnp.zeros((self.num_cell_types, self.num_cell_types))
        for key, val in self.interaction_params.items():
            cell_types = key.split("-")
            self.J = self.J.at[int(cell_types[0]), int(cell_types[1])].set(val.item())
            self.J = self.J.at[int(cell_types[1]), int(cell_types[0])].set(
                val.item()
            )  # symmetric matrix

        # the following is used to scale and shift the J matrix
        # hopefully this improves learning dynamics since we can now learn the scale and mean of J separately
        # from the relative differences in interaction energy
        self.gamma_J = gamma_J
        self.bias_J = bias_J
        self.differentiable = False  # this model is not differentiable with respect to the input cpm, but it is differentiable with respect to its parameters.
        self.offset = offset
        self.offset_scale = offset_scale

        if self.offset is None or self.offset_scale is None:
            warnings.warn('offset or offset_scale None! this might lead to issues when loading eqx weights from disk if these were not None at time of saving. please verify if the weights are loaded correctly')
        print('Initialized CellsortHamiltonian')

    def __call__(self, cpm: jnp.ndarray):
        """
        forward pass of the symbolic Hamiltonian
        :param cpm: the cpm lattice (shape [2, h, w]), where channel 0 is the cell id channel, and channel 1 is the cell type channel
        :return: the energy of the system
        """
        eps = 1e-3
        cpm = jnp.asarray(cpm).astype(int)
        ham = 0
        volumes = self.calculate_cell_volumes(cpm, self.num_cell_ids)
        # volume constraint, dont take medium volume into account hence [1:], take softplus of the lamb coefficient to ensure positivity
        ham = ham + jnp.sum(
            ((volumes[1:] - self.v_pref) ** 2) #* (volumes[1:] > 0)  # only count cells with nonzero volume as these are the only cells that exist
        ) * (self._get_vol_cons_strength() + eps)
        h, w = cpm.shape[-2], cpm.shape[-1]
        J_symmetric = self._get_J()
        ham = ham + self.get_interaction_energy(
            cpm, J_symmetric, h, w, neighbor_order=2
        )
        # add an offset to the energy -- might help in the case fo regularization to center energy around 0,
        # and constant offsets do not matter for the dynamics.
        ham = ham + self.get_offset()
        return ham

    def get_offset(self):
        offset = self.offset if self.offset is not None else 0.0
        offset_scale = self.offset_scale if self.offset_scale is not None else 1.0
        out = offset * offset_scale
        return out

    def calculate_cell_volumes(self, cpm: jnp.ndarray, num_cell_ids: int):
        """
        calculate the volumes of the cells in the CPM lattice
        :param cpm: the CPM lattice -- channel 0 is the cell id, 1 cell type
        :return: jnp array of volumes of the cells, s.t. columes[cell_id] = volume of cell cell_id
        """

        cell_id_range = jnp.arange(num_cell_ids)
        volumes = self.vmap_calculate_cell_vol(cell_id_range, cpm)
        return volumes

    @staticmethod
    @partial(jax.vmap, in_axes=(0, None))
    def vmap_calculate_cell_vol(cell_id: int, cpm: jnp.ndarray):
        """
        calculate the volume of a single cell id
        :param cpm: the CPM lattice -- channel 0 is the cell id, 1 cell type
        :param cell_id: the cell id
        :return: the volume of the cell
        """
        cell_ids = cpm[0]
        mask = cell_ids == cell_id
        volume = jnp.sum(mask)
        return volume

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3, 4))
    def get_interaction_energy(cpm, J, h, w, neighbor_order=2):
        ii, jj = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
        energies = CellsortHamiltonian.vmap_interaction_for_loc(
            ii.flatten(), jj.flatten(), cpm, J, h, w, neighbor_order, False
        )
        offsets = CellsortHamiltonian.get_neighbor_offsets(neighbor_order)
        # We normalize by the number of neighbors. note: we have the convention to count both (i,j) as well as (j,i).
        # The length of offsets is half of the number of neighbors since we avoid the symmetric cases.
        # So we normalize by dividing by the number of neighbors (ie 8 for neighbor_order=2), and then multiply
        # the interaction energy by 2 to account for the symmetric cases.
        # which is the same as just dividing by len(offsets) = #neighbors/2
        interaction_energy = jnp.sum(energies) / len(offsets)
        return interaction_energy

    @staticmethod
    @partial(jax.vmap, in_axes=(0, 0, None, None, None, None, None, None))
    def vmap_interaction_for_loc(
        i, j, cpm, J, h, w, neighbor_order, return_symmetric=False
    ):
        """
        calculate the interaction energy for a single pixel at position i,j in the cpm lattice
        :param i, j: pixel location
        :param cpm: cpm lattice
        :param J: contact energy matrix, shape (num_cell_types, num_cell_types)
        :param h: height of the cpm lattice
        :param w: width of the cpm lattice
        :param neighbor_order: CPM neighbor order, see e.g. https://morpheus.gitlab.io/faq/modeling/neighborhoods/
        :param return_symmetric: whether to consider both (i,j) and (j,i) as neighbors (True), or count only one (False).
        :return: contact energy for the pixel at position i,j
        """
        cell_id = cpm[0][i, j]
        cell_type = cpm[1][i, j]
        offsets = CellsortHamiltonian.get_neighbor_offsets(
            neighbor_order, return_symmetric
        )
        energy = 0
        for offset_i, offset_j in offsets:
            # assume periodic boundary conditions:
            i_neigh = jnp.mod(i + offset_i, h).astype(int)
            j_neigh = jnp.mod(j + offset_j, w).astype(int)

            energy = energy + jnp.where(
                cell_id != cpm[0][i_neigh, j_neigh],
                J[cell_type, cpm[1][i_neigh, j_neigh]],
                0,
            )
        return energy

    @staticmethod
    def get_neighbor_offsets(neighbor_order=2, return_symmetric=False):
        # we return only half of the neighbors for each pixel to avoid counting every interaction twice!
        if neighbor_order == 1:
            arr_nonsymmetric = jnp.array(
                [[0, 1], [1, 0]]
            )  # jnp.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        elif neighbor_order == 2:
            arr_nonsymmetric = jnp.array([[0, 1], [1, 0], [1, 1], [1, -1]])
            # jnp.array([[0, 1], [0, -1], [1, 0], [-1, 0],[1, 1], [1, -1], [-1, 1], [-1, -1]])
        else:
            raise NotImplementedError(
                f"neighbor order {neighbor_order} not implemented"
            )

        if return_symmetric:
            return jnp.concatenate([arr_nonsymmetric, -1.0 * arr_nonsymmetric], axis=0)
        return arr_nonsymmetric

    def delta_energy(
        self, cpm, i_flip, j_flip, new_cell_id, new_cell_type, old_energy=None
    ):
        """
        calculate the energy difference of flipping the cell at position i,j
        shamelessly copied Leon's jax cpm implementation and made some minor adaptations to use the existing functions
        that we had already implemented -- see https://github.com/lhillma/differentiable-cpm.
        """
        cpm = jnp.asarray(cpm).astype(int)
        i, j = i_flip, j_flip
        lamb = self._get_vol_cons_strength()
        v_pref = self.v_pref
        J_symmetric = self._get_J()  # symmetric interaction matrix
        old_cell_id = cpm[0][i, j]
        old_volume_old_cell, old_volume_new_cell = self.vmap_calculate_cell_vol(
            jnp.array([old_cell_id, new_cell_id]), cpm
        )

        new_volume_old_cell = old_volume_old_cell - 1
        new_volume_new_cell = old_volume_new_cell + 1

        d_volume_energy_old_cell = jax.lax.cond(
            old_cell_id == 0,
            lambda: jnp.array([0.0]),
            lambda: lamb
            * (
                (new_volume_old_cell - v_pref) ** 2
                - (old_volume_old_cell - v_pref) ** 2
            ),
        )
        d_volume_energy_new_cell = jax.lax.cond(
            new_cell_id == 0,
            lambda: jnp.array([0.0]),
            lambda: lamb
            * (
                (new_volume_new_cell - v_pref) ** 2
                - (old_volume_new_cell - v_pref) ** 2
            ),
        )

        # prevent cells disappearing by returning inf energy for cells with 0 volume that had nonzero volume before:
        zero_vol_penalize = jax.lax.cond(
            jnp.logical_or(
                jnp.logical_and(new_volume_old_cell == 0, old_volume_old_cell > 0),
                jnp.logical_and(new_volume_new_cell == 0, old_volume_new_cell > 0)
            ),
            lambda: jnp.inf,
            lambda: 0.0,
        )

        old_interaction_energy_ij = self.vmap_interaction_for_loc(
            jnp.array([i]),
            jnp.array([j]),
            cpm,
            J_symmetric,
            cpm.shape[-2],
            cpm.shape[-1],
            2,
            True,
        )[0]

        cpm_proposed = cpm.at[:, i, j].set(jnp.array([new_cell_id, new_cell_type]))
        new_interaction_energy = self.vmap_interaction_for_loc(
            jnp.array([i]),
            jnp.array([j]),
            cpm_proposed,
            J_symmetric,
            cpm.shape[-2],
            cpm.shape[-1],
            2,
            True,
        )[0]

        offsets = CellsortHamiltonian.get_neighbor_offsets(2, return_symmetric=True)

        return (
            d_volume_energy_old_cell
            + d_volume_energy_new_cell
            + 2 * new_interaction_energy / len(offsets)
            - 2 * old_interaction_energy_ij / len(offsets)
            + zero_vol_penalize
        )

    def _get_J(self):
        J_symmetric = (self.J.T + self.J) / 2.0
        J_out = jax.nn.softplus(self.gamma_J) * J_symmetric + self.bias_J
        J_out = J_out.at[0, 0].set(0.0)  # set the medium-medium interaction to 0 (medium = cell 0 = background)
        return J_out

    def _get_vol_cons_strength(self):
        return jax.nn.softplus(self.lamb)

    def get_metrics(self):
        params = {}
        J = self._get_J()
        for i in range(J.shape[0]):
            for j in range(i, J.shape[1]):
                params[f"J_{i}_{j}"] = J[i, j]
        params["lamb_vol"] = self._get_vol_cons_strength()
        params['v_pref'] = self.v_pref
        params['offset'] = self.get_offset()
        return params


class ExternalFieldHamiltonian(CellsortHamiltonian):
    field: jnp.ndarray
    field_coupling: jnp.ndarray

    def __init__(
        self,
        key: jax.random.PRNGKey,
        cfg: ConfigDict,
        field: jnp.ndarray = None,
        field_coupling: jnp.ndarray = None,
        **kwargs
    ):
        """
        External field Hamiltonian that adds an external field energy term to the CellsortHamiltonian.
        :param key: jax prng key
        :param cfg: config containing at least num_cell_ids and num_cell_types
        :param field: external field to be applied to the cells, shape (h, w) where h and w are the height and width of the cpm lattice
        :param field_coupling: shape (num_cell_types,) that defines how strongly the external field couples to each cell type.
        :param kwargs: other init kwargs for CellsortHamiltonian
        """
        super().__init__(key, cfg, **kwargs)
        self.field = field if field is not None else jnp.zeros(cfg.grid_size)
        self.field_coupling = field_coupling if field_coupling is not None else jnp.zeros(cfg.num_cell_types)
        self.differentiable = False
        print('Initialized ExternalFieldHamiltonian')

    def _get_external_field_energy(self, i: int, j: int, cell_type: int):
        return self.field[i, j] * self.field_coupling[cell_type]

    def delta_energy(
        self, cpm, i_flip, j_flip, new_cell_id, new_cell_type, old_energy=None
    ):
        delta_energy = super().delta_energy(
            cpm, i_flip, j_flip, new_cell_id, new_cell_type, old_energy
        )

        delta_potential = self._get_external_field_energy(
            i_flip, j_flip, new_cell_type
        ) - self._get_external_field_energy(i_flip, j_flip, cpm[1][i_flip, j_flip])

        return delta_energy + delta_potential

    def __call__(self, cpm: jnp.ndarray):
        # misc model inputs is not used for this model.
        energy = super().__call__(cpm)
        external_field_energy = jnp.sum(
            self.field * self.field_coupling[cpm[1].astype(int)]
        )

        return energy + external_field_energy


class DifferentiableCellsortHamiltonian(CellsortHamiltonian):

    differentiable: bool

    def __init__(self, *args, **kwargs):
        """
        Identical to CellsortHamiltonian, but made 'differentiable' with respect to the input cpm.
        One-hot encodes the input and can calculate the gradient fo the energy wrt the one-hot encoded space.
        Downside is that this will be slower than the normal CellsortHamiltonian,
        since it operates on the larger one-hot encoded tensor.
        """
        super().__init__(*args, **kwargs)
        self.differentiable = True

    def __call__(self, cpm):
        """
        forward pass of the symbolic Hamiltonian
        cpm: the cpm lattice (shape [2, h, w]), where channel 0 is the cell id channel, and channel 1 is the cell type channel
        :return: the energy of the system
        """

        cell_id_onehot, cell_type_onehot = self.get_onehot_input(cpm)
        ham = self.calc_energy_on_onehot(cell_id_onehot, cell_type_onehot)

        return ham + self.get_offset()

    def get_onehot_input(self, cpm: jnp.ndarray):

        # first, one-hot encode the cell_id input
        cell_id_onehot = jax.nn.one_hot(
            cpm[0], self.num_cell_ids
        )  # shape (h, w, num_cell_ids)
        cell_type_onehot = jax.nn.one_hot(
            cpm[1], self.num_cell_types
        )  # shape (h, w, num_cell_types)

        return cell_id_onehot, cell_type_onehot

    def calc_energy_on_onehot(self, cell_id_onehot: jnp.ndarray, cell_type_onehot: jnp.ndarray):
        """
        forward pass of the symbolic Hamiltonian, implemented differentiable with respect to the one-hot encoded input
        Because for differentiating we need to operate on the one-hot encoded input, the shape is different than
        the input shape of CellSortHamiltonian. Further, the implementation is less efficient because of operating
        on the larger one-hot encoded tensor.
        :cell_id_onehot: the one-hot encoded cell id lattice -- shape [h, w, num_cell_ids]
        :cell_type_onehot: the one-hot encoded cell type lattice -- shape [h, w, num_cell_types]
        :return: the energy of the system
        """

        # for the differentiable variant, we expect the cpm lattice to be one-hot encoded!
        assert cell_id_onehot.shape[-1] == self.num_cell_ids
        assert cell_type_onehot.shape[-1] == self.num_cell_types
        eps = 1e-3


        # func for getting the type oh vector for a cell_id:
        get_type_oh_for_id = lambda cell_id: jnp.max(
            (cell_id_onehot[..., cell_id] == 1)[..., None] * cell_type_onehot,
            axis=(0, 1),
        )
        id_to_type_matrix = jax.vmap(get_type_oh_for_id)(
            jnp.arange(self.num_cell_ids)
        )  # shape (num_cell_ids, num_cell_types) -- id_to_type_matrix[cell_id, cell_type] = 1 if cell_id has type cell_type else 0
        # recalculate the cell type from th cell_id so that gradients flow to cell_id (which is the real input we care about)
        cell_type_onehot_from_id = jnp.einsum(
            "hwi,ij->hwj", cell_id_onehot, id_to_type_matrix
        )  # shape(h, w, num_cell_types)

        cell_type_onehot = cell_type_onehot_from_id

        ham = 0
        volumes = self.calculate_cell_volumes(cell_id_onehot)

        ham = ham + jnp.sum(
            ((volumes[1:] - self.v_pref) ** 2) #* (volumes[1:] > 0)  # only count cells with nonzero volume as these are the only cells that exist
        ) * (self._get_vol_cons_strength() + eps) # volume constraint, dont take medium volume into account hence 1:

        h, w = cell_id_onehot.shape[0], cell_id_onehot.shape[1]
        J_symmetric = self._get_J()
        ham = ham + self.get_interaction_energy(
            cell_id_onehot,
            cell_type_onehot,
            J_symmetric,
            h,
            w,
            self.num_cell_ids,
            self.num_cell_types,
            neighbor_order=2,
        )

        return ham

    def calculate_cell_volumes(
        self, cell_id_onehot: jnp.ndarray, num_cell_ids=None
    ):
        """
        calculate the volumes of the cells in the CPM lattice
        :param cpm: one-hot encoding of the cell_id lattice -- shape [h, w, num_cell_ids]
        :param num_cell_ids: not used for this function since we can infer it from the shape of the one-hto input
        :return: jnp array of volumes of the cells, s.t. arr[cell_id] = volume
        """
        volumes = jnp.sum(cell_id_onehot, axis=(0, 1))
        return volumes

    @staticmethod
    @partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
    def get_interaction_energy(
        cell_id_onehot: jnp.array,
        cell_type_onehot: jnp.array,
        J,
        h,
        w,
        num_cell_ids,
        num_cell_types,
        neighbor_order=2,
    ):
        ii, jj = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")

        # func for getting the type oh vector for a cell_id:
        get_type_oh_for_id = lambda cell_id: jnp.max(
            (cell_id_onehot[..., cell_id] == 1)[..., None] * cell_type_onehot,
            axis=(0, 1),
        )
        id_to_type_matrix = jax.vmap(get_type_oh_for_id)(jnp.arange(num_cell_ids))

        # we now define contact energies at the cell_id level instead of cell_type level so that gradients can flow to the cell_id input:
        J_cell_id_cell_id = id_to_type_matrix @ J @ id_to_type_matrix.T

        # fill the diagonal with 0s:
        J_cell_id_cell_id = jnp.fill_diagonal(J_cell_id_cell_id, 0.0, inplace=False)

        energies = DifferentiableCellsortHamiltonian.vmap_interaction_for_loc(
            ii.flatten(),
            jj.flatten(),
            cell_id_onehot,
            J_cell_id_cell_id,
            h,
            w,
            neighbor_order,  # J+J.T / 2 to ensure that the matrix remains symmetric
        )
        offsets = CellsortHamiltonian.get_neighbor_offsets(neighbor_order)
        # We normalize by the number of neighbors. note: we have the convention to count both (i,j) as well as (j,i).
        # The length of offsets is half of the number of neighbors since we avoid the symmetric cases.
        # So we normalize by dividing by the number of neighbors (ie 8 for neighbor_order=2), and then multiply
        # the interaction energy by 2 to account for the symmetric cases.
        # which is the same as just dividing by len(offsets) = #neighbors/2
        interaction_energy = jnp.sum(energies) / len(offsets)
        return interaction_energy

    @staticmethod
    @partial(jax.vmap, in_axes=(0, 0, None, None, None, None, None))
    def vmap_interaction_for_loc(
        i, j, cell_id_onehot, J_cell_id_cell_id, h, w, neighbor_order
    ):
        id_onehot_ij = cell_id_onehot[i, j]  # shape [num_cell_ids]
        offsets = CellsortHamiltonian.get_neighbor_offsets(neighbor_order)
        energy = 0
        for offset_i, offset_j in offsets:
            # assume periodic boundary conditions:
            i_neigh = jnp.mod(i + offset_i, h).astype(int)
            j_neigh = jnp.mod(j + offset_j, w).astype(int)
            id_oh_neigh = cell_id_onehot[i_neigh, j_neigh]
            energy = energy + id_onehot_ij.T @ J_cell_id_cell_id @ id_oh_neigh
        return energy


    def delta_energy(self, cpm, i_flip, j_flip, new_cell_id, new_cell_type, old_energy=None):
        # propose the new cpm
        cpm = jnp.asarray(cpm).astype(int)
        cpm_proposed = cpm.at[:, i_flip, j_flip].set(
            jnp.array([new_cell_id, new_cell_type])
        )
        # calculate the hamiltionian delta
        new_energy = self.__call__(cpm_proposed)[0]
        old_energy = self.__call__(cpm)[0] if old_energy is None else old_energy
        delta = new_energy - old_energy  # how much increase in energy?
        return delta


class NeuralHamiltonian(HamiltonianBaseClass):
    num_cell_ids: int
    num_cell_types: int
    nh_layers: list
    to_emb_before_agg: Callable
    conv_module_layers: list
    mlp_layers: list
    pools: list
    pools_conv: list
    embedding_module: Callable
    differentiable: bool
    mask_interactions: bool  # used for rebuttal experiments



    def __init__(
        self,
        key: jax.random.PRNGKey,
        cfg: ConfigDict,
        num_layers_per_block: int = 1,
        spatial_downsampling_per_block: Union[int, Tuple[int]] = 1,
        node_emb_dims: tuple[int] = (8,),
        edge_emb_dims: tuple[int] = (8,),
        emb_dim_before_agg: int = 8,
        emb_dims_conv: tuple[int] = tuple(),
        spatial_downsampling_per_block_conv: Union[int, Tuple[int]] = 1,
        num_layers_per_block_conv:int = 1,
        emb_dims_mlp: tuple[int] = (8,),
        activation: Callable = jax.nn.silu,
        mask_interactions=None,
        use_residual=False,
        embedding_module=nn.Conv2d,
        embedding_module_kwargs=None,
        kernel_size=3,
        **kwargs,
    ):
        """
        Neural Hamiltonian model
        :param key: jax PRNG key
        :param cfg: configuration dict containing at least num_cell_ids and num_cell_types
        :param num_layers_per_block: number of layers per NH layer block
        :param spatial_downsampling_per_block: spatial downsampling factor per block;
         int to have uniform downsampling for all blocks, or tuple/list for specifying per-block downsampling rates
        :param node_emb_dims: tuple of node embedding dimensions per block
        :param edge_emb_dims: tuple of edge embedding dimensions (A in figure 2 of the paper) per block
        :param emb_dim_before_agg: embedding dimension before the global pooling step
        :param emb_dims_conv: tuple of embedding dimensions for convolutional layers after pooling over all cells, if any
        :param spatial_downsampling_per_block_conv: spatial downsampling factor per block for the convolutional layers
            int to have uniform downsampling for all blocks, or tuple/list for specifying per-block downsampling rates
        :param num_layers_per_block_conv: number of layers per convolutional block
        :param emb_dims_mlp: tuple of embedding dimensions for the MLP layers after the convolutional layers (or global pooling layers)
        :param activation: activation function to use
        :param mask_interactions: whether to mask out any interaction information between cells in the NH layers
        :param use_residual: whether to use residual connections
        :param embedding_module: the embedding module to use for the initial embedding of the cpm lattice
        :param embedding_module_kwargs: dict of additional kwargs for the embedding module, e.g. kernel_size, padding, etc.
        :param kernel_size: kernel size for Conv layers
        """
        super().__init__()

        if embedding_module_kwargs is None:
            embedding_module_kwargs = dict(kernel_size=kernel_size, padding='SAME')

        if 'use_residual' not in kwargs.keys():
            kwargs['use_residual'] = use_residual  # also pass this down to the griddeepset layers

        # initialize some stuff to store:
        num_cell_ids = cfg.num_cell_ids
        num_cell_types = cfg.num_cell_types
        self.nh_layers = []
        self.num_cell_ids = num_cell_ids
        self.num_cell_types = num_cell_types
        self.differentiable = True

        # initial cell-wise embedding module:
        key, use_key = jax.random.split(key)
        self.embedding_module = embedding_module(
            key=use_key, in_channels=1+num_cell_types, out_channels=node_emb_dims[0],
            **embedding_module_kwargs
        )


        # start defining the NH layers:
        key, use_key = jax.random.split(key)
        num_blocks = len(node_emb_dims)
        keys = jax.random.split(use_key, num_blocks)
        all_channels_nh = (node_emb_dims[0], *node_emb_dims)  # first block as same emb dim input and output

        # define pool2d layers:
        if not hasattr(spatial_downsampling_per_block, '__len__'):
            spatial_downsampling_per_block = [spatial_downsampling_per_block]*num_blocks
        self.pools = [
            eqx.nn.MaxPool2d(
            kernel_size=s,
            stride=s,
            ) for s in spatial_downsampling_per_block
        ]

        # define NH blocks
        for i in range(num_blocks):
            self.nh_layers.append(
                NHBlock(
                    key=keys[i],
                    in_channels=all_channels_nh[i],
                    num_layers=num_layers_per_block,
                    node_emb_dim=all_channels_nh[i + 1],
                    edge_emb_dim=edge_emb_dims[i],
                    activation=activation,
                    kernel_size=kernel_size,
                    **kwargs,
                )
            )



         # define embedding before aggregation:
        key, use_key = jax.random.split(key)
        self.to_emb_before_agg = eqx.nn.Conv2d(
            key=use_key,
            in_channels=all_channels_nh[-1],
            out_channels=emb_dim_before_agg,
            kernel_size=1,
        )

        # define convolutional postprocessing layers (used only for 1 NH layer + CNN model in paper):
        if not hasattr(spatial_downsampling_per_block_conv, '__len__'):
            spatial_downsampling_per_block_conv = [spatial_downsampling_per_block_conv] * len(emb_dims_conv)
        self.conv_module_layers = []
        all_channels_conv = (emb_dim_before_agg, *emb_dims_conv)
        for i in range(len(emb_dims_conv)):
            layers_this_block = []
            for j in range(num_layers_per_block_conv):
                key, use_key = jax.random.split(key)
                layers_this_block.append(
                    eqx.nn.Conv2d(
                        key=use_key,
                        in_channels=all_channels_conv[i if j == 0 else i+1],
                        out_channels=all_channels_conv[i + 1],
                        kernel_size=kernel_size,
                        padding='SAME'
                    )
                )
                layers_this_block.append(activation)
            key, use_key = jax.random.split(key)
            convblock = ConvNet(layers_this_block, use_residual=use_residual, key=use_key)
            self.conv_module_layers.append(convblock)
        self.pools_conv = [
            eqx.nn.MaxPool2d(
                kernel_size=s,
                stride=s,
            ) for s in spatial_downsampling_per_block_conv
        ]

        # MLP head:
        self.mlp_layers = []
        key, use_key = jax.random.split(key)
        all_channels_mlp = (all_channels_conv[-1], *emb_dims_mlp)
        for i in range(len(emb_dims_mlp)):
            layers_this = []
            key, use_key = jax.random.split(key)
            layers_this.append(
                eqx.nn.Linear(all_channels_mlp[i], all_channels_mlp[i + 1], key=use_key)
            )
            layers_this.append(activation)
            key, use_key = jax.random.split(key)
            self.mlp_layers.append(MLP(layers_this, use_residual=use_residual, key=use_key))
        key, use_key = jax.random.split(key)
        self.mlp_layers.append(eqx.nn.Linear(all_channels_mlp[-1], 1, key=use_key))

        self.mask_interactions = mask_interactions

        print("NeuralHamiltonian initialized")

    def __call__(self, cpm: jnp.ndarray):
        """
        forward pass of the symbolic Hamiltonian
        cpm: the cpm lattice (shape [2, h, w]), where channel 0 is the cell id channel, and channel 1 is the cell type channel
        :return: the energy of the system
        """

        cell_id_onehot, cell_type_onehot = self.get_onehot_input(cpm)
        ham = self.calc_energy_on_onehot(cell_id_onehot, cell_type_onehot)

        return ham

    def get_onehot_input(self, cpm: jnp.ndarray):
        # first, one-hot encode the cell_id input towards the shape (num_cells, 1, h, w)
        cell_id_onehot = jax.nn.one_hot(
            cpm[0], self.num_cell_ids
        )  # shape (h, w, num_cell_ids)
        cell_type_onehot = jax.nn.one_hot(
            cpm[1], self.num_cell_types
        )  # shape (h, w, num_cell_types)

        return cell_id_onehot, cell_type_onehot

    def calc_energy_on_onehot(self, cell_id_onehot, cell_type_onehot):
        """
        fw pass of the neural hamiltonian
        :param cell_id_onehot: one-hot encoding of the cell_id channel (channel 0) in the cpm array
        :param cell_type_onehot: one-hot encoding of the cell_type channel (channel 1) in the cpm array
        :return: energy
        """

        x = self._apply_nh_layers(cell_id_onehot, cell_type_onehot)

        # finally, mapping to a scalar hamiltonian function:
        # first, map the embedding to a possibly higher dim to prevent a bottleneck:
        assert x.shape[-1] > 0 and x.shape[-2] > 0, 'shape of x is not as expected'
        x = jax.vmap(self.to_emb_before_agg)(
            x
        )  # shape (num_cell_ids, c, h, w) ->(num_cell_ids, emb_dim_before_agg, h, w)

        # perform a permutation invariant aggregation, preserving the spatial axes:
        x = x.sum(axis=0)

        # process the permutation invariant embedding with conv layers, if any are specified:
        for i, layer in enumerate(self.conv_module_layers):
            x = layer(x)
            pool = self.pools_conv[i]
            x = pool(x)

        # perform spatial aggregation
        x = x.sum(axis=(-2, -1))  # shape (emb_dim_before_agg,)

        # finally, map to the scalar energy:
        for layer in self.mlp_layers:
            x = layer(x)
        ham = x

        return ham

    def _apply_nh_layers(self, cell_id_onehot, cell_type_onehot
                         ):

        # we need to get a differentiable one-hot encoding of the cell type as a function of the cell_id,
        # in case we need the grad of the hamiltonian wrt cell_id input
        get_type_oh_for_id = lambda cell_id: jnp.max(
            (cell_id_onehot[..., cell_id] == 1)[..., None] * cell_type_onehot,
            axis=(0, 1),
        )
        id_to_type_matrix = jax.vmap(get_type_oh_for_id)(
            jnp.arange(self.num_cell_ids)
        )  # shape (num_cell_ids, num_cell_types)
        cell_type_onehot_from_id = jnp.einsum(
            "hwi,ij->hwj", cell_id_onehot, id_to_type_matrix
        )  # shape(h, w, num_cell_types)

        cell_id_onehot = jnp.transpose(cell_id_onehot, (2, 0, 1))[:, None]  # shape (num_cell_ids, 1, h, w)
        cell_type_onehot = jnp.transpose(cell_type_onehot_from_id, (2, 0, 1))[None,
                           :]  # shape (1, num_cell_types, h, w)

        input_to_model = jnp.concatenate(
            [cell_id_onehot, jnp.repeat(cell_type_onehot, self.num_cell_ids, axis=0)],
            axis=1,
        )  # shape (num_cell_ids, 1+num_cell_types, h, w)
        # note:
        # - input_to_model[cell_id, 0, i,j] = 1 iff cpm[0, i, j] == cell_id.
        # - input_to_model[cell_id, 1+type, i,j] = 1 iff cpm[1, i, j] == type

        if self.mask_interactions:
            input_to_model_masked = input_to_model[:, 0:1] * input_to_model  # each cell only sees itself and its own type
            input_to_model = input_to_model_masked

        # first call the embedding module independently on each cell
        x = jax.vmap(self.embedding_module)(input_to_model)

        # processing by the NH layers:
        for i, layer in enumerate(self.nh_layers):
            x = layer(x)
            pool = self.pools[i]
            x = jax.vmap(pool)(x)  # spatial downsampling - node wise
        return x

    def delta_energy(self, cpm, i_flip, j_flip, new_cell_id, new_cell_type, old_energy=None):
        # propose the new cpm
        cpm = jnp.asarray(cpm).astype(int)
        cpm_proposed = cpm.at[:, i_flip, j_flip].set(
            jnp.array([new_cell_id, new_cell_type])
        )
        # calculate the hamiltionian delta
        new_energy = self.__call__(cpm_proposed)[0]
        old_energy = self.__call__(cpm)[0] if old_energy is None else old_energy
        delta = new_energy - old_energy  # how much increase in energy?
        return delta

    def get_metrics(self):
        result = {}
        return result

class NeuralClosureHamiltonian(NeuralHamiltonian):
    basis_model: callable
    weight_basis: jnp.ndarray
    weight_neural: jnp.ndarray

    def __init__(self, key, cfg, basis_model=DifferentiableCellsortHamiltonian,
                 basis_model_kwargs: dict = {},
                 weight_basis=jnp.array([1.]),
                 weight_neural=jnp.array([1.]), **kwargs):
        """
        Neural Hamiltonian as closure term on a basis_model callable (e.g. CellsortHamiltonian).
        :param key: jax PRNG key
        :param cfg: configdict
        :param basis_model: Callable that implements a basis model, e.g. CellsortHamiltonian, on top of which the NH acts
        :param basis_model_kwargs: any kwargs to init the basis model
        :param weight_basis: weight of basis model term (jnp or np array -> trained by optimizer, float -> constant)
        :param weight_neural: weight of neural closure term (jnp or np array -> trained by optimizer, float -> constant)
        :param kwargs: any other kwargs to pass to the NeuralHamiltonian init
        """
        key, use_key = jax.random.split(key)
        super().__init__(key=use_key, cfg=cfg, **kwargs)
        self.basis_model = basis_model(key=key, cfg=cfg, **basis_model_kwargs)
        self.weight_basis = weight_basis
        self.weight_neural = weight_neural
        assert self.basis_model.differentiable, 'basis model must be differentiable!'
        assert self.differentiable, 'I must be differentiable!'
        print('Neural Closure Hamiltonian initialized')

    def calc_energy_on_onehot(self, cell_id_onehot, cell_type_onehot):
        # basis model energy + neural closure term
        ham_basis = self.basis_model.calc_energy_on_onehot(cell_id_onehot, cell_type_onehot)
        ham_neural = super().calc_energy_on_onehot(cell_id_onehot, cell_type_onehot)
        ham = ham_basis * self.weight_basis + ham_neural * self.weight_neural
        return ham

    def get_metrics(self):
        result = super().get_metrics()
        result = {**result, **self.basis_model.get_metrics()}
        result = {**result,
                  'weight_neural': self.weight_neural,
                  'weight_basis': self.weight_basis}
        return result



class SimpleConvNeuralHamiltonian(NeuralHamiltonian, HamiltonianBaseClass):

    conv_layers: list
    differentiable: bool


    def __init__(
        self,
        key: jax.random.PRNGKey,
        cfg: ConfigDict,
        num_layers_per_block: int = 1,
        spatial_downsampling_per_block: int = 1,
        node_emb_dims: tuple[int] = (8,),
        emb_dim_before_agg: int = 8,
        emb_dims_mlp: tuple[int] = (8,),
        activation: Callable = jax.nn.silu,
        use_residual=False,
        embedding_module=nn.Conv2d,
        embedding_module_kwargs=None,
        **kwargs,
    ):
        """
        just a straightforward CNN operating on the one-hot encoded cpm input.
        :param key:
        :param cfg:
        :param num_layers_per_block:
        :param spatial_downsampling_per_block:
        :param node_emb_dims:
        :param emb_dim_before_agg:
        :param emb_dims_mlp:
        :param activation:
        :param use_residual:
        :param embedding_module:
        :param embedding_module_kwargs:
        :param num_hist_observations:
        :param kwargs:
        """
        eqx.Module.__init__(self)
        if embedding_module_kwargs is None:
            embedding_module_kwargs = dict(kernel_size=1, padding='SAME')

        if 'use_residual' not in kwargs.keys():
            kwargs['use_residual'] = use_residual  # also pass this down to the griddeepset layers
        num_cell_ids = cfg.num_cell_ids
        num_cell_types = cfg.num_cell_types
        self.conv_layers = []
        self.num_cell_ids = num_cell_ids
        self.num_cell_types = num_cell_types

        key, use_key = jax.random.split(key)
        self.embedding_module = embedding_module(
            key=use_key,
            in_channels=num_cell_types + num_cell_ids,
            out_channels=node_emb_dims[0],
            **embedding_module_kwargs
        )

        num_blocks = len(node_emb_dims)

        key, use_key = jax.random.split(key)
        keys = jax.random.split(use_key, num_blocks*num_layers_per_block)
        all_channels = (node_emb_dims[0], *node_emb_dims)  # first block as same emb dim input and output
        if not hasattr(spatial_downsampling_per_block, '__len__'):
            spatial_downsampling_per_block = [spatial_downsampling_per_block]*num_blocks
        self.pools = [eqx.nn.MaxPool2d(
            kernel_size=s,
            stride=s,
        ) for s in spatial_downsampling_per_block
        ]

        for i in range(num_blocks):
            layers_this = []
            for j in range(num_layers_per_block):
                key, use_key = jax.random.split(key)
                layers_this.append(
                    eqx.nn.Conv2d(all_channels[i] if j == 0 else all_channels[i + 1],
                                  all_channels[i + 1], kernel_size=3, key=use_key, padding='same')
                )
                layers_this.append(activation)
            key, use_key = jax.random.split(key)
            self.conv_layers.append(ConvNet(layers_this, use_residual=use_residual, key=use_key))

        key, use_key = jax.random.split(key)
        self.to_emb_before_agg = eqx.nn.Conv2d(
            key=use_key,
            in_channels=all_channels[-1],
            out_channels=emb_dim_before_agg,
            kernel_size=1,
        )

        self.mlp_layers = []
        key, use_key = jax.random.split(key)
        all_channels_mlp = (emb_dim_before_agg, *emb_dims_mlp)
        for i in range(len(emb_dims_mlp)):
            layers_this = []
            key, use_key = jax.random.split(key)
            layers_this.append(
                eqx.nn.Linear(all_channels_mlp[i], all_channels_mlp[i + 1], key=use_key)
            )
            layers_this.append(activation)
            key, use_key = jax.random.split(key)
            self.mlp_layers.append(MLP(layers_this, use_residual=use_residual, key=use_key))
        key, use_key = jax.random.split(key)
        self.mlp_layers.append(eqx.nn.Linear(all_channels_mlp[-1], 1, key=use_key))
        self.differentiable = True

        # initialize the below to dummy variables since they are not used:
        self.nh_layers = None
        self.conv_module_layers = None
        self.pools_conv=None
        self.mask_interactions = None
        print("Simple non-equivariant Convolutional NeuralHamiltonian initialized")

    def calc_energy_on_onehot(self, cell_id_onehot, cell_type_onehot, **kwargs):

        for k, v in kwargs.items():
            warnings.warn(f'ignoring input {k} in {self.__class__.__name__} since this model does not support it')

        # we need to get a differentiable one-hot encoding of the cell type as a function of the cell_id,
        # in case we need the grad of the hamiltonian wrt cell_id input
        get_type_oh_for_id = lambda cell_id: jnp.max(
            (cell_id_onehot[..., cell_id] == 1)[..., None] * cell_type_onehot,
            axis=(0, 1),
        )
        id_to_type_matrix = jax.vmap(get_type_oh_for_id)(
            jnp.arange(self.num_cell_ids)
        )  # shape (num_cell_ids, num_cell_types)
        cell_type_onehot_from_id = jnp.einsum(
            "hwi,ij->hwj", cell_id_onehot, id_to_type_matrix
        )  # shape(h, w, num_cell_types)

        cell_type_onehot = jnp.transpose(cell_type_onehot_from_id, (2, 0, 1))  # shape (num_cell_types, h, w)
        cell_id_onehot = jnp.transpose(cell_id_onehot, (2, 0, 1))  # shape (num_cell_ids, h, w)

        input_to_model = jnp.concatenate(
            [cell_id_onehot, cell_type_onehot],
            axis=0,
        )  # shape (num_cell_ids, 1+num_cell_types, h, w)

        # first call the embedding module independently on each cell
        x = self.embedding_module(input_to_model)

        # processing by the conv layers:
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            pool = self.pools[i]
            x = pool(x)  # spatial downsampling

        x = self.to_emb_before_agg(x)  # shape (num_cell_ids, emb_dim_before_agg, h, w)
        # perform the aggregation -- needs to be invariant to translations
        x = x.sum(axis=(-2, -1))  # shape (emb_dim_before_agg,)
        # finally, map to the scalar energy:
        for i, layer in enumerate(self.mlp_layers):
            x = layer(x)
        ham = x
        # jax.debug.print('energy: {}, shape: {}', ham, ham.shape)
        return ham

    def get_metrics(self):
        result = {}
        return result
