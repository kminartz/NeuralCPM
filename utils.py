import os
import pickle as pkl
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt
try:
    import wandb
    WANDB_IMPORTED = True
except ImportError:
    print('wandb not imported, cannot log to wandb')
    WANDB_IMPORTED = False
from scipy.ndimage import label, binary_dilation, generate_binary_structure
print('importing utils')



###### IO related utils:
def make_dir_safe(dir):
    """
    create a directory if it does not exist
    :param dir: the directory
    :return: the directory
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def load_model_weights(config, model, num_parent_directories=0, path=None):
    """

    :param config: root config dict
    :param model: model to load the weights into
    :param num_parent_directories: number of parent directories to go up to, from where to find the model weight path
    :return: model with loaded weights
    """
    if path is None:
        path = config.training_config.model_weight_path
    for _ in range(num_parent_directories):
        path = os.path.join('..', path)
    path = os.path.abspath(path)
    print('loading model from path:', path)

    # if we saved something as scalar, but initialized it as shape 1 array, we need to convert the int to the array
    def saved_arr_init_scalar(init_val, loaded_val):
        out = False
        if isinstance(init_val, jnp.ndarray):
            out = len(jnp.array(loaded_val).shape) == 0 and init_val.shape == (1,)
        return out

    def to_arr_if_needed(init_val, loaded_val):
        if saved_arr_init_scalar(init_val, loaded_val):
            return jnp.array([loaded_val]).astype(init_val.dtype)
        return loaded_val

    filter_spec = lambda f, x: to_arr_if_needed(x, eqx.default_deserialise_filter_spec(f, x))

    model = eqx.tree_deserialise_leaves(path, model, filter_spec=filter_spec)
    return model

def save_model_weights(config, model, suffix=''):
    path = config.training_config.model_weight_path
    base = os.path.basename(path)
    n, ext = os.path.splitext(base)
    path = os.path.join(os.path.dirname(path), n + suffix + ext)
    # print('saving model to path:', os.path.abspath(path))
    eqx.tree_serialise_leaves(path, model)
    return path


def load_data_from_file(path):
    """
    load data from a file npy/npz file
    :param path: path to file
    :return: np array with a datapoint
    """
    if path.endswith('.npz'):
        data = np.load(path)['data']  # this is tested and works
    elif path.endswith('.npy'):
        data = np.load(path)  # did not test these yet
    else:
        raise NotImplementedError('file format not implemented!')

    if data.shape[-1] == 2:  # we have channels last -- original shape of data: (time, h, w, c)
        data = data.transpose(0, 3, 1, 2)  # now (time, channels, h, w)
    assert data.shape[1] == 2, 'data shape not as expected!'
    assert len(data.shape) == 4, 'data shape not as expected!'
    assert data.shape[-2] == data.shape[-1], 'expected square grid!'
    return data  # shape: (time, channels, h, w) where channels = 2 (cell id and cell type)


###### config, initialization, and command line parsing utils:
def parse_str_to_int_or_float(str):
    try:
        return int(str)
    except ValueError:
        return float(str)

def load_config(cfg_str, remaining_args=None):
    import configs
    cfg_module = getattr(configs, cfg_str)
    cfg_dict = cfg_module.get_config(remaining_args)
    return cfg_dict

def overwrite_config(cfg, remaining_args):
    """
    overwrite the config with the arguments given in the command line
    :param cfg: the config dict
    :param remaining_args: the remaining arguments from the command line
    :return: the updated (in place) config dict
    """
    for arg in remaining_args:  # any argument given as --kwarg=x after the config file will be parsed
        # and added to the config dict or overwrite the parameters in the config dict it they are already present
        arg: str
        arg = arg.strip('-')
        k, v = arg.split('=')
        try:
            v = parse_str_to_int_or_float(v)
        except:
            v = v
        try:
            keys_nested = k.split('.')
        except:
            keys_nested = [k]

        # go in the nested dict to update the values of the (sub-)dict
        d_to_update = cfg
        for k in keys_nested:
            if k not in d_to_update and k != keys_nested[-1]:
                d_to_update[k] = {}
            if k == keys_nested[-1]:
                if k in d_to_update.keys():
                    try:
                        if isinstance(d_to_update[k], jnp.ndarray):
                            v = jnp.array([v])
                        else:
                            v = type(d_to_update[k])(v)  # cast to the type of the original value
                    except:
                        print(f'could not convert {v} to type of {d_to_update[k]} -- trying continuing with original type')
                d_to_update[k] = v
                print(f'overwriting config key {k} with value {v}')
            d_to_update = d_to_update[k]

    return cfg


def initialize_model(config, key=None, **kwargs):
    if key is None:
        if 'PRNGKey' in config.keys():
            key = config.PRNGKey
        else:
            key = jax.random.PRNGKey(0)
    model = config.model(key, cfg=config, **kwargs)
    return model

###### CPM data wrangling and calculation utils:
def concat_type_from_dict(cpm: np.ndarray, id_to_type_dict: dict, axis=-1):
    """
    Concatenate the cell type information from a dict to the cpm array
    Adds 1 to each type since we assume type 0 to be reserved for the medium, but
    Morpheus assumes type 0 to be the first cell type (there is no type for medium in morpheus)
    :param cpm: np array with at each pixel the cell id occupying that pixel
    :param id_to_type_dict: dict or series with as keys the cell id and as values the type of that cell
    :return: np.ndarray with additional channel conveying the type info
    """
    cpm_out = np.zeros(cpm.shape)
    cpm_out += cpm
    type = np.zeros_like(cpm_out)
    if 0 in id_to_type_dict.values():
        # we have a cell type with type 0, but this should be reserved for background -> add 1 to all types
        id_to_type_dict = {k: v + 1 for k, v in id_to_type_dict.items()}

    for cell_id, cell_type in id_to_type_dict.items():
        # print(cell_id, cell_type)
        mask = cpm_out == cell_id
        type[mask] = int(cell_type)  # add 1 to avoid 0 values - 0 is reserved for background
        # print()
    cpm_out = np.concatenate([cpm_out, type], axis=axis)
    # print(np.unique(cpm_out[...,1]))
    return cpm_out



######## neighborhood calculation related utils:
def get_filter(neighbor_order=2, propagate_self=True):

    # get kernel size for neighbor order: see https://morpheus.gitlab.io/faq/modeling/neighborhoods/
    even = neighbor_order % 2 == 0
    if even:
        kernel_size = neighbor_order + 1
    else:
        kernel_size = neighbor_order + 2

    conv_filter: np.ndarray = np.ones(shape=(kernel_size, kernel_size),
                      dtype=float)  # neighborhood filter for propagating the categorical counts to all neighboring pixels

    if not even:
        # all entries at the outermost ring that are not radial are zero:
        conv_filter[0, :] = 0
        conv_filter[-1, :] = 0
        conv_filter[:, 0] = 0
        conv_filter[:, -1] = 0
        # radial at edges are 1:
        mid = neighbor_order // 2 + 1  # note: zero-indexing
        conv_filter[mid, 0] = 1
        conv_filter[mid, -1] = 1
        conv_filter[0, mid] = 1
        conv_filter[-1, mid] = 1
    else:
        mid = neighbor_order // 2

    if not propagate_self:
        conv_filter[mid, mid] = 0
    return conv_filter




def propagate_onehot_to_neighbors(onehot_arr, neighbor_order=2, pad_mode='periodic', propagate_self=True):
    conv_filter = get_filter(neighbor_order, propagate_self)
    if pad_mode == 'periodic':
        pad_mode = 'wrap'
    p_shape = tuple(
        round(
        (s - 1) / 2 + (2*s-1) / 1000
        ) for s in conv_filter.shape
    )
    onehot_id_padded = jnp.pad(onehot_arr, pad_width=(p_shape, p_shape, (0, 0)), mode=pad_mode)
    neighbor_prop = jsp.signal.convolve(onehot_id_padded, conv_filter[..., None], mode='valid')
    assert neighbor_prop.shape == onehot_arr.shape
    return neighbor_prop



pertubations = jnp.array([[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]])

def check_if_boundary(grid, i, j):
    return jax.numpy.clip(jax.vmap(lambda c: (grid[0, i + c[0], j + c[1]] != grid[0, i, j]).astype(float))(pertubations).sum(), a_max=1)

def create_boundary_mask(grid):
    """
    Create a mask that indicates for each pixel whether it is on the boundary of a cell.
    :param grid: grid with at first channel the cell_id int
    :return:
    """
    height = grid.shape[-2]
    width = grid.shape[-1]
    return jax.vmap(lambda i: jax.vmap(lambda j: check_if_boundary(grid, i, j))(jnp.arange(height)))(jnp.arange(width))

def update_boundary_mask(grid, boundary_mask, i, j):
    boundary_mask = boundary_mask.at[i, j].set(check_if_boundary(grid, i, j))
    for p in pertubations:
        boundary_mask = boundary_mask.at[i + p[0], j + p[1]].set(check_if_boundary(grid, i + p[0], j + p[1]))
    return boundary_mask

def update_boundary_mask_multiple_sites(grid, boundary_mask, i, j):
    boundaries = jax.vmap(check_if_boundary, in_axes=(None, 0,0))(grid, i, j)
    boundary_mask = boundary_mask.at[i, j].set(boundaries)
    for p in pertubations:
        boundaries_pertub = jax.vmap(check_if_boundary, in_axes=(None, 0,0))(grid, i + p[0], j + p[1])
        boundary_mask = boundary_mask.at[i + p[0], j + p[1]].set(boundaries_pertub)
    return boundary_mask

def aggregate_neighbour_ids(one_hot_cell_ids, i, j):
    '''
    Take a point on the grid, return a one hot array with
    the neighbouring cell ids
    '''
    neighbours_incl_self = jax.vmap(
        lambda c: (one_hot_cell_ids[i + c[0], j + c[1], :]).astype(float)
    )(pertubations).sum(axis=0).clip(max=1.)
    neighbours = neighbours_incl_self - one_hot_cell_ids[i, j, :]
    return neighbours


########### Logging-related utils
class Logger:

    def __init__(self,
        project = 'NeuralCPM',
        run_name = 'unnamed run',
        wandb_config={},
        model_metrics_to_log=[],
        sampler_metrics_to_log=[],
        data_metrics_to_log=[]):

        self.log_to_wandb = True
        if 'log_to_wandb' in wandb_config:
            self.log_to_wandb = wandb_config.pop('log_to_wandb')
        if not WANDB_IMPORTED:
            self.log_to_wandb = False

        if self.log_to_wandb:
            self.run = wandb.init(
                project=project,
                config=wandb_config,
                name=run_name,
                entity='neuralcpm'
            )
        self.model_metrics_to_log = model_metrics_to_log
        self.sampler_metrics_to_log = sampler_metrics_to_log
        self.data_metrics_to_log = data_metrics_to_log

    def log(self, iteration=None, model_metrics=None, sampler_metrics=None, data_metrics=None, model_weights_path=None,
            model_weights_path_ema=None, **kwargs):
        if self.log_to_wandb:
            self.run.log(
                self.process_model_metrics(model_metrics, iteration) | self.process_sampler_metrics(sampler_metrics) | self.process_data_metrics(data_metrics) | kwargs,
                step=iteration)
            if model_weights_path is not None:
                self.run.log_model(model_weights_path, name=os.path.basename(model_weights_path))
            if model_weights_path_ema is not None:
                self.run.log_model(model_weights_path_ema, name=os.path.basename(model_weights_path_ema))
        else:
            print(self.process_model_metrics(model_metrics, iteration) | self.process_sampler_metrics(sampler_metrics) | self.process_data_metrics(data_metrics) | kwargs)
        plt.close()

    def process_model_metrics(self, metrics, iteration):
        result = {}
        if 'loss' in self.model_metrics_to_log or self.model_metrics_to_log == 'all':
            result['Loss'] = metrics.pop('Negative Log Likelihood')
            result['Energy Training Data'] = metrics.pop('Energy Training Data')
            result['Energy Generated Data'] = metrics.pop('Energy Generated Data')
            print(
                f"Step {iteration} - Loss: {jnp.round(result['Loss'], 3)}, Energy Data: {jnp.round(result['Energy Training Data'], 3)}, Energy Generated: {jnp.round(result['Energy Generated Data'], 3)}")

        if self.model_metrics_to_log == 'all':  # log miscellaneous parameters returned from the model
            result = {**result, **metrics}

        if 'params' in self.model_metrics_to_log or self.model_metrics_to_log == 'all':
            if 'params' in metrics.keys():
                # result['model parameters'] = np.asarray(jax.flatten_util.ravel_pytree(result['params'])[0])
                result['parameters'] = wandb.Histogram(np_histogram=np.histogram(np.asarray(jax.flatten_util.ravel_pytree(result['params'])[0])))
                result.pop('params')

        if 'Gradients' in self.model_metrics_to_log or self.model_metrics_to_log == 'all':
            if 'Gradients' in metrics.keys():
                # result['model gradients'] = np.asarray(jax.flatten_util.ravel_pytree(result['Gradients'])[0])
                result['gradients'] = wandb.Histogram(np_histogram=np.histogram(np.asarray(jax.flatten_util.ravel_pytree(result['Gradients'])[0])))
                result.pop('Gradients')

        return result

    def process_sampler_metrics(self, metrics):
        result = {}
        total_flips = metrics['flips'].sum(axis=1).mean() if 'flips' in metrics else 0
        runtime = metrics['runtime'] if 'runtime' in metrics else 0

        # generic logging
        if self.sampler_metrics_to_log == 'all':
            keys = metrics.keys()
        else:
            keys = self.sampler_metrics_to_log
        for k in keys:
            v = metrics[k]
            try:
                result[k] = v if not isinstance(v, jnp.ndarray) else v.mean().item()
            except:
                result[k] = 0

        # custom metric specific postprocessing where necessary:
        if 'runtime per kernel pass' in self.sampler_metrics_to_log or self.sampler_metrics_to_log == 'all':
            result['runtime per kernel pass'] = runtime / metrics['num steps'].mean()
        if 'runtime per flip' in self.sampler_metrics_to_log or self.sampler_metrics_to_log == 'all':
            result['runtime per flip'] = runtime / total_flips
        if 'flips per second' in self.sampler_metrics_to_log or self.sampler_metrics_to_log == 'all':
            result['flips per second'] = total_flips / runtime
        if 'flips' in self.sampler_metrics_to_log or self.sampler_metrics_to_log == 'all':
            result['total flips'] = total_flips
        return result

    def process_data_metrics(self, metrics):
        result = {}
        # compare generated and real samples
        if 'compare samples' in self.data_metrics_to_log or self.data_metrics_to_log == 'all':
            if 'cpm_history' in metrics.keys():
                fig, ax = plt.subplots(3, 6, figsize=(40, 30))
            else:
                fig, ax = plt.subplots(3, 4, figsize=(40, 30))
            for i in range(min(min(3, len(metrics['real_data'])), len(metrics['generated_data']))):
                ax[i, 0].imshow(metrics['real_data'][i][0], cmap='plasma')
                ax[i, 0].set_title('real data cell id')
                ax[i, 1].imshow(metrics['generated_data'][i][0], cmap='plasma')
                ax[i, 1].set_title('generated data cell id')
                ax[i, 2].imshow(metrics['real_data'][i][1], cmap='plasma')
                ax[i, 2].set_title('real data cell type')
                ax[i, 3].imshow(metrics['generated_data'][i][1], cmap='plasma')
                ax[i, 3].set_title('generated data cell type')
                if 'cpm_history' in metrics.keys():
                    for j in range(2):
                        ax[i, 4 + j].imshow(metrics['cpm_history'][i][j][0], cmap='plasma')
                        ax[i, 4 + j].set_title(f'cpm history {j}')
            result['compare samples'] = plt
        return result





############# CPM simulaiton running related utils

def run_cpm_from_init_state(init_cpm, sampler, energy_fn, num_outer_steps, key):
    """
    Run a Cellular Potts Model (CPM) simulation from an initial state, saving num_outer_steps states, each spaced num_inner_steps apart.
    :param init_cpm: initial state
    :param sampler: sampler to use for the simulation
    :param energy_fn: energy function to use for the simulation
    :param num_outer_steps: number of outer steps to run the simulation, each step corresponds to a saved state and a full run of the provided sampler
    the full sampler runs are performed sequentially, intializing from the last state of the previous run
    :param key: jax PRNG key for random number generation
    :return: all_states: all states of the CPM
             all_energies: all energies of the CPM
    """

    state = sampler.init(key, x=init_cpm, previous_state=init_cpm)   # (cpm, energy, boundary_mask)
    def body_fn(state, key):
        (cpm, energy, boundary_mask), metrics = sampler.sample(key, energy_fn, state)
        return (cpm, energy, boundary_mask), (cpm, energy)

    # get num_outer_steps states each spaced num_inner_steps spin-flip-attempts apart
    final_all, (all_states, all_energies) = jax.lax.scan(body_fn, state, jax.random.split(key, num_outer_steps))

    return all_states, all_energies


def get_id_to_type_vec(cpm, num_cell_ids):
    # get the id to type vector from the cpm array
    cell_ids = cpm[0]
    cell_types = cpm[1]
    def get_type_for_cell_id(cell_id):
        return jnp.max(cell_types * (cell_ids == cell_id))

    id_to_type_vec = jax.vmap(get_type_for_cell_id)(jnp.arange(0, num_cell_ids))
    return id_to_type_vec


def get_points_init_state_numpy(key, num_cells=50, id_to_type_dict=None, grid_shape=(100,100), init_radius=25.):
    if id_to_type_dict is None:
        id_to_type_dict = {i: (i-1) // (num_cells/2) + 1 for i in range(1, num_cells+1)}
    cpm = jnp.zeros(shape=(2, *grid_shape)).astype(int)
    # initialize cells at random positions within a radius from the center of the grid:
    center = np.array([[grid_shape[0]//2], [grid_shape[1]//2]])
    possible_locs = np.array(
        [[j.item() for j in i] for i in np.unravel_index(np.arange(grid_shape[0]*grid_shape[1]), grid_shape)]
    )
    # filter out the locations where the cells are not allowed to be placed (larger than the init_radius from the center)
    possible_locs = possible_locs[:, np.linalg.norm(possible_locs - center, axis=0) < init_radius]
    # convert to jax array:
    possible_locs = jnp.array(possible_locs)

    key, use_key = jax.random.split(key)
    idx = jax.random.choice(use_key, possible_locs, (num_cells,), replace=False, axis=1)
    cpm = cpm.at[0, idx[0], idx[1]].set(jnp.arange(1, num_cells+1))
    cpm = cpm.at[1, idx[0], idx[1]].set(jnp.array([id_to_type_dict[i] for i in range(1, num_cells+1)]))

    return cpm


def vol_of_cell(cell_id, cpm):
    return (cpm[0] == cell_id).sum()

def sample_neighbor(key, sampled_x, sampled_y, cpm, min_cell_volume=0):
    # sample a neighbour that is not the same value
    key, use_key = jax.random.split(key)

    neighbours = jax.vmap(
        lambda p: cpm[
            0,
            jnp.mod(sampled_x + p[0], cpm[0].shape[0]),  # modulo for periodic boundary conditions
            jnp.mod(sampled_y + p[1], cpm[0].shape[1])
        ]
    )(pertubations)

    # do not select neighbors that are the same cell id:
    probs_unnorm = (neighbours != cpm[0, sampled_x, sampled_y]).astype(float)
    # add a tiny probability because it is better to select a same-cell neighboring pixel than to pick a pixel that might lead the cell to go to too low volume:
    # (this is only relevant if all neighbors get prob 0 and jax is forced to pick one of them)
    probs_unnorm = probs_unnorm + 1e-6
    # do not select neighbors with a volume that might go too small:
    volumes_neighbour = jax.vmap(lambda x: vol_of_cell(x, cpm))(neighbours)
    probs_unnorm = probs_unnorm * (volumes_neighbour > min_cell_volume).astype(float)

    probs = probs_unnorm / probs_unnorm.sum()
    sampled_idx = jax.random.choice(use_key, jnp.arange(len(pertubations)), p=probs)
    sampled_pertubation = pertubations[sampled_idx]
    neighbour_x = sampled_x + sampled_pertubation[0]
    neighbour_y = sampled_y + sampled_pertubation[1]
    return neighbour_x, neighbour_y, probs[sampled_idx]



################# Expimeriment and visualization related utils:

def _draw_borders_on_type_channel(arr):
    arr = np.asarray(arr)  # shape (..., 2, h ,w)
    # upsample the array so we can draw the cell borders on the type channel in a bit higher resolution:
    arr = arr.repeat(2, axis=-2).repeat(2, axis=-1)
    borderx = np.diff(arr[..., 0,:,:], axis=-1, append=0) != 0
    bordery = np.diff(arr[..., 0, :, :], axis=-2, append=0) != 0
    arr[..., 1, :, :] += 1
    arr[..., 1, :, :][borderx] = 0
    arr[..., 1, :, :][bordery] = 0
    return arr

def arr_to_im(arr, is_one_hot_enc=False, colors=None):

    if not is_one_hot_enc: # shape arr: (2, h, w)
        arr = jax.nn.one_hot(arr[1], len(np.unique(arr[1])))  #(h, w, c)

    if colors is None:
        colors = [
            np.array([[0.,0.,0.]]),# black
            np.array([[0.,0.,0.25]]),# dark blue
            np.array([[4.,95.,208.]]) / 255,  #light blue
            np.array([[255.,95., 31.]]) / 255, # orange
            np.array([[0.,1.,0.]]), #  green
            np.array([[1.,0.,0.]]), #  red
            np.array([[1.,1.,0.]]) #  yellow
        ]
    to_plot = np.zeros((*arr.shape[:2], 3))  #(h, w, c)
    for i in range(arr.shape[2]):
        to_plot[arr[..., i] == 1.] += colors[i]

    return to_plot

def plot_cell_image(datapoint, ax=None, colors=None):
    if ax is None:
        fig, ax = plt.subplots()

    datapoint = _draw_borders_on_type_channel(datapoint)
    to_plot = arr_to_im(datapoint, colors=colors)
    ax.imshow(to_plot)
    return to_plot


def plot_cell_trajectory_data(data, num_samples, ts_to_plot, axs=None, colors=None):
    if axs is None:
        fig, axs = plt.subplots(num_samples, len(ts_to_plot))
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            ax = axs[i, j]
            plotted = plot_cell_image(data[i, ts_to_plot[j]], ax, colors=colors)


def calculate_all_cell_volumes(cpm, num_cells):
    id_oh = jax.nn.one_hot(cpm[0], num_cells) # (h,w,c)
    vols = jnp.sum(id_oh, axis=(0, 1))
    return vols

def count_num_fragmented(cpm, num_cells, neighborhood_order=2):
    assert neighborhood_order > 0, 'expected neighborhood order of at least 1'
    kernel = get_filter(min(neighborhood_order, 2), propagate_self=True)
    num_fragments = np.zeros(num_cells - 1)
    for c in range(1, num_cells):
        this_cell = (cpm[0] == c).astype(cpm.dtype)
        if neighborhood_order > 2:
            this_cell = binary_dilation(this_cell, generate_binary_structure(2,
                                                                             2 if neighborhood_order % 2 == 0  else 1
                                                                             ),
                                        iterations=neighborhood_order - 2
                                        )
        labeled_arr, num_clusters_found_this_type = label(this_cell, structure=kernel)
        num_fragments[c-1] = num_clusters_found_this_type

    num_fragmented = (num_fragments != 1).sum()
    return num_fragmented

def get_cells_com(cpm, num_cells):
    id_oh = jax.nn.one_hot(cpm[0], num_cells)  # (h,w,c)
    cell_vols = jnp.sum(id_oh, axis=(0, 1))
    h_range = jnp.arange(cpm.shape[1]).reshape(-1, 1, 1)
    w_range = jnp.arange(cpm.shape[2]).reshape(1, -1, 1)

    coms_h = jnp.sum(h_range * id_oh, axis=(0, 1)) / cell_vols
    coms_w = jnp.sum(w_range * id_oh, axis=(0, 1)) / cell_vols
    cell_coms = jnp.stack([coms_w, coms_h], axis=-1)  # w -> x, h->y
    return cell_coms


#### Miscellaneous utils:

def pad_array(x, target_shape, num_spatial_dims=2):
    if x.shape[-num_spatial_dims:] != target_shape[-num_spatial_dims:]:  # adjust the computation graph to match the shapes
        amounts_to_pad = [target_shape[-(i + 1)] - x.shape[-(i + 1)] for i in range(num_spatial_dims)]
        pads = [(int(np.floor(pad / 2)), int(np.ceil(pad / 2))) for pad in amounts_to_pad]
        pads = [(0, 0) for _ in range(len(x.shape) - len(pads))] + pads
        # x is shape channels, dim1, dim2, ...
        x = jnp.pad(x, pads, mode='constant', constant_values=0)  # pad all spatial dims except channels
    return x
