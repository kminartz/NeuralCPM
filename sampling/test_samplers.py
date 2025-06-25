from sampling import initializers, samplers, transition_kernels
import unittest
from ml_collections import ConfigDict
import utils
import os
from models.models import CellsortHamiltonian
import jax
KEY = jax.random.PRNGKey(42)

simple_model_config = ConfigDict(
    {
        'num_cell_ids': 51,
        'num_cell_types': 3,
        'grid_size': (100, 100),
    }
)

model = CellsortHamiltonian(KEY, simple_model_config)

data = utils.load_data_from_file(
    os.path.join(
        'data', 'unittest_data', 'all_cpms_1.npz'
    )
)

x = data[-1]
previous_state = data[-2]
id_to_type_vecs = utils.get_id_to_type_vec(data, simple_model_config.num_cell_ids)


def check_if_tk(tk_name):
    tk_class = getattr(transition_kernels, tk_name)
    try:
        is_tk = issubclass(tk_class, transition_kernels.CPMKernelBaseClass)
    except TypeError:
        is_tk = False
    is_not_baseclass = not (tk_class is transition_kernels.CPMKernelBaseClass)
    return is_tk and is_not_baseclass

def check_if_initializer(init_name):
    init_class = getattr(initializers, init_name)
    try:
        is_init = issubclass(init_class, initializers.InitializerBaseClass)
    except TypeError:
        is_init = False
    is_not_baseclass = not (init_class is initializers.InitializerBaseClass)
    return is_init and is_not_baseclass



class TestSamplers(unittest.TestCase):

    def test_samplers(self):
        successes = []
        samplers_tested = []
        for tk_name in dir(transition_kernels):
            if not check_if_tk(tk_name):
                continue
            for init_name in dir(initializers):
                if not check_if_initializer(init_name):
                    continue
                tk_class = getattr(transition_kernels, tk_name)
                init_class = getattr(initializers, init_name)

                # initialize initializer, transition kernel, sampler:
                try:
                    initializer = init_class()
                    tk = tk_class()
                    sampler = samplers.MCMCSampler(initializer, tk, 5)
                except Exception as e:
                    print(f'{type(e)} -- failed to initialize sampler {tk_name} with initializer {init_name}: {e}')
                    successes.append(False)
                    continue
                samplers_tested.append(sampler)

                try:
                    state = sampler.init(KEY, x=x, previous_state=previous_state, x_cell_type_vec=id_to_type_vecs)
                    result = sampler.sample(KEY, energy_fn=model, init_state=state)
                    metrics = result[1]
                    check = metrics['accept'].any() or (metrics['delta'] > 0).all() # we expect at least one successful flip if there is a delta < 0
                    successes.append(check.item())
                except Exception as e:
                    print(f'{type(e)} -- failed to run sampler {tk_name} with initializer {init_name}: {e}')
                    successes.append(False)

        print('failed samplers:', [m for m, s in zip(samplers_tested, successes) if not s])
        print('successful samplers:', [m for m, s in zip(samplers_tested, successes) if s])
        self.assertEqual(successes, [True for _ in range(len(samplers_tested))])


if __name__ == '__main__':
    unittest.main()
