import jax.random
import os
import sys

import models.models as models
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join('..', os.path.dirname(here)))  # add parent directory to path
import utils
import unittest
from ml_collections import ConfigDict


def check_if_model(m):
    model_class = getattr(models, m)
    try:
        is_model = issubclass(model_class, models.HamiltonianBaseClass)
    except TypeError:
        is_model = False
    is_not_baseclass = not (model_class is models.HamiltonianBaseClass)
    return is_model and is_not_baseclass

simple_model_config = ConfigDict(
    {
        'num_cell_ids': 51,
        'num_cell_types': 3,
        'grid_size': (100, 100),
    }
)
KEY = jax.random.PRNGKey(42)

data = utils.load_data_from_file(
    os.path.join(
        'data', 'unittest_data', 'all_cpms_1.npz'
    )
)[-1]


class TestModels(unittest.TestCase):


    def test_energy_calc(self):
        successes = []
        models_tested = []
        for model_name in dir(models):
            if not check_if_model(model_name):
                continue
            models_tested.append(model_name)
            model_class = getattr(models, model_name)
            try:
                model = model_class(cfg=simple_model_config, key=KEY)
            except Exception as e:
                print(f'Exception -- failed to initialize model {model_name}: {e}')
                successes.append(False)
                continue

            try:
                energy = model(data)[0]
                successes.append(isinstance(energy.item(), float))
            except Exception as e:
                print(f'Exception -- failed to run model {model_name}: {e}')
                successes.append(False)

        print('failed models:', [m for m, s in zip(models_tested, successes) if not s])
        self.assertEqual(successes, [True for _ in range(len(models_tested))])

    def test_delta_energy(self):
        successes = []
        models_tested = []
        for model_name in dir(models):
            if not check_if_model(model_name):
                continue
            models_tested.append(model_name)
            model_class = getattr(models, model_name)
            try:
                model = model_class(cfg=simple_model_config, key=KEY)
            except Exception as e:
                print(f'Exception -- failed to initialize model {model_name}: {e}')
                successes.append(False)
                continue

            try:
                current_energy = model(data)
                delta_energy = model.delta_energy(data, 0, 0, data[0,0,0], data[1,0,0], # changes nothing to state -> delta energy should be 0
                                                  old_energy=current_energy)
                successes.append(abs(delta_energy.item()) < 1e-5)  # should be close to zero
            except Exception as e:
                print(f'Exception -- failed to run model {model_name}: {e}')
                successes.append(False)

        print('failed models:', [m for m, s in zip(models_tested, successes) if not s])
        self.assertEqual(successes, [True for _ in range(len(models_tested))])




# testing
if __name__ == "__main__":
    unittest.main()



