# Required Libraries
import numpy as np

# iGWO
from pyMetaheuristic.algorithm import improved_grey_wolf_optimizer
from pyMetaheuristic.utils import graphs
def easom(variables_values = [0, 0]):
    x1, x2     = variables_values
    func_value = -np.cos(x1) * np.cos(x1) * np.exp(-(x1 - np.pi) ** 2 - (x1 - np.pi) ** 2)
    return func_value
parameters = {
        'pack_size': 2,
        'min_values': (0, 0),
        'max_values': (1, 0),
        'iterations': 3,
        'verbose': True
    }
igwo = improved_grey_wolf_optimizer(target_function = easom, **parameters)
# BA - Solution
# iGWO - Solution
variables = igwo[0][:-1]
minimum   = igwo[0][ -1]
print('Variables: ', variables, ' Minimum Value Found: ', minimum)
# Target Function - Values
plot_parameters = {
    'min_values': (0, 0),
    'max_values': (1, 1),
    'step': (0.1, 0.1),
    'solution': [],
    'proj_view': '3D',
    'view': 'notebook'
}
graphs.plot_single_function(target_function = easom, **plot_parameters)