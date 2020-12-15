# Filters

Generalized implementation of some flavours of Bayesian Filters.

This package provides the following filters:

- Augmented Unscented Kalman Filter (AUKF)
- Unscented Kalman Filter

## Usage

All the filters in this package have a common API:

```python
import numpy as np
import time
from Filters import AUKF
from <YOUR_MODEL> import state_transition, output_transition, postprocessing
from <YOUR_SENSOR> import get_measurement

# Initialization
state_dim = 6  # site of state
output_dim = 3  # size of outputs
noise_dim = 3  # size of process noise state variables
deltaT = 0.1  # step time in seconds

filter = AUKF(state_dim, noise_dim, output_dim)

# initialize transition functions
filter.init_io(state_transition, output_transition, postprocessing)

# initialize state
x0 = np.random.rand(state_dim).T  # initial state mean
C0 = 1e4 * np.eye(state_dim)  # initial state covariance

# initialize augmented state
n0 = np.random.rand(noise_dim).T  # initial process noise mean
Q0 = 1e-4 * np.eye(noise_dim)  # initial process noise covariance

filter.init_state(x0, C0, n0, Q0)

while True:
    t0 = time.time()
    measurement, covariance = get_measurement()
    x_estimate, C_estimate = filter.step(deltaT, measurement, covariance)
    print('Estimate: ', x_estimate)
    sleep(deltaT + t0 - time.time())
```