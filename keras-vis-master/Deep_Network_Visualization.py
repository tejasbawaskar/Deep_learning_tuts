from vis.losses import ActivationMaximization
from vis.regularizers import TotalVariation, LPNorm

filter_indices = [1, 2, 3]

# Tuple consists of (loss_function, weight)
# Add regularizers as needed.
losses = [
    (ActivationMaximization(keras_layer, filter_indices), 1),
    (LPNorm(model.input), 10),
    (TotalVariation(model.input), 10)
]

