"""Plot measured best-so-far values; never infer unmeasured outcomes."""

import matplotlib.pyplot as plt
import numpy as np

from .data import Observations


def plot_convergence(observations: Observations, initial_count: int):
    if not 1 <= initial_count <= len(observations.outputs):
        raise ValueError("Initial count must be within the observation history")
    best = np.maximum.accumulate(observations.outputs)
    fig, ax = plt.subplots()
    ax.plot(np.arange(initial_count, len(best) + 1), best[initial_count - 1 :])
    ax.set(xlabel="Total evaluations", ylabel="Best observed output", title="Observed convergence")
    return fig, ax
