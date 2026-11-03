# Direct simulation Monte Carlo

A research implementation of direct simulation Monte Carlo (DSMC) for the
spatially homogeneous Boltzmann equation in two velocity dimensions.

```math
\partial_t f(v)=\frac{1}{\mathrm{Kn}}Q(f,f)(v),
\qquad v\in\mathbb{R}^2,
```

with collision kernel

```math
B(|v-v_*|)=C_\alpha|v-v_*|^\alpha, \qquad \alpha\in[0,1].
```

## Start here

- [`config.py`](config.py): particle count, initial data, collision kernel, and time-stepping parameters.
- [`main.py`](main.py): the particle simulation loop.
- [`numeric_functions.py`](numeric_functions.py): collision updates and numerical helpers.
- [`statistics_functions.py`](statistics_functions.py): sampling routines.
- [`display_engine.py`](display_engine.py): visualization utilities.
- [`solution_animation.gif`](solution_animation.gif): a saved simulation animation.

Install the recorded dependencies from [`requirements.txt`](requirements.txt)
in a virtual environment before running `python main.py` from the repository
root. The dependency versions reflect the original experiment environment.
Review the particle count and time horizon in `config.py` before launching a run.

The `image_*.py` files and accompanying images are separate visualization
experiments. This repository contains research scripts rather than an
installable library; the main script starts the simulation when executed or
imported.
