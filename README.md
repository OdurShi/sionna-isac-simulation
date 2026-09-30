# Sionna ISAC simulation

Simulation and learning-based access-point placement for cell-free integrated sensing and communication.

## Code

- `simulation/main.py`: radio-map example.
- `simulation/generate_offline_database.py`: offline channel database generation.
- `simulation/isac_env.py`: deployment environment.
- `simulation/train.py`: SAC training.
- `simulation/compare_baselines.py`: simulated-annealing and genetic-algorithm comparisons.
- `simulation/main_app.py`: desktop interface.
- `simulation/check_database.py`, `view_heatmap.py`, and `render_topdown.py`: diagnostics and visualization.
- `scenes/`: supplied scene geometry.
- `docs/figures/`: sample outputs.

## Usage

Use `simulation/` as the working directory so that the original scripts can find their data:

```bash
cd simulation
python test_env.py
python train.py
python compare_baselines.py
```

To launch the interface, run `python main_app.py` from the same directory.

The scripts use NumPy, Matplotlib, SciPy, PyTorch, Gym, TensorFlow, Sionna, CustomTkinter, Open3D, and PyVista. The source archive does not include a pinned environment. Set up dependencies compatible with the `sionna.rt.RadioMapSolver` API before running ray tracing or the interface. GPU/backend setup depends on your Sionna installation.

The supplied NumPy channel databases, candidate positions, and example training curve are retained beside the scripts. Generating a database or training will replace corresponding local outputs. The ray-tracing examples use Sionna's bundled `etoile` scene; local geometry is available separately in `scenes/`.

Python syntax has been checked. GPU simulation, training, and the desktop interface have not been executed as part of this release. Original source notices remain applicable.
