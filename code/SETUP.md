# Simulation setup

Instructions for running the original bat swarm simulation locally.

## Requirements

- WSL2 with Ubuntu (or a compatible Linux environment)
- Python 3.11
- Git

Install Python and the required system packages in Ubuntu:

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-tk
```

`python3.11-venv` enables virtual environments. `python3.11-tk` provides GUI support for dialogs and plots.

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/aleksandrazec/Swarming-bats-and-the-collective-soundscape.git
cd Swarming-bats-and-the-collective-soundscape/code
```

**2. Create and activate a virtual environment**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**3. Install Python dependencies**

```bash
python -m pip install -r requirements.txt
```

**4. Create directories for generated files**

```bash
mkdir -p results output tables
```

## Run the simulation

```bash
python SCRIPT_Swarm.py
```

Simulation parameters, including arena, condition, number of bats, and number of steps, can be changed directly in `SCRIPT_Swarm.py`.

The script saves simulation data in `results/` and displays plots in graphical windows.

## Notes

- Run the script from the `code/` directory because it uses relative file paths.
- The required precomputed acoustic data is included in `library/stored_hrtfs.pck`.
- The virtual environment and generated results are excluded from Git.
- To reactivate the environment in a new terminal, run `source .venv/bin/activate` from `code/`.
