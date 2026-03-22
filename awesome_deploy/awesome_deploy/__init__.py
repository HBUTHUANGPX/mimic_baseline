"""Package entry point for the MuJoCo sim-to-sim deployment toolkit."""

import os

# Expose the installed package root so config code can build asset paths
# without assuming the current working directory.
AWESOME_DIR = os.path.abspath(os.path.dirname(__file__))
