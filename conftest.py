"""Root conftest — adds the project root to sys.path so all imports resolve."""
import sys
from pathlib import Path

# Ensure project root is on the path regardless of how pytest is invoked.
sys.path.insert(0, str(Path(__file__).parent))
