import sys
from pathlib import Path

# Get the absolute path to the package directory (where __init__.py is)
package_path: str = str(Path(__file__).resolve().parent)

# Add package_path to sys.path (more reliable than PYTHONPATH)
if package_path not in sys.path:
    sys.path.insert(0, package_path)

from src.conv import Conv
from src.fold import Fold, Unfold
