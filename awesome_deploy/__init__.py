from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
_INNER_PACKAGE_DIR = Path(__file__).resolve().parent / "awesome_deploy"
if str(_INNER_PACKAGE_DIR) not in __path__:
    __path__.append(str(_INNER_PACKAGE_DIR))

from .awesome_deploy import AWESOME_DIR

__all__ = ["AWESOME_DIR"]
