import os
import sys
import importlib
from types import ModuleType


def load_local_medvae_module() -> ModuleType:
    """
    Load the local ./MedVAE source tree as the `medvae` package.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    local_medvae_root = os.path.join(project_root, "MedVAE")
    local_medvae_pkg = os.path.join(local_medvae_root, "medvae")

    if not os.path.isdir(local_medvae_pkg):
        raise ModuleNotFoundError(
            f"Local MedVAE package not found at expected path: {local_medvae_pkg}"
        )

    if local_medvae_root in sys.path:
        sys.path.remove(local_medvae_root)
    sys.path.insert(0, local_medvae_root)

    loaded = sys.modules.get("medvae")
    loaded_file = os.path.abspath(getattr(loaded, "__file__", "")) if loaded else ""
    local_prefix = local_medvae_root + os.sep

    if loaded is not None and not loaded_file.startswith(local_prefix):
        for module_name in list(sys.modules):
            if module_name == "medvae" or module_name.startswith("medvae."):
                del sys.modules[module_name]

    module = importlib.import_module("medvae")
    module_file = os.path.abspath(getattr(module, "__file__", ""))
    if not module_file.startswith(local_prefix):
        raise ImportError(f"Loaded medvae from unexpected path: {module_file}")

    return module


def get_local_mvae_class():
    """
    Return the MVAE class from the local ./MedVAE source tree.
    """
    module = load_local_medvae_module()
    return module.MVAE
