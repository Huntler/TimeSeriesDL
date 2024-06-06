from torch import optim
from torch.optim.optimizer import Optimizer

_optimizer_register = {
    "Adam": optim.Adam,
    "Adagrad": optim.Adagrad,
    "SparseAdam": optim.SparseAdam,
    "RMSprop": optim.RMSprop,
    "ASGD": optim.ASGD,
    "SGD": optim.SGD,
    "Adadelta": optim.Adadelta,
    "Adamax": optim.Adamax,
    "AdamW": optim.AdamW,
}

def get_optimizer_by_name(name: str) -> Optimizer:
    """Get optimizer by name.

    Args:
        name (str): Name of the optimizer to be retrieved

    Raises:
        ValueError: If the optimizer is unknown
        
    Returns:
        Optimizer: Optimizer
    """
    if name not in _optimizer_register:
        raise ValueError(f"{name} is not a valid optimizer")

    return _optimizer_register[name]
