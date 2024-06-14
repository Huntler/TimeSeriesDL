"""Module loads utilities, e.g. the config manager instance."""
from.config import config
from .loss import get_loss_by_name
from .optimizer import get_optimizer_by_name
from .activations import get_activation_from_string
