"""This module normalizes a tensor either to a range of 0 to 1 or by standardizing."""
from typing import Tuple
import torch


class TensorNormalizer:
    """This class normalizes a tensor either to a range of 0 to 1 or by standardizing.
    """
    def __init__(self, standardize: bool = True):
        """Initializes the normalizer with an optional standardization flag.

        Args:
            standardize (bool): Whether to standardize the tensor (default: True).
        """
        self.standardize = standardize

        # For z-score standardizing
        self.center = None
        self.std = None

        # For 0 to 1 normalizing
        self.mi = None
        self.range = None
    
    def _check(self, X: torch.tensor, been_fit: bool = False) -> None:
        """
        Checks if the input tensor X is a 2D tensor and has been fit.
        
        Args:
            X (torch.tensor): The input tensor to check.
            been_fit (bool): A flag indicating whether the normalizer has been fit. Default: False.
        
        Returns:
            None
        """
        assert len(X.shape) == 2
        if been_fit:
            if self.standardize: assert self.center is not None and self.std is not None
            else: assert self.range is not None and self.mi is not None
    
    def fit(self, X: torch.tensor) -> "TensorNormalizer":
        """
        Fits the normalizer to the input tensor X. If standardize is True,
        calculates the mean and standard deviation of X. If False, calculates
        the minimum and maximum values of X.

        Args:
            X (torch.tensor): The input tensor to fit.
        
        Returns:
            TensorNormalizer: The normalizer itself after fitting.
        """
        self._check(X)
        if self.standardize:
            self.center = X.mean(axis=0)
            self.std = X.std(axis=0)
        else:
            self.mi = X.min(axis=0)[0]
            self.range = X.max(axis=0)[0] - self.mi
        return self
    
    def transform(self, X: torch.tensor) -> torch.tensor:
        """
        Transforms the input tensor X using the calculated center and std or mi and range.
        If standardize is True, returns (X - self.center) / self.std.
        If False, returns (X - self.mi) / self.range.

        Args:
            X (torch.tensor): The input tensor to transform.
        
        Returns:
            torch.tensor: The transformed tensor.
        """
        self._check(X, been_fit=True)
        return (X - self.center) / self.std if self.standardize else (X - self.mi) / self.range
    
    def fit_transform(self, X: torch.tensor) -> Tuple["TensorNormalizer", torch.tensor]:
        """
        Fits the normalizer to the input tensor X and then transforms it.

        Args:
            X (torch.tensor): The input tensor to fit and transform.
        
        Returns:
            Tuple[TensorNormalizer, torch.tensor]: A tuple containing the normalizer itself after fitting and transforming, and the transformed tensor.
        """
        self.fit(X)
        return self, self.transform(X)
    
    def inverse_transform(self, X_scaled: torch.tensor) -> torch.tensor:
        """
        Inverses the transformation of a scaled tensor X_scaled. If standardize is True,
        returns (X_scaled * self.std) + self.center.
        If False, returns (X_scaled * self.range) + self.mi.

        Args:
            X_scaled (torch.tensor): The input scaled tensor to inverse transform.
        
        Returns:
            torch.tensor: The inverted transformed tensor.
        """
        self._check(X_scaled, been_fit=True)
        return (X_scaled * self.std) + self.center if self.standardize else (X_scaled * self.range) + self.mi
    
    def set_keep_columns(self, indices: int) -> None:
        """
        Sets the indices to keep for center, std, mi and range.

        Args:
            indices (int): The indices to keep.

        Returns:
            None
        """
        if self.standardize:
            self.center = self.center[indices]
            self.std = self.std[indices]
        else:
            self.mi = self.mi[indices]
            self.range = self.range[indices]
