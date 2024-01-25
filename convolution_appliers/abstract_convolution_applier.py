import abc
import dataclasses
import numpy as np


@dataclasses.dataclass
class AbstractConvolutionApplier(abc.ABC):
    """Abstract class for applying a kernel convolution"""
    kernel : np.ndarray

    @abc.abstractmethod
    def apply(self, image_path : str, output_path : str):
        """Apply a kernel filter to an image"""
        return


