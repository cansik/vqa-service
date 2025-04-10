from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseVQABackend(ABC):

    @abstractmethod
    def process(self, image: np.ndarray, questions: List[str]) -> List[str]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
