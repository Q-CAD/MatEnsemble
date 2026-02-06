"""
A strategy class allowing the manager (SuperFluxManager) to use a
different strategy for processing futures based on the parameters given to it
at run time
"""

import flux

from abc import ABC, abstractmethod


class FutureProcessingStrategy(ABC):
    @abstractmethod
    def process_futures(self, buffer_time) -> None:
        pass
