__author__ = ["Soumendu Bagchi", "Kaleb Duchesneau"]
__package__ = "matensemble"

from . import manager
from .manager import SuperFluxManager
from . import fluxlet
from . import logger
from . import strategy

__all__ = ["manager", "fluxlet", "logger", "strategy", "SuperFluxManager"]

"""
MatEnsemble

MatEnsemble is an adaptive workflow manager designed for high throughput HPC
workflows/tasks

The package consists of the 'manager' which holds the main SuperFluxManager
class where the bulk of the task management logic resides. A 'fluxlet' which
handles job submission and usage of the Flux API's. And a 'logger' which handles
organizing the output of the program, the status and more general
logs.

The strategy sub-package holds the two Abstract Base Classes TaskSubmissionStrategy
and FutureProcessingStrategy. (interfaces) and the implementations of them. These
strategies are used in the SuperFluxManager's poolexecutor method to determine 
how jobs are submitted and how the Future objects are processed. 

"""
