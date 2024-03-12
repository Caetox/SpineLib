# workaround for loading external py file in Slicer

import importlib.util
import sys
from typing import Any, Tuple
import os

sys.dont_write_bytecode = True

def from_module_import(module_name: str, *elements: Tuple[str]) -> Tuple[Any]:
    """ Import any module hosted inside "__file__/Resources/Scripts" by string name """
    module_file = f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(os.path.dirname(__file__), 'Resources', 'Scripts', module_file))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return tuple(sys.modules[module_name].__dict__[el] for el in elements)

from_module_import("vtk_convenience")



# import Registration Library Files

from .SegmentationImage import *
from .SlicerTools import *
from .FacetJointAlignment import *
from .IVD_Center import *
from .Properties import *
from .VertebralBody import *
from .Spine import *
from .UpApproximator import *
from .Vertebra import *