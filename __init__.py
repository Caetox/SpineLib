
import slicer
import os
import platform
from slicer.util import pip_install

if platform.system() == 'Windows':
    requirements_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    requirements = open(requirements_path, 'r').read()
    slicer.util.pip_install(requirements)
    if os.getenv('matplotlibback') is None:
        print('Installing matplotlib backend wxAgg')
        slicer.util.pip_install('wxPython')
        print('Setting matplotlib backend to wxAgg')
        os.environ['matplotlibback'] = 'wxAgg'

# required extensions
extensions = ['SlicerIGT',
              'SlicerVMTK',
              ]

# install required extensions
import slicer
for extensionName in extensions:
    em = slicer.app.extensionsManagerModel()
    em.interactive = False  # prevent display of popups
    restart = True
    if not em.installExtensionFromServer(extensionName, restart):
        raise ValueError(f"Failed to install {extensionName} extension")


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



# import Library Files
from .ProgressBarManager import *
from .SegmentationImage import *
from .SlicerTools import *
from .FacetJointAlignment import *
from .IVD_Center import *
from .Properties import *
from .ShapeDecomposition import *
from .VertebralBody import *
from .Spine import *
from .UpApproximator import *
from .Vertebra import *
from .LigamentLandmarks import *
from .SymmetryPlane import *