from dataclasses import dataclass
import numpy as np
import vtk
from typing import Dict, Tuple


'''Curve properties'''
@dataclass
class Curve:
    geometry:             vtk.vtkPolyData
    points:               np.array
    regression:           np.array
    min_point:            np.array
    max_point:            np.array

    

'''
Oriention of a vertebra object as axes of a RAS-coordinatesystem
RAS -> R:right, A:anterior, S:superior.
'''
@dataclass
class Orientation:
    r: np.array
    a: np.array
    s: np.array


'''
Size of a vertebra object.
width  -> size in R-direction,
length -> size in A-direction,
height -> size in S-direction.
'''
@dataclass
class Size:
    width:  float
    depth: float
    height: float


'''
Landmarks on the surface of the vertebral body.
'''
@dataclass
class Landmarks:
    superior_posterior :  np.array
    superior_anterior  :  np.array
    inferior_posterior :  np.array
    inferior_anterior  :  np.array
    superior_left      :  np.array
    superior_right     :  np.array
    inferior_left      :  np.array
    inferior_right     :  np.array