from dataclasses import dataclass
import numpy as np
import vtk
import vtk_convenience as conv
from enum import IntEnum, auto
import SpineLib
import slicer


class Spine:

    def __init__(self,
                 geometries: vtk.vtkPolyData,
                 max_angle: float
                 )-> None:
        self.orientation    = self.init_spine_orientation(geometries)
        self.vertebrae      = [
            SpineLib.Vertebra(
                spineGeometries=geometries,
                geometry=g,
                spineOrientation=self.orientation,
                max_angle=max_angle
                ) for g in geometries]


    '''
    Calculate orientation of the spine.
    The RAS-coordinates represent axes of a local object coordinate system.
    RAS -> R:right, A:anterior, S:superior
    '''
    # TODO: add calc method
    def init_spine_orientation(self, geometries):

        # # get centers of mass for all vertebrae to approximate the spinal curvature
        # centers_of_mass = [np.array(conv.calc_center_of_mass(g)) for g in geometries]

        # L1 = centers_of_mass[-1]
        # L3 = centers_of_mass[2]
        # L5 = centers_of_mass[0]

        # lineL5L1 = L1 - L5
        
        # # s: normalized vector L1-L5
        # s = conv.normalize(lineL5L1)

        # # a: normalized, orthogonal projection of L3 to s
        # projection = L1 + np.dot(L3 - L1, lineL5L1) / np.dot(lineL5L1, lineL5L1) * lineL5L1
        # a = conv.normalize(L3 - projection)

        # # r: normalized cross product of s and a
        # # r equals cross product, since we operate in right-handed coordinate system
        # r = conv.normalize(np.cross(a,s))

        r = np.array([1,0,0])
        a = np.array([0,1,0])
        s = np.array([0,0,1])


        return SpineLib.Orientation(r=r, a=a, s=s)
    
def showEndplates(body):

    
    center = np.array(conv.calc_center_of_mass(body))
    direction = np.array([0.0,0.0,1.0])

    endplates = conv.eliminate_misaligned_faces(
        polydata=body, center=center, direction=direction, max_angle=45.0
    )

    # seperate into superior and inferior endplate by cutting with a horizontal plane
    superior_endplate = conv.clip_plane(endplates, plane_origin=center, plane_normal=direction)
    inferior_endplate = conv.clip_plane(endplates, plane_origin=center, plane_normal=-direction)

    # filter with connectivity filter to remove outliers (only the largest connected region remains)
    endplates = [superior_endplate,inferior_endplate]
    for i, endplate in enumerate(endplates):
        connectivity_filter = vtk.vtkPolyDataConnectivityFilter()
        connectivity_filter.SetInputData(endplate)
        connectivity_filter.SetExtractionModeToLargestRegion()
        connectivity_filter.Update()
        endplates[i] = connectivity_filter.GetOutput()
    
    # # Create a model node in Slicer
    sup_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode')
    sup_node.SetAndObservePolyData(superior_endplate)
    inf_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode')
    inf_node.SetAndObservePolyData(inferior_endplate)



