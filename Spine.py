from dataclasses import dataclass
import numpy as np
import vtk
import vtk_convenience as conv
from enum import IntEnum, auto
import SpineLib
import slicer


class Spine:

    def __init__(self,
                 model_nodes,
                 geometries: vtk.vtkPolyData,
                 indices: np.ndarray,
                 max_angle: float,
                 calculate_orientation: bool
                 )-> None:
        self.orientation    = self.init_spine_orientation(geometries, calculate_orientation)
        self.vertebrae      = [
            SpineLib.Vertebra(
                modelNode=m,
                spineGeometries=geometries,
                geometry=g,
                index=i,
                spineOrientation=self.orientation,
                max_angle=max_angle
                ) for m,g,i in zip(model_nodes, geometries, indices)]


    '''
    Calculate orientation of the spine.
    The RAS-coordinates represent axes of a local object coordinate system.
    RAS -> R:right, A:anterior, S:superior
    '''
    def init_spine_orientation(self, geometries, calculate_orientation=False):

        if calculate_orientation:

            # get centers of mass for all vertebrae to approximate the spinal curvature
            centers_of_mass = [np.array(conv.calc_center_of_mass(g)) for g in geometries]

            top = centers_of_mass[-1]
            middle = centers_of_mass[len(centers_of_mass)//2]
            bottom = centers_of_mass[0]
            lineDown = top - bottom
            
            # s: normalized vector from top to bottom
            s = conv.normalize(lineDown)

            # a: normalized, orthogonal projection of middle vertebra to s
            projection = top + np.dot(middle - top, lineDown) / np.dot(lineDown, lineDown) * lineDown
            a = conv.normalize(middle - projection)

            # r: normalized cross product of s and a
            # r equals cross product, since we operate in right-handed coordinate system
            r = conv.normalize(np.cross(a,s))
        
        else:
            # Default orientation in RAS coordinate system
            r = np.array([1,0,0])
            a = np.array([0,1,0])
            s = np.array([0,0,1])


        return SpineLib.Orientation(r=r, a=a, s=s)

# Show endplates of the vertebrae in Slicer
def showEndplates(body):
    
    center = np.array(conv.calc_center_of_mass(body))
    direction = np.array([0.0,0.0,1.0])

    # get the endplates = surface whose normals are aligned with the superior-inferior direction
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



