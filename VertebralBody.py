from dataclasses import dataclass
import numpy as np
import vtk
import vtk_convenience as conv
from typing import Dict, Tuple
from vtk.util import numpy_support
import slicer
import SpineLib


class VertebralBody:
    def __init__(self,
                 body: vtk.vtkPolyData,
                 center:np.array,
                 orientation: SpineLib.Orientation,
                 max_angle: float
                ) -> None:
          
          # extract endplates
          self.superior_endplate = VertebralBody._extract_endplate(body=body, center=center, direction=orientation.s, max_angle=max_angle)
          self.inferior_endplate = VertebralBody._extract_endplate(body=body, center=center, direction=-orientation.s, max_angle=max_angle)
          
          # get sagittal curves
          self.superior_sagittal_curve = VertebralBody._extract_curve(self.superior_endplate, plane_origin=center, plane_normal=orientation.r, curve_direction=orientation.a)
          self.inferior_sagittal_curve = VertebralBody._extract_curve(self.inferior_endplate, plane_origin=center, plane_normal=orientation.r, curve_direction=orientation.a)

          # get center of sagittal curves
          sagittal_center = np.mean([self.superior_sagittal_curve.min_point, self.superior_sagittal_curve.max_point, self.inferior_sagittal_curve.min_point, self.inferior_sagittal_curve.max_point],axis=0)
          
          # get frontal curves (with sagittal curve center as plane_origin)
          self.superior_frontal_curve = VertebralBody._extract_curve(self.superior_endplate, plane_origin=sagittal_center, plane_normal=orientation.a, curve_direction=orientation.r)
          self.inferior_frontal_curve = VertebralBody._extract_curve(self.inferior_endplate, plane_origin=sagittal_center, plane_normal=orientation.a, curve_direction=orientation.r)



    def _extract_endplate(body: vtk.vtkPolyData, center: np.array, direction: np.array, max_angle: float):

        # extract points, where normals are similar to the given direction
        endplate = conv.eliminate_misaligned_faces(
            polydata=body, center=center, direction=direction, max_angle=max_angle
        )

        # filter with connectivity filter to remove outliers (only the largest connected region remains)
        connectivity_filter = vtk.vtkPolyDataConnectivityFilter()
        connectivity_filter.SetInputData(endplate)
        connectivity_filter.SetExtractionModeToLargestRegion()
        connectivity_filter.Update()
        endplate = connectivity_filter.GetOutput()

        return endplate
    
    
    def _extract_curve(
            endplate: vtk.vtkPolyData, plane_origin: np.array, plane_normal: np.array, curve_direction: np.array) -> SpineLib.Curve:
        geometry = conv.cut_plane(endplate, plane_origin, plane_normal)
        regression = conv.calc_main_component(geometry)
        regression = (regression if regression.dot(curve_direction) >= 0 else -regression)
        point_array = numpy_support.vtk_to_numpy(geometry.GetPoints().GetData())
        mean = np.mean(point_array, axis=0)
        points = conv.sorted_points(geometry, regression)
        projections = np.dot(points - mean, regression)
        min_point_rg = mean + np.min(projections) * regression
        max_point_rg = mean + np.max(projections) * regression
        points = conv.sorted_points(geometry, regression)
        min_point = np.array(points[0])
        max_point = np.array(points[-1])

        return SpineLib.Curve(geometry=geometry, points=points, regression=regression, min_point=min_point, max_point=max_point)