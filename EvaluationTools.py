import vtk
import numpy as np
import slicer
import os
import pyvista
import math
from dataclasses import dataclass
from slicer.util import loadModel
from vtk.util.numpy_support import vtk_to_numpy
import open3d as o3d

import SpineLib



'''
Evaluation metrics for scalar array
'''
def scalars_difference_metrics(source_values, target_values):

    differences = np.abs(np.subtract(source_values, target_values))
    mean = np.mean(differences)
    std  = np.std(differences)
    rmse = math.sqrt(np.square(np.subtract(source_values,target_values)).mean())

    return mean, std, rmse

'''
Evaluation metrics for 3D points array
'''
def points_difference_metrics(source_values, target_values):

    differences = np.linalg.norm(np.subtract(source_values,target_values), axis=1)
    mean = np.mean(differences)
    std  = np.std(differences)
    rmse = math.sqrt(np.square(np.subtract(source_values,target_values)).mean())

    return mean, std, rmse

'''
Calculate model to model distance (to surface) with pyvista
'''
def modelToModel_surface_distances(vtk_source, vtk_target):

    source = pyvista.PolyData(vtk_source)
    target = pyvista.PolyData(vtk_target)
        
    closest_cells, closest_points = source.find_closest_cell(target.points, return_closest_point=True)
    d_exact = np.linalg.norm(target.points - closest_points, axis=1)
    mean = np.mean(d_exact)
    std  = np.std(d_exact)
    rmse = math.sqrt(np.square(np.subtract(closest_points,target.points)).mean())

    return mean, std, rmse

'''
Calculate pointcloud distance with ICP
'''
def icp_distances(vtk_source, vtk_target):

    # Create a pointclouds from the vtk vertices
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(np.asarray(vtk_source.GetPoints().GetData()))
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(np.asarray(vtk_target.GetPoints().GetData()))

    threshold = 10000
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold)

    return evaluation.fitness, evaluation.inlier_rmse
