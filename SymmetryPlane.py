import os
import vtk
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import *
import numpy as np
from scipy import stats
from scipy.spatial import distance
from scipy.optimize import least_squares
import csv
from pathlib import Path
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
import open3d as o3d

from scipy.linalg import lstsq
from skspatial.objects import Plane, Points

import SpineLib
import vtk_convenience as conv






class SymmetryPlane:

    def __init__(self,
                 model_node,
                 geometry: vtk.vtkPolyData,
                 numIterations: int,
                ) -> None:
        
        self.symmetry_plane = SymmetryPlane.__fit_symmetry_plane(geometry=geometry, numIterations=numIterations)

        # visualize
        planeNode = SpineLib.SlicerTools.createMarkupsPlaneNode(self.symmetry_plane.GetOrigin(), self.symmetry_plane.GetNormal(), "symmetryPlane", 100, 100)
        planeNode.GetDisplayNode().SetSelectedColor(0, 1, 0)
          


    def __fit_symmetry_plane(geometry: vtk.vtkPolyData, numIterations: int):

        centerOfMass = conv.calc_center_of_mass(geometry)
        points = vtk_to_numpy(geometry.GetPoints().GetData())
        
        # initial plane
        p0 = SymmetryPlane.calcInitalPlane(points, centerOfMass)
        #inital_plane_node = SpineLib.SlicerTools.createMarkupsPlaneNode(p0.GetOrigin(), p0.GetNormal(), "initial plane", 100, 100)

        mirroredGeometry = SymmetryPlane.mirrorWithPlane(geometry, p0)
        pFin = None

        for i in range(numIterations):

            print(f'Iteration {i+1}/{numIterations}')
            
            mirroredGeometryPoints = vtk_to_numpy(mirroredGeometry.GetPoints().GetData())
            transformationMatrix = SymmetryPlane.registerWithVanilaICP(mirroredGeometryPoints, points)
            mirroredModel = SpineLib.SlicerTools.createModelNode(mirroredGeometry, "mirroredModel", color=[1.0, 0.0, 0.0], opacity=0.5)

            mirroredGeometry = conv.transform_polydata(mirroredGeometry, transformationMatrix)
            rigisteredModel = SpineLib.SlicerTools.createModelNode(mirroredGeometry, "registeredModel", color=[0.0, 1.0, 0.0], opacity=0.5)
            mirroredGeometryPoints = vtk_to_numpy(mirroredGeometry.GetPoints().GetData())
            SymmetryPlane.evaluateRegistrationICP(mirroredGeometryPoints, points)
            
            middlePoints = SymmetryPlane.getMiddlePoints(points, mirroredGeometryPoints, viz=False)
            p_i = SymmetryPlane.fitPlaneLeastSquered(middlePoints, centerOfMass, i)

            mirroredGeometry = SymmetryPlane.mirrorWithPlane(geometry, p_i)
            pFin = p_i

        lineNode = SpineLib.SlicerTools.markupsLineNode("Right Symm", np.array(centerOfMass), np.add(np.array(centerOfMass), (np.array(pFin.GetNormal())*100)))
        lineNode.GetDisplayNode().SetSelectedColor(0, 1, 0)
        lineNode.GetDisplayNode().SetTextScale(0)
        
        return pFin



    def calcInitalPlane(points, centerOfMass):
        '''
        points: nparray. Takes numpy array and calculates the pca axes to define the initial symmetry plane (sagittal plane in our case)
        centerOfMass: tuple. It is used to display the axes
        returns: inital sagittal plane
        '''
        lineNodeNames = ['e1', 'e2', 'e3']
        lineNodes = []
        lineStartPos = np.asarray(centerOfMass)
        eigenvects, eigenvals = conv.pca_eigenvectors(points)
        #print(f'eigenvals:{eigenvals}')
        i = 0
        #planeNormal = eigenvects[1]
        sorted_eigenvects = conv.closest_vector(eigenvects, [1,0,0])
        #print(sorted_eigenvects)
        planeNormal = sorted_eigenvects[0]

        lineNode = SpineLib.SlicerTools.markupsLineNode("PCA planeNormal", lineStartPos, lineStartPos + (planeNormal*100))
        lineNode.GetDisplayNode().SetSelectedColor(0, 0, 1)
        lineNode.GetDisplayNode().SetTextScale(0)

        # for lineNodeName in lineNodeNames:
        #     e = eigenvects[i]
        #     #we take e2 as normal to sagittal plane
        #     lineEndPos = lineStartPos + (e*100)
        #     lineNode = SpineLib.SlicerTools.markupsLineNode(lineNodeName, lineStartPos, lineEndPos)
        #     lineNodes.append(lineNode)
        #     i +=1 

        plane = vtk.vtkPlane()
        plane.SetOrigin(centerOfMass)
        plane.SetNormal(planeNormal)

        return plane

    
    def mirrorWithPlane(polydata, plane):  

        normal = plane.GetNormal()
        origin = plane.GetOrigin()
        
        mirrorMatrix = vtk.vtkMatrix4x4()
        mirrorMatrix.SetElement(0, 0, 1 - 2 * normal[0] * normal[0])
        mirrorMatrix.SetElement(0, 1, - 2 * normal[0] * normal[1])
        mirrorMatrix.SetElement(0, 2, - 2 * normal[0] * normal[2])
        mirrorMatrix.SetElement(1, 0, - 2 * normal[0] * normal[1])
        mirrorMatrix.SetElement(1, 1, 1 - 2 * normal[1] * normal[1])
        mirrorMatrix.SetElement(1, 2, - 2 * normal[1] * normal[2])
        mirrorMatrix.SetElement(2, 0, - 2 * normal[0] * normal[2])
        mirrorMatrix.SetElement(2, 1, - 2 * normal[1] * normal[2])
        mirrorMatrix.SetElement(2, 2, 1 - 2 * normal[2] * normal[2])
        
        translateWorldToPlane = [0,0,0] 
        vtk.vtkMath.Add(translateWorldToPlane, origin, translateWorldToPlane)
        translatePlaneOToWorld = [0,0,0]
        vtk.vtkMath.Add(translatePlaneOToWorld, origin, translatePlaneOToWorld)
        vtk.vtkMath.MultiplyScalar(translatePlaneOToWorld ,-1)
        
        mirrorTransform = vtk.vtkTransform()
        mirrorTransform.SetMatrix(mirrorMatrix)
        mirrorTransform.PostMultiply()
        mirrorTransform.Identity()
        mirrorTransform.Translate(translatePlaneOToWorld)
        mirrorTransform.Concatenate(mirrorMatrix)
        mirrorTransform.Translate(translateWorldToPlane)
        mirrorFilter = vtk.vtkTransformFilter()
        mirrorFilter.SetTransform(mirrorTransform)
        mirrorFilter.SetInputData(polydata)
        
        reverseNormalFilter = vtk.vtkReverseSense()
        reverseNormalFilter.SetInputConnection(mirrorFilter.GetOutputPort())
        reverseNormalFilter.Update()

        return reverseNormalFilter.GetOutput()


    def getMiddlePoints(originalPoints, mirroredPoints, viz=True):
        
        if originalPoints.shape[0] > 10000:
            inds = np.random.choice(originalPoints.shape[0], 10000)
            originalPoints = np.take(originalPoints,inds, axis=0)
            mirroredPoints = np.take(mirroredPoints,inds, axis=0)

        middlepoints = (originalPoints + mirroredPoints)/2.0

        if viz:

            vtk_points = vtk.vtkPoints()
            vtk_points.SetData(vtk.util.numpy_support.numpy_to_vtk(middlepoints))

            # Create the vtkPolyData object.
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(vtk_points)

            # Create the vtkSphereSource object.
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(2.0)

            # Create the vtkGlyph3D object.
            glyph = vtk.vtkGlyph3D()
            glyph.SetInputData(polydata)
            glyph.SetSourceConnection(sphere.GetOutputPort())

            pointCloudModelNode = slicer.modules.models.logic().AddModel(glyph.GetOutputPort())

        return middlepoints


        
    def fitPlaneLeastSquered(points, centerOfMass, i):
        plane = Plane.best_fit(Points(points))
        normal = plane.normal.round(3)
        # print(plane.normal.round(3))
        # print(type(normal))

        plane = vtk.vtkPlane()
        plane.SetOrigin(centerOfMass)
        plane.SetNormal(normal)

        return plane


    def evaluateRegistrationICP(originalPoints, mirroredPoints):
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(originalPoints)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(mirroredPoints)
        threshold = 10000
        evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold)

        print(f'evaluation.fitness={evaluation.fitness}, evaluation.inlier_rmse={evaluation.inlier_rmse}')
        return evaluation.fitness, evaluation.inlier_rmse


    def registerWithVanilaICP(originalPoints, mirroredPoints):
        threshold = 100000 #mm
        trans_init = np.identity(4)
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(originalPoints)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(mirroredPoints)
        reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())
        icp = reg_p2p.transformation
        transformationMatrix = vtk.vtkMatrix4x4()

        # Populate the VTK matrix with values from the NumPy matrix
        for i in range(4):
            for j in range(4):
                transformationMatrix.SetElement(i, j, icp[i, j])

        return transformationMatrix