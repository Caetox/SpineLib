import vtk
import slicer
from slicer.ScriptedLoadableModule import *
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import open3d as o3d
from skspatial.objects import Plane, Points
import vtk_convenience as conv

''' Fit a symmetry plante to the geometry using ICP and mirroring'''
def fit_symmetry_plane(geometry: vtk.vtkPolyData, numIterations: int):

    centerOfMass = conv.calc_center_of_mass(geometry)
    points = vtk_to_numpy(geometry.GetPoints().GetData())
    
    # set initial plane to sagittal plane
    p0 = vtk.vtkPlane()
    p0.SetOrigin(centerOfMass)
    p0.SetNormal(np.array([1.0, 0.0, 0.0]))

    # mirror the geometry with the initial plane
    mirroredGeometry = mirrorWithPlane(geometry, p0)
    pFin = None

    # fit the symmetry plane with n iterations
    for i in range(numIterations):
        
        # get points and calculate registration transformation with ICP
        mirroredGeometryPoints = vtk_to_numpy(mirroredGeometry.GetPoints().GetData())
        transformationMatrix = registerWithICP(mirroredGeometryPoints, points, pointToPlane=True)
        mirroredGeometry = conv.transform_polydata(mirroredGeometry, transformationMatrix)
        mirroredGeometryPoints = vtk_to_numpy(mirroredGeometry.GetPoints().GetData())

        # get the middle points of the original and mirrored points
        middlePoints = getMiddlePoints(points, mirroredGeometryPoints, viz=False)

        # fit plane to the middle points
        p_i = fitPlaneLeastSquered(middlePoints, centerOfMass, i)

        # mirror with new plane
        mirroredGeometry = mirrorWithPlane(geometry, p_i)
        pFin = p_i
    
    return pFin

''' Mirror the polydata with a plane.'''
def mirrorWithPlane(polydata, plane):  

    normal = plane.GetNormal()
    origin = plane.GetOrigin()
    
    # setup mirror matrix
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
    
    # create the mirror transform
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

''' Get the middle points of the original and mirrored points.
    Middle points = between each original point and its mirrored counterpart.'''
def getMiddlePoints(originalPoints, mirroredPoints, viz=True):
    
    # randomly sample points if there are too many
    if originalPoints.shape[0] > 10000:
        inds = np.random.choice(originalPoints.shape[0], 10000)
        originalPoints = np.take(originalPoints,inds, axis=0)
        mirroredPoints = np.take(mirroredPoints,inds, axis=0)

    # calculate the middle points
    middlepoints = (originalPoints + mirroredPoints)/2.0

    # visualize
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


''' Fit a plane to the middle points of the original and mirrored points using least squares method.'''
def fitPlaneLeastSquered(points, centerOfMass, i):
    plane = Plane.best_fit(Points(points))
    normal = plane.normal.round(3)
    plane = vtk.vtkPlane()
    plane.SetOrigin(centerOfMass)
    plane.SetNormal(normal)

    return plane

''' Evaluate the registration of the original points and the mirrored points.'''
def evaluateRegistrationICP(originalPoints, mirroredPoints):
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(originalPoints)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(mirroredPoints)
    threshold = 10000
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold)

    return evaluation.fitness, evaluation.inlier_rmse

''' Register the original points and the mirrored points with ICP.'''
def registerWithICP(originalPoints, mirroredPoints, pointToPlane=False):
    threshold = 100000 #mm
    trans_init = np.identity(4)
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(originalPoints)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(mirroredPoints)
    if pointToPlane:
        source.estimate_normals()
        target.estimate_normals()
        reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPlane())
    else:
        reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())
    icp = reg_p2p.transformation
    transformationMatrix = vtk.vtkMatrix4x4()

    # Populate the VTK matrix with values from the NumPy matrix
    for i in range(4):
        for j in range(4):
            transformationMatrix.SetElement(i, j, icp[i, j])

    return transformationMatrix
