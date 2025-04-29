import vtk
import numpy as np
import slicer
import os
from dataclasses import dataclass
from slicer.util import loadModel
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
import logging
import logging

import vtk_convenience as conv
import SpineLib


class SlicerTools:

    '''
    Remeshing model
    '''
    def remesh(model, factor):
        remesh = vtk.vtkQuadricDecimation()
        remesh.SetInputData(model.GetPolyData())
        remesh.SetTargetReduction(factor)
        remesh.Update()
        return remesh.GetOutput()

    '''
    Create model node
    '''
    def createModelNode(polydata, name, color=[1,0,0], opacity=1.0):
        modelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', name)
        modelNode.SetAndObservePolyData(polydata)
        modelNode.CreateDefaultDisplayNodes()
        modelDisplayNode = modelNode.GetDisplayNode()
        modelDisplayNode.SetColor(color)
        modelDisplayNode.SetOpacity(opacity)
        return modelNode

    '''
    Transform objects with matrix.
    vtObjects: list containing Slicer nodes such as model and corresposponding landmarks, sourceimage (CT/MRI) or source labelmap
    '''
    def transformVertebraObjects(transformMatrix, vtObjects):
            
            transformNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode')
            transformNode.SetMatrixTransformToParent(transformMatrix)
            for vtObject in vtObjects:
                if (vtObject is not None):
                    vtObject.SetAndObserveTransformNodeID(transformNode.GetID())
                    vtObject.HardenTransform()
            slicer.mrmlScene.RemoveNode(transformNode)


    '''
    Transform one object with transformation matrix.
    vtObject: Slicer node to be transformed
    '''
    def transformOneObject(transformMatrix, vtObject):
            
            transformNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode')
            transformNode.SetMatrixTransformToParent(transformMatrix)
            if (vtObject is not None):
                vtObject.SetAndObserveTransformNodeID(transformNode.GetID())
                vtObject.HardenTransform()
            slicer.mrmlScene.RemoveNode(transformNode)
    
    '''
    create point cloud as Slicer node from a numpy array (x,3)
    '''
    def createPointCloudNode(points, radius=2.0):
            
        # Create the vtkPoints object.
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(vtk.util.numpy_support.numpy_to_vtk(points))

        # Create the vtkPolyData object.
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)

        # Create the vtkSphereSource object.
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(radius)

        # Create the vtkGlyph3D object.
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(polydata)
        glyph.SetSourceConnection(sphere.GetOutputPort())


        pointCloudModelNode = slicer.modules.models.logic().AddModel(glyph.GetOutputPort())
            



    '''
    Approximate the surface normal of a given coordinate on a polydata.
    '''
    def approx_surface_normal(polydata, coordinate):   

        pointLocator = vtk.vtkPointLocator()
        pointLocator.SetDataSet(polydata)
        pointLocator.BuildLocator()
        result = vtk.vtkIdList()

        # find closest point
        pointLocator.FindClosestNPoints(1, coordinate, result)
        pointId = result.GetId(0)

        # get point normal
        normal = np.array(list(conv.iter_normals(polydata)))[pointId]
        normal = normal/np.linalg.norm(normal)

        return normal
    

    '''
    Compute center of mass for each model and create markup fiducials.
    '''
    def centroidFiducialsForModels(modelNodes):

        # create fiducial point list node
        pointListNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "CentroidMarkupsNode")

        # add model CoMs to point list node
        for modelNode in modelNodes:
            polydata = modelNode.GetPolyData().GetPoints().GetData()
            centroid_ras = np.average(vtk_to_numpy(polydata), axis=0)
            pointListNode.AddFiducialFromArray(centroid_ras, modelNode.GetName())
        
        return pointListNode
    
    '''
    Create a markups line node from start and end coordinates
    '''
    def markupsCurveNode(pointListNode):

        # get control points
        coordinates = slicer.util.arrayFromMarkupsControlPoints(pointListNode)

        # create curve node
        curveNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", "Curve")
        curveNode.SetCurveTypeToCardinalSpline()
        slicer.util.updateMarkupsControlPointsFromArray(curveNode, coordinates)

        return curveNode

    '''
    Create a markups line node for a markups fiducial point list.
    '''
    def markupsLineNode(lineNodeName, lineStartPos, lineEndPos):

        lineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", lineNodeName)
        lineNode.AddControlPointWorld(lineStartPos, f'start_{lineNodeName}')
        lineNode.AddControlPointWorld(lineEndPos, f'end_{lineNodeName}')
        lineNode.GetDisplayNode().SetTextScale(0.0)


        return lineNode
    

    '''
    For a set of vtkMRMLModelNodes, create oriented bounding boxes as vtkMRMLROINodes.
    The orientation of the bounding boxes are calculated from the curveNode.
    '''
    # TODO: seperate functionality for getting orientation and creating bounding boxes (apply bounding box function to individual vertebrae)
    # TODO: seperate functionality for creating vtk Bounding Box and adding vtkMRMLROINode
    
    def orientedBoundingBoxes(modelNodes, curveNode):

        orientedBoxes = []
    
        for i in range(len(modelNodes)):
            modelNode = modelNodes[i]

            # transform cloned node to align with world coordinate system
            transformMatrix = vtk.vtkMatrix4x4()
            curveNode.GetCurvePointToWorldTransformAtPointIndex(curveNode.GetCurvePointIndexFromControlPointIndex(i),transformMatrix)
            transformMatrix.Invert()
            SpineLib.TransformationTools.transformVertebraObjects(transformMatrix, [modelNode])

            # for all points of the model: get min and max coordinate values
            points = slicer.util.arrayFromModelPoints(modelNode)
            min_vals = np.min(points, axis=0)
            max_vals = np.max(points, axis=0)
            roi_size = np.subtract(max_vals,min_vals)
            roi_radius = np.divide(roi_size,2)
            roi_center = np.add(min_vals,roi_radius)

            # create axis-aligned bounding box
            aabb=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
            aabb.GetDisplayNode().SetHandlesInteractive(False)
            aabb.SetName("AABB_" + str(modelNode.GetName()))
            aabb.SetCenter(roi_center)
            aabb.SetSize(roi_size)
            aabbDisplayNode = aabb.GetDisplayNode()
            aabbDisplayNode.SetSelectedColor(1,0,1)
            aabbDisplayNode.SetOpacity(0.5)

            # transform bounding box to align with object
            transformMatrix.Invert()
            SpineLib.TransformationTools.transformVertebraObjects(transformMatrix, [modelNode, aabb])

            orientedBoxes.append(aabb)
        
        return orientedBoxes


    '''
    Load all files (stl and json) to the slicer scene.
    '''
    def loadObjectsInDirectory(directory_path):

        files = slicer.util.getFilesInDirectory(directory_path)
        nodes = []

        for file in files:
            if file.endswith(".stl"):
                node = slicer.util.loadModel(file)
            if file.endswith(".json"):
                node = slicer.util.loadMarkups(file)
            nodes.append(node)

        return nodes
    

    '''
    Given a set of nodes, get all nodes whose name contains the passed name.
    If provided, sort the nodes with the sortingKeys:
    The order or nodes will be matched with the order of keys, where the key is a substring of the nodes name.
    '''
    def getSortedNodesByName(nodes, sortingKeys, name):

        nodesByName = [node for node in nodes if name in node.GetName()]
        nodesByKeys = [node for node in nodesByName for key in sortingKeys if key+"_" in node.GetName()]
        nodeNames   = [nodeByKey.GetName() for nodeByKey in nodesByKeys]
        sortedNames = sorted(nodeNames, key=lambda x: next((sortingKeys.index(key) for key in sortingKeys if key in x), float('inf')))
        sortedNodes = [nodeByKeys for sortedName in sortedNames for nodeByKeys in nodesByKeys if nodeByKeys.GetName() == sortedName]

        return sortedNodes
    
    '''
    Remove all passed nodes from the slicer scene
    '''
    def removeNodes(node_structure):
        if isinstance(node_structure, list):
            for item in node_structure:
                SpineLib.SlicerTools.removeNodes(item)
        else:
            # Single node
            slicer.mrmlScene.RemoveNode(node_structure)


    '''
    Return verticies of the mesh from model node (3D) as numpy array
    '''
    def pointsFromModelNode_asNumPy(modelNode):

        assert modelNode is not None
        polydata = modelNode.GetPolyData()
        pointsData = polydata.GetPoints().GetData()
        return vtk_to_numpy(pointsData)

    
    '''
    Add a model containing vtkGlyph3D objects that represent a point cloud 
    Return pointCloudModelNode
    '''
    def drawPointCloudFromNumpyArray(points, radius=1.5):
        
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(numpy_to_vtk(points))

        # Create the vtkPolyData object.
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)

        # Create the vtkSphereSource object.
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(radius)

        # Create the vtkGlyph3D object.
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(polydata)
        glyph.SetSourceConnection(sphere.GetOutputPort())


        pointCloudModelNode = slicer.modules.models.logic().AddModel(glyph.GetOutputPort())
        return pointCloudModelNode
    
    '''
    Resample curve to given number of points
    '''
    def resampleCurve(curveNode, numberOfPoints):

        sampleDist = curveNode.GetCurveLengthWorld() / (numberOfPoints-1)
        curveNode.ResampleCurveWorld(sampleDist)

    '''
    Create curve an resample to given number of points
    '''
    def createResampledCurve(controlPoints, numberOfPoints, name="MarkupsCurve", color=[1,0,0]):

        curve = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsCurveNode', name)
        curve.SetCurveTypeToPolynomial()
        curve.GetDisplayNode().SetTextScale(0.0)
        curve.GetDisplayNode().SetSelectedColor(color)
        slicer.util.updateMarkupsControlPointsFromArray(curve, np.array(controlPoints))
        SpineLib.SlicerTools.resampleCurve(curve, numberOfPoints)

        return curve
    
    '''
    Create markups point list node from numpy array
    '''
    def createMarkupsFiducialNode(points, name="MarkupsFiducial", color=[0,0,1], glyphScale=3.0):

        markupsNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', name)
        markupsNode.GetDisplayNode().SetSelectedColor(color)
        markupsNode.GetDisplayNode().SetTextScale(0.0)
        markupsNode.GetDisplayNode().SetGlyphScale(glyphScale)
        for point in points:
            markupsNode.AddFiducialFromArray(point, name)
        
        return markupsNode
    
    '''
    Create markups plane node from origin and normal
    '''
    def createMarkupsPlaneNode(origin, normal, name="MarkupsPlane", width=50, length=50, color=[1,0,0], opacity=0.5):

        markupsNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsPlaneNode', name)
        markupsNode.SetOrigin(origin)
        markupsNode.SetNormal(normal)
        markupsNode.SetSize(width, length)
        markupsNode.GetDisplayNode().SetHandlesInteractive(False)
        markupsNode.GetDisplayNode().SetSelectedColor(color)
        markupsNode.GetDisplayNode().SetOpacity(opacity)
        
        return markupsNode
    
    '''
    Set model color node for scalars
    '''
    def setModelColorTable(modelNode, colorTableName):
        """Set the color table of the model node to the specified color table."""
        displayNode = modelNode.GetDisplayNode()
        if displayNode:
            colorNode = slicer.util.getNode(colorTableName)
            if colorNode:
                displayNode.SetAndObserveColorNodeID(colorNode.GetID())
            else:
                logging.error(f"Color table {colorTableName} not found.")