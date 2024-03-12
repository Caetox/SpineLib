import slicer
import SegmentStatistics
import vtk
import numpy as np

import SpineLib


# Find IVD center in a spine segmentation image

class IVD_Center:

    def ivd_center_markups(segmentationNode, segmentationVolumeNode):
        centroids       =   SpineLib.IVD_Center.addObbCentroids(segmentationNode)
        curveModel      =   SpineLib.IVD_Center.addCurve(centroids)
        curveSegment    =   SpineLib.IVD_Center.curveModelToSegment(curveModel, segmentationVolumeNode)
        IVDP_Node       =   SpineLib.IVD_Center.addIVDPlanes(curveSegment, segmentationVolumeNode)
        intersection    =   SpineLib.IVD_Center.intersectCurveIVDP(segmentationVolumeNode, IVDP_Node, curveSegment)
        IVD_Center      =   SpineLib.IVD_Center.addIvdCenter(intersection)



    def addObbCentroids(segmentationNode):

        # compute oriented bounding box for each vertebra
        segStatLogic = SegmentStatistics.SegmentStatisticsLogic()
        segStatLogic.getParameterNode().SetParameter("Segmentation", segmentationNode.GetID())
        segStatLogic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.obb_origin_ras.enabled",str(True))
        segStatLogic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.obb_diameter_mm.enabled",str(True))
        segStatLogic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.obb_direction_ras_x.enabled",str(True))
        segStatLogic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.obb_direction_ras_y.enabled",str(True))
        segStatLogic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.obb_direction_ras_z.enabled",str(True))
        segStatLogic.computeStatistics()
        stats = segStatLogic.getStatistics()

        # add markup fiducials
        pointListNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "OBB_Centroid_MarkupsNode")
        pointListNode.CreateDefaultDisplayNodes()

        # Draw ROI for each oriented bounding box
        for segmentId in stats["SegmentIDs"]:

            # Get bounding box
            obb_origin_ras = np.array(stats[segmentId,"LabelmapSegmentStatisticsPlugin.obb_origin_ras"])
            obb_diameter_mm = np.array(stats[segmentId,"LabelmapSegmentStatisticsPlugin.obb_diameter_mm"])
            obb_direction_ras_x = np.array(stats[segmentId,"LabelmapSegmentStatisticsPlugin.obb_direction_ras_x"])
            obb_direction_ras_y = np.array(stats[segmentId,"LabelmapSegmentStatisticsPlugin.obb_direction_ras_y"])
            obb_direction_ras_z = np.array(stats[segmentId,"LabelmapSegmentStatisticsPlugin.obb_direction_ras_z"])
            # Create ROI
            segment = segmentationNode.GetSegmentation().GetSegment(segmentId)
            roi=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
            roi.SetName(segment.GetName() + " OBB")
            roi.GetDisplayNode().SetHandlesInteractive(False)
            roi.GetDisplayNode().SetOpacity(0.0)
            roi.SetSize(obb_diameter_mm)
            # Position and orient ROI using a transform
            obb_center_ras = obb_origin_ras+0.5*(obb_diameter_mm[0] * obb_direction_ras_x + obb_diameter_mm[1] * obb_direction_ras_y + obb_diameter_mm[2] * obb_direction_ras_z)
            boundingBoxToRasTransform = np.row_stack((np.column_stack((obb_direction_ras_x, obb_direction_ras_y, obb_direction_ras_z, obb_center_ras)), (0, 0, 0, 1)))
            boundingBoxToRasTransformMatrix = slicer.util.vtkMatrixFromArray(boundingBoxToRasTransform)
            roi.SetAndObserveObjectToNodeMatrix(boundingBoxToRasTransformMatrix)

            # add obbCentroid Fiducials
            pointListNode.AddFiducialFromArray(obb_center_ras, "OBBCentroid")

            return pointListNode


    def addCurve(sourceNode):
        
        # create model node
        curveModel = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', 'CurveModel')

        # 'activate' Curve Maker Module
        mainWindow = slicer.util.mainWindow()
        mainWindow.moduleSelector().selectModule('CurveMaker')
        mainWindow.moduleSelector().selectModule('Registration')

        # create Curve with Curve maker widget
        cml = slicer.modules.CurveMakerWidget.logic
        cml.SourceNode = sourceNode
        cml.DestinationNode = curveModel
        cml.TubeRadius = 1
        cml.generateCurveOnce()

        return curveModel


    def addIVDPlanes(curveSegment, volumeNode):

        # Clone the segmentation node
        shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        itemIDToClone = shNode.GetItemByDataNode(curveSegment)
        clonedItemID = slicer.modules.subjecthierarchy.logic().CloneSubjectHierarchyItem(shNode, itemIDToClone)
        IVDP_Node = shNode.GetItemDataNode(clonedItemID)
        clonedSegmentation = IVDP_Node.GetSegmentation()
        IVDP_Node.SetName("IVDPs")

        # Grow From Seeds (Segment Editor)
        segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
        segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        segmentEditorNode = slicer.vtkMRMLSegmentEditorNode()
        slicer.mrmlScene.AddNode(segmentEditorNode)
        segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
        segmentEditorWidget.setSegmentationNode(IVDP_Node)
        segmentEditorWidget.setMasterVolumeNode(volumeNode)
        segmentEditorWidget.setActiveEffectByName("Grow from seeds")
        effect = segmentEditorWidget.activeEffect()
        effect.setParameter("SeedLocalityFactor", 10.0)
        effect.self().onPreview()
        effect.self().onApply()

        # add Margin to each segment
        inputSegmentIDs = vtk.vtkStringArray()
        IVDP_Node.GetDisplayNode().GetVisibleSegmentIDs(inputSegmentIDs)

        for index in range(inputSegmentIDs.GetNumberOfValues()):
            segmentID = inputSegmentIDs.GetValue(index)
            segmentEditorWidget.setCurrentSegmentID(segmentID)
            segmentEditorWidget.setActiveEffectByName("Margin")
            segmentEditorNode.SetOverwriteMode(2)
            segmentEditorNode.SetMaskMode(0)
            effect = segmentEditorWidget.activeEffect()
            effect.setParameter("MarginSizeMm", 0.8)
            effect.self().onApply()

        # get intersection of vertebrae pairs
        for index in range(inputSegmentIDs.GetNumberOfValues()):
            try:
                segmentID = inputSegmentIDs.GetValue(index)
                modifierSegmentID = inputSegmentIDs.GetValue(index+1)
                segmentEditorWidget.setCurrentSegmentID(segmentID)
                segmentEditorWidget.setActiveEffectByName("Logical operators")
                effect = segmentEditorWidget.activeEffect()
                effect.setParameter("Operation", "INTERSECT")
                effect.setParameter("ModifierSegmentID", modifierSegmentID)
                effect.self().onApply()
            except ValueError:
                segment = clonedSegmentation.GetSegment(segmentID)
                clonedSegmentation.RemoveSegment(segmentID)
        
        return IVDP_Node

    
    def curveModelToSegment(curveModel, volumeNode):

        # convert model to segment
        curveSegment = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode', 'CurveSegment')
        curveSegment.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
        slicer.modules.segmentations.logic().ImportModelToSegmentationNode(curveModel, curveSegment)
        # curveModel.GetDisplayNode().SetOpacity(0.2)
        # curveSegment.GetDisplayNode().SetOpacity3D(0.5)
        segmentation = curveSegment.GetSegmentation()
        segmentation.GetSegment(segmentation.GetSegmentIdBySegmentName("CurveModel")).SetColor(1,1,0)

        return curveSegment


    def intersectCurveIVDP(volumeNode, IVDP_Node, curveSegment):

        curveSegmentation = curveSegment.GetSegmentation()

        # Segment Editor 
        segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
        segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        segmentEditorNode = slicer.vtkMRMLSegmentEditorNode()
        slicer.mrmlScene.AddNode(segmentEditorNode)
        segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
        segmentEditorWidget.setSegmentationNode(IVDP_Node)
        segmentEditorWidget.setMasterVolumeNode(volumeNode)

        inputSegmentIDs = vtk.vtkStringArray()
        IVDP_Node.GetDisplayNode().GetVisibleSegmentIDs(inputSegmentIDs)

        # intersect Curve with each (grown) Segment
        for index in range(inputSegmentIDs.GetNumberOfValues()):
            segmentID = inputSegmentIDs.GetValue(index)
            if (segmentID != "Curve"):
                curveSegmentID = curveSegmentation.GetNthSegmentID(0)
                IVDP_Node.GetSegmentation().CopySegmentFromSegmentation(curveSegmentation, curveSegmentID)
                curveSegmentCopyID = IVDP_Node.GetSegmentation().GetSegmentIdBySegmentName("Model")
                segmentEditorWidget.setCurrentSegmentID(curveSegmentCopyID)
                segmentEditorWidget.setActiveEffectByName("Logical operators")
                effect = segmentEditorWidget.activeEffect()
                effect.setParameter("Operation", "INTERSECT")
                effect.setParameter("ModifierSegmentID", segmentID)
                effect.self().onApply()
                curveCopy = IVDP_Node.GetSegmentation().GetSegment(curveSegmentCopyID)
                curveCopy.SetName("IVD_Centroid")
            IVDP_Node.RemoveSegment(segmentID)

        return IVDP_Node
        
        #slicer.mrmlScene.RemoveNode(curveSegment)


    def addIvdCenter(intersection):

        # Compute centroids
        segStatLogic = SegmentStatistics.SegmentStatisticsLogic()
        segStatLogic.getParameterNode().SetParameter("Segmentation", intersection.GetID())
        segStatLogic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.centroid_ras.enabled", str(True))
        segStatLogic.computeStatistics()
        stats = segStatLogic.getStatistics()

        # Place a markup point in each centroid
        pointListNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "IVD_Centroids_MarkupsNode")
        pointListNode.CreateDefaultDisplayNodes()
        pointListNode.GetDisplayNode().SetSelectedColor(1,1,0)
        for segmentId in stats["SegmentIDs"]:
            centroid_ras = stats[segmentId,"LabelmapSegmentStatisticsPlugin.centroid_ras"]
            segmentName = intersection.GetSegmentation().GetSegment(segmentId).GetName()
            pointListNode.AddFiducialFromArray(centroid_ras, segmentName)

        return pointListNode
        
        #slicer.mrmlScene.RemoveNode(intersection)