import numpy as np
import slicer
import pandas as pd
import SpineLib

class FacetJointAlignment:

    '''
    Align the facet joints for L1-L2 to attain realistic facet joint space width
    '''
    def align(filepath, vertebraModels, vertebraIDs, lib_vertebraIDs, facetJointsMarkupNodes, FJA_SourceMarkupNodes):

        indices = [lib_vertebraIDs.index(id) for id in vertebraIDs]

        # read measurements from file
        j = pd.read_csv(filepath, header=None, index_col=0, comment='#').to_numpy()
        jointSpaces = [j[index] for index in indices[:-1]]
        jointSpaces  = np.concatenate((jointSpaces, jointSpaces), axis=1)
        
        # access FiducialRegistrationWizard Module
        frw = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLFiducialRegistrationWizardNode')
        frw.SetRegistrationModeToWarping()

        for vt in range (1,len(vertebraIDs)):

            # nodes
            vertebraModel           =   vertebraModels[vt]
            facetJoints             =   facetJointsMarkupNodes[vt-1]
            sourcePoints            =   FJA_SourceMarkupNodes[vt]

            sourcePositions = slicer.util.arrayFromMarkupsControlPoints(facetJoints)[0:10]
            jointTransformPositions = []

            for jointArea in range(0,10):
                point = sourcePositions[jointArea]
                normal = SpineLib.SlicerTools.approx_surface_normal(vertebraModels[vt-1].GetPolyData(), point)
                jointTransformPositions.append(np.add(point, np.multiply(jointSpaces[vt-1][jointArea], normal)))


            # clone fiducial node to To_Points pointlist
            shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
            itemIDToClone = shNode.GetItemByDataNode(sourcePoints)
            clonedItemID = slicer.modules.subjecthierarchy.logic().CloneSubjectHierarchyItem(shNode, itemIDToClone)
            targetPoints = shNode.GetItemDataNode(clonedItemID)

            # set new joint positions to To_Points pointlist
            for t in range(0, len(jointTransformPositions)):
                targetPoints.SetNthControlPointPosition(t, jointTransformPositions[t])


            # transform joint
            jointTransformNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode', "TransformJoint_" + str(vertebraIDs[vt]))
            frw.SetOutputTransformNodeId(jointTransformNode.GetID())
            frw.SetAndObserveFromFiducialListNodeId(sourcePoints.GetID())
            frw.SetAndObserveToFiducialListNodeId(targetPoints.GetID())

            vertebraModel.SetAndObserveTransformNodeID(jointTransformNode.GetID())
            vertebraModel.HardenTransform()

            slicer.mrmlScene.RemoveNode(jointTransformNode)
            slicer.mrmlScene.RemoveNode(targetPoints)
