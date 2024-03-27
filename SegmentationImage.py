import slicer
import skimage
from skimage import measure
import numpy as np
import vtk

import SpineLib


class SegmentationImage:

    '''
    Create vtkMRMLModelNodes in slicer for a segmentation image (numpy).
    For each segmented region, an individual model is added.
    '''
    def createModelsFromNumpy(imageFilePath: str,
                              volumeNode: slicer.vtkMRMLScalarVolumeNode,
                              ):

        # load numpy image
        imageArray = np.load(imageFilePath)

        # spacing = volumeNode.GetSpacing()
        # directions = np.zeros([3, 3])
        # volumeNode.GetIJKToRASDirections(directions)
        # origin = volumeNode.GetOrigin()

        # # convert numpy image to segmentation volume
        # segmentationVolumeNode  =   SpineLib.SegmentationImage.numpy_to_volume(imageArray,origin,spacing,directions)
        
        # # create segments from segmentation volume
        # #segmentationImage       =   slicer.util.arrayFromVolume(segmentationVolumeNode)
        segmentationNode        =   SpineLib.SegmentationImage.segmentImg_to_segments(imageArray, volumeNode)

        # convert segments to models
        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        exportFolderItemId = shNode.CreateFolderItem(shNode.GetSceneItemID(), "Segments")
        slicer.modules.segmentations.logic().ExportAllSegmentsToModels(segmentationNode, exportFolderItemId)

        # get all models in exportFolderItemId
        children = vtk.vtkIdList()
        nodes = []
        shNode.GetItemChildren(exportFolderItemId, children)
        for i in range(children.GetNumberOfIds()):
            child = children.GetId(i)
            nodes.append(shNode.GetItemDataNode(child))

        slicer.mrmlScene.RemoveNode(segmentationNode)

        return nodes

    '''
    Create a vtkMRMLScalarVolumeNode from a numpy image, with the provided origin, spacing and direction.
    '''
    def numpy_to_volume(numpyImage, origin, spacing, directions):

        # create vktImages
        data_vtk = vtk.util.numpy_support.numpy_to_vtk(num_array=numpyImage.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
        imageData = vtk.vtkImageData()
        imageData.SetDimensions(numpyImage.shape[::-1])
        imageData.GetPointData().SetScalars(data_vtk)

        # create slicer image volume
        volumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "SegmentationVolumeNode")
        volumeNode.SetOrigin(origin)
        volumeNode.SetSpacing(spacing)
        volumeNode.SetIJKToRASDirections(directions)
        volumeNode.SetAndObserveImageData(imageData)
        #volumeNode.CreateDefaultDisplayNodes()
        volumeNode.CreateDefaultStorageNode()
        slicer.util.setSliceViewerLayers(background = volumeNode)
        slicer.util.resetSliceViews()

        return volumeNode


    '''
    Create segments from a segmentation image
    '''
    def segmentImg_to_segments(image_array, volumeNode):

        label_image         =   SpineLib.SegmentationImage.filtered_label_image(image_array)
        sorted_labels       =   SpineLib.SegmentationImage.get_sorted_labels(label_image)
        segmentationNode    =   SpineLib.SegmentationImage.create_segments(volumeNode, label_image, sorted_labels)

        return segmentationNode
    

    '''
    Filter the label image to seperate connected regions and to remove non-relevant small objects
    '''
    @classmethod
    def filtered_label_image(cls, image):

        # apply morphological opening
        image_opened = skimage.morphology.binary_opening(image)

        # create temporary label image for threshold computation
        (temp_label_img, num) = measure.label(image_opened, return_num=True)
        label_sizes = [np.count_nonzero(temp_label_img == x) for x in range(1,num+1)]
        mean = np.mean(label_sizes, axis=0)
        sd = np.std(label_sizes, axis=0)
        print("SD: ", sd)
        print("Mean: ", mean)
        threshold = mean - sd

        # remove small objects
        image_filtered = skimage.morphology.remove_small_objects(image_opened, threshold)

        # create final label image
        label_img = measure.label(image_filtered)

        return label_img
    

    '''
    sort labels by height
    '''
    @classmethod
    def get_sorted_labels(cls, label_image):
        # TODO: consider different orientation

        # sort labels by height
        labels = []
        l = 1
        for region in skimage.measure.regionprops(label_image=label_image):
            labels.append((l, region.centroid[0], region.centroid[1], region.centroid[2]))
            l = l+1
        labels.sort(key=lambda x: x[2])
        sorted_labels = [label[0] for label in labels]

        # only take first 6 elements (S1, L5, L4, L3, L2, L1)
        sorted_labels = sorted_labels[:6]


        return sorted_labels
    

    """
    Create a segmentation node in Slicer and add a segment for each label
    """
    @classmethod
    def create_segments(cls, volumeNode, label_image, sorted_labels):

        segmentationNode = slicer.vtkMRMLSegmentationNode()
        slicer.mrmlScene.AddNode(segmentationNode)
        segmentationNode.CreateDefaultDisplayNodes()
        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
        segmentationNode.SetName("VertebraSegmentation")

        vertebraNames = ["S1", "L5", "L4", "L3", "L2", "L1"]
        vNameCounter = 0

        # create a new segment for each label
        for x in range(0,len(vertebraNames)):
            segmentName = vertebraNames[x]
            segmentationNode.GetSegmentation().AddEmptySegment(segmentName)
            segmentId = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segmentName)
            segmentArray = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segmentId, volumeNode)
            segmentArray[:] = 0
            segmentArray[label_image == sorted_labels[x]] = 1
            slicer.util.updateSegmentBinaryLabelmapFromArray(segmentArray, segmentationNode, segmentId, volumeNode)
            vNameCounter = vNameCounter + 1

        segmentationNode.CreateClosedSurfaceRepresentation()
        displayNode = segmentationNode.GetDisplayNode()
        displayNode.SetOpacity3D(0.5)

        return segmentationNode

        
