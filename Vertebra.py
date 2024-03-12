from dataclasses import dataclass
import numpy as np
import vtk
import vtk_convenience as conv
from vtk.util import numpy_support
from typing import Dict, Tuple
import SpineLib
import slicer

class Vertebra:

    def __init__(self,
                 spineGeometries:   vtk.vtkPolyData         = None,
                 geometry:          vtk.vtkPolyData         = None,
                 spineOrientation:  SpineLib.Orientation  = None,
                 max_angle:         float                   = None,
                 landmarks:         SpineLib.Landmarks    = None
                 ) -> None:
        
        self.landmarks       = landmarks
        self.spineGeometries = spineGeometries

        # calculate landmarks, if not provided
        if (self.landmarks == None):
            self.geometry       	      = geometry
            self.center                   = np.array(conv.calc_center_of_mass(self.geometry))
            self.orientation              = Vertebra._init_orientation(self.spineGeometries, self.geometry, self.center, spineOrientation)
            self.body                     = SpineLib.VertebralBody(body=self.geometry, center=self.center, orientation=self.orientation, max_angle=max_angle)
            self.landmarks                = Vertebra._init_landmarks(body=self.body)
        
        self.center                   = np.mean(list(vars(self.landmarks).values()), axis=0)
        self.size, self.orientation   = Vertebra._init_properties(landmarks=self.landmarks)
        self.objectToWorldMatrix      = Vertebra._init_objectToWorldMatrix(self.center, self.size, self.orientation)



    '''
    Calculate orientation of the vertebra.
    The vectors represent the RAS-coordinates of a local object coordinate system.
    RAS -> R:right, A:anterior, S:superior
    '''
    def _init_orientation(
            spineGeometries: vtk.vtkPolyData,
            geometry: vtk.vtkPolyData,
            center: np.array,
            spineOrientation: SpineLib.Orientation
            ) -> SpineLib.Orientation:

        up_approximator         = SpineLib.UpApproximator(spineGeometries)
        oriented_bounding_box   = conv.calc_obb(geometry)[1:]

        r, flip = conv.closest_vector(oriented_bounding_box, spineOrientation.r)
        r       = conv.normalize(flip * np.array(r))
        s       = up_approximator(center)
        s       = conv.normalize(round(np.dot(s, spineOrientation.s)) * np.array(s))
        a       = conv.normalize(np.cross(s, r))

        return SpineLib.Orientation(r=r,a=a,s=s)
    

    '''
    Set landmarks of the vertebral body as the minimal and maximal curve points.
    For each endplate there are 4 landmarks, marking the anterior, posterior, right and left.
    '''
    def _init_landmarks(body:SpineLib.VertebralBody) -> SpineLib.Landmarks:

        superior_posterior = body.superior_sagittal_curve.min_point
        superior_anterior  = body.superior_sagittal_curve.max_point
        inferior_posterior = body.inferior_sagittal_curve.min_point
        inferior_anterior  = body.inferior_sagittal_curve.max_point
        superior_left      = body.superior_frontal_curve.min_point
        superior_right     = body.superior_frontal_curve.max_point
        inferior_left      = body.inferior_frontal_curve.min_point
        inferior_right     = body.inferior_frontal_curve.max_point

        landmarks = [superior_posterior, superior_anterior, inferior_posterior, inferior_anterior,
                     superior_left, superior_right, inferior_left, inferior_right]
        
        return SpineLib.Landmarks(*landmarks)


    '''
    Calculate orientation and size of the vertebra from surface landmarks.
    '''
    def _init_properties(landmarks: SpineLib.Landmarks):

    # averaged vectors between landmarks
        r_vector = np.average(
            [landmarks.superior_right, landmarks.inferior_right],  axis=0) - np.average(
            [landmarks.superior_left,  landmarks.inferior_left],   axis=0)
        a_vector = np.average(
            [landmarks.superior_anterior,  landmarks.inferior_anterior],   axis=0) - np.average(
            [landmarks.superior_posterior, landmarks.inferior_posterior],  axis=0)
        s_vector =  np.average(
            [landmarks.superior_posterior, landmarks.superior_anterior, landmarks.superior_left, landmarks.superior_right], axis=0) - np.average(
            [landmarks.inferior_posterior, landmarks.inferior_anterior, landmarks.inferior_left, landmarks.inferior_right], axis=0)
        

        # object size
        depth  = np.linalg.norm(a_vector)
        width  = np.linalg.norm(r_vector)
        height = np.linalg.norm(s_vector)

        # object orientations
        r_orientation = conv.normalize(r_vector)
        a_orientation = conv.normalize(a_vector)
        s_orientation = conv.normalize(np.cross(r_orientation, a_orientation)) # take cross product, so that all orientation vectors are orthogonal
        
        orientation = SpineLib.Orientation(
            r=r_orientation,
            a=a_orientation,
            s=s_orientation)
        
        size = SpineLib.Size(
            height=height,
            depth=depth,
            width=width)
        
        return size, orientation
    

    '''
    Calculate the object to world matrix
    '''
    def _init_objectToWorldMatrix(center: np.array, size: SpineLib.Size, orientation: SpineLib.Orientation) -> vtk.vtkMatrix4x4:

        # setup rotation matrix with object orientations
        rotationMatrix = vtk.vtkMatrix4x4()
        rotationMatrix.Identity()
        for i in range(3):
            for j in range(3):
                rotationMatrix.SetElement(i, j, [orientation.r, orientation.a, orientation.s][i][j])
        rotationMatrix.Invert()

        # setup translation matrix with object center
        translationMatrix = vtk.vtkMatrix4x4()
        translationMatrix.Identity()
        translationMatrix.SetElement(0,3, center[0])
        translationMatrix.SetElement(1,3, center[1])
        translationMatrix.SetElement(2,3, center[2])

        # setup scaling matrix with object size
        scalingMatrix = vtk.vtkMatrix4x4()
        scalingMatrix.Identity()
        scalingMatrix.SetElement(0,0, size.width)
        scalingMatrix.SetElement(1,1, size.depth)
        scalingMatrix.SetElement(2,2, size.height)

        # final registration matrix = translation matrix * rotation matrix * scaling matrix
        objectToWorldMatrix = vtk.vtkMatrix4x4()
        objectToWorldMatrix.Identity()
        vtk.vtkMatrix4x4.Multiply4x4(translationMatrix, rotationMatrix, objectToWorldMatrix)
        vtk.vtkMatrix4x4.Multiply4x4(objectToWorldMatrix, scalingMatrix, objectToWorldMatrix)
        
        return objectToWorldMatrix
     
        

