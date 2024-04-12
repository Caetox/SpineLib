from dataclasses import dataclass
import numpy as np
import os
import vtk
import vtk_convenience as conv
from vtk.util import numpy_support
from typing import Dict, Tuple
import slicer
import SpineLib


class LigamentLandmarks:   

    def __init__(self,
                 spine:            SpineLib.Spine           = None,
                 ) -> None:
        
        #self.all_pll    = LigamentLandmarks._detect_all_pll(spine)
        self.facet_lm   = LigamentLandmarks._detect_facet_landmarks(spine)
        #self.spinous_lm = LigamentLandmarks._detect_spinous_landmarks(spine)
        #self.flavum_lm  = LigamentLandmarks._detect_flavum_landmarks(spine)

        
    '''
    Detect anterior longitudinal ligament (ALL) and posterior longitudinal ligament (PLL) landmarks
    '''
    def _detect_all_pll(spine: SpineLib.Spine) -> Dict[int, np.ndarray]:
        
        for vt, vertebra in enumerate(spine.vertebrae):

            # calculate the spinal canal width
            landmarks   = vertebra.shapeDecomposition.landmarks
            spinal_canal_width = np.linalg.norm(landmarks["left_pedicle_medial"] - landmarks["right_pedicle_medial"])
            print("Spinal canal width: ", spinal_canal_width)

            plane_origins = np.linspace(landmarks["left_pedicle_medial"], landmarks["right_pedicle_medial"], 7, axis=0)

            sup_intersections = [conv.cut_plane(vertebra.body.superior_endplate, plane_origin, vertebra.orientation.r) for plane_origin in plane_origins]
            inf_intersections = [conv.cut_plane(vertebra.body.inferior_endplate, plane_origin, vertebra.orientation.r) for plane_origin in plane_origins]

            ligament_landmarks_markupsNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', "LigamentLandmarks")
            ligament_landmarks_markupsNode.GetDisplayNode().SetTextScale(0.0)
            ligament_landmarks_markupsNode.GetDisplayNode().SetSelectedColor(1, 0, 0)
            ligament_landmarks = []

            for i in sup_intersections + inf_intersections:
                SpineLib.SlicerTools.createModelNode(i, "SupIntersection", color=[1.0, 0.0, 0.0], opacity=0.7)
                regression = conv.calc_main_component(i)
                regression = (regression if regression.dot(vertebra.orientation.a) >= 0 else -regression)
                points = conv.sorted_points(i, regression)
                ligament_landmarks.append(np.array(points[0]))
                ligament_landmarks.append(np.array(points[-1]))

            for lm in ligament_landmarks:
                ligament_landmarks_markupsNode.AddControlPoint(lm)
            
            return ligament_landmarks




    '''
    Get contact points between two polydata
    '''
    def get_contact_points(point_list_1, point_list_2):

        contact_points = []
        for point in point_list_1:
            distances = np.linalg.norm(point_list_2 - point, axis=1)
            contact_points.append(point_list_2[np.argmin(distances)])
  
        return contact_points
    



    '''
    Detect facet joint landmarks
    '''
    def _detect_facet_landmarks(spine: SpineLib.Spine) -> Dict[int, np.ndarray]:

        for inferior_vertebra, superior_vertebra in zip(spine.vertebrae, spine.vertebrae[1:]):

            inf_AS_polydatas  = [inferior_vertebra.shapeDecomposition.process_polydata["ASL"], inferior_vertebra.shapeDecomposition.process_polydata["ASR"]]
            sup_AI_polydatas  = [superior_vertebra.shapeDecomposition.process_polydata["AIL"], superior_vertebra.shapeDecomposition.process_polydata["AIR"]]
            medial_directions = [inferior_vertebra.orientation.r, -inferior_vertebra.orientation.r]

            for inf_AS_polydata, sup_AI_polydata, medial_direction in zip(inf_AS_polydatas, sup_AI_polydatas, medial_directions):

                # get facet contact surface
                inf_AS_points = numpy_support.vtk_to_numpy(inf_AS_polydata.GetPoints().GetData())
                sup_AI_points = numpy_support.vtk_to_numpy(sup_AI_polydata.GetPoints().GetData())

                contact_points_inf = LigamentLandmarks.get_contact_points(sup_AI_points, inf_AS_points)
                contact_points_markup = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', "ContactPoints")
                contact_points_markup.GetDisplayNode().SetTextScale(0.0)
                contact_points_markup.GetDisplayNode().SetSelectedColor(1, 0, 0)
                contact_points_markup.GetDisplayNode().SetGlyphScale(1.0)
                for cp in contact_points_inf:
                    contact_points_markup.AddControlPoint(cp)


                # Compute best fit plane
                points = vtk.vtkPoints()
                for point in contact_points_inf:
                    points.InsertNextPoint(point)
                center = [0.0, 0.0, 0.0]
                normal = [0.0, 0.0, 1.0]
                vtk.vtkPlane.ComputeBestFittingPlane(points, center, normal)
                normal = np.array(normal)
                # check if normal or inverse of normal is more similar with medial_direction
                if np.dot(normal, np.average([medial_direction, -inferior_vertebra.orientation.a], axis=0)) < 0:
                    normal = -normal

                # local plane directions
                u = inferior_vertebra.orientation.s
                n = np.array(normal)
                n_norm = np.sqrt(sum(n**2))
                proj_of_u_on_n = (np.dot(u, n)/n_norm**2)*n
                facet_up = u - proj_of_u_on_n
                facet_medial = normal
                facet_LR = np.cross(facet_up, facet_medial)

                # filter polydata to get facet surface
                facet_polydata = conv.eliminate_misaligned_faces(inf_AS_polydata, center, normal, 45.0)
                facet_polydata = conv.filterLargestRegion(facet_polydata)
                facet_center = np.mean(numpy_support.vtk_to_numpy(facet_polydata.GetPoints().GetData()), axis=0)

                # Display best fit plane as a markups plane
                planeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsPlaneNode')
                planeNode.SetCenter(facet_center)
                planeNode.SetNormal(normal)
                planeNode.SetSize(20, 20)
                planeNode.GetDisplayNode().SetOpacity(0.5)
                planeNode.GetDisplayNode().SetHandlesInteractive(False)
                SpineLib.SlicerTools.createModelNode(facet_polydata, "Facet", color=[1.0, 0.0, 0.0], opacity=0.7)

                lr_intersection = conv.cut_plane(facet_polydata, facet_center, facet_up)
                lr_sorted = conv.sorted_points(lr_intersection, facet_LR)
                lr_1 = lr_sorted[0]
                lr_2 = lr_sorted[-1]
                is_intersection = conv.cut_plane(facet_polydata, facet_center, facet_LR)
                is_sorted = conv.sorted_points(is_intersection, facet_up)
                is_1 = is_sorted[0]
                is_2 = is_sorted[-1]

                inf_ligament_markup = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', "InferiorVertebraLigamentLandmarks")
                inf_ligament_markup.GetDisplayNode().SetTextScale(0.0)
                inf_ligament_markup.GetDisplayNode().SetSelectedColor(0, 0, 1)
                inf_ligament_landmarks = [lr_1, lr_2, is_1, is_2]
                for lm in inf_ligament_landmarks:
                    inf_ligament_markup.AddControlPoint(lm)

                sup_ligament_landmarks = []
                for lm in inf_ligament_landmarks:
                    # find closest point in superior vertebra AIL to the landmark
                    closest_point = sorted(sup_AI_points, key=lambda p: np.linalg.norm(p - lm))[0]
                    sup_ligament_landmarks.append(closest_point)

                sup_ligament_markup = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', "SuperiorVertebraLigamentLandmarks")
                sup_ligament_markup.GetDisplayNode().SetTextScale(0.0)
                sup_ligament_markup.GetDisplayNode().SetSelectedColor(0, 1, 1)
                for lm in sup_ligament_landmarks:
                    sup_ligament_markup.AddControlPoint(lm)


        # for vt, vertebra in enumerate(spine.vertebrae):

        #     landmarks = vertebra.shapeDecomposition.landmarks
        #     facet_landmarks = {}
        #     for side in ["left", "right"]:
        #         facet_landmarks[side] = {}
        #         for direction in ["medial", "lateral"]:
        #             facet_landmarks[side][direction] = conv.cut_plane(vertebra.facet_joints[side][direction], landmarks[f"{side}_pedicle_{direction}"], vertebra.orientation.r)
        #             SpineLib.SlicerTools.createModelNode(facet_landmarks[side][direction], f"{side.capitalize()}Facet{direction.capitalize()}", color=[0.0, 1.0, 0.0], opacity=0.7)

        # return facet_landmarks

    
    def _detect_spinous_landmarks(spine):
        
        for vt, vertebra in enumerate(spine.vertebrae):

            spinous_polydata = vertebra.shapeDecomposition.process_polydata["S"]

            upper_spinous = conv.eliminate_misaligned_faces(spinous_polydata, vertebra.center, vertebra.orientation.s, 60.0)
            lower_spinous = conv.eliminate_misaligned_faces(spinous_polydata, vertebra.center, -vertebra.orientation.s, 60.0)
            upper_spinous_points = numpy_support.vtk_to_numpy(upper_spinous.GetPoints().GetData())
            lower_spinous_points = numpy_support.vtk_to_numpy(lower_spinous.GetPoints().GetData())

            # # create curve for upper_spinous_points
            # upper_spinous_curve = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsCurveNode', "UpperSpinousCurve")
            # upper_spinous_curve.GetDisplayNode().SetTextScale(0.0)
            # upper_spinous_curve.GetDisplayNode().SetSelectedColor(1, 0, 0)
            # slicer.util.updateMarkupsControlPointsFromArray(upper_spinous_curve, upper_spinous_points)

            SpineLib.SlicerTools.createModelNode(upper_spinous, "UpperSpinous", color=[1.0, 0.0, 0.0], opacity=0.8)
            SpineLib.SlicerTools.createModelNode(lower_spinous, "LowerSpinous", color=[1.0, 0.0, 0.0], opacity=0.8)

            sorted_upper_spinous = conv.sorted_points(upper_spinous, conv.calc_main_component(upper_spinous))
            # create curve for upper_spinous_points
            upper_spinous_curve = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsCurveNode', "UpperSpinousCurve")
            upper_spinous_curve.SetCurveTypeToPolynomial()
            upper_spinous_curve.GetDisplayNode().SetTextScale(0.0)
            upper_spinous_curve.GetDisplayNode().SetSelectedColor(1, 0, 0)
            slicer.util.updateMarkupsControlPointsFromArray(upper_spinous_curve, np.array(sorted_upper_spinous))

            sorted_lower_spinous = conv.sorted_points(lower_spinous, conv.calc_main_component(lower_spinous))
            # create curve for lower_spinous_points
            lower_spinous_curve = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsCurveNode', "LowerSpinousCurve")
            lower_spinous_curve.SetCurveTypeToPolynomial()
            lower_spinous_curve.GetDisplayNode().SetTextScale(0.0)
            lower_spinous_curve.GetDisplayNode().SetSelectedColor(1, 0, 0)
            slicer.util.updateMarkupsControlPointsFromArray(lower_spinous_curve, np.array(sorted_lower_spinous))

            # resamplingNumber = 8
            # sampleDist = upper_spinous_curve.GetCurveLengthWorld() / (resamplingNumber-1)
            # upper_spinous_curve.ResampleCurveWorld(sampleDist)

            # Compute best fit plane
            points = vtk.vtkPoints()
            for point in np.concatenate((np.array(upper_spinous_points), np.array(lower_spinous_points))):
                points.InsertNextPoint(point)
            center = [0.0, 0.0, 0.0]
            normal = [0.0, 0.0, 1.0]
            vtk.vtkPlane.ComputeBestFittingPlane(points, center, normal)
            #Display best fit plane as a markups plane
            planeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsPlaneNode')
            planeNode.SetCenter(center)
            planeNode.SetNormal(normal)
            planeNode.SetSize(50, 50)
            planeNode.GetDisplayNode().SetOpacity(0.5)
            planeNode.GetDisplayNode().SetHandlesInteractive(False)




            # # Create a vtkPoints object to hold the points
            # points = vtk.vtkPoints()

            # # Add the points to the vtkPoints object
            # for point in upper_spinous_points:
            #     points.InsertNextPoint(point)

            # # Fit a polynomial spline to the points
            # spline = vtk.vtkCardinalSpline()
            # spline.SetLeftConstraint(2)  # Free slope at the left endpoint
            # spline.SetRightConstraint(2)  # Free slope at the right endpoint
            # spline.SetClosed(False)  # Not a closed spline

            # for i in range(points.GetNumberOfPoints()):
            #     x, y, z = points.GetPoint(i)  # Assuming points are (x, y, z)
            #     spline.AddPoint(i, x, y, z)



            # # Resample the spline to have n points
            # n_points = 10
            # splineFilter = vtk.vtkSplineFilter()
            # splineFilter.SetInputData(upper_spinous)
            # splineFilter.SetSpline(spline)
            # splineFilter.SetNumberOfOutputPoints(n_points)
            # splineFilter.Update()

            # points = splineFilter.GetOutput().GetPoints()
            # print(points)

            # mkup = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', "Curve")
            # mkup.GetDisplayNode().SetTextScale(0.0)
            # mkup.GetDisplayNode().SetSelectedColor(1, 0, 1)
            # for i in range(points.GetNumberOfPoints()):
            #     mkup.AddControlPoint(points.GetPoint(i))

            ###################################################################################################################################################
    
    '''
    Detect ligamentum flavum landmarks
    '''
    def _detect_flavum_landmarks(spine: SpineLib.Spine) -> Dict[int, np.ndarray]:

        for vt, vertebra in enumerate(spine.vertebrae):
            
            # canal_markups = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', "Canal")
            # canal_markups.GetDisplayNode().SetTextScale(0.0)
            # canal_markups.GetDisplayNode().SetSelectedColor(0, 1, 1)
            # # canal_path = self.runDijkstra(shapeDecomposition.processes_poly, left_sphere_intersection_closest_point, right_sphere_intersection_closest_point)
            # # canal_path_points = numpy_support.vtk_to_numpy(canal_path.GetData())


            # # extract posterior canal
            # # extract points, where normals are similar to the given direction
            # posterior_canal = conv.eliminate_misaligned_faces(polydata=shapeDecomposition.processes_poly, center=[0,0,0], direction=vertebra.orientation.a, max_angle=60.0)
            # connectivity_filter = vtk.vtkPolyDataConnectivityFilter()
            # connectivity_filter.SetInputData(posterior_canal)
            # connectivity_filter.SetExtractionModeToLargestRegion()
            # connectivity_filter.Update()
            # posterior_canal = connectivity_filter.GetOutput()
            # cleaner = vtk.vtkCleanPolyData()
            # cleaner.SetInputData(posterior_canal)
            # cleaner.SetTolerance(0.001)  # Adjust tolerance as needed
            # cleaner.PointMergingOn()
            # cleaner.Update()

            # posterior_canal = cleaner.GetOutput()
            # SpineLib.SlicerTools.createModelNode(posterior_canal, "PosteriorCanal", color=[1.0, 0.0, 0.0], opacity=1.0)

            # # compute geodesic path with dijkstra
            # dijkstraPoints = self.runDijkstra(posterior_canal, left_sphere_intersection_highest_point, right_sphere_intersection_highest_point)
            # #dijkstraPoints = self.runDijkstra(shapeDecomposition.processes_poly, left_superior_articular_segment_highest_point, right_superior_articular_segment_highest_point)

            # path = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', "Path")
            # path.GetDisplayNode().SetTextScale(0.0)
            # path.GetDisplayNode().SetSelectedColor(1, 0, 1)
            # #path.GetDisplayNode().SetOpacity(0.5)
            # path.GetDisplayNode().SetGlyphScale(1.0)
            # for dp in range(dijkstraPoints.GetNumberOfPoints()):
            #     path.AddControlPoint(dijkstraPoints.GetPoint(dp))




            # # find intersection points
            # parametricSpline = vtk.vtkParametricSpline()
            # parametricSpline.SetPoints(dijkstraPoints)
            # parametricFunctionSource = vtk.vtkParametricFunctionSource()
            # parametricFunctionSource.SetParametricFunction(parametricSpline)
            # parametricFunctionSource.Update()

            # sup_flavum = [conv.cut_plane(parametricFunctionSource.GetOutput(), plane_origin, vertebra.orientation.r).GetPoint(0) for plane_origin in plane_origins]
            # #intersection = conv.cut_plane(parametricFunctionSource.GetOutput(), plane_origins[3], vertebra.orientation.r)

            # flavum = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', "Flavum")
            # flavum.GetDisplayNode().SetTextScale(0.0)
            # flavum.GetDisplayNode().SetSelectedColor(1, 0, 0)
            # for sf in sup_flavum:
            #     try:
            #         flavum.AddControlPoint(sf)
            #     except:
            #         pass

            pass
        pass