from dataclasses import dataclass
import numpy as np
import os
import vtk
import vtk_convenience as conv
from vtk.util import numpy_support
from typing import Dict, Tuple
import slicer
import SpineLib
import pyvista


class LigamentLandmarks:   

    def __init__(self,
                 spine:            SpineLib.Spine           = None,
                 spine_indices:    np.ndarray               = None,
                 ) -> None:
        
        try:
            self.all_pll    = LigamentLandmarks._detect_all_pll(spine, spine_indices)
            self.CL 	    = LigamentLandmarks._detect_CL(spine, spine_indices)
            self.ISL        = LigamentLandmarks._detect_ISL(spine, spine_indices)
            self.LF         = LigamentLandmarks._detect_LF(spine, spine_indices)
            self.ITL        = LigamentLandmarks._detect_ITL(spine, spine_indices)
            self.SSL        = LigamentLandmarks._detect_SSL(spine, spine_indices)

        except Exception as e:
            print(e)
            print("Error in LigamentLandmarks")
            return None

    '''
    Detect intertransverse ligament landmarks
    '''
    def _detect_ITL(spine: SpineLib.Spine, spine_indices: np.ndarray) -> Dict[int, np.ndarray]:
            
        for vt, vertebra in enumerate(spine.vertebrae):
            
            left_transverse_polydata = vertebra.shapeDecomposition.process_polydata["TL"]
            left_transverse_points = numpy_support.vtk_to_numpy(left_transverse_polydata.GetPoints().GetData())
            left_ITL = sorted(left_transverse_points, key=(lambda p: np.array(p).dot(vertebra.orientation.r)))[0]

            right_transverse_polydata = vertebra.shapeDecomposition.process_polydata["TR"]
            right_transverse_points = numpy_support.vtk_to_numpy(right_transverse_polydata.GetPoints().GetData())
            right_ITL = sorted(right_transverse_points, key=(lambda p: np.array(p).dot(vertebra.orientation.r)))[-1]
            
            ITL_landmarks = [left_ITL, right_ITL]
            markupNode = SpineLib.SlicerTools.createMarkupsFiducialNode(ITL_landmarks, "ITL", color=[0.5, 0, 0.75])

    '''
    Detect supraspinous ligament landmarks
    '''
    def _detect_SSL(spine: SpineLib.Spine, spine_indices: np.ndarray) -> Dict[int, np.ndarray]:
        
        for vt, vertebra in enumerate(spine.vertebrae):
            SSL_landmark = vertebra.shapeDecomposition.process_landmarks["S"]
            markupNode = SpineLib.SlicerTools.createMarkupsFiducialNode([SSL_landmark], "SSL", color=[0.5, 0, 0.25])
        
    '''
    Detect anterior longitudinal ligament (ALL) and posterior longitudinal ligament (PLL) landmarks
    '''
    def _detect_all_pll(spine: SpineLib.Spine, spine_indices: np.ndarray) -> Dict[int, np.ndarray]:
        
        for vt, vertebra in enumerate(spine.vertebrae):

            factor_cervical = 0.2
            factor_thor_lumbar = 0.1

            if (spine_indices[vt] >= 17):
                factor = factor_cervical
            else:
                factor = factor_thor_lumbar

            left_medial = vertebra.shapeDecomposition.landmarks["left_pedicle_medial"]
            right_medial = vertebra.shapeDecomposition.landmarks["right_pedicle_medial"]
            canal_width = np.linalg.norm(left_medial - right_medial)
            canal_vector = right_medial - left_medial
            left_origin = left_medial + factor * canal_vector
            right_origin = right_medial - factor * canal_vector


            # superior landmarks
            superior_boundary = conv.extractBoundary(vertebra.body.superior_endplate)
            superior_clipped = conv.clip_plane(superior_boundary, left_origin, vertebra.orientation.r)
            superior_clipped = conv.clip_plane(superior_clipped, right_origin, -vertebra.orientation.r)
            anterior_clipped = conv.clip_plane(superior_clipped, vertebra.center, vertebra.orientation.a)
            posterior_clipped = conv.clip_plane(superior_clipped, vertebra.center, -vertebra.orientation.a)

            sorted_anterior = conv.sorted_points(list(numpy_support.vtk_to_numpy(anterior_clipped.GetPoints().GetData())), vertebra.orientation.r)
            sorted_posterior = conv.sorted_points(list(numpy_support.vtk_to_numpy(posterior_clipped.GetPoints().GetData())), vertebra.orientation.r)

            sup_all_curve = SpineLib.SlicerTools.createResampledCurve(sorted_anterior, 7, name="ALL", color=[1, 0, 0])
            sup_pll_curve = SpineLib.SlicerTools.createResampledCurve(sorted_posterior, 7, name="PLL", color=[1, 0, 0])


            # inferior landmarks
            inferior_boundary = conv.extractBoundary(vertebra.body.inferior_endplate)
            inferior_clipped = conv.clip_plane(inferior_boundary, left_origin, vertebra.orientation.r)
            inferior_clipped = conv.clip_plane(inferior_clipped, right_origin, -vertebra.orientation.r)
            anterior_clipped = conv.clip_plane(inferior_clipped, vertebra.center, vertebra.orientation.a)
            posterior_clipped = conv.clip_plane(inferior_clipped, vertebra.center, -vertebra.orientation.a)

            sorted_anterior = conv.sorted_points(list(numpy_support.vtk_to_numpy(anterior_clipped.GetPoints().GetData())), vertebra.orientation.r)
            sorted_posterior = conv.sorted_points(list(numpy_support.vtk_to_numpy(posterior_clipped.GetPoints().GetData())), vertebra.orientation.r)

            inf_all_curve = SpineLib.SlicerTools.createResampledCurve(sorted_anterior, 7, name="ALL", color=[1, 0, 0])
            inf_pll_curve = SpineLib.SlicerTools.createResampledCurve(sorted_posterior, 7, name="PLL", color=[1, 0, 0])

            




    # '''
    # Get contact points between two polydata
    # '''
    # def get_contact_points(point_list_1, point_list_2):

    #     contact_points = []
    #     for point in point_list_1:
    #         distances = np.linalg.norm(point_list_2 - point, axis=1)
    #         contact_points.append(point_list_2[np.argmin(distances)])
  
    #     return contact_points
    
    
    '''
    Project points onto plane
    '''
    def project_points(points, plane_center, plane_normal):
            
            plane_normal = plane_normal / np.linalg.norm(plane_normal)
            projected_points = []
            for point in points:
                projected_points.append(point - np.dot(point - plane_center, plane_normal) * plane_normal)
            
            return np.array(projected_points)


    '''
    Detect facet joint landmarks
    '''
    def _detect_CL(spine: SpineLib.Spine, spine_indices: np.ndarray) -> Dict[int, np.ndarray]:

        for inferior_vertebra, superior_vertebra in zip(spine.vertebrae, spine.vertebrae[1:]):

            inf_AS_polydatas  = [inferior_vertebra.shapeDecomposition.process_polydata["ASL"], inferior_vertebra.shapeDecomposition.process_polydata["ASR"]]
            sup_AI_polydatas  = [superior_vertebra.shapeDecomposition.process_polydata["AIL"], superior_vertebra.shapeDecomposition.process_polydata["AIR"]]
            medial_directions = [inferior_vertebra.orientation.r, -inferior_vertebra.orientation.r]

            for inf_AS_polydata, sup_AI_polydata, medial_direction in zip(inf_AS_polydatas, sup_AI_polydatas, medial_directions):

                # get facet contact polydata
                inf_contact_polydata = conv.get_contact_polydata(sup_AI_polydata, inf_AS_polydata)
                sup_contact_polydata = conv.get_contact_polydata(inf_AS_polydata, sup_AI_polydata)

                combined_surface = conv.polydata_append(inf_contact_polydata, sup_contact_polydata)
                facet_polydata = conv.polydata_convexHull(combined_surface)
                #SpineLib.SlicerTools.createModelNode(facet_polydata, "Facet", color=[1.0, 0.0, 0.0], opacity=0.7)



                # # get facet contact surface
                # inf_AS_points = numpy_support.vtk_to_numpy(inf_AS_polydata.GetPoints().GetData())
                # sup_AI_points = numpy_support.vtk_to_numpy(sup_AI_polydata.GetPoints().GetData())
                # contact_points_inf = LigamentLandmarks.get_contact_points(sup_AI_points, inf_AS_points)
                # contact_points_sup = LigamentLandmarks.get_contact_points(inf_AS_points, sup_AI_points)
                # SpineLib.SlicerTools.createMarkupsFiducialNode(contact_points_inf, "ContactPoints", color=[1, 0, 0], glyphScale=1.0)
                # SpineLib.SlicerTools.createMarkupsFiducialNode(contact_points_sup, "ContactPoints", color=[0, 1, 0], glyphScale=1.0)

                # Compute best fit plane
                #planeCenter, planeNormal = conv.fitPlane(np.concatenate((contact_points_inf, contact_points_sup)))
                planeCenter, planeNormal = conv.fitPlane(numpy_support.vtk_to_numpy(inf_contact_polydata.GetPoints().GetData()))
                normal = np.array(planeNormal)
                # check if normal or inverse of normal is more similar with medial_direction
                if np.dot(normal, np.average([medial_direction, -inferior_vertebra.orientation.a], axis=0)) < 0:
                    normal = -normal
                #planeNode = SpineLib.SlicerTools.createMarkupsPlaneNode(planeCenter, normal, "FittedPlane", 20, 20, opacity=0.7)

                # local plane directions
                u = inferior_vertebra.orientation.s
                n = np.array(normal)
                n_norm = np.sqrt(sum(n**2))
                proj_of_u_on_n = (np.dot(u, n)/n_norm**2)*n
                facet_up = u - proj_of_u_on_n
                facet_medial = normal
                facet_LR = np.cross(facet_up, facet_medial)

                # # projected points
                # lr_sorted = conv.sorted_points(list(inf_contact_points_projected_up), facet_LR)
                # lr_len = np.linalg.norm(lr_sorted[0] - lr_sorted[-1])
                # is_sorted = conv.sorted_points(list(inf_contact_points_projected_up), facet_up)
                # is_len = np.linalg.norm(is_sorted[0] - is_sorted[-1])

                # inf_contact_points_projected_up = LigamentLandmarks.project_points(contact_points_inf, planeCenter, facet_up)
                # SpineLib.SlicerTools.createMarkupsFiducialNode(inf_contact_points_projected_up, "ProjectedPoints", color=[1, 0, 0], glyphScale=1.0)
                # lr_sorted = conv.sorted_points(list(inf_contact_points_projected_up), facet_LR)
                # lr_1 = lr_sorted[0]
                # lr_2 = lr_sorted[-1]
                # inf_contact_points_projected_up = LigamentLandmarks.project_points(contact_points_inf, planeCenter, facet_LR)
                # SpineLib.SlicerTools.createMarkupsFiducialNode(inf_contact_points_projected_up, "ProjectedPoints", color=[1, 0, 0], glyphScale=1.0)
                # is_sorted = conv.sorted_points(list(inf_contact_points_projected_up), facet_up)
                # is_1 = is_sorted[0]
                # is_2 = is_sorted[-1]

                # inf_ligament_landmarks = [lr_1, lr_2, is_1, is_2]
                # inf_ligament_markup = SpineLib.SlicerTools.createMarkupsFiducialNode(inf_ligament_landmarks, "InferiorVertebraLigamentLandmarks", color=[0, 0, 1])

                ###############

                # # filter polydata to get inferior facet surface
                # inf_contact_polydata = conv.eliminate_misaligned_faces(inf_AS_polydata, planeCenter, normal, 30.0)
                # inf_contact_polydata = conv.filterLargestRegion(inf_contact_polydata)
                inf_facet_center = np.mean(numpy_support.vtk_to_numpy(inf_contact_polydata.GetPoints().GetData()), axis=0)
                # SpineLib.SlicerTools.createModelNode(inf_facet_polydata, "Facet", color=[1.0, 0.0, 0.0], opacity=0.7)

                lr_intersection = conv.cut_plane(inf_contact_polydata, inf_facet_center, facet_up)
                lr_sorted = conv.sorted_points(lr_intersection, facet_LR)
                lr_1 = lr_sorted[0]
                lr_2 = lr_sorted[-1]
                is_intersection = conv.cut_plane(inf_contact_polydata, inf_facet_center, facet_LR)
                is_sorted = conv.sorted_points(is_intersection, facet_up)
                is_1 = is_sorted[0]
                is_2 = is_sorted[-1]

                inf_ligament_landmarks = [lr_1, lr_2, is_1, is_2]
                inf_ligament_markup = SpineLib.SlicerTools.createMarkupsFiducialNode(inf_ligament_landmarks, "InferiorVertebraLigamentLandmarks", color=[0, 0, 1])

                # filter polydata to get superior facet surface
                # sup_contact_polydata = conv.eliminate_misaligned_faces(sup_AI_polydata, planeCenter, -normal, 30.0)
                # sup_contact_polydata = conv.filterLargestRegion(sup_contact_polydata)
                sup_facet_center = np.mean(numpy_support.vtk_to_numpy(sup_contact_polydata.GetPoints().GetData()), axis=0)
                #SpineLib.SlicerTools.createModelNode(sup_facet_polydata, "Facet", color=[1.0, 0.0, 0.0], opacity=0.7)

                lr_intersection = conv.cut_plane(sup_contact_polydata, sup_facet_center, facet_up)
                lr_sorted = conv.sorted_points(lr_intersection, facet_LR)
                lr_1 = lr_sorted[0]
                lr_2 = lr_sorted[-1]
                is_intersection = conv.cut_plane(sup_contact_polydata, sup_facet_center, facet_LR)
                is_sorted = conv.sorted_points(is_intersection, facet_up)
                is_1 = is_sorted[0]
                is_2 = is_sorted[-1]

                sup_ligament_landmarks = [lr_1, lr_2, is_1, is_2]
                sup_ligament_markup = SpineLib.SlicerTools.createMarkupsFiducialNode(sup_ligament_landmarks, "SuperiorVertebraLigamentLandmarks", color=[0, 1, 1])



                # # get closest points on superior facet to the inferior facet landmarks
                # sup_ligament_landmarks = [sorted(sup_AI_points, key=lambda p: np.linalg.norm(p - lm))[0] for lm in inf_ligament_landmarks]
                # sup_ligament_markup = SpineLib.SlicerTools.createMarkupsFiducialNode(sup_ligament_landmarks, "SuperiorVertebraLigamentLandmarks", color=[0, 1, 1])



        # for vt, vertebra in enumerate(spine.vertebrae):

        #     landmarks = vertebra.shapeDecomposition.landmarks
        #     facet_landmarks = {}
        #     for side in ["left", "right"]:
        #         facet_landmarks[side] = {}
        #         for direction in ["medial", "lateral"]:
        #             facet_landmarks[side][direction] = conv.cut_plane(vertebra.facet_joints[side][direction], landmarks[f"{side}_pedicle_{direction}"], vertebra.orientation.r)
        #             SpineLib.SlicerTools.createModelNode(facet_landmarks[side][direction], f"{side.capitalize()}Facet{direction.capitalize()}", color=[0.0, 1.0, 0.0], opacity=0.7)

        # return facet_landmarks

    
    def _detect_ISL(spine: SpineLib.Spine, spine_indices: np.ndarray):
        
        for vt, vertebra in enumerate(spine.vertebrae):
            
            # filter data points
            spinous_polydata = vertebra.shapeDecomposition.process_polydata["S"]

            # remesh polydata
            spinous_polydata = conv.polydata_remesh(spinous_polydata, 2, 5000)

            spinous_centerline = vertebra.shapeDecomposition.centerlines["S"]
            spinous_centerline.GetDisplayNode().SetVisibility(1)

            controlPoints = np.array(spinous_centerline.GetCurvePointsWorld().GetData())
            # fit a line to the control points
            main_comp = conv.calc_main_component(list(controlPoints))
            main_comp = conv.closest_vector([main_comp], np.average([vertebra.orientation.s, vertebra.orientation.a], axis=0))
            main_comp = conv.normalize(main_comp[0]*main_comp[1])
            local_r = np.cross(main_comp, vertebra.orientation.s)
            local_s = np.cross(local_r, main_comp)

            superior_spinous_polydata = conv.eliminate_misaligned_faces(spinous_polydata, controlPoints.mean(axis=0), local_s, 35.0)
            inferior_spinous_polydata = conv.eliminate_misaligned_faces(spinous_polydata, controlPoints.mean(axis=0), -local_s, 35.0)

            #SpineLib.SlicerTools.markupsLineNode("LocalR", controlPoints.mean(axis=0), controlPoints.mean(axis=0) + 25*local_s)

            superior_intersection = conv.cut_plane(superior_spinous_polydata, controlPoints.mean(axis=0), local_r)
            superior_intersection_points = numpy_support.vtk_to_numpy(superior_intersection.GetPoints().GetData())
            inferior_intersection = conv.cut_plane(inferior_spinous_polydata, controlPoints.mean(axis=0), -local_r)
            inferior_intersection_points = numpy_support.vtk_to_numpy(inferior_intersection.GetPoints().GetData())
            #SpineLib.SlicerTools.createModelNode(intersection_points, "IntersectionPoints", color=[1.0, 0.0, 0.0], opacity=1.0)

            # fit curve to intersection points
            sorted_superior_points = conv.sorted_points(list(superior_intersection_points), main_comp)
            sorted_inferior_points = conv.sorted_points(list(inferior_intersection_points), main_comp)
            superior_curveNode = SpineLib.SlicerTools.createResampledCurve(sorted_superior_points, 12, name="Superior_SpinousCurve", color=[1, 0, 0])
            inferior_curveNode = SpineLib.SlicerTools.createResampledCurve(sorted_inferior_points, 12, name="Inferior_SpinousCurve", color=[1, 0, 0])
            superior_ISL_samples = slicer.util.arrayFromMarkupsControlPoints(superior_curveNode)[:-4]
            inferior_ISL_samples = slicer.util.arrayFromMarkupsControlPoints(inferior_curveNode)[:-4]

            SpineLib.SlicerTools.removeNodes([superior_curveNode, inferior_curveNode])

            superior_ISL = conv.find_closest_points(spinous_polydata, superior_ISL_samples)
            inferior_ISL = conv.find_closest_points(spinous_polydata, inferior_ISL_samples) 

            SpineLib.SlicerTools.createMarkupsFiducialNode(superior_ISL, "Superior_ISL", color=[0.0, 0.0, 1.0])
            SpineLib.SlicerTools.createMarkupsFiducialNode(inferior_ISL, "Inferior_ISL", color=[0.0, 0.0, 1.0])






            # upper_spinous = conv.eliminate_misaligned_faces(spinous_polydata, vertebra.center, vertebra.orientation.s, 60.0)
            # lower_spinous = conv.eliminate_misaligned_faces(spinous_polydata, vertebra.center, -vertebra.orientation.s, 60.0)
            # SpineLib.SlicerTools.createModelNode(upper_spinous, "UpperSpinous", color=[1.0, 0.0, 0.0], opacity=0.8)
            # SpineLib.SlicerTools.createModelNode(lower_spinous, "LowerSpinous", color=[1.0, 0.0, 0.0], opacity=0.8)

            # # create curve for upper_spinous_points
            # sorted_upper_spinous = conv.sorted_points(upper_spinous, conv.calc_main_component(upper_spinous))
            # SpineLib.SlicerTools.createMarkupsFiducialNode(sorted_upper_spinous, "UpperSpinousLandmarks", color=[0, 0, 1])
            # SpineLib.SlicerTools.createResampledCurve(sorted_upper_spinous, 8, name="UpperSpinousCurve", color=[1, 0, 0])

            # # create curve for lower_spinous_points
            # sorted_lower_spinous = conv.sorted_points(lower_spinous, conv.calc_main_component(lower_spinous))
            # SpineLib.SlicerTools.createMarkupsFiducialNode(sorted_lower_spinous, "LowerSpinousLandmarks", color=[0, 0, 1])
            # SpineLib.SlicerTools.createResampledCurve(sorted_lower_spinous, 8, name="LowerSpinousCurve", color=[1, 0, 0])


            # # Compute best fit plane
            # points = np.concatenate((numpy_support.vtk_to_numpy(upper_spinous.GetPoints().GetData()), numpy_support.vtk_to_numpy(lower_spinous.GetPoints().GetData())))
            # planeCenter, planeNormal = conv.fitPlane(points)

            # #Display best fit plane as a markups plane
            # planeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsPlaneNode')
            # planeNode.SetCenter(planeCenter)
            # planeNode.SetNormal(planeNormal)
            # planeNode.SetSize(50, 50)
            # planeNode.GetDisplayNode().SetOpacity(0.5)
            # planeNode.GetDisplayNode().SetHandlesInteractive(False)

            # # find closest points on polydata to the curve




    '''
    Detect ligamentum flavum landmarks
    '''
    def _detect_LF(spine: SpineLib.Spine, spine_indices: np.ndarray) -> Dict[int, np.ndarray]:

        for vt, vertebra in enumerate(spine.vertebrae):
            
            factor_cervical = 0.4
            factor_thor_lumbar = 0.2

            if (spine_indices[vt] >= 17):
                factor = factor_cervical
            else:
                factor = factor_thor_lumbar

            left_medial = vertebra.shapeDecomposition.landmarks["left_pedicle_medial"]
            right_medial = vertebra.shapeDecomposition.landmarks["right_pedicle_medial"]
            canal_width = np.linalg.norm(left_medial - right_medial)
            canal_vector = right_medial - left_medial
            left_origin = left_medial + factor * canal_vector
            right_origin = right_medial - factor * canal_vector
            polydata = vertebra.shapeDecomposition.processes

            # run dijkstra
            flavum_curvePoints = conv.runDijkstra(polydata, np.array(left_medial), np.array(right_medial))


            # clip to left and right
            parametricSpline = vtk.vtkParametricSpline()
            parametricSpline.SetPoints(flavum_curvePoints)
            parametricFunctionSource = vtk.vtkParametricFunctionSource()
            parametricFunctionSource.SetParametricFunction(parametricSpline)
            parametricFunctionSource.Update()

            # left half
            left_clipped = conv.clip_plane(parametricFunctionSource.GetOutput(), vertebra.center, -vertebra.orientation.r)
            right_clipped = conv.clip_plane(parametricFunctionSource.GetOutput(), vertebra.center, vertebra.orientation.r)

            left_flavum_Points = numpy_support.vtk_to_numpy(left_clipped.GetPoints().GetData())
            right_flavum_Points = numpy_support.vtk_to_numpy(right_clipped.GetPoints().GetData())

            left_flavum_curve = SpineLib.SlicerTools.createResampledCurve(left_flavum_Points, 7, name="FlavumCurve", color=[1, 0, 1])
            right_flavum_curve = SpineLib.SlicerTools.createResampledCurve(right_flavum_Points, 7, name="FlavumCurve", color=[1, 0, 1])

            left_flavum_Points = slicer.util.arrayFromMarkupsControlPoints(left_flavum_curve)[1:-1]
            right_flavum_Points = slicer.util.arrayFromMarkupsControlPoints(right_flavum_curve)[1:-1]

            
            slicer.util.updateMarkupsControlPointsFromArray(left_flavum_curve, np.array(left_flavum_Points))
            slicer.util.updateMarkupsControlPointsFromArray(right_flavum_curve, np.array(right_flavum_Points))



            # SpineLib.SlicerTools.createMarkupsFiducialNode(left_flavum_Points, "FlavumLandmarksL", color=[1, 0, 0])
            # SpineLib.SlicerTools.createMarkupsFiducialNode(right_flavum_Points, "FlavumLandmarksR", color=[1, 0, 0])







            # # CLIPPING

            # # create curve
            # parametricSpline = vtk.vtkParametricSpline()
            # parametricSpline.SetPoints(flavum_curvePoints)
            # parametricFunctionSource = vtk.vtkParametricFunctionSource()
            # parametricFunctionSource.SetParametricFunction(parametricSpline)
            # parametricFunctionSource.Update()

            # clipped = conv.clip_plane(parametricFunctionSource.GetOutput(), left_origin, vertebra.orientation.r)
            # clipped = conv.clip_plane(clipped, right_origin, -vertebra.orientation.r)
            # #clipped = conv.clip_plane(clipped, right_keypoint, vertebra.orientation.r)

            # flavum_curvePoints = numpy_support.vtk_to_numpy(clipped.GetPoints().GetData())
            # curveNode = SpineLib.SlicerTools.createResampledCurve(flavum_curvePoints, 7, name="FlavumCurve", color=[1, 0, 1])





            # # create curve node
            # curveNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsCurveNode', "FlavumCurve")
            # slicer.util.updateMarkupsControlPointsFromArray(curveNode, numpy_support.vtk_to_numpy(clipped.GetPoints().GetData()))


            # # find intersection points
            # plane_origins = np.linspace(vertebra.shapeDecomposition.landmarks["left_pedicle_medial"], vertebra.shapeDecomposition.landmarks["right_pedicle_medial"], 13, axis=0)
            # flavum_landmarks = [conv.cut_plane(parametricFunctionSource.GetOutput(), plane_origin, vertebra.orientation.r).GetPoint(0) for plane_origin in plane_origins[3:-3]]

            # # create markups node
            # flavum_markupsNode = SpineLib.SlicerTools.createMarkupsFiducialNode(flavum_landmarks, "FlavumLandmarks", color=[1, 0, 0])

            # SpineLib.SlicerTools.createResampledCurve(flavum_landmarks, 7, name="FlavumCurve", color=[1, 0, 1])

        pass