from dataclasses import dataclass
import numpy as np
import os
import vtk
import vtk_convenience as conv
from vtk.util import numpy_support
from typing import Dict, Tuple
import slicer
import SpineLib
import traceback


class LigamentLandmarks:   

    def __init__(self,
                 spine:            SpineLib.Spine           = None,
                 ) -> None:
        
        try:
            self._detect_ALL_PLL(spine)
            self._detect_CL(spine)
            self._detect_ISL(spine)
            self._detect_LF(spine)
            self._detect_ITL(spine) 
            self._detect_SSL(spine)

        except Exception as e:
            print(e)
            print("Error in LigamentLandmarks")
            traceback.print_exc()
            return None
        

    '''
    Detect intertransverse ligament landmarks
    '''
    def _detect_ITL(self, spine: SpineLib.Spine):
            
        for vt, vertebra in enumerate(spine.vertebrae):
            
            left_transverse_polydata = vertebra.shapeDecomposition.process_polydata["TL"]
            left_transverse_points = numpy_support.vtk_to_numpy(left_transverse_polydata.GetPoints().GetData())
            left_ITL = sorted(left_transverse_points, key=(lambda p: np.array(p).dot(vertebra.orientation.r)))[0]

            right_transverse_polydata = vertebra.shapeDecomposition.process_polydata["TR"]
            right_transverse_points = numpy_support.vtk_to_numpy(right_transverse_polydata.GetPoints().GetData())
            right_ITL = sorted(right_transverse_points, key=(lambda p: np.array(p).dot(vertebra.orientation.r)))[-1]
            
            ITL_landmarks = [left_ITL, right_ITL]
            markupNode = SpineLib.SlicerTools.createMarkupsFiducialNode(ITL_landmarks, "ITL", color=[0.5, 0, 0.75])

            vertebra.ligament_landmarks["ITL"] = ITL_landmarks


    '''
    Detect supraspinous ligament landmarks
    '''
    def _detect_SSL(self, spine: SpineLib.Spine):
        
        for vt, vertebra in enumerate(spine.vertebrae):
            SSL_landmark = vertebra.shapeDecomposition.process_landmarks["S"]
            markupNode = SpineLib.SlicerTools.createMarkupsFiducialNode([SSL_landmark], "SSL", color=[0.5, 0, 0.25])

            vertebra.ligament_landmarks["SSL"] = [SSL_landmark]

        
    '''
    Detect anterior longitudinal ligament (ALL) and posterior longitudinal ligament (PLL) landmarks
    '''
    def _detect_ALL_PLL(self, spine: SpineLib.Spine):
        
        for vt, vertebra in enumerate(spine.vertebrae):

            factor_cervical = 0.2
            factor_thor_lumbar = 0.1

            if (vertebra.index >= 17):
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

            sup_all = slicer.util.arrayFromMarkupsControlPoints(sup_all_curve)
            sup_pll = slicer.util.arrayFromMarkupsControlPoints(sup_pll_curve)


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

            inf_all = slicer.util.arrayFromMarkupsControlPoints(inf_all_curve)
            inf_pll = slicer.util.arrayFromMarkupsControlPoints(inf_pll_curve)

            SpineLib.SlicerTools.removeNodes([sup_all_curve, sup_pll_curve, inf_all_curve, inf_pll_curve])

            ALL_landmarks = np.concatenate((sup_all, inf_all))
            PLL_landmarks = np.concatenate((sup_pll, inf_pll))

            SpineLib.SlicerTools.createMarkupsFiducialNode(ALL_landmarks, "ALL", color=[0.5, 0, 0.75])
            SpineLib.SlicerTools.createMarkupsFiducialNode(PLL_landmarks, "PLL", color=[0.5, 0, 0.75])

            vertebra.ligament_landmarks["ALL"] = ALL_landmarks
            vertebra.ligament_landmarks["PLL"] = PLL_landmarks


            




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
    def project_points(self, points, plane_center, plane_normal):
            
            plane_normal = plane_normal / np.linalg.norm(plane_normal)
            projected_points = []
            for point in points:
                projected_points.append(point - np.dot(point - plane_center, plane_normal) * plane_normal)
            
            return np.array(projected_points)


    '''
    Detect facet joint landmarks
    '''
    def _detect_CL(self, spine: SpineLib.Spine):

        for inferior_vertebra, superior_vertebra in zip(spine.vertebrae, spine.vertebrae[1:]):

            inf_AS_polydatas  = [inferior_vertebra.shapeDecomposition.process_polydata["ASL"], inferior_vertebra.shapeDecomposition.process_polydata["ASR"]]
            sup_AI_polydatas  = [superior_vertebra.shapeDecomposition.process_polydata["AIL"], superior_vertebra.shapeDecomposition.process_polydata["AIR"]]
            medial_directions = [inferior_vertebra.orientation.r, -inferior_vertebra.orientation.r]

            for inf_AS_polydata, sup_AI_polydata, medial_direction, side in zip(inf_AS_polydatas, sup_AI_polydatas, medial_directions, ["left", "right"]):

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
                # facet_medial = conv.closest_vector([facet_LR], medial_direction)
                # print("............")
                # print("Facet LR: ", facet_LR)
                # print("Facet medial: ", facet_medial)
                # if (facet_medial[1] != 0):
                #     facet_medial = conv.normalize(facet_medial[0]*facet_medial[1])
                # else:
                #     facet_medial = conv.normalize(facet_medial[0])
                #     print("its zero")

                #facet_medial = facet_LR if np.dot(facet_LR, medial_direction) >= np.dot(-facet_LR, medial_direction) else -facet_LR
                facet_medial = facet_LR if np.dot(facet_LR, medial_direction) >= 0 else -facet_LR

                print("Facet medial: ", facet_medial)

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
                lr_sorted = conv.sorted_points(lr_intersection, facet_medial)
                lr_1 = lr_sorted[0]
                lr_2 = lr_sorted[-1]
                is_intersection = conv.cut_plane(inf_contact_polydata, inf_facet_center, facet_medial)
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
                lr_sorted = conv.sorted_points(lr_intersection, facet_medial)
                lr_1 = lr_sorted[0]
                lr_2 = lr_sorted[-1]
                is_intersection = conv.cut_plane(sup_contact_polydata, sup_facet_center, facet_medial)
                is_sorted = conv.sorted_points(is_intersection, facet_up)
                is_1 = is_sorted[0]
                is_2 = is_sorted[-1]

                sup_ligament_landmarks = [lr_1, lr_2, is_1, is_2]
                sup_ligament_markup = SpineLib.SlicerTools.createMarkupsFiducialNode(sup_ligament_landmarks, "SuperiorVertebraLigamentLandmarks", color=[0, 1, 1])

                inferior_vertebra.ligament_landmarks["CL_" + side + "_sup"] = inf_ligament_landmarks
                superior_vertebra.ligament_landmarks["CL_" + side + "_inf"] = sup_ligament_landmarks

            
  

        # for vt, vertebra in enumerate(spine.vertebrae):

        #     landmarks = vertebra.shapeDecomposition.landmarks
        #     facet_landmarks = {}
        #     for side in ["left", "right"]:
        #         facet_landmarks[side] = {}
        #         for direction in ["medial", "lateral"]:
        #             facet_landmarks[side][direction] = conv.cut_plane(vertebra.facet_joints[side][direction], landmarks[f"{side}_pedicle_{direction}"], vertebra.orientation.r)
        #             SpineLib.SlicerTools.createModelNode(facet_landmarks[side][direction], f"{side.capitalize()}Facet{direction.capitalize()}", color=[0.0, 1.0, 0.0], opacity=0.7)

        # return facet_landmarks

    
    def _detect_ISL(self, spine: SpineLib.Spine):
        
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

            ISL_landmarks = np.concatenate((superior_ISL, inferior_ISL))

            SpineLib.SlicerTools.createMarkupsFiducialNode(superior_ISL, "Superior_ISL", color=[0.0, 0.0, 1.0])
            SpineLib.SlicerTools.createMarkupsFiducialNode(inferior_ISL, "Inferior_ISL", color=[0.0, 0.0, 1.0])

            vertebra.ligament_landmarks["ISL"] = ISL_landmarks







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


    def flavum_curve(self, vertebra, left_keypoint, right_keypoint):

        polydata = vertebra.shapeDecomposition.processes

        # run dijkstra
        flavum_curvePoints = conv.runDijkstra(polydata, np.array(left_keypoint), np.array(right_keypoint))

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
        flavum_points = np.concatenate((left_flavum_Points, right_flavum_Points))

        SpineLib.SlicerTools.removeNodes([left_flavum_curve, right_flavum_curve])

        return flavum_points

    '''
    Detect ligamentum flavum landmarks
    '''
    def _detect_LF(self, spine: SpineLib.Spine):

        for vt, vertebra in enumerate(spine.vertebrae):
            
            factor_cervical = 0.4
            factor_thor_lumbar = 0.2

            if (vertebra.index >= 17):
                factor = factor_cervical
            else:
                factor = factor_thor_lumbar

            left_medial = vertebra.shapeDecomposition.landmarks["left_pedicle_medial"]
            right_medial = vertebra.shapeDecomposition.landmarks["right_pedicle_medial"]
            canal_width = np.linalg.norm(left_medial - right_medial)
            canal_vector = right_medial - left_medial
            left_origin = left_medial + factor * canal_vector
            right_origin = right_medial - factor * canal_vector

            # CL_landmarks :-> [ CL_inf_l, CL_sup_l, CL_inf_r, CL_sup_r ]

            #print("Index: ", vertebra.index)

            # upper flavum curve
            if "CL_left_sup" in vertebra.ligament_landmarks and "CL_right_sup" in vertebra.ligament_landmarks:
                left_keypoint = vertebra.ligament_landmarks["CL_left_sup"][1]
                right_keypoint = vertebra.ligament_landmarks["CL_right_sup"][1]
                upper_flavum_points = self.flavum_curve(vertebra, left_keypoint, right_keypoint)
                SpineLib.SlicerTools.createMarkupsFiducialNode(upper_flavum_points, "Ligamentum Flavum", color=[0.5, 0, 0.25])
                vertebra.ligament_landmarks["LF_sup"] = upper_flavum_points   
            

            # lower flavum curve
            if "CL_left_inf" in vertebra.ligament_landmarks and "CL_right_inf" in vertebra.ligament_landmarks:
                left_keypoint = vertebra.ligament_landmarks["CL_left_inf"][1]
                right_keypoint = vertebra.ligament_landmarks["CL_right_inf"][1]
                lower_flavum_points = self.flavum_curve(vertebra, left_keypoint, right_keypoint)
                SpineLib.SlicerTools.createMarkupsFiducialNode(lower_flavum_points, "Ligamentum Flavum", color=[0.5, 0, 0.25])
                vertebra.ligament_landmarks["LF_inf"] = lower_flavum_points



            
            # slicer.util.updateMarkupsControlPointsFromArray(left_flavum_curve, np.array(left_flavum_Points))
            # slicer.util.updateMarkupsControlPointsFromArray(right_flavum_curve, np.array(right_flavum_Points))



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
