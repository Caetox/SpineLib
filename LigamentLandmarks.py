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
                 spine:               SpineLib.Spine              = None,
                 progressBarManager:  SpineLib.ProgressBarManager = None

                 ) -> None:
        
        self.progressBarManager = progressBarManager
        
        try:
            self._detect_ALL_PLL(spine, self.progressBarManager)
            self._detect_CL     (spine, self.progressBarManager)
            self._detect_ISL    (spine, self.progressBarManager)
            self._detect_LF     (spine, self.progressBarManager)
            self._detect_ITL    (spine, self.progressBarManager) 
            self._detect_SSL    (spine, self.progressBarManager)

        except Exception as e:
            print(e)
            print("Error in LigamentLandmarks")
            traceback.print_exc()
            return None
        
        #self._add_connections(spine)

        # Test


        
    def _add_connections(self, spine:SpineLib.Spine):

        for inferior_vertebra, superior_vertebra in zip(spine.vertebrae, spine.vertebrae[1:]):

            # def connections
            connections = []
            try:
                for p1, p2 in zip(inferior_vertebra.ligament_landmarks["superiorALL"], superior_vertebra.ligament_landmarks["inferiorALL"]):
                    connections.append([p1,p2])
            except Exception as e:
                print(e)
            try:
                for p1, p2 in zip(inferior_vertebra.ligament_landmarks["superiorPLL"], superior_vertebra.ligament_landmarks["inferiorPLL"]):
                    connections.append([p1, p2])
            except Exception as e:
                print(e)

            for c in connections:
                SpineLib.SlicerTools.markupsLineNode("longitudinal ligaments", c[0], c[1])




    '''
    Detect intertransverse ligament landmarks
    '''
    def _detect_ITL(self, spine: SpineLib.Spine = None, progressBarManager: SpineLib.ProgressBarManager = None):
            
        for vt, vertebra in enumerate(spine.vertebrae):

            print("Detecting ITL landmarks for " + vertebra.name + " ...")

            # left_transverse_polydata = vertebra.shapeDecomposition.process_polydata["TL"]
            # left_transverse_points = numpy_support.vtk_to_numpy(left_transverse_polydata.GetPoints().GetData())
            # left_ITL = sorted(left_transverse_points, key=(lambda p: np.array(p).dot(vertebra.orientation.r)))[0]

            # right_transverse_polydata = vertebra.shapeDecomposition.process_polydata["TR"]
            # right_transverse_points = numpy_support.vtk_to_numpy(right_transverse_polydata.GetPoints().GetData())
            # right_ITL = sorted(right_transverse_points, key=(lambda p: np.array(p).dot(vertebra.orientation.r)))[-1]
            
            # ITL_landmarks = [left_ITL, right_ITL]
            # markupNode = SpineLib.SlicerTools.createMarkupsFiducialNode(ITL_landmarks, "ITL", color=[0.5, 0, 0.75])

            # vertebra.ligament_landmarks["ITL"] = ITL_landmarks




            # get left transverse data
            left_transverse_polydata = vertebra.shapeDecomposition.process_polydata["TL"]
            left_transverse_centerline = vertebra.shapeDecomposition.centerlines["TL"]

            # get main component of the centerline
            controlPoints = np.array(left_transverse_centerline.GetCurvePointsWorld().GetData())
            #controlPoints = controlPoints[:len(controlPoints)//2]
            main_comp = conv.calc_main_component(list(controlPoints))
            main_comp = conv.closest_vector([main_comp], np.average((-vertebra.orientation.r, -vertebra.orientation.a), axis=0))
            main_comp = conv.normalize(main_comp[0]*main_comp[1])

            # intersect polydata with line from first control point, in direction of main component
            left_ITL_landmark = conv.get_intersection_points(left_transverse_polydata, controlPoints[0], controlPoints[0]+main_comp*100)


            # get right data
            right_transverse_polydata = vertebra.shapeDecomposition.process_polydata["TR"]
            right_transverse_centerline = vertebra.shapeDecomposition.centerlines["TR"]

            # get main component of the centerline
            controlPoints = np.array(right_transverse_centerline.GetCurvePointsWorld().GetData())
            controlPoints = controlPoints[:len(controlPoints)//2]
            main_comp = conv.calc_main_component(list(controlPoints))
            main_comp = conv.closest_vector([main_comp], np.average((vertebra.orientation.r, -vertebra.orientation.a), axis=0))
            main_comp = conv.normalize(main_comp[0]*main_comp[1])

            # intersect polydata with line from first control point, in direction of main component
            right_ITL_landmark = conv.get_intersection_points(right_transverse_polydata, controlPoints[0], controlPoints[0]+main_comp*100)            

            ITL_landmarks = [left_ITL_landmark, right_ITL_landmark]
            markupNode = SpineLib.SlicerTools.createMarkupsFiducialNode(ITL_landmarks, "ITL", color=[0.5, 0, 0.25])

            vertebra.ligament_landmarks["ITL"] = ITL_landmarks


            slicer.app.processEvents()
            progressBarManager.updateProgress()


    '''
    Detect supraspinous ligament landmarks
    '''
    def _detect_SSL(self, spine: SpineLib.Spine = None, progressBarManager: SpineLib.ProgressBarManager = None):
        
        print("Detecting SSL landmarks ...")

        for vt, vertebra in enumerate(spine.vertebrae):

            # get spinous data
            spinous_polydata = vertebra.shapeDecomposition.process_polydata["S"]
            spinous_centerline = vertebra.shapeDecomposition.centerlines["S"]

            # get main component of the centerline
            controlPoints = np.array(spinous_centerline.GetCurvePointsWorld().GetData())
            controlPoints = controlPoints[:len(controlPoints)//2]
            main_comp = conv.calc_main_component(list(controlPoints))
            main_comp = conv.closest_vector([main_comp], -np.average([vertebra.orientation.s, vertebra.orientation.a], axis=0))
            main_comp = conv.normalize(main_comp[0]*main_comp[1])

            # intersect polydata with line from first control point, in direction of main component
            SSL_landmark = conv.get_intersection_points(spinous_polydata, controlPoints[0], controlPoints[0]+main_comp*100)
            vertebra.ligament_landmarks["SSL"] = [SSL_landmark]

            
            markupNode = SpineLib.SlicerTools.createMarkupsFiducialNode([SSL_landmark], "SSL", color=[0.5, 0, 0.25])

            slicer.app.processEvents()

        progressBarManager.updateProgress()



    '''
    Detect anterior longitudinal ligament (ALL) and posterior longitudinal ligament (PLL) landmarks
    '''
    def _detect_ALL_PLL(self, spine: SpineLib.Spine = None, progressBarManager: SpineLib.ProgressBarManager = None):
        
        for vt, vertebra in enumerate(spine.vertebrae):

            print("Detecting ALL and PLL landmarks for " + vertebra.name + " ...")

            # TODO: change factors for width, this is only average lumbar width
            avg_ALL_width = 37.1
            #avg_ALL_width = 25.0
            avg_body_width = 52.0
            avg_PLL_width = 24.0
            avg_canal_width = 30.0

            # ratio of ligament to body width
            ratio_ALL = avg_ALL_width / avg_body_width
            factor_ALL = ratio_ALL/2

            ratio_PLL = avg_PLL_width / avg_canal_width
            factor_PLL = (1-ratio_PLL)/2

            left_origin_ALL = vertebra.center - factor_ALL * vertebra.orientation.r * vertebra.size.width
            right_origin_ALL = vertebra.center + factor_ALL * vertebra.orientation.r * vertebra.size.width

            left_medial = vertebra.shapeDecomposition.landmarks["left_pedicle_medial"]
            right_medial = vertebra.shapeDecomposition.landmarks["right_pedicle_medial"]
            canal_width = np.linalg.norm(left_medial - right_medial)
            canal_vector = right_medial - left_medial

            left_origin_PLL = left_medial + factor_PLL * canal_vector
            right_origin_PLL = right_medial - factor_PLL * canal_vector


            # factor_cervical = 0.2
            # factor_thor_lumbar = 0.1

            # if (vertebra.index >= 17):
            #     factor = factor_cervical
            # else:
            #     factor = factor_thor_lumbar

                
            # left_medial = vertebra.shapeDecomposition.landmarks["left_pedicle_medial"]
            # right_medial = vertebra.shapeDecomposition.landmarks["right_pedicle_medial"]
            # canal_width = np.linalg.norm(left_medial - right_medial)
            # canal_vector = right_medial - left_medial
            # left_origin = left_medial + factor * canal_vector
            # right_origin = right_medial - factor * canal_vector

            for side in ["superior", "inferior"]:
                endplate = getattr(vertebra.body, side+"_endplate")
                boundary = conv.extractBoundary(endplate)

                anterior_clipped = conv.clip_plane(boundary, vertebra.center, vertebra.orientation.a)
                posterior_clipped = conv.clip_plane(boundary, vertebra.center, -vertebra.orientation.a)

                clipped_all = conv.clip_plane(anterior_clipped, left_origin_ALL, vertebra.orientation.r)
                clipped_all = conv.clip_plane(clipped_all, right_origin_ALL, -vertebra.orientation.r)

                clipped_pll = conv.clip_plane(posterior_clipped, left_origin_PLL, vertebra.orientation.r)
                clipped_pll = conv.clip_plane(clipped_pll, right_origin_PLL, -vertebra.orientation.r)

                sorted_anterior_all = conv.sorted_points(list(numpy_support.vtk_to_numpy(clipped_all.GetPoints().GetData())), vertebra.orientation.r)
                sorted_posterior_pll = conv.sorted_points(list(numpy_support.vtk_to_numpy(clipped_pll.GetPoints().GetData())), vertebra.orientation.r)

                all_curve = SpineLib.SlicerTools.createResampledCurve(sorted_anterior_all, 7, name="ALL", color=[1, 0, 0])
                pll_curve = SpineLib.SlicerTools.createResampledCurve(sorted_posterior_pll, 7, name="PLL", color=[1, 0, 0])

                all = slicer.util.arrayFromMarkupsControlPoints(all_curve)
                pll = slicer.util.arrayFromMarkupsControlPoints(pll_curve)

                SpineLib.SlicerTools.removeNodes([all_curve, pll_curve])

                vertebra.ligament_landmarks[side+"ALL"] = all
                vertebra.ligament_landmarks[side+"PLL"] = pll

            for name in ["superiorALL", "superiorPLL", "inferiorALL", "inferiorPLL"]:
                SpineLib.SlicerTools.createMarkupsFiducialNode(vertebra.ligament_landmarks[name], name, color=[0.5, 0, 0.75])
            
            progressBarManager.updateProgress()
        



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
    def _detect_CL(self, spine: SpineLib.Spine = None, progressBarManager: SpineLib.ProgressBarManager = None):

        for inferior_vertebra, superior_vertebra in zip(spine.vertebrae, spine.vertebrae[1:]):

            print("Detecting CL landmarks for " + inferior_vertebra.name + " ...")

            inf_AS_polydatas  = [inferior_vertebra.shapeDecomposition.process_polydata["ASL"], inferior_vertebra.shapeDecomposition.process_polydata["ASR"]]
            sup_AI_polydatas  = [superior_vertebra.shapeDecomposition.process_polydata["AIL"], superior_vertebra.shapeDecomposition.process_polydata["AIR"]]
            # SpineLib.SlicerTools.createModelNode(inf_AS_polydatas[0], "Facet", color=[1.0, 0.0, 0.0], opacity=0.7)
            # SpineLib.SlicerTools.createModelNode(inf_AS_polydatas[1], "Facet", color=[1.0, 0.0, 0.0], opacity=0.7)
            # SpineLib.SlicerTools.createModelNode(sup_AI_polydatas[0], "Facet", color=[1.0, 0.0, 0.0], opacity=0.7)
            # SpineLib.SlicerTools.createModelNode(sup_AI_polydatas[1], "Facet", color=[1.0, 0.0, 0.0], opacity=0.7)
            medial_directions = [inferior_vertebra.orientation.r, -inferior_vertebra.orientation.r]

            for inf_AS_polydata, sup_AI_polydata, medial_direction, side in zip(inf_AS_polydatas, sup_AI_polydatas, medial_directions, ["left", "right"]):

                # get facet contact polydata
                inf_contact_polydata = conv.get_contact_polydata(sup_AI_polydata, inf_AS_polydata)
                sup_contact_polydata = conv.get_contact_polydata(inf_AS_polydata, sup_AI_polydata)

                # SpineLib.SlicerTools.createModelNode(inf_contact_polydata, "FacetContact", color=[1.0, 0.0, 0.0], opacity=0.7)
                # SpineLib.SlicerTools.createModelNode(sup_contact_polydata, "FacetContact", color=[1.0, 0.0, 0.0], opacity=0.7)

                #combined_surface = conv.polydata_append(inf_contact_polydata, sup_contact_polydata)
                #facet_polydata = conv.polydata_convexHull(combined_surface)
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

                #print("Facet medial: ", facet_medial)

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
            
            progressBarManager.updateProgress()

            
  

        # for vt, vertebra in enumerate(spine.vertebrae):

        #     landmarks = vertebra.shapeDecomposition.landmarks
        #     facet_landmarks = {}
        #     for side in ["left", "right"]:
        #         facet_landmarks[side] = {}
        #         for direction in ["medial", "lateral"]:
        #             facet_landmarks[side][direction] = conv.cut_plane(vertebra.facet_joints[side][direction], landmarks[f"{side}_pedicle_{direction}"], vertebra.orientation.r)
        #             SpineLib.SlicerTools.createModelNode(facet_landmarks[side][direction], f"{side.capitalize()}Facet{direction.capitalize()}", color=[0.0, 1.0, 0.0], opacity=0.7)

        # return facet_landmarks

    
    def _detect_ISL(self, spine: SpineLib.Spine = None, progressBarManager: SpineLib.ProgressBarManager = None):
        
        for vt, vertebra in enumerate(spine.vertebrae):

            print("Detecting ISL landmarks for " + vertebra.name + " ...")

            # TODO: fix local_s direction
            
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
            # local_s is the local superior direction of the centerline, therefor of the spinous process,
            # this is used for filtering, since the direction is different for e.g cervical/thoracic/lumbar vertebrae
            local_s = np.cross(local_r, main_comp)

            # # filter polydata
            superior_spinous_polydata = conv.eliminate_misaligned_faces(spinous_polydata, controlPoints.mean(axis=0),  local_s, 45.0)
            superior_intersection = conv.cut_plane(superior_spinous_polydata, controlPoints.mean(axis=0), local_r)
            superior_intersection_points = numpy_support.vtk_to_numpy(superior_intersection.GetPoints().GetData())
            sorted_superior_points = conv.sorted_points(list(superior_intersection_points), main_comp)
            superior_curveNode = SpineLib.SlicerTools.createResampledCurve(sorted_superior_points, 7, name="Superior_SpinousCurve", color=[1, 0, 0])
            superior_ISL_samples = slicer.util.arrayFromMarkupsControlPoints(superior_curveNode)[1:-1]
            superior_ISL = conv.find_closest_points(spinous_polydata, superior_ISL_samples)


            # # find individual intersection points
            # superior_intersection_points = [value for c in controlPoints if (value := conv.get_intersection_points(superior_spinous_polydata, c, c* vertebra.orientation.s*100)) is not None]
            # inferior_intersection_points = [value for c in controlPoints if (value := conv.get_intersection_points(inferior_spinous_polydata, c, c*-vertebra.orientation.s*100)) is not None]


            # # for cervical spine, fit with symmetry plane
            # if (vertebra.index >= 17):
            #     symmetry_plane = vertebra.symmetry_plane

            #     # superior ISL points
            #     superior_spinous_intersection = conv.cut_plane(superior_spinous_polydata, symmetry_plane.GetOrigin(), symmetry_plane.GetNormal())
            #     superior_spinous_intersection_points = numpy_support.vtk_to_numpy(superior_spinous_intersection.GetPoints().GetData())
            #     superior_sorted_intersection_points = conv.sorted_points(list(superior_spinous_intersection_points), vertebra.orientation.a)
            #     superior_curveNode = SpineLib.SlicerTools.createResampledCurve(superior_sorted_intersection_points, 12, name="Superior_SpinousCurve", color=[1, 0, 0])
            #     superior_ISL_samples = slicer.util.arrayFromMarkupsControlPoints(superior_curveNode)[:-4]

            #     # inferior ISL points
            #     inferior_spinous_intersection = conv.cut_plane(inferior_spinous_polydata, symmetry_plane.GetOrigin(), symmetry_plane.GetNormal())
            #     inferior_spinous_intersection_points = numpy_support.vtk_to_numpy(inferior_spinous_intersection.GetPoints().GetData())
            #     inferior_sorted_intersection_points = conv.sorted_points(list(inferior_spinous_intersection_points), vertebra.orientation.a)
            #     inferior_curveNode = SpineLib.SlicerTools.createResampledCurve(inferior_sorted_intersection_points, 12, name="Inferior_SpinousCurve", color=[1, 0, 0])
            #     inferior_ISL_samples = slicer.util.arrayFromMarkupsControlPoints(inferior_curveNode)[:-4]

                
            #     SpineLib.SlicerTools.removeNodes([superior_curveNode, inferior_curveNode])

            #     superior_ISL = conv.find_closest_points(spinous_polydata, superior_ISL_samples)
            #     inferior_ISL = conv.find_closest_points(spinous_polydata, inferior_ISL_samples)

            #     ISL_landmarks = np.concatenate((superior_ISL, inferior_ISL))

            #     SpineLib.SlicerTools.createMarkupsFiducialNode(superior_ISL, "Superior_ISL", color=[0.0, 0.0, 1.0])
            #     SpineLib.SlicerTools.createMarkupsFiducialNode(inferior_ISL, "Inferior_ISL", color=[0.0, 0.0, 1.0])

            #     vertebra.ligament_landmarks["ISL"] = ISL_landmarks




            # for thoracic and lumbar spine, fit with skeleton line
            #else:

                #SpineLib.SlicerTools.markupsLineNode("LocalR", controlPoints.mean(axis=0), controlPoints.mean(axis=0) + 25*local_s)

            ########################################
            inferior_spinous_polydata = conv.eliminate_misaligned_faces(spinous_polydata, controlPoints.mean(axis=0), -local_s, 45.0)
            # cut off with superior data
            superior_front_point = slicer.util.arrayFromMarkupsControlPoints(superior_curveNode)[-1]
            inferior_spinous_polydata = conv.clip_plane(inferior_spinous_polydata, superior_front_point, -vertebra.orientation.a)

            inferior_intersection = conv.cut_plane(inferior_spinous_polydata, controlPoints.mean(axis=0), -local_r)
            inferior_intersection_points = numpy_support.vtk_to_numpy(inferior_intersection.GetPoints().GetData())
            sorted_inferior_points = conv.sorted_points(list(inferior_intersection_points), main_comp)
            inferior_curveNode = SpineLib.SlicerTools.createResampledCurve(sorted_inferior_points, 7, name="Inferior_SpinousCurve", color=[1, 0, 0])
            inferior_ISL_samples = slicer.util.arrayFromMarkupsControlPoints(inferior_curveNode)[1:-1]
            inferior_ISL = conv.find_closest_points(spinous_polydata, inferior_ISL_samples)
            ########################################

            #SpineLib.SlicerTools.createModelNode(intersection_points, "IntersectionPoints", color=[1.0, 0.0, 0.0], opacity=1.0)

            # superior_intersection_points = [conv.get_intersection_points(spinous_polydata, c, c*local_s*100) for c in controlPoints]
            # inferior_intersection_points = [conv.get_intersection_points(spinous_polydata, c, c*-local_s*100) for c in controlPoints]

            # fit curve to intersection points


            # create markup nodes
            ISL_landmarks = np.concatenate((superior_ISL, inferior_ISL))
            SpineLib.SlicerTools.removeNodes([superior_curveNode, inferior_curveNode])
            SpineLib.SlicerTools.createMarkupsFiducialNode(superior_ISL, "Superior_ISL", color=[0.0, 0.0, 1.0])
            SpineLib.SlicerTools.createMarkupsFiducialNode(inferior_ISL, "Inferior_ISL", color=[0.0, 0.0, 1.0])

            # add landmarks to dict
            vertebra.ligament_landmarks["ISL"] = ISL_landmarks

                # for i, c in enumerate(controlPoints):
                #     transformMatrix = vtk.vtkMatrix4x4()
                #     spinous_centerline.GetCurvePointToWorldTransformAtPointIndex(spinous_centerline.GetCurvePointIndexFromControlPointIndex(i),transformMatrix)

                









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


            progressBarManager.updateProgress()


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
    def _detect_LF(self, spine: SpineLib.Spine = None, progressBarManager: SpineLib.ProgressBarManager = None):

        for vt, vertebra in enumerate(spine.vertebrae):

            print("Detecting LF landmarks for " + vertebra.name + " ...")
            
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


            progressBarManager.updateProgress()



            
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
