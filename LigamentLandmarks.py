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
                 spine: SpineLib.Spine,
                 progressBarManager: SpineLib.ProgressBarManager = None
                 ) -> None:
        
        # progress bar in slicer widget
        self.progressBarManager = progressBarManager
        
        # number of landmarks for ligament groups
        num_ll  = 7 # should be even
        num_cl  = 4 # fixed
        num_isl = 7 
        num_lf  = 3 # should be even
        num_itl = 1 # fixed
        num_ssl = 1 # fixed

        try:
            self._detect_ALL_PLL(spine, num_ll,  self.progressBarManager)
            self._detect_CL     (spine, num_cl,  self.progressBarManager)
            self._detect_ISL    (spine, num_isl, self.progressBarManager)
            self._detect_LF     (spine, num_lf,  self.progressBarManager)
            self._detect_ITL    (spine, num_itl, self.progressBarManager) 
            self._detect_SSL    (spine, num_ssl, self.progressBarManager)

        except Exception as e:
            print(e)
            print("Error in LigamentLandmarks")
            traceback.print_exc()
            return None
        
        #self._add_connections(spine)


    # add connections to visualize the ligaments force vectors
    def _add_connections(self,
                         spine:SpineLib.Spine
                         ) -> bool:

        for inferior_vertebra, superior_vertebra in zip(spine.vertebrae, spine.vertebrae[1:]):

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

            return True
        

    '''
    Detect intertransverse ligament (ITL) landmarks
    '''
    def _detect_ITL(self,
                    spine: SpineLib.Spine,
                    resampling_num: int = 1,
                    progressBarManager: SpineLib.ProgressBarManager = None
                    ) -> bool:
            
        for vt, vertebra in enumerate(spine.vertebrae):

            try:
                print("Detecting ITL landmarks for " + vertebra.name + " ...")

                if vertebra.name in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]: # for cervical spine, get the geometric extremes of the transverse processes segments

                    left_transverse_polydata = vertebra.shapeDecomposition.process_polydata["TL"]
                    left_transverse_points = numpy_support.vtk_to_numpy(left_transverse_polydata.GetPoints().GetData())
                    left_ITL_landmark = sorted(left_transverse_points, key=(lambda p: np.array(p).dot(vertebra.orientation.r)))[0]

                    right_transverse_polydata = vertebra.shapeDecomposition.process_polydata["TR"]
                    right_transverse_points = numpy_support.vtk_to_numpy(right_transverse_polydata.GetPoints().GetData())
                    right_ITL_landmark = sorted(right_transverse_points, key=(lambda p: np.array(p).dot(vertebra.orientation.r)))[-1]

                else: # for thoracic and lumbar spine, extrapolate the ITL centerline curve and get the intersection points with the transverse processes
                    
                    # LEFT ITL landmark #
                    # get left transverse data
                    left_transverse_polydata = vertebra.shapeDecomposition.process_polydata["TL"]
                    left_transverse_centerline = vertebra.shapeDecomposition.centerlines["TL"]

                    # get main component of the centerline
                    controlPoints = np.array(left_transverse_centerline.GetCurvePointsWorld().GetData())
                    controlPoints = controlPoints[:len(controlPoints)//2]
                    main_comp = conv.calc_main_component(list(controlPoints)) # TODO: add a method in conv for this main_comp: fit_direction_vector (input: point, approximated direction vector, normalize:bbool. output: direction vector)
                    main_comp = conv.closest_vector([main_comp], np.average((-vertebra.orientation.r, -vertebra.orientation.a), axis=0))
                    main_comp = conv.normalize(main_comp[0]*main_comp[1])

                    # intersect polydata with line from first control point, in direction of main component
                    left_ITL_landmark = conv.get_intersection_points(left_transverse_polydata, controlPoints[0], controlPoints[0]+main_comp*100)

                    # RIGHT ITL landmark #
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

                # add landmarks to dict
                vertebra.ligament_landmarks["ITL_L"] = [left_ITL_landmark]
                vertebra.ligament_landmarks["ITL_R"] = [right_ITL_landmark]
                

            except Exception as e:
                print(e)
                print("Error in ITL detection")
                traceback.print_exc()

            # process events and update progress bar in slicer widget
            slicer.app.processEvents()
            progressBarManager.updateProgress()

        return True


    '''
    Detect supraspinous ligament (SSL) landmarks
    '''
    def _detect_SSL(self,
                    spine: SpineLib.Spine,
                    resampling_num: int = 1,
                    progressBarManager: SpineLib.ProgressBarManager = None
                    ) -> bool:
        
        print("Detecting SSL landmarks ...")

        for vt, vertebra in enumerate(spine.vertebrae):
            try:
                
                # extrapolate the SSL centerline curve and get the intersection points with the spinous process
                # get spinous data
                spinous_polydata = vertebra.shapeDecomposition.process_polydata["S"]
                spinous_centerline = vertebra.shapeDecomposition.centerlines["S"]

                # get main component of the centerline
                controlPoints = np.array(spinous_centerline.GetCurvePointsWorld().GetData())
                controlPoints = controlPoints[:len(controlPoints)//2]
                main_comp = conv.calc_main_component(list(controlPoints))
                main_comp = conv.closest_vector([main_comp], -np.average([vertebra.orientation.s, vertebra.orientation.a], axis=0))
                main_comp = conv.normalize(main_comp[0]*main_comp[1])

                SSL_landmark = conv.get_intersection_points(spinous_polydata, controlPoints[0], controlPoints[0]+main_comp*100)
                vertebra.ligament_landmarks["SSL"] = [SSL_landmark]
            
            except Exception as e:
                print(e)
                print("Error in SSL detection")
                traceback.print_exc()

            slicer.app.processEvents()
        
        # update progress bar in slicer widget
        progressBarManager.updateProgress()

        return True



    '''
    Detect anterior longitudinal ligament (ALL) and posterior longitudinal ligament (PLL) landmarks
    '''
    def _detect_ALL_PLL(self,
                        spine: SpineLib.Spine,
                        resampling_num: int, 
                        progressBarManager: SpineLib.ProgressBarManager = None
                        ) -> bool:
        
        for vt, vertebra in enumerate(spine.vertebrae):

            try:

                print("Detecting ALL and PLL landmarks for " + vertebra.name + " ...")

                # measurements for average lumbar spine
                avg_ALL_width = 30
                avg_body_width = 52.0
                avg_PLL_width = 20.0
                avg_canal_width = 30.0

                # calulate the dimensions for ALL and PLL
                ratio_ALL  = avg_ALL_width / avg_body_width
                factor_ALL = ratio_ALL/2
                ratio_PLL  = avg_PLL_width / avg_canal_width
                factor_PLL = (1-ratio_PLL)/2
                left_origin_ALL  = vertebra.center - factor_ALL * vertebra.orientation.r * vertebra.size.width
                right_origin_ALL = vertebra.center + factor_ALL * vertebra.orientation.r * vertebra.size.width
                left_medial  = vertebra.shapeDecomposition.landmarks["left_pedicle_medial"]
                right_medial = vertebra.shapeDecomposition.landmarks["right_pedicle_medial"]
                canal_vector = right_medial - left_medial
                left_origin_PLL = left_medial + factor_PLL * canal_vector
                right_origin_PLL = right_medial - factor_PLL * canal_vector

                # calculate ALL and PLL for superior and inferior parts
                for side in ["superior", "inferior"]:

                    # get endplate boundary
                    endplate = getattr(vertebra.body, side+"_endplate")           
                    boundary = conv.extractBoundary(endplate)

                    # clip into anterior and posterior parts
                    anterior_clipped = conv.clip_plane(boundary, vertebra.center, vertebra.orientation.a)
                    posterior_clipped = conv.clip_plane(boundary, vertebra.center, -vertebra.orientation.a)

                    # clip left and right (restrict width)
                    clipped_all = conv.clip_plane(anterior_clipped, left_origin_ALL, vertebra.orientation.r)
                    clipped_all = conv.clip_plane(clipped_all, right_origin_ALL, -vertebra.orientation.r)
                    clipped_pll = conv.clip_plane(posterior_clipped, left_origin_PLL, vertebra.orientation.r)
                    clipped_pll = conv.clip_plane(clipped_pll, right_origin_PLL, -vertebra.orientation.r)

                    # sort points in lateral direction
                    sorted_anterior_all = conv.sorted_points(list(numpy_support.vtk_to_numpy(clipped_all.GetPoints().GetData())), vertebra.orientation.r)
                    sorted_posterior_pll = conv.sorted_points(list(numpy_support.vtk_to_numpy(clipped_pll.GetPoints().GetData())), vertebra.orientation.r)

                    # fit curve from points and resample to equidistant points
                    all_curve = SpineLib.SlicerTools.createResampledCurve(sorted_anterior_all, resampling_num, name="ALL", color=[1, 0, 0])
                    pll_curve = SpineLib.SlicerTools.createResampledCurve(sorted_posterior_pll, resampling_num, name="PLL", color=[1, 0, 0])
                    all = slicer.util.arrayFromMarkupsControlPoints(all_curve)
                    pll = slicer.util.arrayFromMarkupsControlPoints(pll_curve)

                    SpineLib.SlicerTools.removeNodes([all_curve, pll_curve])

                    # add landmarks to dict
                    side_name = "S" if side == "superior" else "I"
                    vertebra.ligament_landmarks["ALL_"+side_name] = all
                    vertebra.ligament_landmarks["PLL_"+side_name] = pll
                
            except Exception as e:
                print(e)
                print("Error in ALL/PLL detection")
                traceback.print_exc()

            # update progress bar in slicer widget
            progressBarManager.updateProgress()

        return True
        


    '''
    Detect capsular ligament (CL) landmarks
    '''
    def _detect_CL(self,
                   spine: SpineLib.Spine,
                   resampling_num: int,
                   progressBarManager: SpineLib.ProgressBarManager = None):

        for inferior_vertebra, superior_vertebra in zip(spine.vertebrae, spine.vertebrae[1:]):

            try:
                print("Detecting CL landmarks for " + inferior_vertebra.name + " ...")

                inf_AS_polydatas  = [inferior_vertebra.shapeDecomposition.process_polydata["ASL"], inferior_vertebra.shapeDecomposition.process_polydata["ASR"]]
                sup_AI_polydatas  = [superior_vertebra.shapeDecomposition.process_polydata["AIL"], superior_vertebra.shapeDecomposition.process_polydata["AIR"]]
                medial_directions = [inferior_vertebra.orientation.r, -inferior_vertebra.orientation.r]

                for inf_AS_polydata, sup_AI_polydata, medial_direction, side in zip(inf_AS_polydatas, sup_AI_polydatas, medial_directions, ["L", "R"]):

                    # get facet contact polydata
                    inf_contact_polydata = conv.get_contact_polydata(sup_AI_polydata, inf_AS_polydata)
                    sup_contact_polydata = conv.get_contact_polydata(inf_AS_polydata, sup_AI_polydata)

                    # Compute best fit plane
                    planeCenter, planeNormal = conv.fitPlane(numpy_support.vtk_to_numpy(inf_contact_polydata.GetPoints().GetData()))
                    normal = np.array(planeNormal)
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
                    facet_medial = facet_LR if np.dot(facet_LR, medial_direction) >= 0 else -facet_LR

                    # calculate the facet center, intersect with facet-up-plane and facet-medial-plane, define landmarks as outer points of the intersections
                    # inferior CL landmarks
                    inf_facet_center = np.mean(numpy_support.vtk_to_numpy(inf_contact_polydata.GetPoints().GetData()), axis=0)
                    lr_intersection = conv.cut_plane(inf_contact_polydata, inf_facet_center, facet_up)
                    lr_sorted = conv.sorted_points(lr_intersection, facet_medial)
                    lr_1 = lr_sorted[0]
                    lr_2 = lr_sorted[-1]
                    is_intersection = conv.cut_plane(inf_contact_polydata, inf_facet_center, facet_medial)
                    is_sorted = conv.sorted_points(is_intersection, facet_up)
                    is_1 = is_sorted[0]
                    is_2 = is_sorted[-1]

                    inf_ligament_landmarks = [lr_1, lr_2, is_1, is_2]

                    # superior CL landmarks
                    sup_facet_center = np.mean(numpy_support.vtk_to_numpy(sup_contact_polydata.GetPoints().GetData()), axis=0)
                    lr_intersection = conv.cut_plane(sup_contact_polydata, sup_facet_center, facet_up)
                    lr_sorted = conv.sorted_points(lr_intersection, facet_medial)
                    lr_1 = lr_sorted[0]
                    lr_2 = lr_sorted[-1]
                    is_intersection = conv.cut_plane(sup_contact_polydata, sup_facet_center, facet_medial)
                    is_sorted = conv.sorted_points(is_intersection, facet_up)
                    is_1 = is_sorted[0]
                    is_2 = is_sorted[-1]

                    sup_ligament_landmarks = [lr_1, lr_2, is_1, is_2]

                    # add landmarks to dict
                    inferior_vertebra.ligament_landmarks["CL_S_" + side] = inf_ligament_landmarks
                    superior_vertebra.ligament_landmarks["CL_I_" + side] = sup_ligament_landmarks
                
                
            except Exception as e:
                print(e)
                print("Error in CL detection")
                traceback.print_exc()
                
            # process events and update progress bar in slicer widget
            slicer.app.processEvents()
            progressBarManager.updateProgress()

        return True
        

    

    ''' Detect interspinous ligament (ISL) landmarks'''
    def _detect_ISL(self,
                    spine: SpineLib.Spine,
                    resampling_num: int,
                    progressBarManager: SpineLib.ProgressBarManager = None
                    ) -> bool:
        
        for vt, vertebra in enumerate(spine.vertebrae):

            try:

                print("Detecting ISL landmarks for " + vertebra.name + " ...")

                # filter data points
                spinous_polydata = vertebra.shapeDecomposition.process_polydata["S"]

                # remesh polydata
                spinous_polydata = conv.polydata_remesh(spinous_polydata, 2, 5000)
                spinous_centerline = vertebra.shapeDecomposition.centerlines["S"]
                spinous_centerline.GetDisplayNode().SetVisibility(1)

                # fit a line to the control points
                controlPoints = np.array(spinous_centerline.GetCurvePointsWorld().GetData())
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
                superior_curveNode = SpineLib.SlicerTools.createResampledCurve(sorted_superior_points, resampling_num+2, name="Superior_SpinousCurve", color=[1, 0, 0])
                superior_ISL_samples = slicer.util.arrayFromMarkupsControlPoints(superior_curveNode)[1:-1]
                superior_ISL = conv.find_closest_points(spinous_polydata, superior_ISL_samples)


                ########################################
                inferior_spinous_polydata = conv.eliminate_misaligned_faces(spinous_polydata, controlPoints.mean(axis=0), -local_s, 45.0)
                # cut off with superior data
                superior_front_point = slicer.util.arrayFromMarkupsControlPoints(superior_curveNode)[-1]
                inferior_spinous_polydata = conv.clip_plane(inferior_spinous_polydata, superior_front_point, -vertebra.orientation.a)

                inferior_intersection = conv.cut_plane(inferior_spinous_polydata, controlPoints.mean(axis=0), -local_r)
                inferior_intersection_points = numpy_support.vtk_to_numpy(inferior_intersection.GetPoints().GetData())
                sorted_inferior_points = conv.sorted_points(list(inferior_intersection_points), main_comp)
                inferior_curveNode = SpineLib.SlicerTools.createResampledCurve(sorted_inferior_points, resampling_num+2, name="Inferior_SpinousCurve", color=[1, 0, 0])
                inferior_ISL_samples = slicer.util.arrayFromMarkupsControlPoints(inferior_curveNode)[1:-1]
                inferior_ISL = conv.find_closest_points(spinous_polydata, inferior_ISL_samples)
                ########################################

                SpineLib.SlicerTools.removeNodes([superior_curveNode, inferior_curveNode])

                # add landmarks to dict
                vertebra.ligament_landmarks["ISL_S"] = superior_ISL
                vertebra.ligament_landmarks["ISL_I"] = inferior_ISL


            except Exception as e:
                print(e)
                print("Error in ISL detection")
                traceback.print_exc()

            # process events and update progress bar in slicer widget
            progressBarManager.updateProgress()
            slicer.app.processEvents()

        return True


    '''
    Fit a curve between two points on a surface of a vertebra.
    The curve is constructed from a dijkstra path between the two points'''
    def flavum_curve(self,
                     vertebra: SpineLib.Vertebra,
                     left_keypoint,
                     right_keypoint,
                     middle_keypoint = None,
                     resampling_num: int = 7
                     ) -> np.ndarray: 

        # get the segmentation of the processes
        polydata = vertebra.shapeDecomposition.processes

        # run dijkstra
        if middle_keypoint is None:
            # get the dijkstra path between left and right keypoint
            flavum_curvePoints = conv.runDijkstra(polydata, np.array(left_keypoint), np.array(right_keypoint))

        else:
            # get the dijkstra path between left and middle keypoint, and middle and right keypoint
            flavum_curvePoints_left = conv.runDijkstra(polydata, np.array(left_keypoint), np.array(middle_keypoint))
            flavum_curvePoints_right = conv.runDijkstra(polydata, np.array(middle_keypoint), np.array(right_keypoint))

            # concatenate left and right points
            flavum_curvePoints = vtk.vtkPoints()
            for i in range(flavum_curvePoints_right.GetNumberOfPoints()):
                flavum_curvePoints.InsertNextPoint(flavum_curvePoints_right.GetPoint(i))
            for i in range(flavum_curvePoints_left.GetNumberOfPoints()):
                flavum_curvePoints.InsertNextPoint(flavum_curvePoints_left.GetPoint(i))

        # clip to left and right
        parametricSpline = vtk.vtkParametricSpline()
        parametricSpline.SetPoints(flavum_curvePoints)
        parametricFunctionSource = vtk.vtkParametricFunctionSource()
        parametricFunctionSource.SetParametricFunction(parametricSpline)
        parametricFunctionSource.Update()
        left_clipped = conv.clip_plane(parametricFunctionSource.GetOutput(), vertebra.center, -vertebra.orientation.r)
        right_clipped = conv.clip_plane(parametricFunctionSource.GetOutput(), vertebra.center, vertebra.orientation.r)
        left_flavum_Points = numpy_support.vtk_to_numpy(left_clipped.GetPoints().GetData())
        right_flavum_Points = numpy_support.vtk_to_numpy(right_clipped.GetPoints().GetData())
        left_flavum_curve = SpineLib.SlicerTools.createResampledCurve(left_flavum_Points, resampling_num+2, name="FlavumCurve", color=[1, 0, 1])
        right_flavum_curve = SpineLib.SlicerTools.createResampledCurve(right_flavum_Points, resampling_num+2, name="FlavumCurve", color=[1, 0, 1])
        left_flavum_Points = slicer.util.arrayFromMarkupsControlPoints(left_flavum_curve)[1:-1]
        right_flavum_Points = slicer.util.arrayFromMarkupsControlPoints(right_flavum_curve)[1:-1]
        flavum_points = np.concatenate((right_flavum_Points, left_flavum_Points))

        SpineLib.SlicerTools.removeNodes([left_flavum_curve, right_flavum_curve])

        return flavum_points


    '''
    Detect ligamentum flavum landmarks
    '''
    def _detect_LF(self,
                   spine: SpineLib.Spine,
                   resampling_num: int,
                   progressBarManager: SpineLib.ProgressBarManager = None
                   ) -> bool:

        for vt, vertebra in enumerate(spine.vertebrae):

            try:

                print("Detecting LF landmarks for " + vertebra.name + " ...")

                if vertebra.name in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]: # cervical vertebra, with CL landmarks

                    # upper flavum curve
                    if "CL_S_L" in vertebra.ligament_landmarks and "CL_S_R" in vertebra.ligament_landmarks:
                        left_keypoint = vertebra.ligament_landmarks["CL_S_L"][1]
                        right_keypoint = vertebra.ligament_landmarks["CL_S_R"][1]
                        middle_keypoint = vertebra.ligament_landmarks["ISL_S"][-1]
                        upper_flavum_points = self.flavum_curve(vertebra, left_keypoint, right_keypoint, middle_keypoint=middle_keypoint, resampling_num=resampling_num)
                        vertebra.ligament_landmarks["LF_S"] = upper_flavum_points 
                    else:
                        print("vertebra ", vertebra.name, ": CL landmarks not found for upper flavum curve")
                    
                    # lower flavum curve
                    if "CL_I_L" in vertebra.ligament_landmarks and "CL_I_R" in vertebra.ligament_landmarks:
                        left_keypoint = vertebra.ligament_landmarks["CL_I_L"][1]
                        right_keypoint = vertebra.ligament_landmarks["CL_I_R"][1]
                        middle_keypoint = vertebra.ligament_landmarks["ISL_I"][-1]
                        lower_flavum_points = self.flavum_curve(vertebra, left_keypoint, right_keypoint, middle_keypoint=middle_keypoint, resampling_num=resampling_num)
                        vertebra.ligament_landmarks["LF_I"] = lower_flavum_points
                    else:
                        print("vertebra ", vertebra.name, ": CL landmarks not found for lower flavum curve")
                

                else: # thoracic and lumbar, with pedicle landmarks
                    left_keypoint = vertebra.shapeDecomposition.landmarks["left_pedicle_superior"]
                    right_keypoint = vertebra.shapeDecomposition.landmarks["right_pedicle_superior"]
                    upper_flavum_points = self.flavum_curve(vertebra, left_keypoint, right_keypoint, resampling_num=resampling_num) 

                    left_keypoint = vertebra.shapeDecomposition.landmarks["left_pedicle_inferior"]
                    right_keypoint = vertebra.shapeDecomposition.landmarks["right_pedicle_inferior"]
                    lower_flavum_points = self.flavum_curve(vertebra, left_keypoint, right_keypoint, middle_keypoint=vertebra.ligament_landmarks["ISL_I"][-1], resampling_num=resampling_num)
                    
                    vertebra.ligament_landmarks["LF_I"] = lower_flavum_points
                    vertebra.ligament_landmarks["LF_S"] = upper_flavum_points  

                    
            except Exception as e:
                print(e)
                print("Error in LF detection")
                traceback.print_exc()

            # process events and update progress bar in slicer widget
            progressBarManager.updateProgress()
            slicer.app.processEvents()
        
        return True