from dataclasses import dataclass
import numpy as np
import os
import vtk
import vtk_convenience as conv
from vtk.util import numpy_support
from typing import Dict, Tuple
import seaborn as sns
from scipy.signal import find_peaks, argrelextrema
from sklearn.cluster import KMeans
import matplotlib
#matplotlib.use('Qt5Agg')

matplotlibBack = os.getenv('matplotlibback')
if matplotlibBack == None:
    print(f'matplotlibBack is probably not set. matplotlibBack={matplotlibBack}')
else: 
    print(f'matplotlibBack:{matplotlibBack}')
matplotlib.use(matplotlibBack)

import matplotlib.pyplot as plt
import SpineLib
import slicer
import ExtractCenterline


class ShapeDecomposition:   

    def __init__(self,
                 geometry:           vtk.vtkPolyData             = None,
                 center:             np.array                    = None,
                 size:               SpineLib.Size               = None,
                 orientation:        SpineLib.Orientation        = None,
                 symmetry_plane:     vtk.vtkPlane                = None,
                 index:              int                         = None,
                 original_model:     vtk.vtkPolyData             = None,
                 with_lamina:        bool                        = None,
                 progressBarManager:  SpineLib.ProgressBarManager = None
                 ) -> None:
        
        lib_vertebraIDs = ["L5", "L4", "L3", "L2", "L1",
                           "T13", "T12", "T11", "T10", "T9", "T8", "T7", "T6", "T5", "T4", "T3", "T2", "T1",
                           "C7", "C6", "C5", "C4", "C3", "C2", "C1"]
        
        print("Semantic Segmentation of " + lib_vertebraIDs[index] + " ...")

        self.geometry = geometry
        self.threshold, self.body, self.processes        = ShapeDecomposition._pdf_decomposition(geometry, center, size, orientation, index)
        self.landmarks                                   = ShapeDecomposition._landmarks(geometry, center, size, orientation, self.threshold, index)
        self.segmented_geometry, self.process_polydata, self.process_landmarks, self.centerlines = ShapeDecomposition._segment_processes(geometry, self.processes, self.body, self.landmarks, orientation, symmetry_plane, index, with_lamina)

        # optional: if original model (not preprocessed) is given, use that for segmentation
        if original_model is not None:
            orig_body = conv.clip_sphere(original_model, self.landmarks["body_front"], self.threshold, InsideOut=True)
            self.label_Model, _ = ShapeDecomposition.centerline_segmentation(original_model, orig_body, self.centerlines, index)
            # SpineLib.SlicerTools.createModelNode(self.label_Model, "label_model")

        # progress bar in slicer widget
        if progressBarManager is not None: progressBarManager.updateProgress()

            

    '''
    Calculate orientation of the vertebra.
    The vectors represent the RAS-coordinates of a local object coordinate system.
    RAS -> R:right, A:anterior, S:superior
    '''
    def _pdf_decomposition(
            geometry:          vtk.vtkPolyData         = None,
            center:            np.array                = None,
            size:              SpineLib.Size           = None,
            orientation:       SpineLib.Orientation    = None,
            index:             int                     = None,
            ):

        landmark = conv.get_intersection_points(geometry, center, (center + (orientation.a * size.depth)))
        # for cervical spine, take the center of the geometry as landmark
        if (index >= 18):
            landmark = (center + landmark) / 2.0

        points = numpy_support.vtk_to_numpy(geometry.GetPoints().GetData())
        distances = np.linalg.norm(points - landmark, axis=1)

        # Calculate the KDE
        kde = sns.kdeplot(distances, color='skyblue', legend=False)
        x_vals = kde.get_lines()[0].get_xdata()
        y_vals = kde.get_lines()[0].get_ydata()
        
        # Find extrema points (peaks and valleys)
        peaks, _ = find_peaks(y_vals)
        valleys = argrelextrema(y_vals, np.less)[0]

        # Calculate the second derivative of the KDE
        second_derivative = np.gradient(np.gradient(y_vals, x_vals), x_vals)

        # Filter extrema into minima and maxima
        minima_points = [valley for valley in valleys if y_vals[valley] < y_vals[valley + 1]]
        maxima_points = [peak for peak in peaks if y_vals[peak] > y_vals[peak + 1]]

        # Sort maxima by height (descending order)
        sorted_maxima_indices = np.argsort(y_vals[maxima_points])[::-1]
        highest_maxima_x = [x_vals[maxima_points[i]] for i in sorted_maxima_indices[:2]]

        # Find the lowest minimum between the maxima (consider edge cases)
        lowest_min_between_maxima = None

        for min_point in minima_points:
            # Ensure minimum is strictly between the X values of the maxima
            
            if x_vals[min_point] < max(highest_maxima_x) and x_vals[min_point] > min(highest_maxima_x):
            # Update if lower than current minimum
                if (lowest_min_between_maxima is None) or (y_vals[min_point] < y_vals[lowest_min_between_maxima]):
                    lowest_min_between_maxima = min_point
            
        # Highlight inflection, minima, and maxima points on the plot with different colors
        plt.ioff()
        plt.plot(x_vals, y_vals, color='skyblue')
        plt.xlabel("Distance")
        plt.ylabel("Probability Density")
        
        # segment by filtering polydata
        threshold = x_vals[lowest_min_between_maxima]
        plt.clf()
        vertebral_body = conv.filter_point_ids(geometry, condition=lambda vertex: (distances[vertex] > threshold))
        processes = conv.filter_point_ids(geometry, condition=lambda vertex: distances[vertex] < threshold)

        return threshold, vertebral_body, processes
    



    '''
    Find Spinal landmarks
    '''
    def _landmarks(
            geometry:          vtk.vtkPolyData         = None,
            center:            np.array                = None,
            size:              SpineLib.Size           = None,
            orientation:       SpineLib.Orientation    = None,
            threshold:         float                   = None,
            index:             int                     = None,
            ):
        
        # find body-front and canal center landmarks
        landmarks = {}
        body_front = conv.get_intersection_points(geometry, center, (center + (orientation.a * size.depth)))
        if (index >= 18):
            body_front = (center + body_front) / 2.0
        canal_center = body_front - orientation.a * threshold
        
        landmarks['canal_center']           = canal_center
        landmarks['body_front']             = body_front

        # find the pedicle landmarks
        for side, side_orientation, in zip(["left", "right"], [-orientation.r, orientation.r]):
            clipped_geometry        = conv.clip_plane(geometry, canal_center, side_orientation)
            pedicle_intersection    = conv.cut_sphere(clipped_geometry, body_front, threshold)
            intersection_com        = np.mean(numpy_support.vtk_to_numpy(pedicle_intersection.GetPoints().GetData()), axis=0)
            lr_intersection         = conv.cut_plane(pedicle_intersection, intersection_com, orientation.s)
            left_lr_sorted          = conv.sorted_points(lr_intersection, -side_orientation)
            is_intersection         = conv.cut_plane(pedicle_intersection, intersection_com, side_orientation)
            is_sorted               = conv.sorted_points(is_intersection, orientation.s)

            landmarks[f"{side}_pedicle_medial"]    = np.array(left_lr_sorted[-1])
            landmarks[f"{side}_pedicle_lateral"]   = np.array(left_lr_sorted[0])
            landmarks[f"{side}_pedicle_com"]       = np.array(intersection_com)
            landmarks[f"{side}_pedicle_superior"]  = np.array(is_sorted[-1])
            landmarks[f"{side}_pedicle_inferior"]  = np.array(is_sorted[0])

        return landmarks
    

    

    '''
    Segment the processes of the vertebra
    '''
    def _segment_processes(
            geometry:          vtk.vtkPolyData         = None,
            processes:         vtk.vtkPolyData         = None,
            body:              vtk.vtkPolyData         = None,
            landmarks:         Dict                    = None,
            orientation:       SpineLib.Orientation    = None,
            symmetry_plane:    vtk.vtkPlane            = None,
            index:             int                     = None,
            with_lamina:       bool                    = True,
            ):

        # compute centerline of the lamina
        centerline_lamina = ShapeDecomposition.centerline(processes, landmarks["left_pedicle_com"], landmarks["right_pedicle_com"], index, "Lamina")
        centerline_lamina_points = centerline_lamina.GetCurvePointsWorld()
        centerline_lamina_points = numpy_support.vtk_to_numpy(centerline_lamina_points.GetData())
        centerline_lamina_samples = [centerline_lamina_points[i] for i in range(0, len(centerline_lamina_points), len(centerline_lamina_points)//6)]

        # recalulate the lamina centerline as centerline from second to second last point
        centerline_lamina = ShapeDecomposition.centerline(processes, centerline_lamina_samples[1], centerline_lamina_samples[-2], index, "Lamina new")
        process_endpoints = {"TL": [], "ASL": [], "AIL": [], "S": [], "AIR": [], "ASR": [], "TR": []}
        process_endpoints = {"TL":  landmarks["left_pedicle_lateral"],
                             "ASL": centerline_lamina_samples[1],
                             "AIL": centerline_lamina_samples[2],
                             "S":   centerline_lamina_samples[3],
                             "AIR": centerline_lamina_samples[4],
                             "ASR": centerline_lamina_samples[5],
                             "TR":  landmarks["right_pedicle_lateral"]}
        
        
        # initial segmentation of processes
        initial_segmented_polydata, process_polydata, process_label_ids, process_points, process_landmarks = ShapeDecomposition.clustering(processes, orientation, symmetry_plane, index)

        for name, point in process_landmarks.items():
            # find closest centerline_lamina_point to each landmark
            distances = np.linalg.norm(centerline_lamina_points - point, axis=1)
            closest_lamina_point = centerline_lamina_points[np.argmin(distances)]
            process_endpoints[name] = closest_lamina_point

        # centerlines
        centerlines = {"TL": [], "ASL": [], "AIL": [], "S": [], "AIR": [], "ASR": [], "TR": []}
        for name, point in process_landmarks.items():
            centerlines[name] = ShapeDecomposition.centerline(processes, point, process_endpoints[name], index, name)
        
        # add lamina centerline to centerlines, then it is used for segmentation
        if with_lamina:
            centerlines["Lamina"] = centerline_lamina

        # final segmentation of processes
        segmented_polydata, process_polydata = ShapeDecomposition.centerline_segmentation(geometry, body, centerlines, index)
        
        return segmented_polydata, process_polydata, process_landmarks, centerlines



    ''' K-Means clustering of points '''
    def k_means(points, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(points)
        cluster_points = {}
        for label in range(n_clusters):
            cluster_points[label] = [points[i] for i, l in enumerate(labels) if l == label]
        cluster_centers = kmeans.cluster_centers_

        return labels, cluster_centers


    ''' Clustering of the vertebra processes using K-Means and rule-based determination of key landmarks'''
    def clustering(polydata, orientation, symmetry_plane, index):

        polydata_points = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
        process_label_ids = {}
        landmarks = {}

        # thoracic and lumbar spine w/out T12 and T11
        if (index < 18 and index != 5 and index != 6 and index != 7):

            labels, cluster_centers = ShapeDecomposition.k_means(polydata_points, 8)

            ######################################### Find process clusters ################################################################

            process_label_ids["TL"], tl_cluster_index = ShapeDecomposition.find_cluster_label_ids(labels, cluster_centers, cluster_centers, key=(lambda p: np.array(p).dot(orientation.r)))
            process_label_ids["TR"], tr_cluster_index = ShapeDecomposition.find_cluster_label_ids(labels, cluster_centers, cluster_centers, key=(lambda p: np.array(p).dot(-orientation.r)))
            process_label_ids["S"], s_cluster_index   = ShapeDecomposition.find_cluster_label_ids(labels, cluster_centers, cluster_centers, key=(lambda p: np.array(p).dot(orientation.a)))

            # remaining clusters
            re_cluster_centers = [element for i, element in enumerate(cluster_centers) if i not in [tl_cluster_index, tr_cluster_index, s_cluster_index]]
            central_center = sorted(re_cluster_centers, key=(lambda p: np.array(p).dot(orientation.r)))[len(re_cluster_centers)//2]
            central_cluster_index = np.where(np.all(cluster_centers == central_center, axis=1))[0][0]
            re_cluster_centers = [element for i, element in enumerate(cluster_centers) if i not in [tl_cluster_index, tr_cluster_index, s_cluster_index, central_cluster_index]]

            superior_clusters = sorted(re_cluster_centers, key=(lambda p: np.array(p).dot(orientation.s)), reverse=True)[:2]
            inferior_clusters = sorted(re_cluster_centers, key=(lambda p: np.array(p).dot(-orientation.s)), reverse=True)[:2]

            process_label_ids["ASL"], asl_cluster_index = ShapeDecomposition.find_cluster_label_ids(labels, cluster_centers, superior_clusters, key=(lambda p: np.array(p).dot(orientation.r)))
            process_label_ids["ASR"], asr_cluster_index = ShapeDecomposition.find_cluster_label_ids(labels, cluster_centers, superior_clusters, key=(lambda p: np.array(p).dot(-orientation.r)))
            process_label_ids["AIL"], ail_cluster_index = ShapeDecomposition.find_cluster_label_ids(labels, cluster_centers, inferior_clusters, key=(lambda p: np.array(p).dot(orientation.r)))
            process_label_ids["AIR"], air_cluster_index = ShapeDecomposition.find_cluster_label_ids(labels, cluster_centers, inferior_clusters, key=(lambda p: np.array(p).dot(-orientation.r)))

            ######################### Find landmarks #################################################################################
            process_points = {}
            for name in process_label_ids.keys():
                process_points[name] = [polydata_points[i] for i in process_label_ids[name]]

            
            # lumbar spine
            if(index <= 4):
                landmarks["ASL"] = sorted(process_points["ASL"], key=(lambda p: np.array(p).dot(np.average([orientation.s, orientation.s, -orientation.a, -orientation.r], axis=0))))[-1]
                landmarks["ASR"] = sorted(process_points["ASR"], key=(lambda p: np.array(p).dot(np.average([orientation.s, orientation.s, -orientation.a, orientation.r], axis=0))))[-1]
                landmarks["AIL"] = sorted(process_points["AIL"], key=(lambda p: np.array(p).dot(np.average([orientation.s], axis=0))))[0]
                landmarks["AIR"] = sorted(process_points["AIR"], key=(lambda p: np.array(p).dot(np.average([orientation.s], axis=0))))[0]
                landmarks["S"]   = sorted(process_points["S"], key=(lambda p: np.array(p).dot(orientation.a)))[0]
                landmarks["TL"]  = sorted(process_points["TL"], key=(lambda p: np.array(p).dot(orientation.r)))[0]
                landmarks["TR"]  = sorted(process_points["TR"], key=(lambda p: np.array(p).dot(orientation.r)))[-1]

            # thoracic spine
            else:
                landmarks["ASL"] = sorted(process_points["ASL"], key=(lambda p: np.array(p).dot(np.average([orientation.s], axis=0))))[-1]
                landmarks["ASR"] = sorted(process_points["ASR"], key=(lambda p: np.array(p).dot(np.average([orientation.s], axis=0))))[-1]
                landmarks["AIL"] = sorted(process_points["AIL"], key=(lambda p: np.array(p).dot(np.average([-orientation.s, orientation.a, -orientation.r], axis=0))))[-1]
                landmarks["AIR"] = sorted(process_points["AIR"], key=(lambda p: np.array(p).dot(np.average([-orientation.s, orientation.a, orientation.r], axis=0))))[-1]
                landmarks["S"]   = sorted(process_points["S"], key=(lambda p: np.array(p).dot(orientation.a)))[0]
                landmarks["TL"]  = sorted(process_points["TL"], key=(lambda p: np.array(p).dot(orientation.r)))[0]
                landmarks["TR"]  = sorted(process_points["TR"], key=(lambda p: np.array(p).dot(orientation.r)))[-1]


        # thoracic spine T12
        elif (index == 5 or index == 6 or index == 7):
            labels, cluster_centers = ShapeDecomposition.k_means(polydata_points, 5)

            ######################################### Find process clusters ################################################################
            process_label_ids["S"], s_cluster_index   = ShapeDecomposition.find_cluster_label_ids(labels, cluster_centers, cluster_centers, key=(lambda p: np.array(p).dot(orientation.a)))
            re_cluster_centers = [element for i, element in enumerate(cluster_centers) if i not in [s_cluster_index]]

            superior_clusters = sorted(re_cluster_centers, key=(lambda p: np.array(p).dot(orientation.s)), reverse=True)[:2]
            inferior_clusters = sorted(re_cluster_centers, key=(lambda p: np.array(p).dot(-orientation.s)), reverse=True)[:2]

            process_label_ids["ASL"], asl_cluster_index = ShapeDecomposition.find_cluster_label_ids(labels, cluster_centers, superior_clusters, key=(lambda p: np.array(p).dot(orientation.r)))
            process_label_ids["ASR"], asr_cluster_index = ShapeDecomposition.find_cluster_label_ids(labels, cluster_centers, superior_clusters, key=(lambda p: np.array(p).dot(-orientation.r)))
            process_label_ids["AIL"], ail_cluster_index = ShapeDecomposition.find_cluster_label_ids(labels, cluster_centers, inferior_clusters, key=(lambda p: np.array(p).dot(orientation.r)))
            process_label_ids["AIR"], air_cluster_index = ShapeDecomposition.find_cluster_label_ids(labels, cluster_centers, inferior_clusters, key=(lambda p: np.array(p).dot(-orientation.r)))
            

            ######################### Find landmarks #################################################################################
            process_points = {}
            for name in process_label_ids.keys():
                process_points[name] = [polydata_points[i] for i in process_label_ids[name]]
        
            else:
                landmarks["S"]   = sorted(process_points["S"], key=(lambda p: np.array(p).dot(orientation.a)))[0]
                landmarks["ASL"] = sorted(process_points["ASL"], key=(lambda p: np.array(p).dot(orientation.s)))[-1]
                landmarks["ASR"] = sorted(process_points["ASR"], key=(lambda p: np.array(p).dot(orientation.s)))[-1]
                landmarks["AIL"] = sorted(process_points["AIL"], key=(lambda p: np.array(p).dot(np.average([-orientation.s, -orientation.r], axis=0))))[-1]
                landmarks["AIR"] = sorted(process_points["AIR"], key=(lambda p: np.array(p).dot(np.average([-orientation.s, orientation.r], axis=0))))[-1]
                landmarks["TL"]  = sorted(process_points["ASL"], key=(lambda p: np.array(p).dot(np.average([-orientation.a, -orientation.r], axis=0))))[-1]
                landmarks["TR"]  = sorted(process_points["ASR"], key=(lambda p: np.array(p).dot(np.average([-orientation.a, orientation.r], axis=0))))[-1]


        # cervical spine
        else:
            labels, cluster_centers = ShapeDecomposition.k_means(polydata_points, 3)

            ######################################### Find process clusters ################################################################

            process_label_ids["S"], s_cluster_index   = ShapeDecomposition.find_cluster_label_ids(labels, cluster_centers, cluster_centers, key=(lambda p: np.array(p).dot(orientation.a)))
            re_cluster_centers = [element for i, element in enumerate(cluster_centers) if i not in [s_cluster_index]]
            process_label_ids["L"], l_cluster_index = ShapeDecomposition.find_cluster_label_ids(labels, cluster_centers, re_cluster_centers, key=(lambda p: np.array(p).dot(orientation.r)))
            process_label_ids["R"], r_cluster_index = ShapeDecomposition.find_cluster_label_ids(labels, cluster_centers, re_cluster_centers, key=(lambda p: np.array(p).dot(-orientation.r)))
            

            ######################### Find landmarks #################################################################################
            process_points = {}
            for name in process_label_ids.keys():
                process_points[name] = [polydata_points[i] for i in process_label_ids[name]]


            # cervical spine
            if (index !=5):
                symmetry_intersection = conv.cut_plane(polydata, symmetry_plane.GetOrigin(), symmetry_plane.GetNormal())
                symmetry_intersection_points = numpy_support.vtk_to_numpy(symmetry_intersection.GetPoints().GetData())
                sorted_symmetry_intersection_points = conv.sorted_points(list(symmetry_intersection_points), orientation.a)

                landmarks["S"]   = sorted(sorted_symmetry_intersection_points, key=(lambda p: np.array(p).dot(orientation.a)))[0]
                landmarks["ASL"] = sorted(process_points["L"], key=(lambda p: np.array(p).dot(np.average([orientation.s, orientation.s, -orientation.r], axis=0))))[-1]
                landmarks["ASR"] = sorted(process_points["R"], key=(lambda p: np.array(p).dot(np.average([orientation.s, orientation.s, orientation.r], axis=0))))[-1]
                landmarks["AIL"] = sorted(process_points["L"], key=(lambda p: np.array(p).dot(orientation.s)))[0]
                landmarks["AIR"] = sorted(process_points["R"], key=(lambda p: np.array(p).dot(orientation.s)))[0]
                landmarks["TL"]  = sorted(process_points["L"], key=(lambda p: np.array(p).dot(orientation.a)))[-1]
                landmarks["TR"]  = sorted(process_points["R"], key=(lambda p: np.array(p).dot(orientation.a)))[-1]


        ############################# Filter polydata ###########################################################################
        process_polydatas = {}
        for name in process_label_ids.keys():
            process_polydatas[name] = conv.filter_point_ids(polydata, condition=lambda vertex: vertex not in process_label_ids[name])

        ############################## Add label Scalar to polydata ####################################################################
        new_labels = np.zeros(len(labels))
        process_names = list(process_label_ids.keys())
        for n in range(len(process_names)):
            for i in process_label_ids[process_names[n]]:
                new_labels[i] = n+1

        vtk_labels = vtk.vtkFloatArray()
        vtk_labels.SetNumberOfComponents(1)
        vtk_labels.SetName("Segments")
        for l in new_labels:
            vtk_labels.InsertNextValue(l)

        polydata.GetPointData().SetScalars(vtk_labels)
        ####################################################################################################################################


        return polydata, process_polydatas, process_label_ids, process_points, landmarks
    


    ''' Find IDs of the clusters '''
    def find_cluster_label_ids(labels, cluster_centers, sorting_cluster_centers, key):
        cluster = sorted(sorting_cluster_centers, key=key)[0]
        cluster_index = np.where(np.all(cluster_centers == cluster, axis=1))[0][0]
        label_ids = [i for i, x in enumerate(labels) if x == cluster_index]
        return label_ids, cluster_index
    
    ''' Calculate the distance of a point to a centerline curve.'''
    def centerline_distance(centerline, point):
        points = centerline.GetCurvePointsWorld()
        points = numpy_support.vtk_to_numpy(points.GetData())
        distances = np.linalg.norm(points - point, axis=1)
        return np.min(distances)

    ''' Segment the polydata based on the vertices distances to the centerlines.'''
    def centerline_segmentation(polydata, body, centerlines, index):
        polydata_points = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
        body_points = numpy_support.vtk_to_numpy(body.GetPoints().GetData())

        # create vtk labels
        vtk_labels = vtk.vtkFloatArray()
        vtk_labels.SetNumberOfComponents(1)
        vtk_labels.SetName("Centerline_Segments")

        # for each vertex, set a label
        for point in polydata_points:
            if point in body_points:
                l = -1 # vertebral body
            else:
                # get the closest centerline, set corresponding label
                centerline_distances = {name: ShapeDecomposition.centerline_distance(centerlines[name], point) for name in centerlines.keys()}
                closest_centerline = min(centerline_distances, key=centerline_distances.get)
                l = list(centerlines.keys()).index(closest_centerline)
            vtk_labels.InsertNextValue(l+1)

        # add labels to polydata
        polydata.GetPointData().SetScalars(vtk_labels)

        # get individual process polydata
        ids = {name: [i for i in range(polydata.GetNumberOfPoints()) if polydata.GetPointData().GetScalars().GetValue(i)-1 != list(centerlines.keys()).index(name)] for name in centerlines.keys()}
        process_polydatas = {name: conv.filter_point_ids(polydata, condition=lambda vertex: vertex in ids[name]) for name in centerlines.keys()}

        return polydata, process_polydatas

    

    ''' Extract centerline from polydata using the ExtractCenterline module.'''
    def centerline(polydata, startPoint, endPoints, index, name):
        
        pointMarkup = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', "Points")
        pointMarkup.GetDisplayNode().SetTextScale(0.0)
        pointMarkup.GetDisplayNode().SetSelectedColor(0, 0, 1)
        pointMarkup.AddControlPoint(startPoint)
        pointMarkup.AddControlPoint(endPoints)

        # access the ExtractCenterline logic
        extractLogic = ExtractCenterline.ExtractCenterlineLogic()
        targetNumberOfPoints = 5000.0
        decimationAggressiveness = 4
        subdivideInputSurface = False
        preprocessedPolyData = extractLogic.preprocess(polydata, targetNumberOfPoints, decimationAggressiveness, subdivideInputSurface)

        # Extract the centerline
        try:
            centerlineCurveNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", "Centerline curve"+str(name)+str(index))
            centerlineCurveNode.GetDisplayNode().SetTextScale(0.0)
            centerlinePolyData, voronoiDiagramPolyData = extractLogic.extractCenterline(preprocessedPolyData, pointMarkup)
            centerlinePropertiesTableNode = None
            extractLogic.createCurveTreeFromCenterline(centerlinePolyData, centerlineCurveNode, centerlinePropertiesTableNode)
            centerlineCurveNode.GetDisplayNode().SetVisibility(0)

        except:
            print("Centerline extraction failed")

        SpineLib.SlicerTools.removeNodes([pointMarkup])
        return centerlineCurveNode
    