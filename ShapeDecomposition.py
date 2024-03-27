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

print(f'{os.getenv('matplotlibback')}')

matplotlib.use(os.getenv('matplotlibback'))
import matplotlib.pyplot as plt
import SpineLib
import slicer
import ExtractCenterline


class ShapeDecomposition:   

    def __init__(self,
                 geometry:          vtk.vtkPolyData         = None,
                 center:            np.array                = None,
                 size:              SpineLib.Size           = None,
                 orientation:       SpineLib.Orientation    = None,
                 ) -> None:
        
        self.threshold, self.body, self.processes = ShapeDecomposition._pdf_decomposition(geometry, center, size, orientation)
        self.landmarks = ShapeDecomposition._landmarks(geometry, center, size, orientation, self.threshold)
        self.segmented_processes, self.process_landmarks = ShapeDecomposition._segment_processes(self.processes, self.landmarks, center, size, orientation)

    '''
    Segment the processes of the vertebra
    '''
    def _segment_processes(
            processes:         vtk.vtkPolyData         = None,
            landmarks:         Dict                    = None,
            center:            np.array                = None,
            size:              SpineLib.Size           = None,
            orientation:       SpineLib.Orientation    = None,
            ):
        
        left_pedicle_medial = landmarks["left_pedicle_medial"]
        right_pedicle_medial = landmarks["right_pedicle_medial"]
        curve = ShapeDecomposition.centerline(processes, left_pedicle_medial, right_pedicle_medial)
        sampledPoints = curve.GetCurvePointsWorld()
        sampledPoints = numpy_support.vtk_to_numpy(sampledPoints.GetData())
        canalPoints = [sampledPoints[i] for i in range(0, len(sampledPoints), len(sampledPoints)//4)]
        process_endpoints = {"TL":  left_pedicle_medial,
                                "ASL": canalPoints[0],
                                "AIL": canalPoints[1],
                                "S":   canalPoints[2],
                                "AIR": canalPoints[3],
                                "ASR": canalPoints[4],
                                "TR":  right_pedicle_medial}
        
        # segmentation of processes
        initial_segmented_polydata, process_label_ids, process_points, landmarks = ShapeDecomposition.clustering(processes, orientation, n_clusters=8)

        # centerlines
        centerlines = {"TL": [], "ASL": [], "AIL": [], "S": [], "AIR": [], "ASR": [], "TR": []}
        for name, point in landmarks.items():
            centerlines[name] = ShapeDecomposition.centerline(processes, point, process_endpoints[name])

        segmented_polydata = ShapeDecomposition.centerline_segmentation(initial_segmented_polydata, centerlines)
        
        return segmented_polydata, landmarks



    '''
    Find Spinal Canal landmarks
    '''
    def _landmarks(
            geometry:          vtk.vtkPolyData         = None,
            center:            np.array                = None,
            size:              SpineLib.Size           = None,
            orientation:       SpineLib.Orientation    = None,
            threshold:         float                   = None,
            ):
        
        landmarks = {}
        
        body_front = conv.get_intersection_points(geometry, center, (center + (orientation.a * size.depth)))
        canal_center = body_front - orientation.a * threshold

        left_geometry                       = conv.clip_plane(geometry, canal_center, -orientation.r)
        left_pedicle_intersection           = conv.cut_sphere(left_geometry, body_front, threshold)
        left_pedicle_intersection_points    = numpy_support.vtk_to_numpy(left_pedicle_intersection.GetPoints().GetData())
        left_pedicle_intersection_distances = np.linalg.norm(left_pedicle_intersection_points - canal_center, axis=1)
        left_pedicle_medial_point           = left_pedicle_intersection_points[np.argmin(left_pedicle_intersection_distances)]
        left_pedicle_com                    = np.mean(left_pedicle_intersection_points, axis=0)

        right_geometry                      = conv.clip_plane(geometry, canal_center, orientation.r)
        right_pedicle_intersection          = conv.cut_sphere(right_geometry, body_front, threshold)
        right_pedicle_intersection_points   = numpy_support.vtk_to_numpy(right_pedicle_intersection.GetPoints().GetData())
        right_pedicle_intersection_distances= np.linalg.norm(right_pedicle_intersection_points - canal_center, axis=1)
        right_pedicle_medial_point          = right_pedicle_intersection_points[np.argmin(right_pedicle_intersection_distances)]
        right_pedicle_com                   = np.mean(right_pedicle_intersection_points, axis=0)

        landmarks['canal_center']           = canal_center
        landmarks['body_front']             = body_front
        landmarks['left_pedicle_medial']    = left_pedicle_medial_point
        landmarks['right_pedicle_medial']   = right_pedicle_medial_point
        landmarks['left_pedicle_com']       = left_pedicle_com
        landmarks['right_pedicle_com']      = right_pedicle_com

        return landmarks
    

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
            ):

        landmark = conv.get_intersection_points(geometry, center, (center + (orientation.a * size.depth)))
        points = numpy_support.vtk_to_numpy(geometry.GetPoints().GetData())
        
        distances = np.linalg.norm(points - landmark, axis=1)
        #distances = np.linalg.norm(points - center, axis=1)

        # Calculate the KDE
        kde = sns.kdeplot(distances, color='skyblue', legend=False)
        x_vals = kde.get_lines()[0].get_xdata()
        y_vals = kde.get_lines()[0].get_ydata()
        
        # Find extrema points (peaks and valleys)
        peaks, _ = find_peaks(y_vals)
        valleys = argrelextrema(y_vals, np.less)[0]

        # Calculate the second derivative of the KDE
        second_derivative = np.gradient(np.gradient(y_vals, x_vals), x_vals)
        inflection_points = np.where(np.diff(np.sign(second_derivative)))[0]
        downward_inflection_points = [point for point in inflection_points if y_vals[point] > y_vals[point + 1]]
        upward_inflection_points = [point for point in inflection_points if y_vals[point] < y_vals[point + 1]]

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
        plt.scatter(x_vals[downward_inflection_points], y_vals[downward_inflection_points], color='red', marker='o', label='Downward Inflection Points')
        plt.scatter(x_vals[upward_inflection_points], y_vals[upward_inflection_points], color='blue', marker='x', label='Upward Inflection Points')
        plt.scatter(x_vals[minima_points], y_vals[minima_points], color='orange', marker='o', label='Minima Points')
        plt.scatter(x_vals[maxima_points], y_vals[maxima_points], color='purple', marker='x', label='Maxima Points')
        if lowest_min_between_maxima is not None:
            plt.scatter(x_vals[lowest_min_between_maxima], y_vals[lowest_min_between_maxima], marker='s', color='magenta', s=80, label='Pre-Max Minimum')
        plt.legend()

        #plt.show(block=False)
        #plt.show()
            
        threshold = x_vals[lowest_min_between_maxima]
        # threshold = x_vals[minima_points[0]]
        # threshold = x_vals[downward_inflection_points[0]]
        plt.clf()

        vertebral_body = conv.filter_point_ids(geometry, condition=lambda vertex: distances[vertex] > threshold)
        processes = conv.filter_point_ids(geometry, condition=lambda vertex: distances[vertex] < threshold)
        
        return threshold, vertebral_body, processes
    


    def centerline(polydata, startPoint, endPoints):
        
        pointMarkup = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', "Points")
        pointMarkup.AddControlPoint(startPoint)
        pointMarkup.AddControlPoint(endPoints)

        extractLogic = ExtractCenterline.ExtractCenterlineLogic()
        targetNumberOfPoints = 5000.0
        decimationAggressiveness = 4 # I had to lower this to 3.5 in at least one case to get it to work, 4 is the default in the module
        subdivideInputSurface = False
        preprocessedPolyData = extractLogic.preprocess(polydata, targetNumberOfPoints, decimationAggressiveness, subdivideInputSurface)

        # Extract the centerline
        try:
            centerlineCurveNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", "Centerline curve")
            centerlineCurveNode.GetDisplayNode().SetTextScale(0.0)
            centerlinePolyData, voronoiDiagramPolyData = extractLogic.extractCenterline(preprocessedPolyData, pointMarkup)
            centerlinePropertiesTableNode = None
            extractLogic.createCurveTreeFromCenterline(centerlinePolyData, centerlineCurveNode, centerlinePropertiesTableNode)
        except:
            print("Centerline extraction failed")

        SpineLib.SlicerTools.removeNodes([pointMarkup])
        return centerlineCurveNode
    

    def clustering(polydata, orientation, n_clusters=8):

        ######################################### CLustering with k means ##############################################################
        polydata_points = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(polydata_points)
        cluster_points = {}
        for label in range(n_clusters):
            cluster_points[label] = [polydata_points[i] for i, l in enumerate(labels) if l == label]
        cluster_centers = kmeans.cluster_centers_


        ######################################### Find process clusters ################################################################
        process_label_ids = {"TL": [], "ASL": [], "AIL": [], "S": [], "AIR": [], "ASR": [], "TR": []}

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


        ######################### Find landmarks #################################################################################
        process_points = {}
        for name in process_label_ids.keys():
            process_points[name] = [polydata_points[i] for i in process_label_ids[name]]

        landmarks = {"TL": [], "ASL": [], "AIL": [], "S": [], "AIR": [], "ASR": [], "TR": []}
        landmarks["ASL"] = sorted(process_points["ASL"], key=(lambda p: np.array(p).dot(np.average([orientation.s, -orientation.a, -orientation.r], axis=0))))[-1]
        #landmarks["ASL"] = sorted(process_points["ASL"], key=(lambda p: np.array(p).dot(np.average([orientation.s], axis=0))))[-1]
        landmarks["ASR"] = sorted(process_points["ASR"], key=(lambda p: np.array(p).dot(np.average([orientation.s, -orientation.a, orientation.r], axis=0))))[-1]
        #landmarks["ASR"] = sorted(process_points["ASR"], key=(lambda p: np.array(p).dot(np.average([orientation.s], axis=0))))[-1]
        landmarks["AIL"] = sorted(process_points["AIL"], key=(lambda p: np.array(p).dot(np.average([-orientation.s, -orientation.a, -orientation.r], axis=0))))[-1]
        landmarks["AIR"] = sorted(process_points["AIR"], key=(lambda p: np.array(p).dot(np.average([-orientation.s, -orientation.a, orientation.r], axis=0))))[-1]
        landmarks["S"]   = sorted(process_points["S"], key=(lambda p: np.array(p).dot(orientation.a)))[0]
        landmarks["TL"]  = sorted(process_points["TL"], key=(lambda p: np.array(p).dot(orientation.r)))[0]
        landmarks["TR"]  = sorted(process_points["TR"], key=(lambda p: np.array(p).dot(orientation.r)))[-1]

        #return dijkstraPoints
        return polydata, process_label_ids, process_points, landmarks
    
    def find_cluster_label_ids(labels, cluster_centers, sorting_cluster_centers, key):
        cluster = sorted(sorting_cluster_centers, key=key)[0]
        cluster_index = np.where(np.all(cluster_centers == cluster, axis=1))[0][0]
        label_ids = [i for i, x in enumerate(labels) if x == cluster_index]
        return label_ids, cluster_index
    

    def centerline_distance(centerline, point):
        points = centerline.GetCurvePointsWorld()
        points = numpy_support.vtk_to_numpy(points.GetData())
        distances = np.linalg.norm(points - point, axis=1)
        return np.min(distances)
    
    def centerline_segmentation(polydata, centerlines):
        polydata_points = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())

        vtk_labels = vtk.vtkFloatArray()
        vtk_labels.SetNumberOfComponents(1)
        vtk_labels.SetName("Centerline_Segments")

        for point in polydata_points:
            centerline_distances = {name: ShapeDecomposition.centerline_distance(centerlines[name], point) for name in centerlines.keys()}
            closest_centerline = min(centerline_distances, key=centerline_distances.get)
            l = list(centerlines.keys()).index(closest_centerline)
            vtk_labels.InsertNextValue(l)

        polydata.GetPointData().SetScalars(vtk_labels)

        return polydata