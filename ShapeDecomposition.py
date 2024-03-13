from dataclasses import dataclass
import numpy as np
import vtk
import vtk_convenience as conv
from vtk.util import numpy_support
from typing import Dict, Tuple
import seaborn as sns
from scipy.signal import find_peaks, argrelextrema
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('WXAgg')
import matplotlib.pyplot as plt
import SpineLib
import slicer


class ShapeDecomposition:

    def __init__(self,
                 geometry:          vtk.vtkPolyData         = None,
                 center:            np.array                = None,
                 size:              SpineLib.Size           = None,
                 orientation:       SpineLib.Orientation    = None,
                 ) -> None:
        
        self.threshold, self.body, self.processes = ShapeDecomposition._pdf_decomposition(geometry, center, size, orientation)



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

        plt.show(block=False)
        #plt.show()
            
        threshold = x_vals[lowest_min_between_maxima]
        # threshold = x_vals[minima_points[0]]
        # threshold = x_vals[downward_inflection_points[0]]
        plt.clf()

        vertebral_body = conv.filter_point_ids(
            geometry, condition=lambda vertex: distances[vertex] > threshold)
        
        processes = conv.filter_point_ids(
            geometry, condition=lambda vertex: distances[vertex] < threshold)

        return threshold, vertebral_body, processes