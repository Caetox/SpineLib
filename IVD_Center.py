import slicer
import SegmentStatistics
import vtk
import numpy as np

import SpineLib
import vtk_convenience as conv


# Find IVD centers

class IVD_Center:

    def create_IVD_centers(curve, geometries):

        ivd_centers = []

        intersection_points_list = []
        curve.SetNumberOfPointsPerInterpolatingSegment(50)
        curve_points = curve.GetCurvePointsWorld()

        # calulate intersection points
        for g in geometries:
            intersection_points = conv.get_curve_intersection_points(g, curve)
            intersection_points_list.append(intersection_points)

        # Compute IVD centers for pairs of consecutive geometries
        for i in range(len(intersection_points_list) - 1):

            last_intersection_point = intersection_points_list[i][-1]
            first_intersection_point = intersection_points_list[i + 1][0]

            approx_disc_center = np.average([last_intersection_point, first_intersection_point], axis=0)
            disc_center_id = curve.GetClosestCurvePointIndexToPositionWorld(approx_disc_center)
            disc_center = [0, 0, 0]
            curve_points.GetPoint(disc_center_id, disc_center)
            ivd_centers.append(disc_center)
        
        return ivd_centers