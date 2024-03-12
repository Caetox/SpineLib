from dataclasses import dataclass
import vtk
import numpy as np
import vtk_convenience as conv
from scipy.interpolate import PchipInterpolator
import SpineLib


class UpApproximator:
    """
    For a set of vertebra geoemtries, guess the most probable up-vector.
    This is achieved by calcuating the Pchip interpolation through all vertebra geometries' centers
    of mass (COM). The up-vector is then approximated by a secant along this Pchip curve.

    Usage:
        # create list of vertebra geometries
        vertebrae_stl_file_list = ["L1.stl", "L2.stl", "T12.stl"]
        vertebrae_geo = [load_stl(stl) for stl in vertebrae_stl_file_list]

        # initialize the approximator
        approximator = UpApproximator(vertebrae_geo)

        # query up-vector at any spot (here: center of mass for T12)
        com_T12 = calc_center_of_mass(vertebrae_geo[2])
        local_up = approximator(com_T12)
    """

    def __init__(self, geometries: vtk.vtkPolyData) -> None:

        centers_of_mass = np.array([np.array(conv.calc_center_of_mass(g)) for g in geometries])

        self.most_significant_column = self.column_with_widest_spread(centers_of_mass)
        centers_of_mass = self.sort_by_column(
            centers_of_mass, column=self.most_significant_column
        )

        interpolator = PchipInterpolator(
            centers_of_mass[:, self.most_significant_column],
            centers_of_mass,
            extrapolate=True,
        )
        self.derivative = interpolator.derivative()

    @classmethod
    def column_with_widest_spread(cls, array: np.ndarray) -> int:
        """
        For a two-dimensional array, find the column with the widest range in value.
        """
        max_positions = np.amax(array, axis=0)
        min_positions = np.amin(array, axis=0)
        spread = max_positions - min_positions
        return spread.tolist().index(max(spread))

    @classmethod
    def sort_by_column(cls, array: np.ndarray, column: int) -> np.ndarray:
        """
        For a two-dimensional array, select a column to sort the array by.
        """
        array = array.copy()
        return array[array[:, column].argsort()]

    def __call__(self, position: np.ndarray) -> np.ndarray:
        """
        Get the most probably up-vector for a 3D position.
        The approximator only takes one axis into consideration, height.
        The height is assumed to be the axis with the widest range in value.

        The result is calculated by a secant, with the two intersections being
        distance 'window_size' apart..

        Keyword arguments:
            position: numpy.ndarray to calculate the local up-vector for.
            window_size: the distance of secant intersections.
        """
        return conv.normalize(
            self.derivative((position[self.most_significant_column],))[0]
        )
