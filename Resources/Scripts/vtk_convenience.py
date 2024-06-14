"""
Convenience functions to quickly prototype vtk applications.
These methods rely heavily on vtkPolyData objects. They make
no use of the piping abilities to construct larger pipelines
for efficient calculations.

If time is critical factor, better not use these or extend on
them.
"""
# TODO: add function descriptions to module docstring
from typing import Union, Generator, Tuple, List, Callable
from math import cos, radians
import numpy as np
import pyvista as pv
import pyacvd
from vtk.util import numpy_support
import pyvista

# pylint: disable=no-name-in-module
from vtk import (
    vtkAlgorithm,
    vtkPolyData,
    vtkSTLReader,
    vtkOBJWriter,
    vtkPolyDataAlgorithm,
    vtkPolyDataMapper,
    vtkActor,
    vtkDataArray,
    vtkPolyDataNormals,
    vtkRemovePolyData,
    vtkIdTypeArray,
    vtkOBBTree,
    vtkCleanPolyData,
    vtkPlane,
    vtkSphere,
    vtkClipPolyData,
    vtkCellLocator,
    vtkCutter,
    vtkCenterOfMass,
    vtkAbstractPolyDataReader,
    vtkOBJReader,
    vtkBoundingBox,
    vtkPoints,
    vtkPointSet,
    vtkAxisActor,
    vtkDijkstraGraphGeodesicPath,
    vtkAppendFilter,
    vtkPointLocator,
    vtkMath,
    vtkPolyDataConnectivityFilter,
    vtkTriangleFilter,
    vtkFillHolesFilter,
    vtkAppendPolyData,
    vtkGeometryFilter,
    vtkDelaunay3D,
    vtkGenericCell,
    vtkFeatureEdges,
    vtkIdFilter,
    vtkTransformPolyDataFilter,
    vtkTransform,
    vtkSmoothPolyDataFilter,
    vtkWindowedSincPolyDataFilter,
    mutable,
)
from numpy import zeros, array, dot, ndarray
from numpy.linalg import norm

VtkAlgorithmOrPolyData = Union[vtkAlgorithm, vtkPolyData]
Tuple3Float = Tuple[float, float, float]
Vector3D = List[float]
OBBType = Tuple[Vector3D, Vector3D, Vector3D, Vector3D]


class Orientation:
    """
    Contain an orientation in 3D space. It consists of one array
    per dimension and a center of mass.
    """

    LATERAL_AXIS = 1, 0, 0
    LONGITUDINAL_AXIS = 0, 1, 0
    SAGITTAL_AXIS = 0, 0, 1


    def __init__(
        self,
        vertebra_body: vtkPolyData,
        lateral_axis=LATERAL_AXIS,
        longitudinal_axis=LONGITUDINAL_AXIS,
        sagittal_axis=SAGITTAL_AXIS,
    ):
        """
        Constructor to calculate the orientation based on a given
        coordinate system.

        Keyword Arguments:
        vertebra_body -- vtk mesh to assign this Orientation to
        lateral_axis -- global sideways axis as 3d tuple
        longitudinal_axis -- global upwards axis as 3d tuple
        sagittal_axis -- global front axis as 3d tuple
        """
        self.center_of_mass = array(calc_center_of_mass(vertebra_body))
        oriented_bounding_box = calc_obb(vertebra_body)[1:]
        oriented_bounding_box_lvl_3 = calc_obb_geometry(vertebra_body, level=3)
        oriented_bounding_box_lvl_3_normals = list(
            iter_normals(oriented_bounding_box_lvl_3)
        )

        lateral_vector, lateral_flip = closest_vector(
            oriented_bounding_box, lateral_axis
        )
        sagittal_vector, sagittal_flip = closest_vector(
            oriented_bounding_box, sagittal_axis
        )
        longitudinal_vector, longitudinal_flip = closest_vector(
            oriented_bounding_box_lvl_3_normals, longitudinal_axis
        )

        self.lateral_vector = normalize(lateral_flip * array(lateral_vector))
        self.sagittal_vector = normalize(sagittal_flip * array(sagittal_vector))
        self.longitudinal_vector = normalize(longitudinal_flip * array(longitudinal_vector))

        self.longitudinal_vector = normalize(array(longitudinal_axis))

    @property
    def up_vector(self) -> vtkAxisActor:
        axis_actor = line_actor(self.center_of_mass, self.center_of_mass + 100. * self.longitudinal_vector)
        axis_actor.SetAxisTypeToY()
        axis_actor.GetAxisLinesProperty().BackfaceCullingOff()
        return axis_actor

def line_actor(point1: ndarray, point2:ndarray) -> vtkAxisActor:
    line_actor = vtkAxisActor()
    line_actor.SetPoint1(point1)
    line_actor.SetPoint2(point2)
    line_actor.GetAxisLinesProperty().SetLineWidth(3)
    return line_actor

def closest_vector(
    vector_list: List[Tuple3Float], vector_to_match: Tuple3Float
) -> Tuple[Tuple3Float, int]:
    """
    Return vector from "vector_list" that is aligned the most with
    "vector_to_match". Returns scalar [-1|1] as indication if closest
    vector and "vector_to_match" point in similar or opposing directions.
    """
    vector_to_match = normalize(array(vector_to_match))
    normalized_vectors = [normalize(array(vector)) for vector in vector_list]

    vector_similarity = [
        dot(vector, vector_to_match) for vector in normalized_vectors
    ]
    vector_similarity_dict = dict(zip(vector_similarity, vector_list))
    greatest_similarity = sorted(vector_similarity_dict, key=abs)[-1]

    return vector_similarity_dict[greatest_similarity], round(greatest_similarity)


def normalize(vector: np.ndarray) -> np.ndarray:
    """Return vector in the direction of the input parameter of length 1."""
    length = norm(vector)
    if length == 0:
        print("Warning: Attempting to normalize a zero vector:", vector)
        return vector
    return vector / length


def eliminate_misaligned_faces(
    polydata: vtkPolyData, center: np.array, direction: ndarray, max_angle: float
) -> vtkPolyData:
    """
    Return vtkPolyData object where all faces facing perpendicular to
    a direction vector are deleted.

    Keyword Arguemts
    polydata - geometry to be modified
    direction - numpy vector, faces pointing in line or opposite direction
    are being kept
    max_angle - all faces with normals more than max_angle diverging from
    'direction' are deleted
    """
    normals = list(iter_normals(polydata))
    normals = array(normals)
    direction = normalize(direction)
    projection = dot(normals, direction)
    perpendicular = np.cross(normals, direction)
    orientation = np.dot(perpendicular, direction)

    max_cos = cos(radians(max_angle))

    cleaned_polydata = filter_point_ids(
        polydata,
        condition=lambda vertex: projection[vertex] < max_cos
    )
    return cleaned_polydata


def cut_plane(
    polydata: vtkPolyData, plane_origin: Tuple3Float, plane_normal: Tuple3Float
) -> vtkPolyData:
    """
    Return vtkPolyData object where each vertex marks an intersection
    between 'polydata's edges and a plane.

    Keyword Arguments:
    polydata - vtk geometry to be intersected
    plane_origin - some point on the cutting plane
    plane_normal - orientation of the cutting plane
    """
    plane = vtkPlane()
    plane.SetOrigin(plane_origin)
    plane.SetNormal(plane_normal)

    cutter = vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(polydata)
    cutter.Update()

    return cutter.GetOutput()

def cut_sphere(
    polydata: vtkPolyData, sphere_origin: Tuple3Float, sphere_radius: Tuple3Float
) -> vtkPolyData:
    """
    Return vtkPolyData object where each vertex marks an intersection
    between 'polydata's edges and a sphere.

    Keyword Arguments:
    polydata - vtk geometry to be intersected
    sphere_origin - center of the cutting sphere
    sphere_radius - radius of the cutting sphere
    """
    sphere = vtkSphere()
    sphere.SetCenter(sphere_origin)
    sphere.SetRadius(sphere_radius)

    cutter = vtkCutter()
    cutter.SetCutFunction(sphere)
    cutter.SetInputData(polydata)
    cutter.Update()

    return cutter.GetOutput()


def calc_center_of_mass(polydata: vtkPolyData) -> Tuple3Float:
    """Return the center of mass of a vtk geometry as float tuple."""
    center_of_mass = vtkCenterOfMass()
    center_of_mass.SetInputData(polydata)
    center_of_mass.SetUseScalarsAsWeights(False)
    center_of_mass.Update()

    return center_of_mass.GetCenter()


def clip_plane(
    polydata: vtkPolyData, plane_origin: Tuple3Float, plane_normal: Tuple3Float
) -> vtkPolyData:
    """
    Return geometry "polydata" with all points below the plane removed.

    Keyword Arguments:
    polydata - vtk geometry to be clipped
    plane_origin - some point on the clipping plane
    plane_normal - orientation of the plane and direction on where vertices
    will be removed
    """
    plane = vtkPlane()
    plane.SetOrigin(*plane_origin)
    plane.SetNormal(*plane_normal)

    clip = vtkClipPolyData()
    clip.SetInputData(polydata)
    clip.SetClipFunction(plane)
    clip.Update()
    return clip.GetOutput()


def clip_sphere(
    polydata: vtkPolyData, sphere_origin: Tuple3Float, sphere_radius: Tuple3Float, InsideOut = False
) -> vtkPolyData:
    """
    Return geometry "polydata" with all points below the plane removed.

    Keyword Arguments:
    polydata - vtk geometry to be clipped
    plane_origin - some point on the clipping plane
    plane_normal - orientation of the plane and direction on where vertices
    will be removed
    """
    sphere = vtkSphere()
    sphere.SetCenter(sphere_origin)
    sphere.SetRadius(sphere_radius)

    clip = vtkClipPolyData()
    clip.SetInputData(polydata)
    clip.SetClipFunction(sphere)
    clip.SetInsideOut(InsideOut)
    clip.Update()
    return clip.GetOutput()


def composite_center(polydatas: List[vtkPolyData]):
    all_points = vtkPoints()
    for poly in polydatas:
        all_points.InsertPoints(all_points.GetNumberOfPoints(), poly.GetNumberOfPoints(), 0, poly.GetPoints())
    
    composite = vtkPointSet()
    composite.SetPoints(all_points)
    return calc_center_of_mass(composite)

    
def calc_bb(polydata: vtkPolyData) -> vtkBoundingBox:
    """
    Return min and max point of an axis-aligned bounding box for
    the given polydata object
    """
    bounding_box = vtkBoundingBox()
    for id_ in range(polydata.GetNumberOfPoints()):
        bounding_box.AddPoint(polydata.GetPoint(id_))
    return bounding_box.GetMinPoint(), bounding_box.GetMaxPoint()


def _calc_obb(polydata: vtkPolyData, max_level: int) -> vtkOBBTree:
    """
    Return oriented bounding box trees main and sublevel boxes as vtkOBBTree.

    Keyword Arguments:
        polydata -- geometry to build bounding box tree
        max_level -- deepest level to calculate for the OBB tree;
          each level splits the above box in half along the
          currently most significant axis
    """
    obb_tree = vtkOBBTree()
    obb_tree.SetDataSet(polydata)
    obb_tree.SetMaxLevel(max_level)
    obb_tree.BuildLocator()

    return obb_tree


def calc_obb(polydata: vtkPolyData) -> OBBType:
    """
    Return oriented bounding box as a tuple (corner, vector1, vector2, vector3,).
    """
    obb = zeros((5, 3))
    obb_tree = _calc_obb(polydata, max_level=0)
    obb_tree.ComputeOBB(polydata, *[obb[i] for i in range(5)])
    obb_data = tuple(obb[i].tolist() for i in range(4))

    return tuple(obb[i].tolist() for i in range(4))


def calc_obb_geometry(polydata: vtkPolyData, level: int = 0) -> vtkPolyData:
    """
    Return oriented bounding box tree's selected sublevel boxes as vtkPolyData
    geometries.

    Keyword Arguments:
        polydata -- geometry to build bounding box tree
        level -- level of the OBB tree to inspect;
          each level splits the above box in half along
          the currently most significant axis
    """
    geometry = vtkPolyData()
    obb_tree = _calc_obb(polydata, max_level=level)
    obb_tree.GenerateRepresentation(level, geometry)
    return geometry


def delete_points(polydata: vtkPolyData, ids: vtkIdTypeArray) -> vtkPolyData:
    """Return geometry with the points given by "ids" eliminated."""
    remove_filter = vtkRemovePolyData()
    remove_filter.SetInputData(polydata)
    remove_filter.SetPointIds(ids)
    remove_filter.Update()
    output = remove_filter.GetOutput()

    cleaned_polydata = polydata_clean(output, tolerance=0.001)

    return cleaned_polydata

def polydata_clean(polydata, tolerance=0.001):
    cleaner = vtkCleanPolyData()
    cleaner.SetInputData(polydata)
    cleaner.SetTolerance(tolerance)
    cleaner.PointMergingOn()
    cleaner.Update()

    return cleaner.GetOutput()


def filter_point_ids(
    polydata: vtkPolyData, condition: Callable[[int], bool]
) -> vtkPolyData:
    """
    Return geometry with only the points fullfilling "condition" method remaining.

    Keyword Arguments:
    polydata - vtk geometry
    condition - unary function returning boolean value. It is passed a point id
    to identify a point from polydata dataset.
    """
    remove_ids = vtkIdTypeArray()
    for id_ in range(polydata.GetNumberOfPoints()):
        if condition(id_):
            remove_ids.InsertNextTuple1(id_)
    filtered_polydata = delete_points(polydata, remove_ids)
    filtered_polydata.Modified()
    cleaned_polydata = polydata_clean(filtered_polydata)

    return cleaned_polydata

def filter_points(
    polydata: vtkPolyData, condition: Callable[[int], bool]
) -> vtkPolyData:
    remove_ids = vtkIdTypeArray()
    for id_ in range(polydata.GetNumberOfPoints()):
        if not condition(polydata.GetPoint(id_)):
            remove_ids.InsertNextTuple1(id_)
    filtered_polydata = delete_points(polydata, remove_ids)
    filtered_polydata.Modified()
    cleaned_polydata = polydata_clean(filtered_polydata)

    return cleaned_polydata


def _calc_normals(polydata: vtkPolyData) -> vtkDataArray:
    """Return normals for all vertices of a vtk geometry."""
    normals = vtkPolyDataNormals()
    normals.SetInputData(polydata)
    normals.SplittingOff()
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    normals.AutoOrientNormalsOn()
    normals.Update()
    return normals.GetOutput().GetPointData().GetNormals()


def iter_points(polydata: vtkPolyData) -> Generator[Tuple3Float, None, None]:
    """Return generator over all vertices as tuple(x, y, z)."""
    for point_id in range(polydata.GetNumberOfPoints()):
        yield polydata.GetPoint(point_id)


def iter_normals(polydata: vtkPolyData) -> Generator[Tuple3Float, None, None]:
    """Return generator over all vertice's normals as tuple(n_x, n_y, n_z)."""
    normals = _calc_normals(polydata)
    for normal_id in range(normals.GetNumberOfTuples()):
        yield normals.GetTuple(normal_id)


def _load_geometry(filename: str, reader: vtkAbstractPolyDataReader) -> vtkPolyData:
    """Load the given poly data file, and return a vtkPolyData object for it."""
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def load_stl(filename: str) -> vtkPolyData:
    """Load the given STL file, and return a vtkPolyData object for it."""
    return _load_geometry(filename, reader=vtkSTLReader())


def load_obj(filename: str) -> vtkPolyData:
    """Load the given STL file, and return a vtkPolyData object for it."""
    return _load_geometry(filename, reader=vtkOBJReader())


def write_obj(polydata: vtkPolyData, filename: str) -> None:
    """Write a PolyDataAlgorithm objects poly data representation to an obj file."""
    writer = vtkOBJWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Update()


def _make_actor(
    input_method: str, input_data: VtkAlgorithmOrPolyData, **properties
) -> vtkActor:
    """
    Return for a given input a vtkActor with properties as specified.
    Function will create a vtkMapper object for you under the hood.

    Keyword Arguments:
    input_method -- available input accessor of vtkPolyDataMapper as STR
      ("SetInputData" or "SetInputConnection")
    input_data -- data to map and act upon (vtkAlgorithm or vtkPolyData)
    properties -- keyword arguments that have their dedicated "Set..."
      accessor function in vtkProperty (i.e. "LineWidth" or "lineWidth" for
      invoking vtkProperty.SetLineWidth). Must obey camel case lettering

    Sample Call:
        cube = vtkCubeSource()
        cube_actor = _make_actor("SetInputConnection", cube.GetOutputPort(), opacity=0.5)

        # OR

        line = vtkLine()
        line_actor = _make_actor("SetInputData", line, opacity=0.5)
    """
    mapper = vtkPolyDataMapper()
    getattr(mapper, input_method)(input_data)

    actor = vtkActor()
    actor_properties = actor.GetProperty()
    for property_, value in properties.items():
        accessor_func_name = f"Set{property_[0:1].capitalize()}{property_[1:]}"
        if value is None:
            getattr(actor_properties, accessor_func_name)()
        else:
            getattr(actor_properties, accessor_func_name)(value)
    actor.SetMapper(mapper)
    return actor


def make_polydata_actor(polydata: vtkPolyData, **properties):
    """Return vtkActor for vtkPolyData. See _make_actor."""
    return _make_actor("SetInputData", polydata, **properties)


def make_algorithm_actor(algorithm: vtkAlgorithm, **properties):
    """Return vtkActor for vtkAlgorithm. See _make_actor."""
    return _make_actor("SetInputConnection", algorithm.GetOutputPort(), **properties)


def calc_main_component(geometry):
    """
    Return the main component of singular value decomposition through
    all vertices of a vtkPolyData object or a list of points.
    """
    if isinstance(geometry, vtkPolyData):
        points = list(iter_points(geometry))
    elif isinstance(geometry, list):
        points = geometry
    else:
        raise ValueError("Input geometry must be either vtkPolyData or a list of points")

    points = np.array(points)
    mean = points.mean(axis=0)
    _1, _2, eigenvector = np.linalg.svd(points - mean)

    return eigenvector[0]


def sorted_points(geometry, main_axis):
    """
    Sort points of a vtkPolyData object or a list of points along a specified main axis.
    """
    if isinstance(geometry, vtkPolyData):
        points = [geometry.GetPoint(id_) for id_ in range(geometry.GetNumberOfPoints())]
    elif isinstance(geometry, list):
        points = geometry
    else:
        raise ValueError("Input geometry must be either vtkPolyData or a list of points")

    main_axis = np.array(main_axis)
    projection = lambda p: np.array(p).dot(main_axis)

    return sorted(points, key=projection)

def get_intersection_points(polydata, point1, point2):

    # Create a locator for the model
    locator = vtkCellLocator()
    locator.SetDataSet(polydata)
    locator.BuildLocator()

    # Initialize intersection points
    intersection_point = None
    t = mutable(0)
    x = [0.0, 0.0, 0.0]
    pcoords = [0.0, 0.0, 0.0]
    subId = mutable(0)

    # Check for intersection between the line segment and the model
    intersect = locator.IntersectWithLine(point1, point2, 0.001, t, x, pcoords, subId)
    if intersect:
        intersection_point = x

    return intersection_point

def get_curve_intersection_points(model, curve_node):
    # Get model and curve objects
    curve = curve_node.GetCurvePointsWorld()

    # Create a locator for the model
    locator = vtkCellLocator()
    locator.SetDataSet(model)
    locator.BuildLocator()

    # Initialize intersection points
    intersection_points = []

    # Iterate through points in the curve
    for i in range(curve.GetNumberOfPoints() - 1):
        # Get two consecutive points on the curve
        point1 = [0, 0, 0]
        point2 = [0, 0, 0]
        curve.GetPoint(i, point1)
        curve.GetPoint(i + 1, point2)

        # Initialize intersection point and parameters
        t = mutable(0)
        x = [0.0, 0.0, 0.0]
        pcoords = [0.0, 0.0, 0.0]
        subId = mutable(0)

        # Check for intersection between the line segment and the model
        intersect = locator.IntersectWithLine(point1, point2, 0.001, t, x, pcoords, subId)
        if intersect:
            intersection_points.append(x)

    return intersection_points


def pca_eigenvectors(points):
    # Compute the covariance matrix and principal components without centering
    Cov = np.cov(points.T)
    eigval, eigvect = np.linalg.eig(Cov.T)

    return eigvect.T, eigval.T


def voxelization(geometry: vtkPolyData, factor):
    bounds = geometry.GetBounds()
    diagonal_length = ((bounds[1] - bounds[0])**2 + (bounds[3] - bounds[2])**2 + (bounds[5] - bounds[4])**2)**0.5
    density = diagonal_length / factor
    voxels = pv.voxelize(geometry, density=density)
    return voxels.points

'''
Find id of the closest vertex to given point
'''
def find_closest_point_id(polydata, point):
    # Create a locator for the model
    locator = vtkPointLocator()
    locator.SetDataSet(polydata)
    locator.BuildLocator()

    # Find the closest point on the model
    closestPointId = locator.FindClosestPoint(point)

    return closestPointId

def find_closest_points(polydata, points):
    py_mesh = pyvista.PolyData(polydata)
    closest_cells, closest_points = py_mesh.find_closest_cell(points, return_closest_point=True)

    return closest_points


def runDijkstra(polydata, point1, point2):

    points = vtkPoints()
    closestPointId = find_closest_point_id(polydata, point1)
    closestPointId1 = find_closest_point_id(polydata, point2)

    appendFilter = vtkAppendFilter()
    appendFilter.MergePointsOn()
        
    #create geodesic path: vtkDijkstraGraphGeodesicPath
    dijkstra = vtkDijkstraGraphGeodesicPath()
    dijkstra.SetInputData(polydata)
    dijkstra.SetStartVertex(closestPointId)
    dijkstra.SetEndVertex(closestPointId1)
    dijkstra.Update()
                    
    pts = dijkstra.GetOutput().GetPoints()

    for i in range(pts.GetNumberOfPoints()):
        p = [0,0,0]
        pts.GetPoint(i,p)
        points.InsertNextPoint(p)
    
    return points


def calcSagittalAngles(vectors):

    angles = []
    for index in range(0,len(vectors)-1):
        v1 = np.array(vectors[index])
        v2 = np.array(vectors[index+1])
        v1[0] = 0.0
        v2[0] = 0.0
        crossProduct = np.cross(v1, v2)
        dotProduct = np.dot(crossProduct, [-1.0, 0.0, 0.0])
        angleRad = vtkMath.AngleBetweenVectors(v1, v2)
        if (dotProduct >= 0): angleRad = np.negative(angleRad)
        angleDeg = np.round(vtkMath.DegreesFromRadians(angleRad),2)
        angles.append(angleDeg)

    return angles

def filterLargestRegion(polydata):

    connectivity_filter = vtkPolyDataConnectivityFilter()
    connectivity_filter.SetInputData(polydata)
    connectivity_filter.SetExtractionModeToLargestRegion()
    connectivity_filter.Update()

    return connectivity_filter.GetOutput()


'''
Fit a plane to a numpy array of points
'''
def fitPlane(points):
    # check i points is numpy array


    if isinstance(points, ndarray) or isinstance(points, list):
        vtk_points = vtkPoints()
        for point in points:
            vtk_points.InsertNextPoint(point)
    elif isinstance(points, vtkPoints):
        vtk_points = points
    else:
        raise ValueError("Input must be either vtkPoints or a list of points")
    
    center = [0.0, 0.0, 0.0]
    normal = [0.0, 0.0, 1.0]
    vtkPlane.ComputeBestFittingPlane(vtk_points, center, normal)

    return center, normal

'''
Remesh polydata
'''
def polydata_remesh(polydata, subdivide=2, clusters=5000):

    tri_filter = vtkTriangleFilter()
    tri_filter.SetInputData(polydata)
    tri_filter.Update()
    inputMesh = pv.wrap(tri_filter.GetOutput())
    clus = pyacvd.Clustering(inputMesh)
    if subdivide > -1:
        clus.subdivide(subdivide)
    clus.cluster(clusters)
    outputMesh = vtkPolyData()
    outputMesh.DeepCopy(clus.create_mesh())

    return outputMesh

'''
Smooth polydata
'''
def polydata_smooth(polydata, method='Taubin', iterations=30, laplaceRelaxationFactor=0.5, taubinPassBand=0.1, boundarySmoothing=True):

    if method == "Laplace":
      smoothing = vtkSmoothPolyDataFilter()
      smoothing.SetRelaxationFactor(laplaceRelaxationFactor)
    else:  # "Taubin"
      smoothing = vtkWindowedSincPolyDataFilter()
      smoothing.SetPassBand(taubinPassBand)
    smoothing.SetBoundarySmoothing(boundarySmoothing)
    smoothing.SetNumberOfIterations(iterations)
    smoothing.SetInputData(polydata)
    smoothing.Update()

    outputMesh = smoothing.GetOutput()

    return outputMesh

'''
Fill holes in polydata
'''
def polydata_fillHoles(polydata, maximumHoleSize=1000.0):

    fill = vtkFillHolesFilter()
    fill.SetInputData(polydata)
    fill.SetHoleSize(maximumHoleSize)

    # Need to auto-orient normals, otherwise holes could appear to be unfilled when
    # only front-facing elements are chosen to be visible.
    normals = vtkPolyDataNormals()
    normals.SetInputConnection(fill.GetOutputPort())
    normals.SetAutoOrientNormals(True)
    normals.Update()

    outputMesh = normals.GetOutput()

    return outputMesh

'''
Polydata append filter
'''
def polydata_append(polydata_1, polydata_2):

    appendFilter = vtkAppendPolyData()
    appendFilter.AddInputData(polydata_1)
    appendFilter.AddInputData(polydata_2)
    appendFilter.Update()
    outputPolydata = polydata_clean(appendFilter.GetOutput())

    return outputPolydata

'''
Convex hull of polydata
'''
def polydata_convexHull(polydata):

    convexHull = vtkDelaunay3D()
    convexHull.SetInputData(polydata)
    outerSurface = vtkGeometryFilter()
    outerSurface.SetInputConnection(convexHull.GetOutputPort())
    outerSurface.Update()
    outputMesh = polydata_clean(outerSurface.GetOutput())

    return outputMesh

'''
Get contact polydata between two polydata
'''
def get_contact_polydata(polydata_1, polydata_2):

    # remesh
    polydata_1 = polydata_remesh(polydata_1, subdivide=2, clusters=5000)
    polydata_2 = polydata_remesh(polydata_2, subdivide=2, clusters=5000)

    points_list_1 = numpy_support.vtk_to_numpy(polydata_1.GetPoints().GetData())
    points_list_2 = numpy_support.vtk_to_numpy(polydata_2.GetPoints().GetData())

    contact_ids = []
    for i, point in enumerate(points_list_1):
        distances = np.linalg.norm(points_list_2 - point, axis=1)
        contact_ids.append(np.argmin(distances))

    # filter polydata
    non_contact_ids = [i for i in range(len(points_list_2)) if i not in contact_ids]
    remove_ids = vtkIdTypeArray()
    for id_ in non_contact_ids:
        remove_ids.InsertNextTuple1(id_)
    contact_polydata = delete_points(polydata_2, remove_ids)

    # fill holes
    contact_polydata = polydata_fillHoles(contact_polydata, maximumHoleSize=1000.0)

    return contact_polydata

'''
Get IDs of boundary edges of polydata
'''
def extractBoundary(polydata):

    edgeFilter = vtkFeatureEdges()
    edgeFilter.SetInputData(polydata)
    edgeFilter.BoundaryEdgesOn()
    edgeFilter.ManifoldEdgesOff()
    edgeFilter.NonManifoldEdgesOff()
    edgeFilter.FeatureEdgesOff()
    edgeFilter.Update()

    edges = edgeFilter.GetOutput()
    edgePoints = numpy_support.vtk_to_numpy(edges.GetPoints().GetData())

    return edges

def transform_polydata(polydata, transformMatrix):
    
        transform = vtkTransform()
        transform.SetMatrix(transformMatrix)
    
        transformFilter = vtkTransformPolyDataFilter()
        transformFilter.SetInputData(polydata)
        transformFilter.SetTransform(transform)
        transformFilter.Update()
    
        return transformFilter.GetOutput()