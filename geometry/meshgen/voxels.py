import trimesh as trm
import numpy as np
import scipy
from scipy.ndimage import convolve, binary_dilation
import meshgen.mesher as mesher

from tqdm import tqdm  # progress bar for parallel ops
from concurrent.futures import ProcessPoolExecutor


def _segment_worker(args):
    """
    Voxelize a mesh segment and return global lattice indices.

    Parameters
    ----------
    args : tuple
        (submesh, voxel_size, bounds_min, target_shape)

    Returns
    -------
    numpy.ndarray
        Array of shape (N, 3) with integer indices in the global lattice.
    """
    submesh, voxel_size, bounds_min, target_shape = args

    if submesh is None or len(submesh.faces) == 0:
        return np.zeros((0, 3), dtype=int)

    vg = submesh.voxelized(voxel_size)
    sparse = vg.sparse_indices
    if sparse.size == 0:
        return np.zeros((0, 3), dtype=int)

    centers = vg.indices_to_points(sparse)
    # Map voxel centers to global lattice indices relative to the original mesh bounds.
    bounds_min = np.asarray(bounds_min, dtype=float)
    rel = (centers - bounds_min) / voxel_size
    global_indices = np.rint(rel).astype(int)
    if target_shape is not None:
        max_idx = np.asarray(target_shape, dtype=int) - 1
        min_idx = np.zeros_like(max_idx)
        global_indices = np.clip(global_indices, min_idx, max_idx)
    return global_indices


def load_mesh(path):
    """
    Load a mesh from a specified file path using trimesh.

    This function takes a file path as input, and uses the trimesh library to load the mesh.
    It supports various mesh formats that trimesh can handle.

    Parameters:
    path (str): The file path of the mesh to be loaded.

    Returns:
    trimesh.Trimesh: A trimesh object representing the loaded mesh.
    """
    # Load the mesh from the given file path using trimesh
    mesh = trm.load(path)
    return mesh


def get_leading_direction(shape):
    """
    Identify the leading direction of a 3D shape based on its dimensions.

    This function calculates which of the three dimensions (x, y, z) of a shape
    is the largest and returns the index of that dimension. It uses numpy's argmax
    function to find the index of the maximum value in the shape dimensions.

    Parameters:
    shape (tuple or list): A 3-element iterable representing the dimensions of the shape.
                           Typically, this would be in the format (x, y, z).

    Returns:
    int: The index of the leading dimension (0 for x, 1 for y, 2 for z).
    """
    # Find the index of the maximum value in shape dimensions
    direction_idx = np.argmax([shape[0], shape[1], shape[2]])

    return direction_idx


def fill_extended_mesh(original_mesh, new_mesh, num_type='bool', expected_in_outs=None):
    """
    Embed the voxelized occupancy into a larger cuboidal domain, then label.

    Pipeline per spec:
    - Start with boolean occupancy from voxelization (fluid=True).
    - Embed this occupancy into a cuboidal domain (LBM-friendly shape).
    - If `num_type == 'bool'`, return the embedded occupancy.
    - If `num_type == 'int'`, convert the embedded occupancy to labels via `label_elements`
      so face tags and wall shells are computed on the final domain shape.

    Parameters:
    - original_mesh: numpy.ndarray (3D), boolean-like occupancy from voxelization.
    - new_mesh: numpy.ndarray (3D), only its shape is used for embedding placement.
    - num_type: 'bool' or 'int' to control output type.
    - expected_in_outs: optional set of {N,E,S,W,B,F} for face tagging (when num_type='int').

    Returns:
    - numpy.ndarray: embedded occupancy (bool) or labeled volume (int) with the same shape as `new_mesh`.
    """
    # Shapes
    original_shape = original_mesh.shape
    new_shape = new_mesh.shape

    # Determine leading direction and the remaining two axes
    direction_idx = get_leading_direction(original_shape)
    remaining_indices = [i for i in range(3) if i != direction_idx]

    # Ensure interior is filled on the original occupancy
    occ = fill_mesh_inside_surface(original_mesh.astype(bool))

    # Prepare an empty occupancy of target size for embedding
    embedded_occ = np.zeros(new_shape, dtype=bool)

    # Embed occupancy into the center along the two non-leading axes
    if remaining_indices == [0, 1]:
        start_0 = (new_shape[0] - original_shape[0]) // 2
        start_1 = (new_shape[1] - original_shape[1]) // 2
        embedded_occ[
            start_0 : start_0 + original_shape[0],
            start_1 : start_1 + original_shape[1],
            :,
        ] = occ
    elif remaining_indices == [0, 2]:
        start_0 = (new_shape[0] - original_shape[0]) // 2
        start_1 = (new_shape[2] - original_shape[2]) // 2
        embedded_occ[
            start_0 : start_0 + original_shape[0],
            :,
            start_1 : start_1 + original_shape[2],
        ] = occ
    else:
        start_0 = (new_shape[1] - original_shape[1]) // 2
        start_1 = (new_shape[2] - original_shape[2]) // 2
        embedded_occ[
            :,
            start_0 : start_0 + original_shape[1],
            start_1 : start_1 + original_shape[2],
        ] = occ

    # Return either the embedded occupancy or labeled integers computed on the final domain
    if num_type == 'bool':
        return embedded_occ

    labeled = label_elements(embedded_occ, expected_in_outs=expected_in_outs, num_type='int')
    return labeled


def get_lbm_shape(original_shape, leading_multiple=128):
    """
    Adjust the shape of a 3D object to make it suitable for Lattice Boltzmann Method (LBM) simulations.

    This function checks if the leading dimension of the original shape is a multiple of `leading_multiple`.
    If not, it raises a ValueError. Then, it adjusts the remaining dimensions to be the smallest
    multiple of 32 that is greater than or equal to the maximum of these dimensions, keeping the
    leading dimension unchanged.

    Parameters:
    original_shape (tuple or list): A 3-element iterable representing the original shape dimensions (x, y, z).
    leading_multiple (int): Required multiple for the leading dimension (default 128).

    Returns:
    tuple: The new shape, adjusted for LBM simulations.

    Raises:
    ValueError: If the leading dimension of the original shape is not a multiple of ``leading_multiple``.
    """
    # Identify the leading direction (dimension) of the shape
    direction_idx = get_leading_direction(original_shape)

    if leading_multiple <= 0:
        raise ValueError("leading_multiple must be a positive integer.")

    # Check if the leading dimension is a multiple of the requested stride, raise an error if not
    if (original_shape[direction_idx] % leading_multiple) != 0:
        error_msg = f"Shape {original_shape} is not suitable for LBM simulation."
        raise ValueError(error_msg)

    # Find the maximum dimension size that is not the leading direction
    # and calculate the smallest multiple of 32 that is greater than or equal to that size
    max_remaining_size = max(
        original_shape[i] for i in range(len(original_shape)) if i != direction_idx
    )
    rounded_size = ((max_remaining_size + 31) // 32) * 32

    # Construct the new shape, adjusting non-leading dimensions
    new_shape = [
        original_shape[i] if i == direction_idx else rounded_size
        for i in range(len(original_shape))
    ]
    new_shape = tuple(new_shape)

    return new_shape


def complete_mesh(original_mesh, num_type='bool', expected_in_outs=None, leading_multiple=128):
    """
    Adjust an original mesh to make it suitable for Lattice Boltzmann Method (LBM) simulations.

    This function first determines a new shape for the mesh that is compatible with LBM simulations
    using the 'get_lbm_shape' function. It then creates an empty mesh of this new shape and embeds
    the original mesh into it using the 'fill_extended_mesh' function.

    Parameters:
    original_mesh (numpy.ndarray): The original mesh to be adjusted.
    num_type (str): 'bool' for occupancy, 'int' for labels.
    expected_in_outs (dict or iterable, optional): Domain faces to tag.
    leading_multiple (int, optional): Required multiple for the leading dimension when embedding
                                      into an LBM-friendly domain. Defaults to 128.

    Returns:
    numpy.ndarray: The new mesh, adjusted in shape and filled, suitable for LBM simulations.
    """
    # Get the shape of the original mesh
    original_shape = original_mesh.shape

    # Determine the new shape suitable for LBM simulations
    new_shape = get_lbm_shape(original_shape, leading_multiple=leading_multiple)

    # Create an empty mesh with the new shape
    empty_mesh = np.zeros(new_shape, dtype=bool)

    if num_type == 'int':
        empty_mesh = empty_mesh.astype(int)

    # Fill the original mesh into the new, empty mesh
    new_mesh = fill_extended_mesh(original_mesh, empty_mesh, num_type=num_type, expected_in_outs=expected_in_outs)

    return new_mesh


def fill_mesh_inside_surface(mesh):
    """
    Fill the internal voids of a 3D mesh.

    This function takes a 3D mesh, represented as a NumPy array, and fills its internal voids.
    It uses the 'binary_fill_holes' function from SciPy's ndimage module to identify and fill
    these voids. The input mesh is expected to be a boolean array, where True indicates the
    presence of the mesh and False indicates empty space.

    Parameters:
    mesh (numpy.ndarray): A 3D boolean array representing the mesh.

    Returns:
    numpy.ndarray: The modified mesh with internal voids filled.
    """
    # Fill the internal voids of the mesh using SciPy's binary_fill_holes function
    mesh[scipy.ndimage.binary_fill_holes(mesh)] = True

    return mesh


def is_leading_dir(direction, leading_direction):
    """
    Check if a given direction is the leading direction.

    Parameters:
    direction (int): The direction to be checked (0 for x, 1 for y, 2 for z).
    leading_direction (int): The leading direction against which to check.

    Returns:
    bool: True if 'direction' is the same as 'leading_direction', otherwise False.
    """
    # Compare the given direction with the leading direction
    return direction == leading_direction


def calculate_margin(original_bounds, new_bounds, leading_direction):
    """
    Calculate the margin between the original and new bounds of a mesh.

    This function calculates the margin for each dimension (x, y, z). For the leading direction,
    the margin is set to [0, 0]. For the other two directions, it calculates the absolute difference
    between the original and new bounds.

    Parameters:
    original_bounds (array-like): The original bounds of the mesh.
    new_bounds (array-like): The new bounds of the mesh after processing.
    leading_direction (int): The index of the leading direction (0 for x, 1 for y, 2 for z).

    Returns:
    numpy.ndarray: A 2D array representing the margins for each direction.
    """
    margin = []
    # Calculate margin for each dimension
    for j in range(3):
        if is_leading_dir(j, leading_direction):
            margin.append([0, 0])
        else:
            margin.append(
                [
                    abs(new_bounds[0][j] - original_bounds[0][j]),
                    abs(new_bounds[1][j] - original_bounds[1][j]),
                ]
            )

    return np.array(margin)


def calculate_segment_direction_and_increment(leading_direction, voxel_size):
    """
    Calculate the direction vector and increment for segmenting a mesh.

    This function determines the direction vector and increment value for segmenting a mesh along
    its leading direction. The direction vector is a unit vector along the leading direction, and
    the increment is calculated based on the voxel size and the leading direction.

    Parameters:
    leading_direction (int): The index of the leading direction (0 for x, 1 for y, 2 for z).
    voxel_size (int): The size of a single voxel.

    Returns:
    tuple: A tuple containing the direction vector and the increment value.
    """
    # Create a direction vector based on the leading direction
    segment_direction = np.array(
        [int(is_leading_dir(i, leading_direction)) for i in range(3)]
    )

    # Calculate increment based on voxel size and leading direction
    segment_increment = np.array(
        [voxel_size * int(is_leading_dir(i, leading_direction)) for i in range(3)]
    )

    return segment_direction, segment_increment


def slice_mesh(mesh, direction, plane_origin, increment):
    """
    Slice a mesh in a specified direction using two parallel planes.

    This function slices a 3D mesh using two parallel planes. The first plane is defined by the
    'plane_origin' point and the 'direction' vector. The second plane is parallel to the first and
    offset by the 'increment' value. The section of the mesh that lies between these two planes is returned.

    Parameters:
    mesh (trimesh.Trimesh): The mesh to be sliced.
    direction (array-like): A 3-element array representing the normal vector of the slicing plane.
    plane_origin (array-like): A 3-element array representing a point on the first slicing plane.
    increment (float): The distance between the two parallel slicing planes.

    Returns:
    trimesh.Trimesh: The portion of the mesh that lies between the two slicing planes.
    """
    # Slice the mesh with the first plane in the given direction
    mesh_one_direction = trm.intersections.slice_mesh_plane(
        mesh, direction, plane_origin
    )

    # Slice the resulting mesh with a second plane, in the opposite direction and offset by increment
    mesh_second_direction = trm.intersections.slice_mesh_plane(
        mesh_one_direction, direction * (-1), plane_origin + increment
    )

    return mesh_second_direction


def split_mesh(mesh, voxel_size, n_segments):
    """
    Split a mesh into segments along the leading axis using face intersections.

    Faces intersecting a segment range are included (with overlap) to avoid gaps at
    boundaries. No geometric slicing is performed, which keeps the implementation
    free of additional dependencies such as Shapely.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh to segment.
    voxel_size : float
        Voxel pitch (unused directly but retained for API symmetry).
    n_segments : int
        Number of desired segments along the leading axis.

    Returns
    -------
    tuple[list[trimesh.Trimesh], list[np.ndarray]]
        Submeshes and dummy margin placeholders (zeros).
    """
    bounds = mesh.bounds
    leading_direction = get_leading_direction(tuple(bounds[1] - bounds[0]))
    n_segments = max(1, int(n_segments))

    axis_min = bounds[0, leading_direction]
    axis_max = bounds[1, leading_direction]
    if axis_max <= axis_min:
        # Degenerate extent; return original mesh as a single segment.
        return [mesh.copy()], [np.zeros((3, 2), dtype=int)]

    edges = np.linspace(axis_min, axis_max, n_segments + 1)
    # Precompute per-face bounds along the leading axis
    tri_vertices = mesh.vertices[mesh.faces]
    tri_min = tri_vertices[:, :, leading_direction].min(axis=1)
    tri_max = tri_vertices[:, :, leading_direction].max(axis=1)

    submeshes = []
    margins = []
    epsilon = 1e-9

    for start, end in zip(edges[:-1], edges[1:]):
        mask = (tri_max >= start - epsilon) & (tri_min <= end + epsilon)
        if not np.any(mask):
            empty = trm.Trimesh(
                vertices=np.empty((0, 3)),
                faces=np.empty((0, 3), dtype=int),
                process=False,
            )
            submeshes.append(empty)
            margins.append(np.zeros((3, 2), dtype=int))
            continue

        segment = mesh.submesh([mask], append=True, repair=False)
        submeshes.append(segment)
        margins.append(np.zeros((3, 2), dtype=int))

    return submeshes, margins


def complete_segment(segment, margins):
    """
    Complete a segment of a mesh by adding margins in each direction.

    This function takes a segment of a mesh and adds specified margins around it in all three
    dimensions (x, y, z). The margins are added by creating a larger array and embedding the
    original segment in it, centered with the given margins.

    Parameters:
    segment (numpy.ndarray): A 3D array representing a segment of the mesh.
    margins (numpy.ndarray): A 2D array where each row represents the margins [start, end]
                             for each dimension (x, y, z).

    Returns:
    numpy.ndarray: The segment with added margins, represented as a larger 3D array.
    """
    a1 = margins[0][0]  # offsets in the x direction
    a2 = margins[0][1]

    b1 = margins[1][0]  # offsets in the y direction
    b2 = margins[1][1]

    c1 = margins[2][0]  # offsets in the z direction
    c2 = margins[2][1]

    # Calculate the new shape including the margins
    new_shape = (
        segment.shape[0] + a1 + a2,
        segment.shape[1] + b1 + b2,
        segment.shape[2] + c1 + c2,
    )

    # Create a new array with the new shape, initialized as empty (False)
    new_array = np.zeros(new_shape, dtype=bool)

    # Embed the original segment into the center of the new array, surrounded by the margins
    new_array[
        a1 : a1 + segment.shape[0],
        b1 : b1 + segment.shape[1],
        c1 : c1 + segment.shape[2],
    ] = segment

    return new_array


def fill_slice(mesh_slice, leading_direction):
    """
    Fill a mesh slice and expand it along the leading direction.

    This function takes a slice of a mesh (represented as a 2D array), fills its internal voids,
    and then expands this slice into a 3D array by adding an extra dimension along the specified
    leading direction. This step is typically used to prepare the slice for processes that require
    3D input or to maintain consistency in the mesh's dimensionality across operations.

    Parameters:
    mesh_slice (numpy.ndarray): A 2D array representing a slice of the mesh.
    leading_direction (int): The axis along which to expand the slice (0 for x, 1 for y, 2 for z).

    Returns:
    numpy.ndarray: A 3D array representing the filled and expanded slice of the mesh.
    """
    # Fill the internal voids of the slice
    mesh_slice = fill_mesh_inside_surface(mesh_slice)

    # Expand the 2D slice into a 3D array along the specified leading direction
    mesh_slice = np.expand_dims(mesh_slice, axis=leading_direction)

    return mesh_slice


def calculate_voxel_size(mesh, res, leading_multiple=128):
    """
    Calculate the voxel size for a mesh based on the specified resolution.

    This function calculates the voxel size needed to represent a mesh at a given resolution. It
    finds the maximum dimension of the mesh and divides it by a factor derived from the resolution.
    The function is used to determine the appropriate voxel size for voxelizing the mesh.

    Parameters:
    mesh (trimesh.Trimesh): The mesh for which the voxel size is being calculated.
    res (float): The desired resolution, used as a factor in the calculation.
    leading_multiple (int, optional): Target multiple for the longest axis voxel count. Defaults to 128.

    Returns:
    float: The calculated voxel size.
    """
    # Calculate the maximum size of the mesh in any dimension
    maxima = [np.max(mesh.bounds[:, i]) - np.min(mesh.bounds[:, i]) for i in range(3)]
    h = np.max(maxima)

    # Choose pitch so the longest axis yields ~leading_multiple*res cells.
    # VoxelGrid commonly yields N = floor(extent / pitch) + 1 along an axis.
    # Use (target-1) in the denominator to drive N ≈ target.
    target = max(1, int(leading_multiple * res))
    denom = max(1, target - 1)
    voxel_size = h / denom

    return voxel_size


def voxelize_elementary(mesh, voxel_size):
    """
    Convert a mesh into a voxelized representation without splitting.

    This function takes a mesh and converts it into a voxelized form using the specified voxel size.
    The voxelization process divides the mesh into cubic cells of the given size. It does not split
    the mesh into separate segments but rather represents the entire mesh in a voxelized format.

    Parameters:
    mesh (trimesh.Trimesh): The mesh to be voxelized.
    voxel_size (float): The edge length of each voxel in the voxelized representation.

    Returns:
    numpy.ndarray: A 3D array representing the voxelized mesh, where each cell is a voxel.
    """
    # Convert the mesh into a voxelized form with the specified voxel size
    return mesh.voxelized(voxel_size).matrix


def process_submesh(submsh, margin, voxel_size, leading_direction):
    """
    Process a single submesh for voxelization.

    This function handles the voxelization of a single mesh segment. It voxelizes the segment,
    completes it by adding necessary margins, and adjusts the slicing based on the leading direction.
    Finally, it fills the slice to ensure a solid segment.

    Parameters:
    submsh (trimesh.Trimesh): The submesh segment to be voxelized.
    margin (numpy.ndarray): The margin to be added around the voxelized segment.
    voxel_size (float): The size of each voxel.
    leading_direction (int): The leading direction for slicing (0 for x, 1 for y, 2 for z).

    Returns:
    numpy.ndarray: A 3D array representing the processed and filled submesh segment.
    """
    # Voxelize the submesh volume; do not fill here to avoid
    # double-filling. We will fill once globally after stitching.
    voxelized_segment = voxelize_elementary(submsh, voxel_size)

    # Embed with margins to align to global coordinates
    comp = complete_segment(voxelized_segment, margin.astype(int))

    # Return full 3D block (no collapsing)
    return comp


def voxelize_with_splitting(mesh, voxel_size, split, num_processes=1, target_bounds=None):
    """
    Voxelize a surface by segmenting along the leading axis and stitching results.

    Each segment is voxelized independently (optionally in parallel) and converted
    to global lattice indices. Segments are stitched into a single occupancy grid,
    the interior is filled once globally, and the shape is normalized to match the
    Trimesh VoxelGrid behavior (`ceil(extent / voxel_size) + 1` per axis).

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Surface mesh to voxelize.
    voxel_size : float
        Target lattice pitch.
    split : int
        Number of segments along the leading axis (values <= 1 fall back to single pass).
    num_processes : int
        Parallel workers for segment voxelization.
    target_bounds : ndarray | None
        Optional bounds to derive the expected dimensions; defaults to mesh.bounds.

    Returns
    -------
    numpy.ndarray
        Boolean occupancy grid with filled interior.
    """
    if target_bounds is None:
        target_bounds = mesh.bounds

    # No-split path: voxelize once and normalize to expected dims derived from bounds and pitch
    if split is None or split <= 1:
        ary = voxelize_elementary(mesh, voxel_size)
        ary = fill_mesh_inside_surface(ary)
        ext = (target_bounds[1] - target_bounds[0])
        exp = (np.ceil(ext / voxel_size).astype(int) + 1).tolist()
        sx, sy, sz = ary.shape
        tx, ty, tz = exp
        if sx >= tx and sy >= ty and sz >= tz:
            return ary[:tx, :ty, :tz]
        # pad if needed to match expected dims
        nx, ny, nz = max(sx, tx), max(sy, ty), max(sz, tz)
        tmp = np.zeros((nx, ny, nz), dtype=ary.dtype)
        tmp[:sx, :sy, :sz] = ary
        return tmp[:tx, :ty, :tz]

    ext = (target_bounds[1] - target_bounds[0])
    target_shape = np.ceil(ext / voxel_size).astype(int) + 1
    target_shape = target_shape.astype(int)
    bounds_min = np.asarray(target_bounds[0], dtype=float)

    n_segments = max(1, int(split))
    submeshes, _ = split_mesh(mesh, voxel_size, n_segments)

    occupancy = np.zeros(tuple(target_shape), dtype=bool)

    segment_args = [(sub, voxel_size, bounds_min, target_shape) for sub in submeshes]
    total_segments = len(segment_args)

    def _accumulate(indices):
        if indices.size == 0:
            return
        mask = (
            (indices[:, 0] >= 0)
            & (indices[:, 0] < target_shape[0])
            & (indices[:, 1] >= 0)
            & (indices[:, 1] < target_shape[1])
            & (indices[:, 2] >= 0)
            & (indices[:, 2] < target_shape[2])
        )
        if not np.any(mask):
            return
        valid = indices[mask]
        occupancy[valid[:, 0], valid[:, 1], valid[:, 2]] = True

    if num_processes > 1 and total_segments > 1:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            iterator = executor.map(_segment_worker, segment_args)
            if total_segments > 1:
                iterator = tqdm(iterator, total=total_segments, desc="Voxelizing segments")
            for indices in iterator:
                _accumulate(indices)
    else:
        iterator = segment_args
        if total_segments > 1:
            iterator = tqdm(iterator, total=total_segments, desc="Voxelizing segments")
        for args in iterator:
            indices = _segment_worker(args)
            _accumulate(indices)

    if occupancy.any():
        # Close small cracks introduced by splitting before global fill.
        occupancy = binary_dilation(occupancy, iterations=1)

    occupancy = fill_mesh_inside_surface(occupancy)
    return occupancy


def voxelize_mesh(name, res=1, split=None, num_processes=1, leading_multiple=128, **kwargs):
    """
    Load a mesh from a file, voxelize it with or without splitting, and adjust it for LBM simulations.

    This function loads a mesh from the specified path and calculates an appropriate voxel size based
    on the mesh's bounds and a resolution factor. It then voxelizes the mesh, either as a whole or by
    splitting it into segments, depending on the 'split' parameter. The voxelized mesh is adjusted
    using the 'complete_mesh' function to make it suitable for LBM simulations.

    Parameters:
    path (str): The file path of the mesh to be loaded and voxelized.
    res (int, optional): A resolution factor for voxelization. Default is 1.
    split (int, optional): The number of segments to split the mesh into before voxelization.
                           If None, the mesh is voxelized without splitting. Default is None.
    num_processes(int, optional): The number of subprocesses the main process should be divided into.
                                  Default 1 = no subprocesses.
    leading_multiple (int, optional): Target multiple applied to the leading axis. The voxel pitch is
                                      chosen so the longest axis has ~leading_multiple * res cells.
                                      Defaults to 128 to preserve existing behavior.


    Returns:
    numpy.ndarray: The voxelized and adjusted mesh, suitable for LBM simulations.
    """

    modified_kwargs = dict(kwargs)
    modified_kwargs["resolution"] = res

    # Generate STL using Gmsh into a temp working directory
    stl_file_path = mesher.gmsh_surface(name, True, **modified_kwargs)

    # Load the mesh from the specified path
    mesh = trm.load(stl_file_path)

    # Calculate the voxel size based on the mesh's bounds and resolution factor
    voxel_size = calculate_voxel_size(mesh, res, leading_multiple=leading_multiple)

    if split is None:
        # Voxelize the mesh without splitting and normalize to expected dims
        output = voxelize_elementary(mesh, voxel_size)
        output = fill_mesh_inside_surface(output)
        ext = (mesh.bounds[1] - mesh.bounds[0])
        exp = (np.ceil(ext / voxel_size).astype(int) + 1).tolist()
        sx, sy, sz = output.shape
        tx, ty, tz = exp
        if sx >= tx and sy >= ty and sz >= tz:
            output = output[:tx, :ty, :tz]
        else:
            nx, ny, nz = max(sx, tx), max(sy, ty), max(sz, tz)
            tmp = np.zeros((nx, ny, nz), dtype=output.dtype)
            tmp[:sx, :sy, :sz] = output
            output = tmp[:tx, :ty, :tz]
    else:
        # Voxelize with splitting directly on the original surface; do not include
        # any auxiliary boxing geometry in the voxelization itself. Stitch results
        # and normalize to the expected dimensions from the original bounds.
        bounds = mesh.bounds
        output = voxelize_with_splitting(mesh, voxel_size, split, num_processes=num_processes, target_bounds=bounds)

    return output


def voxelize_stl(path, res=1, split=None, num_processes=1, leading_multiple=128):
    """
    Voxelize a closed STL surface directly, with optional splitting.

    Parameters:
    - path: path to the input STL file (must be closed/watertight for correct filling).
    - res: resolution scale; longest axis ~ leading_multiple * res voxels.
    - split: if None, voxelize as a single mesh; otherwise, split into this many segments
             along the leading direction and process in parallel.
    - num_processes: parallel workers when splitting.
    - leading_multiple: target multiple for longest axis voxel count (default 128 for backwards
                        compatibility); also used for LBM padding downstream.

    Returns:
    - numpy.ndarray of dtype bool containing occupancy (True=inside/solid)
    """
    # Load mesh (handle scenes by concatenation)
    mesh = trm.load(path)
    if isinstance(mesh, trm.Scene):
        mesh = trm.util.concatenate(tuple(mesh.geometry.values()))

    # Ensure watertight surface if possible
    if hasattr(mesh, "is_watertight") and not mesh.is_watertight:
        try:
            # Attempt a lightweight repair; ignore failures
            trm.repair.fix_normals(mesh)
            trm.repair.fix_winding(mesh)
            trm.repair.fill_holes(mesh)
        except Exception:
            pass

    voxel_size = calculate_voxel_size(mesh, res, leading_multiple=leading_multiple)

    if split is None:
        occ = voxelize_elementary(mesh, voxel_size)
        occ = fill_mesh_inside_surface(occ)
        # Normalize to expected dims derived from original bounds
        ext = (mesh.bounds[1] - mesh.bounds[0])
        exp = (np.ceil(ext / voxel_size).astype(int) + 1).tolist()
        sx, sy, sz = occ.shape
        tx, ty, tz = exp
        if sx >= tx and sy >= ty and sz >= tz:
            occ = occ[:tx, :ty, :tz]
        else:
            nx, ny, nz = max(sx, tx), max(sy, ty), max(sz, tz)
            tmp = np.zeros((nx, ny, nz), dtype=occ.dtype)
            tmp[:sx, :sy, :sz] = occ
            occ = tmp[:tx, :ty, :tz]
        return occ

    # Splitting: segment and voxelize directly from the original mesh, then stitch.
    bounds = mesh.bounds
    return voxelize_with_splitting(
        mesh,
        voxel_size,
        split,
        num_processes=num_processes,
        target_bounds=bounds,
    )


def generate_lbm_mesh(original_mesh, expected_in_outs=None, leading_multiple=128):
    """
    Embed an occupancy grid into an LBM-friendly domain and apply labels.

    Parameters:
    original_mesh (numpy.ndarray): Occupancy grid to embed.
    expected_in_outs (dict or iterable, optional): Domain faces to tag.
    leading_multiple (int, optional): Required multiple for the leading dimension. Defaults to 128.

    Returns:
    numpy.ndarray: Integer labeled volume ready for export.
    """
    lbm_mesh = complete_mesh(
        original_mesh,
        num_type='int',
        expected_in_outs=expected_in_outs,
        leading_multiple=leading_multiple,
    )

    return lbm_mesh


def label_elements(original_mesh, expected_in_outs=None, num_type='bool'):
    """Convert an occupancy grid into labeled volume semantics.

    Semantics:
    - 0 = empty/outside
    - 1 = fluid/interior (occupancy)
    - 2 = wall/solid (one-voxel layer outside the fluid along solid boundaries)
    - 11..16 = optional domain wall tags on the fluid layer for N,E,S,W,B,F

    Notes:
    - This function is intentionally vectorized; no per-voxel Python loops.
    - The input may be boolean (occupancy) or integer with 0/1 values.
    """
    # Normalize input to boolean occupancy for processing
    occ = (original_mesh > 0)

    # Start with zeros (outside/empty)
    labeled = np.zeros(occ.shape, dtype=int)
    # Mark interior fluid
    labeled[occ] = 1

    # Derive a 1-voxel solid shell surrounding fluid: outside cells that neighbor fluid
    kernel = np.ones((3, 3, 3), dtype=int)
    kernel[1, 1, 1] = 0
    # Count fluid neighbors for every cell
    fluid_nbrs = convolve(occ.astype(int), kernel, mode="constant", cval=0)
    solid_shell = (~occ) & (fluid_nbrs > 0)
    labeled[solid_shell] = 2

    # Optional domain wall tags: apply to the last fluid layer on requested faces
    if expected_in_outs is None:
        expected_in_outs = {}

    # Determine fluid extents along each axis and tag by fluid boundary planes
    sx, sy, sz = occ.shape
    x_any = np.any(occ, axis=(1, 2))
    y_any = np.any(occ, axis=(0, 2))
    z_any = np.any(occ, axis=(0, 1))
    x_ids = np.flatnonzero(x_any)
    y_ids = np.flatnonzero(y_any)
    z_ids = np.flatnonzero(z_any)

    # Z faces per updated convention: N=max index, S=min index among fluid
    if 'N' in expected_in_outs and z_ids.size:
        zn = int(z_ids[-1])
        plane = labeled[:, :, zn]
        mask = (plane == 1)
        if np.any(mask):
            plane[mask] = 11
            labeled[:, :, zn] = plane
    if 'S' in expected_in_outs and z_ids.size:
        zs = int(z_ids[0])
        plane = labeled[:, :, zs]
        mask = (plane == 1)
        if np.any(mask):
            plane[mask] = 13
            labeled[:, :, zs] = plane

    # X faces: W=min index, E=max index among fluid
    if 'E' in expected_in_outs and x_ids.size:
        xe = int(x_ids[-1])
        plane = labeled[xe, :, :]
        mask = (plane == 1)
        if np.any(mask):
            plane[mask] = 12
            labeled[xe, :, :] = plane
    if 'W' in expected_in_outs and x_ids.size:
        xw = int(x_ids[0])
        plane = labeled[xw, :, :]
        mask = (plane == 1)
        if np.any(mask):
            plane[mask] = 14
            labeled[xw, :, :] = plane

    # Y faces: F=min index, B=max index among fluid
    if 'B' in expected_in_outs and y_ids.size:
        yb = int(y_ids[-1])
        plane = labeled[:, yb, :]
        mask = (plane == 1)
        if np.any(mask):
            plane[mask] = 15
            labeled[:, yb, :] = plane
    if 'F' in expected_in_outs and y_ids.size:
        yf = int(y_ids[0])
        plane = labeled[:, yf, :]
        mask = (plane == 1)
        if np.any(mask):
            plane[mask] = 16
            labeled[:, yf, :] = plane

    # If integer labels explicitly requested, ensure dtype matches
    if num_type == 'int' and labeled.dtype != int:
        labeled = labeled.astype(int)

    return labeled


def assign_near_walls(original_mesh):
    """Label fluid cells (1) that neighbor walls (2) as 3 using 3D convolution."""
    updated_mesh = original_mesh.copy()
    # 3x3x3 kernel with zero center to count 26-neighborhood
    kernel = np.ones((3, 3, 3), dtype=int)
    kernel[1, 1, 1] = 0
    wall_neighbors = convolve((original_mesh == 2).astype(int), kernel, mode="constant", cval=0) > 0
    mask = (original_mesh == 1) & wall_neighbors
    updated_mesh[mask] = 3
    return updated_mesh


def assign_near_near_walls(original_mesh):
    """Label fluid cells (1) that neighbor label 3 as 4 using 3D convolution."""
    updated_mesh = original_mesh.copy()
    kernel = np.ones((3, 3, 3), dtype=int)
    kernel[1, 1, 1] = 0
    near_neighbors = convolve((original_mesh == 3).astype(int), kernel, mode="constant", cval=0) > 0
    mask = (original_mesh == 1) & near_neighbors
    updated_mesh[mask] = 4
    return updated_mesh


def assign_near_near_near_walls(original_mesh):
    """Label fluid cells (1) that neighbor label 4 as 5 using 3D convolution."""
    updated_mesh = original_mesh.copy()
    kernel = np.ones((3, 3, 3), dtype=int)
    kernel[1, 1, 1] = 0
    nnear_neighbors = convolve((original_mesh == 4).astype(int), kernel, mode="constant", cval=0) > 0
    mask = (original_mesh == 1) & nnear_neighbors
    updated_mesh[mask] = 5
    return updated_mesh


def prepare_voxel_mesh_txt(mesh, expected_in_outs=None, num_type='int', label_faces: bool = True):
    # Start from occupancy → base labels (0/1/2) only. Do NOT set face tags here.
    output_mesh = label_elements(mesh, expected_in_outs=None, num_type=num_type)

    all_directions = {'N', 'E', 'S', 'W', 'B', 'F'}
    if expected_in_outs is None:
        # Preserve existing behaviour: when no domain tags are requested,
        # skip adding extra wall layers.
        active_directions = set(all_directions)
    elif isinstance(expected_in_outs, dict):
        # Accept dict-style inputs (direction -> truthy flag) by selecting
        # only the enabled faces.
        active_directions = {face for face, enabled in expected_in_outs.items() if enabled}
    elif isinstance(expected_in_outs, str):
        active_directions = {expected_in_outs}
    else:
        active_directions = set(expected_in_outs)

    # Any direction not explicitly tagged receives an added wall layer.
    remaining_directions = all_directions - active_directions
    final_mesh = output_mesh.copy()
    for direction in remaining_directions:
        final_mesh = add_additional_wall_layer(final_mesh, direction, 2)

    # Ensure no wall layer (2) sits on requested inlet/outlet planes prior to face-tagging.
    # This avoids spurious near-wall bands under I/O planes and prevents the appearance
    # of a "wall sheet" on boundary faces before tags are applied.
    if expected_in_outs:
        sx, sy, sz = final_mesh.shape
        # Z planes: N = z max, S = z min
        if 'N' in expected_in_outs and sz > 0:
            plane = final_mesh[:, :, -1]
            plane[plane == 2] = 0
            final_mesh[:, :, -1] = plane
        if 'S' in expected_in_outs and sz > 0:
            plane = final_mesh[:, :, 0]
            plane[plane == 2] = 0
            final_mesh[:, :, 0] = plane
        # X planes: W = x min, E = x max
        if 'W' in expected_in_outs and sx > 0:
            plane = final_mesh[0, :, :]
            plane[plane == 2] = 0
            final_mesh[0, :, :] = plane
        if 'E' in expected_in_outs and sx > 0:
            plane = final_mesh[-1, :, :]
            plane[plane == 2] = 0
            final_mesh[-1, :, :] = plane
        # Y planes: F = y min, B = y max
        if 'F' in expected_in_outs and sy > 0:
            plane = final_mesh[:, 0, :]
            plane[plane == 2] = 0
            final_mesh[:, 0, :] = plane
        if 'B' in expected_in_outs and sy > 0:
            plane = final_mesh[:, -1, :]
            plane[plane == 2] = 0
            final_mesh[:, -1, :] = plane

    # Near‑wall bands: promote fluid labels 1→3→4→5, walls (2) stay
    export_mesh = final_mesh.copy()
    export_mesh = assign_near_walls(export_mesh)
    export_mesh = assign_near_near_walls(export_mesh)
    export_mesh = assign_near_near_near_walls(export_mesh)

    # Final overlay: tag requested domain boundary planes intersected with fluid-like voxels
    if label_faces and expected_in_outs:
        sx, sy, sz = export_mesh.shape
        fluid_like = (export_mesh == 1) | (export_mesh == 3) | (export_mesh == 4) | (export_mesh == 5)

        # Z planes (updated convention: N=z max plane, S=z min plane)
        if 'N' in expected_in_outs and sz > 0:
            plane = export_mesh[:, :, -1]
            # Tag if boundary cell is fluid-like OR if the immediate interior neighbor is fluid-like
            neighbor = fluid_like[:, :, -2] if sz > 1 else np.zeros((sx, sy), dtype=bool)
            mask = fluid_like[:, :, -1] | neighbor
            plane[mask] = 11
            export_mesh[:, :, -1] = plane
        if 'S' in expected_in_outs and sz > 0:
            plane = export_mesh[:, :, 0]
            neighbor = fluid_like[:, :, 1] if sz > 1 else np.zeros((sx, sy), dtype=bool)
            mask = fluid_like[:, :, 0] | neighbor
            plane[mask] = 13
            export_mesh[:, :, 0] = plane

        # X planes (W=x min plane, E=x max plane)
        if 'W' in expected_in_outs and sx > 0:
            plane = export_mesh[0, :, :]
            neighbor = fluid_like[1, :, :] if sx > 1 else np.zeros((sy, sz), dtype=bool)
            mask = fluid_like[0, :, :] | neighbor
            plane[mask] = 14
            export_mesh[0, :, :] = plane
        if 'E' in expected_in_outs and sx > 0:
            plane = export_mesh[-1, :, :]
            neighbor = fluid_like[-2, :, :] if sx > 1 else np.zeros((sy, sz), dtype=bool)
            mask = fluid_like[-1, :, :] | neighbor
            plane[mask] = 12
            export_mesh[-1, :, :] = plane

        # Y planes (F=y min plane, B=y max plane)
        if 'F' in expected_in_outs and sy > 0:
            plane = export_mesh[:, 0, :]
            neighbor = fluid_like[:, 1, :] if sy > 1 else np.zeros((sx, sz), dtype=bool)
            mask = fluid_like[:, 0, :] | neighbor
            plane[mask] = 16
            export_mesh[:, 0, :] = plane
        if 'B' in expected_in_outs and sy > 0:
            plane = export_mesh[:, -1, :]
            neighbor = fluid_like[:, -2, :] if sy > 1 else np.zeros((sx, sz), dtype=bool)
            mask = fluid_like[:, -1, :] | neighbor
            plane[mask] = 15
            export_mesh[:, -1, :] = plane

    return export_mesh


def add_additional_wall_layer(mesh, direction, value=2):
    # Determine the shape of the new layer based on the direction
    if direction == 'N':  # z maximum
        new_layer = np.full((mesh.shape[0], mesh.shape[1], 1), value)
        return np.concatenate((mesh, new_layer), axis=2)
    elif direction == 'E':  # x maximum
        new_layer = np.full((1, mesh.shape[1], mesh.shape[2]), value)
        return np.concatenate((mesh, new_layer), axis=0)
    elif direction == 'S':  # z = 0
        new_layer = np.full((mesh.shape[0], mesh.shape[1], 1), value)
        return np.concatenate((new_layer, mesh), axis=2)
    elif direction == 'W':  # x = 0
        new_layer = np.full((1, mesh.shape[1], mesh.shape[2]), value)
        return np.concatenate((new_layer, mesh), axis=0)
    elif direction == 'B':  # y maximum
        new_layer = np.full((mesh.shape[0], 1, mesh.shape[2]), value)
        return np.concatenate((mesh, new_layer), axis=1)
    elif direction == 'F':  # y = 0
        new_layer = np.full((mesh.shape[0], 1, mesh.shape[2]), value)
        return np.concatenate((new_layer, mesh), axis=1)
    else:
        raise ValueError("Direction must be one of 'N', 'E', 'S', 'W', 'B', 'F'.")


if __name__ == "__main__":
    pass
