import os
import tempfile
import shutil

try:
    import gmsh  # type: ignore
except ImportError as exc:  # pragma: no cover - import guard
    gmsh = None  # type: ignore
    _GM_IMPORT_ERROR = exc  # type: ignore


def _require_gmsh():
    """Ensure the gmsh Python module is available before proceeding."""
    if gmsh is None:  # pragma: no cover - simple guard
        raise RuntimeError(
            "Gmsh is required for .geo templated meshing. Install `gmsh` or "
            "use the STL pipeline instead."
        ) from _GM_IMPORT_ERROR


def modify_geo_file(input_file_path, output_file_path, **kwargs):
    """
    Modify a GEO file by replacing placeholders with specified values.

    This function reads a GEO file, replaces specified placeholders with their corresponding values,
    and writes the modified content to a new file. Placeholders in the GEO file are expected to be
    in the format 'DEFINE_VARIABLE', where 'VARIABLE' is the name of the variable to be replaced.

    Parameters:
    input_file_path (str): The path to the original GEO file.
    output_file_path (str): The path where the modified GEO file will be saved.
    **kwargs: Arbitrary keyword arguments where keys are the variable names (as they appear in the
              'DEFINE_VARIABLE' placeholders) and values are the replacement values.

    Example:
    modify_geo_file('original.geo', 'modified.geo', angle=45, length=10)
    """

    # Open and read the content of the original GEO file
    with open(input_file_path, "r") as file:
        file_data = file.read()

    # Iterate through each keyword argument to replace placeholders
    for variable, value in kwargs.items():
        # Construct the placeholder string
        placeholder_variable = "DEFINE_" + variable.upper()
        replace_string = str(value)

        # Replace the placeholder with the actual value
        file_data = file_data.replace(placeholder_variable, replace_string)

    # Write the modified data to the new GEO file
    with open(output_file_path, "w") as file:
        file.write(file_data)


def box_stl(stl_file, x, y, z, dx, dy, dz, voxel_size):
    """
    Generate a boxed STL file from a template and specified parameters.

    This function creates a boxed version of an STL file by modifying a template GEO file with
    specific dimensions and then using Gmsh to generate the mesh and export it as an STL file.
    The template GEO file should contain placeholders for the STL file name and box dimensions.

    Parameters:
    stl_file (str): The name of the STL file (without the '.stl' extension).
    x, y, z (float): The coordinates of the bottom-left-front corner of the box.
    dx, dy, dz (float): The dimensions of the box in the x, y, and z directions, respectively.
    voxel_size (float): The size of the voxels for the mesh.

    Returns:
    str: The path to the generated boxed STL file.
    """
    _require_gmsh()

    # Resolve template/geometries relative to this package directory
    base_dir = os.path.dirname(__file__)
    template_path = os.path.join(base_dir, "geo_templates", "boxed_stl_template.geo")

    # Use a temporary working directory for generated files
    workdir = tempfile.mkdtemp(prefix="meshgen_box_")
    output_file_path = os.path.join(workdir, f"boxed_template_filled.geo")

    # Modify the GEO template with the provided dimensions and voxel size
    # Determine source STL path: accept absolute/relative path or name without extension
    # Accept either a full/relative path to an STL or a bare name (without extension)
    if os.path.isabs(stl_file) or os.sep in str(stl_file) or str(stl_file).lower().endswith('.stl'):
        stl_source = stl_file
    else:
        stl_source = os.path.join(base_dir, "stl_models", f"{stl_file}.stl")

    # Copy the STL source into the working directory using a fixed name
    local_stl = os.path.join(workdir, "source.stl")
    shutil.copyfile(stl_source, local_stl)

    # Fill the boxed template with parameters
    modify_geo_file(
        template_path,
        output_file_path,
        stl_file=os.path.basename(local_stl),
        x=x,
        y=y,
        z=z,
        dx=dx,
        dy=dy,
        dz=dz,
        voxel_size=voxel_size,
    )

    # Initialize Gmsh for mesh generation
    gmsh.initialize()
    try:
        # Open and process the modified GEO file
        gmsh.open(output_file_path)

        # Generate a 2D mesh from the GEO file
        gmsh.model.mesh.generate(2)

        # Export the generated mesh as an STL file
        stl_file_path = os.path.join(workdir, "boxed_output.stl")
        gmsh.write(stl_file_path)
    finally:
        # Finalize Gmsh to free resources
        gmsh.finalize()

    return stl_file_path


def gmsh_surface(name_geo, dependent=False, **kwargs):
    """
    Generate a surface mesh using Gmsh based on a template GEO file and custom parameters.

    This function modifies a template GEO file with provided parameters, then uses Gmsh to generate
    a 2D mesh and exports it as an STL file. It's designed to work with GEO files containing placeholders
    that are replaced by the specified keyword arguments.

    Parameters:
    name_geo (str): The base name of the GEO file (without the '.geo' extension).
    **kwargs: Arbitrary keyword arguments where keys are the variable names (as they appear in the
              placeholders of the GEO file) and values are the replacement values.

    Returns:
    str: The path to the generated STL file.
    """
    _require_gmsh()

    # Resolve template path relative to this package directory
    base_dir = os.path.dirname(__file__)
    template_path = os.path.join(base_dir, "geo_templates", f"{name_geo}_template.geo")

    # Use a temporary working directory
    workdir = tempfile.mkdtemp(prefix="meshgen_geo_")
    filled_geo = os.path.join(workdir, f"{name_geo}_template_filled.geo")

    # Provide a default for DEFINE_H if the template expects it and caller didn't pass it.
    # Heuristic: scale mesh density with resolution if provided; fall back to 1e-3.
    with open(template_path, "r") as _tf:
        template_text = _tf.read()
    filled_kwargs = dict(kwargs)
    if "DEFINE_H" in template_text and ("h" not in filled_kwargs and "H" not in filled_kwargs):
        res = filled_kwargs.get("resolution", 1)
        try:
            # Finer Gmsh near higher voxel resolution; bound to a reasonable minimum
            h_val = max(1e-4, 1e-3 / max(1, float(res)))
        except Exception:
            h_val = 1e-3
        filled_kwargs["h"] = h_val

    # Fill template with provided and inferred parameters
    modify_geo_file(template_path, filled_geo, **filled_kwargs)

    # Initialize Gmsh for mesh generation
    gmsh.initialize()
    try:
        # Open and process the modified GEO file
        gmsh.open(filled_geo)

        # Generate a 2D mesh from the GEO file
        gmsh.model.mesh.generate(2)

        # Export the generated mesh as an STL file
        stl_file_path = os.path.join(workdir, f"{name_geo}.stl")
        gmsh.write(stl_file_path)
    finally:
        # Finalize Gmsh to free resources
        gmsh.finalize()

    return stl_file_path


if __name__ == "__main__":
    pass
