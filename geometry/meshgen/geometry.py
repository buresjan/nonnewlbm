import os
import numpy as np
import json
import hashlib
from meshgen.voxels import voxelize_mesh, voxelize_stl, generate_lbm_mesh, prepare_voxel_mesh_txt
from meshgen.utilities import array_to_textfile


class Geometry:
    def __init__(
        self,
        name=None,
        resolution=1,
        split=None,
        num_processes=1,
        output_dir="output",
        expected_in_outs=None,
        stl_path=None,
        leading_multiple=128,
        **kwargs,
    ):
        """
        Initialize the Geometry class with the specified parameters.

        Parameters:
        name (str): The base name of the .geo template file (without extension).
        resolution (int): The resolution for voxelization.
        split (int, optional): Number of segments to split the mesh into before voxelization.
                               If None, the mesh is voxelized without splitting. Default is None.
        num_processes (int, optional): Number of processes to use for parallel voxelization.
                                       Default is 1 (no parallelism).
        output_dir (str, optional): Directory where output files will be stored. Default is 'output'.
        leading_multiple (int, optional): Target multiple for the leading axis (default 128). Controls
                                          voxel pitch (~leading_multiple*resolution along the longest
                                          axis) and LBM grid padding requirements.
        **kwargs: Additional parameters for customizing the geometry.
        """
        self.name = name
        self.resolution = resolution
        self.split = split
        self.num_processes = num_processes
        self.output_dir = output_dir
        self.kwargs = kwargs
        self.stl_path = stl_path
        self.leading_multiple = leading_multiple

        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Store the voxelized geometry as an attribute
        self.voxelized_mesh = None

        # Store the lbm geometry as an attribute
        self.lbm_mesh = None

        self.expected_in_outs = expected_in_outs

        # Store state as JSON string
        self.state = json.dumps(kwargs)

        state = self.state
        # Generate a name hash
        self.name_hash = hashlib.md5(state.encode()).hexdigest()


    def generate_voxel_mesh(self):
        """
        Generate the voxelized mesh for the geometry based on the .geo template.
        """
        route = "STL" if self.stl_path else ".geo"
        print(f"Generating voxel mesh via {route} route...")
        if self.stl_path:
            self.voxelized_mesh = voxelize_stl(
                self.stl_path,
                res=self.resolution,
                split=self.split,
                num_processes=self.num_processes,
                leading_multiple=self.leading_multiple,
            )
        else:
            self.voxelized_mesh = voxelize_mesh(
                self.name,
                res=self.resolution,
                split=self.split,
                num_processes=self.num_processes,
                leading_multiple=self.leading_multiple,
                **self.kwargs,
            )
        print(f"Voxel mesh generation complete. Shape: {self.voxelized_mesh.shape}")

    def save_voxel_mesh(self, filename="voxel_mesh.npy"):
        """
        Save the voxelized mesh to a file in the specified output directory.

        Parameters:
        filename (str, optional): Name of the file to save the voxel mesh. Default is 'voxel_mesh.npy'.
        """
        if self.voxelized_mesh is not None:
            file_path = os.path.join(self.output_dir, filename)
            np.save(file_path, self.voxelized_mesh)
            print(f"Voxel mesh saved to {file_path}")
        else:
            print(
                "Error: No voxel mesh to save. Generate it first using 'generate_voxel_mesh'."
            )

    def save_voxel_mesh_to_text(self, filename="voxel_mesh.txt"):
        """
        Save the voxelized mesh as a text file in the specified output directory.

        Parameters:
        filename (str, optional): Name of the text file to save the voxel mesh. Default is 'voxel_mesh.txt'.
        """
        if self.voxelized_mesh is not None:
            output_mesh = prepare_voxel_mesh_txt(
                self.voxelized_mesh,
                expected_in_outs=self.expected_in_outs,
                num_type="int",
            )

            # Prepare filenames
            geom_file = os.path.join(self.output_dir, f"geom_{filename}")
            dim_file = os.path.join(self.output_dir, f"dim_{filename}")
            val_file = os.path.join(self.output_dir, f"val_{filename}")

            # Save geometry file
            array_to_textfile(output_mesh, geom_file)

            # Save dimensions file
            with open(dim_file, "w") as f:
                shape = output_mesh.shape
                f.write(f"{shape[0]} {shape[1]} {shape[2]}\n")

            # Create empty values file
            open(val_file, "w").close()

            # Check for special case
            if self.name == "junction_2d":
                angle_file = os.path.join(self.output_dir, f"angle_{filename}")
                lower_angle = self.kwargs.get("lower_angle", "undefined")
                upper_angle = self.kwargs.get("upper_angle", "undefined")
                with open(angle_file, "w") as f:
                    f.write(f"{lower_angle} {upper_angle}\n")

            print(f"Voxel mesh and additional files saved to {self.output_dir}")
        else:
            print(
                "Error: No voxel mesh to save. Generate it first using 'generate_voxel_mesh'."
            )

    def load_voxel_mesh(self, filename="voxel_mesh.npy"):
        """
        Load the voxelized mesh from a file.

        Parameters:
        filename (str, optional): Name of the file to load the voxel mesh from. Default is 'voxel_mesh.npy'.
        """
        file_path = os.path.join(self.output_dir, filename)
        if os.path.exists(file_path):
            self.voxelized_mesh = np.load(file_path)
            print(f"Voxel mesh loaded from {file_path}")
        else:
            print(f"Error: File {file_path} does not exist.")

    def visualize(self):
        """
        Visualize the voxelized mesh using the visualization function from utilities.
        """
        if self.voxelized_mesh is not None:
            # Use absolute package import to avoid import errors when installed
            from meshgen.utilities import vis

            # output_mesh = prepare_voxel_mesh_txt(self.voxelized_mesh, expected_in_outs=self.expected_in_outs,
            #                                      num_type='int')
            #
            # boolean_array = (output_mesh == 3)
            vis(self.voxelized_mesh)
        else:
            print("Error: No voxel mesh to visualize. Generate or load it first.")

    def get_voxel_mesh(self):
        """
        Return the voxelized mesh.

        Returns:
        numpy.ndarray: The voxelized mesh array.
        """
        if self.voxelized_mesh is not None:
            return self.voxelized_mesh
        else:
            print("Error: No voxel mesh available. Generate or load it first.")
            return None

    def generate_lbm_mesh(self):
        """
        Generate the voxelized mesh for the geometry based on the .geo template.
        """
        print(f"Generating LBM mesh for geometry '{self.name}'...")

        if self.voxelized_mesh is None:
            self.lbm_mesh = None
        else:
            self.lbm_mesh = generate_lbm_mesh(
                self.voxelized_mesh,
                expected_in_outs=self.expected_in_outs,
                leading_multiple=self.leading_multiple,
            )

            print(f"LBM mesh generation complete. Shape: {self.lbm_mesh.shape}")

    def save_lbm_mesh_to_text(self, filename="lbm_mesh.txt"):
        """
        Save the voxelized mesh as a text file in the specified output directory.

        Parameters:
        filename (str, optional): Name of the text file to save the voxel mesh. Default is 'voxel_mesh.txt'.
        """
        if self.lbm_mesh is not None:
            # Use the same 3-file format as voxel text export for consistency
            geom_file = os.path.join(self.output_dir, f"geom_{filename}")
            dim_file = os.path.join(self.output_dir, f"dim_{filename}")
            val_file = os.path.join(self.output_dir, f"val_{filename}")

            array_to_textfile(self.lbm_mesh, geom_file)
            with open(dim_file, "w") as f:
                shape = self.lbm_mesh.shape
                f.write(f"{shape[0]} {shape[1]} {shape[2]}\n")
            open(val_file, "w").close()
            print(f"LBM mesh saved to {self.output_dir} in triplet format")
        else:
            print(
                "Error: No voxel mesh to save. Generate it first using 'generate_voxel_mesh'."
            )


if __name__ == "__main__":
    pass
    # geom = Geometry(
    #      name="junction_2d",
    #      resolution=4,
    #      num_processes=4,
    #      offset=0.01,
    #      lower_angle=10,
    #      upper_angle=0,
    #      upper_flare=0.001,
    #      lower_flare=0.002,
    #      h=0.005,
    #      expected_in_outs={"W", "E", "S", "N"},
    # )
    # name_hash = geom.name_hash
    # state = geom.state
    # geom.generate_voxel_mesh()
    # geom.save_voxel_mesh_to_text(f"{name_hash}.txt")
    # geom.visualize()
