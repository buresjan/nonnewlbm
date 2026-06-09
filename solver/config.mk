#use_CUDA = no
use_CUDA = yes
use_MPI = yes

# TCPC geometry is read from a VTK rectilinear grid.
use_VTK = yes

use_CUSPARSE= no
#use_CUSPARSE = yes

use_CYModel = yes
#use_CASSON = yes

use_UMFPACK = no
#use_UMFPACK = yes

# TODO: add support for petsc if needed for lagrange3D
##use_PETSC = no
##use_PETSC = yes

# TCPC flow cases do not use the immersed-boundary Lagrange solver. Leave its
# TNL sparse-matrix path off by default because old TNL matrix templates do not
# compile cleanly with recent CUDA/GCC versions.
use_TNL_LAGRANGE = no

# Override this if pkg-config does not know your VTK installation.
VTK_CONFIG ?= $(shell pkg-config --cflags --libs vtk 2>/dev/null || pkg-config --cflags --libs vtk-9.4 2>/dev/null || echo "-I/usr/include/vtk -lvtkCommonCore -lvtkIOLegacy -lvtkCommonDataModel -lvtkIOXML -lvtksys")


GPU_ARCH ?= sm_75

CXX=g++

ifeq ($(use_CUDA),no)
	use_CUSPARSE=no
endif
