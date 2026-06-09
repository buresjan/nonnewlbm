CPPFLAGS = -I.
CXXFLAGS = -O3 -fopenmp -funroll-loops -std=c++14
LDFLAGS = -lgomp -lrt -lpng

ifeq ($(use_CUDA),yes)
	NVCC=nvcc
	NVCCFLAGS=  -std=c++14 --use_fast_math -O3 -arch $(GPU_ARCH) -Wno-deprecated-gpu-targets
	# flag for debug
	NVCCFLAGS += -g
	# flag for OpenMP with nvcc
	NVCCFLAGS += -Xcompiler -fopenmp
	# disable useless nvcc warning
	#NVCCFLAGS += -Xcudafe --diag_suppress=declared_but_not_referenced -Xcudafe --diag_suppress=code_is_unreachable -Xcudafe --diag_suppress=loop_not_reachable -Xcudafe --diag_suppress=implicit_return_from_non_void_function -Xcudafe --diag_suppress=unsigned_compare_with_zero -Xcudafe --diag_suppress=2906 -Xcudafe --diag_suppress=2913 -Xcudafe --diag_suppress=2886 -Xcudafe --diag_suppress=2929 -Xcudafe --diag_suppress=2977 -Xcudafe --diag_suppress=3057 -Xcudafe --diag_suppress=3124 -Xcudafe --display_error_number
	#NVCCFLAGS += -Xcudafe "\"--diag_suppress=declared_but_not_referenced --diag_suppress=code_is_unreachable --diag_suppress=implicit_return_from_non_void_function --diag_suppress=unsigned_compare_with_zero --diag_suppress=2913 --diag_suppress=2906 --diag_suppress=2929 --diag_suppres=185 --diag_suppress=2977 --display_error_number \""
	# profiling/debugging
	#NVCCFLAGS += -Xptxas -v -lineinfo
	# for optimized MPI synchronizations
	NVCCFLAGS += --default-stream per-thread
else
	NVCC=$(CXX)
	NVCCFLAGS= -std=c++14
endif

# CUDA
ifeq ($(use_CUDA),yes)
	NVCCFLAGS += -DUSE_CUDA
	CXXFLAGS +=  -DUSE_CUDA
endif

# MPI
ifeq ($(use_MPI),yes)
	NVCCFLAGS += --compiler-bindir mpicxx -DHAVE_MPI
	CXXFLAGS += -DHAVE_MPI
	CXX = mpicxx
endif

# VTK
ifeq ($(use_VTK),yes)
	NVCCFLAGS += -DUSE_VTK
	NVCCFLAGS += $(VTK_CONFIG)
	CXXFLAGS +=  -DUSE_VTK
	CXXFLAGS += $(VTK_CONFIG)
	LDFLAGS += $(VTK_CONFIG)
endif

# CYMODEL
ifeq ($(use_CYModel),yes)
	NVCCFLAGS += -D USE_CYMODEL
	CXXFLAGS +=  -D USE_CYMODEL
endif


# CASSON
ifeq ($(use_CASSON),yes)
	NVCCFLAGS += -D USE_CASSON
	CXXFLAGS +=  -D USE_CASSON
endif

# TNL flags. Keep this overrideable so the repository does not need to vendor
# a machine-local TNL checkout.
TNL_DIR ?= ../tnl_submodule
TNL_INCLUDE_DIRS := -I $(TNL_DIR)/src/ -I $(TNL_DIR)/src/3rdparty/
NVCCFLAGS += -DHAVE_OPENMP -DNDEBUG $(TNL_INCLUDE_DIRS)
CXXFLAGS  += -DHAVE_OPENMP -DNDEBUG $(TNL_INCLUDE_DIRS)
ifeq ($(use_TNL_LAGRANGE),yes)
	NVCCFLAGS += -DUSE_TNL
	CXXFLAGS += -DUSE_TNL
endif
ifeq ($(use_CUDA),yes)
	NVCCFLAGS += --expt-relaxed-constexpr --expt-extended-lambda
	NVCCFLAGS += -DHAVE_CUDA
endif

ifeq ($(use_CUSPARSE),yes)
	NVCCFLAGS += -DUSE_CUSPARSE
	CXXFLAGS +=  -DUSE_CUSPARSE
#	NVCCFLAGS +=  -lcublas_static -lcusparse_static -lculibos
	NVCCFLAGS +=  -lcublas -lcusparse -lculibos
endif

# UMFPACK
ifeq ($(use_UMFPACK),yes)
	NVCCFLAGS += -DUSE_UMFPACK
	NVCCFLAGS += -lblas -lamd -lumfpack
	CXXFLAGS +=  -DUSE_UMFPACK
	CXXFLAGS += -lblas -lamd -lumfpack
endif
