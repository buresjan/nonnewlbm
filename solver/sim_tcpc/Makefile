defaults: all

include ../config.mk
include ../flags.mk

CPPFLAGS += -I../lbm3d
LDFLAGS += -L ../lbm3d -L .
NVCCFLAGS += -I../lbm3d

CPP_SOURCES = $(wildcard *.cpp) $(wildcard ../lbm3d/*.cpp)
CU_SOURCES = $(wildcard *.cu) $(wildcard ../lbm3d/*.cu)
#SOURCES = $(CPP_SOURCES) $(CU_SOURCES)
#SIMS = $(wildcard sim?*.cu) $(wildcard ../lbm3d/sim?*.cu)
SIMS_LBM3D = $(wildcard ../lbm3d/sim?*.cu)
SIMS = $(wildcard sim?*.cu)
TARGETS = $(SIMS:%.cu=%)

# OBJS are *.o except sim?*.o
OBJS = $(filter-out $(SIMS:%.cu=%.o) $(SIMS_LBM3D:%.cu=%.o), $(CU_SOURCES:%.cu=%.o))  $(CPP_SOURCES:%.cpp=%.o)

all: $(TARGETS)

clean:
	rm -f *.o *.d $(TARGETS) Makefile_autoflags.h

%.o: %.cpp
	$(CXX) -MD -MP $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

%.o: %.cu
ifeq ($(use_CUDA),yes)
	$(NVCC) -M $(NVCCFLAGS) -o $(@:%.o=%.d) $<
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c -o $@ $<
else
#use CXX instead of NVCC --- important for using lgomp and fopenmp for CPU based LBM
	$(CXX) -MD -MP $(CPPFLAGS) $(CXXFLAGS) -xc++ -c -o $@ $<
endif

$(TARGETS): % : %.o $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)

-include $(CPP_SOURCES:%.cpp=%.d)
-include $(CU_SOURCES:%.cu=%.d)
