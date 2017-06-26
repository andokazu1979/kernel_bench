##############################
# Default variables
##############################

# CXX: C++ compiler command
CXX = g++

# CXXFLAGS: C++ compiler options
CXXFLAGS = 

# LDFLAGS: Linker options
LDFLAGS =

# EXEC: Executable
EXEC = kernel_bench

# OBJS: object files to link
OBJS = kernel_bench.o kernel.o timer.o 

##############################
# Options
##############################

# CBLAS support
CBLAS=OFF

# Eigen support
EIGEN=OFF

# CUDA support
CUDA=OFF

ifeq ($(CBLAS),ON)
  CXX = icpc
  CXXFLAGS += -DCBLAS
  LDFLAGS += -mkl
endif

ifeq ($(EIGEN),ON)
  CXX = icpc
  CXXFLAGS += -DEIGEN -I ~/eigen-eigen-XXXXXXXXXXXX
endif

ifeq ($(CUDA),ON)
  NVCC = nvcc
  NVCCFLAGS += -DCUDA -arch=sm_60
  CXXFLAGS += -DCUDA
  LDFLAGS += -L/usr/local/cuda/lib64 -lcuda -lcudart
  OBJS += kernel_cuda.o
endif

##############################
# Make rules
##############################

# Link all objects and create executable binary file
$(EXEC): $(OBJS)
	$(CXX) -o $@ $(LDFLAGS) $^

# Make rule for c++ source code
%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $^

# Make rule for CUDA source code
%.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) $^

# Target for clean generated files
clean:
	rm -f *.o *.lst $(EXEC)
