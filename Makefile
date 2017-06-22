##############################
# Default variables
##############################

# CXX: C++ compiler command
CXX = g++

# CXXFLAGS: C++ compiler options
CXXFLAGS = 

# EXEC: Executable
EXEC = kernel_bench

# OBJS: object files to link
OBJS = kernel_bench.o kernel.o timer.o kernel_cuda.o 

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
  CXXFLAGS += -DCBLAS -mkl
endif

ifeq ($(EIGEN),ON)
  CXX = icpc
  CXXFLAGS += -DEIGEN -I ~/eigen-eigen-XXXXXXXXXXXX
endif

ifeq ($(CUDA),ON)
  NVCC = nvcc
  NVCCFLAGS += -DCUDA
  CXXFLAGS += -DCUDA -L/usr/local/cuda/lib64 -lcuda -lcudart 
endif

##############################
# Make rules
##############################

# Link all objects and create executable binary file
$(EXEC): $(OBJS)
	$(CXX) -o $@ $(CXXFLAGS) $^

# Make rule for c++ source code
%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $^

# Make rule for CUDA source code
%.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) $^

# Target for clean generated files
clean:
	rm -f *.o *.lst $(EXEC)
