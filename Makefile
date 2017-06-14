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
OBJS = kernel_bench.o kernel.o timer.o

##############################
# Options
##############################

# CBLAS support
CBLAS=OFF

# Eigen support
EIGEN=OFF

ifeq ($(CBLAS),ON)
  CXX = icpc
  CXXFLAGS += -DCBLAS -mkl
endif

ifeq ($(EIGEN),ON)
  CXX = icpc
  CXXFLAGS += -DEIGEN -I ~/eigen-eigen-XXXXXXXXXXXX
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

# Target for clean generated files
clean:
	rm -f *.o *.lst $(EXEC)
