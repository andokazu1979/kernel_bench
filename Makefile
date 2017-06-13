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
