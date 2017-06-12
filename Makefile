# CXX: C++ compiler command
CXX = g++

# CXXFLAGS: C++ compiler options
CXXFLAGS = 

# EXEC: Executable
EXEC = kernel_bench

# OBJS: object files to link
OBJS = kernel_bench.o kernel.o timer.o

$(EXEC): $(OBJS)
	$(CXX) -o $@ $(CXXFLAGS) $^

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $^

clean:
	rm -f *.o *.lst $(EXEC)
