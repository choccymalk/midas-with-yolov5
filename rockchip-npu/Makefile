# Makefile for rockchip_npu

# Compiler
CXX = g++

# Compiler flags (adjust as needed)
CXXFLAGS = -std=c++11 -Wall -Iinclude -I/opt/homebrew/opt/opencv/include/opencv4/opencv2

# Linker flags (adjust OpenCV path if needed)
LDFLAGS = `pkg-config --libs opencv4` # Using pkg-config for OpenCV libraries

# Source files
SOURCES = $(shell find src -name "*.cpp")
SOURCES += $(shell find . -name "*.cpp" -not -path "src/*") # find cpp files in the root directory
SOURCES += $(shell find . -name "*.c")  # find c files in the root directory
SOURCES += $(shell find include -name "*.hpp") # Find hpp files in the include directory
SOURCES += $(shell find . -name "*.hpp" -not -path "include/*") # Find hpp files in the root directory
SOURCES += $(shell find . -name "*.h")  # find h files in the root directory
SOURCES += $(shell find include -name "*.h") # Find h files in the include directory

# Object files
OBJECTS = $(SOURCES:.cpp=.o)
OBJECTS = $(OBJECTS:.c=.o) # handle .c files as well

# Executable name
EXECUTABLE = rockchip_npu

# Default target
all: $(EXECUTABLE)

# Compile object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.c
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link object files to create the executable
$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(LDFLAGS) $(OBJECTS) -o $(EXECUTABLE)

# Clean object files and executable
clean:
	rm -f $(OBJECTS) $(EXECUTABLE)

run:
	./$(EXECUTABLE)