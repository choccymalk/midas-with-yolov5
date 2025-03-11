# clang-linux-toolchain.cmake - Toolchain file for Clang cross-compiling to Linux

# -------- General Configuration --------
# Specify the target system
SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_VERSION 1) # A dummy value; can be overridden

# Specify the target architecture (replace with your target, e.g., arm64, armv7, x86_64)
SET(CMAKE_SYSTEM_PROCESSOR aarch64) # EXAMPLE:  Change to your target architecture

# -------- Compiler Locations --------
#  These paths *must* be absolute
#  Replace with the *actual* paths to your cross-compilation tools.

# **IMPORTANT:** If you are using a pre-built toolchain (e.g., from a distribution package
# or a buildroot/Yocto SDK), you usually want to use the compilers that come with that
# toolchain.  This toolchain should also set up the correct sysroot.
# Otherwise, you might have to build your own Clang/LLVM toolchain specifically for
# cross-compilation.

SET(CMAKE_C_COMPILER   /opt/homebrew/opt/llvm/bin/clang)   # EXAMPLE: Replace with your Clang C compiler path
SET(CMAKE_CXX_COMPILER /opt/homebrew/opt/llvm/bin/clang++) # EXAMPLE: Replace with your Clang C++ compiler path
SET(CMAKE_AR           /opt/homebrew/opt/llvm/bin/llvm-ar)  # EXAMPLE: Replace with your LLVM archiver path
SET(CMAKE_RANLIB       /opt/homebrew/opt/llvm/bin/llvm-ranlib) # EXAMPLE: Replace with your LLVM ranlib path
SET(CMAKE_LINKER       /opt/homebrew/opt/lld/bin/lld) #EXAMPLE: use lld. use /opt/cross/bin/ld if necessary

# -------- Sysroot Location --------
# Sysroot is the directory containing the target system's headers and libraries.
# This path *must* be absolute.

SET(CMAKE_SYSROOT /opt/homebrew/opt/llvm/)  # EXAMPLE:  Replace with your target sysroot path

# -------- Search Path Configuration --------
# These settings control how CMake finds programs, libraries, and headers.

# Don't search for programs (like 'make') on the target system
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# Do search for libraries and headers in the target environment
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# -------- Target-Specific Flags (Optional) --------
#  You can add target-specific compiler and linker flags here.
#  For example, you might want to set the CPU architecture or enable specific features.

# Example for ARM:
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8")

# Example for specific optimization
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Ofast")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")

# Example flags to help statically link musl if that's what you're using
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static -fuse-ld=/opt/cross/bin/ld")
# set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static -fuse-ld=/opt/cross/bin/ld")

# --------- Custom Commands ------------
# If you need to perform specific actions during the build process (e.g., copying files
# to the target system), you can add custom commands here.

# -------- Important Notes --------
# 1.  Replace the placeholder paths with the *actual* paths to your cross-compilation tools and sysroot.
# 2.  Ensure that the Clang version you are using is compatible with the target system.
# 3.  You may need to adjust the target-specific flags depending on your hardware and software requirements.
# 4. This file assumes a basic Linux system.  For embedded systems or systems with specific
#    toolchain setups (e.g., Buildroot, Yocto), you might need to adapt this file further.
