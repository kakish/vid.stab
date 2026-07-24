# to use intel compiler: cd cmake; CC=icc CXX=icpc cmake ../
#  very fast: faster than orc code at imgcompare without any options.
#  library needs libimf.so (link statically?)

cmake_minimum_required (VERSION 2.6)
project (vid.stab)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/../CMakeModules/")

include (FindSSE)

option(USE_OMP "use parallelization use OMP" ON)

option(USE_MPS "Enable Apple Metal Performance Shaders acceleration" ON)
if(USE_MPS AND NOT APPLE)
  set(USE_MPS OFF)
endif()
if(USE_MPS)
  enable_language(OBJCXX)
  add_definitions(-DUSE_MPS)
endif()

# Default to debug builds if no explicit build type specified.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Debug")
endif()

add_definitions(-Wall -Wno-pointer-sign -DTESTING)
add_compile_options($<$<COMPILE_LANGUAGE:C>:-std=gnu99>)

if(SSE2_FOUND)
add_definitions( -DUSE_SSE2 -msse2 -ffast-math -fno-show-column ) # -DUSE_SSE2_ASM
endif()

if(USE_OMP)
  set(VIDSTAB_OMP_C_FLAGS "-fopenmp")
  set(VIDSTAB_OMP_LIB "gomp")
  # AppleClang has no native OpenMP runtime (unlike upstream Clang) --
  # bare -fopenmp/-lgomp silently fail to compile/link there. Use
  # Homebrew's libomp instead, when available, mirroring the root
  # CMakeLists.txt's find_package(OpenMP) detection.
  if(APPLE AND CMAKE_C_COMPILER_ID MATCHES "Clang")
    find_program(HOMEBREW_EXECUTABLE brew)
    if(HOMEBREW_EXECUTABLE)
      execute_process(COMMAND ${HOMEBREW_EXECUTABLE} --prefix libomp
                       OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
                       OUTPUT_STRIP_TRAILING_WHITESPACE
                       RESULT_VARIABLE HOMEBREW_LIBOMP_RESULT
                       ERROR_QUIET)
    endif()
    if(HOMEBREW_LIBOMP_RESULT EQUAL 0 AND EXISTS "${HOMEBREW_LIBOMP_PREFIX}/include/omp.h")
      set(VIDSTAB_OMP_C_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
      set(VIDSTAB_OMP_LIB "${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib")
    else()
      message(WARNING "USE_OMP requested but no usable OpenMP runtime found for AppleClang (Homebrew libomp not found) -- disabling")
      set(USE_OMP OFF)
    endif()
  endif()
endif()
if(USE_OMP)
  add_definitions(${VIDSTAB_OMP_C_FLAGS} -DUSE_OMP)
endif()

# Make sure the compiler can find include files from transcode
include_directories (../src)

set(TEST_SOURCES tests.c testutils.c testframework.c ../src/vsvector.c
  ../src/transform.c ../src/transformfloat.c ../src/transformfixedpoint.c
  ../src/libvidstab.c ../src/transformtype.c ../src/frameinfo.c
  ../src/serialize.c ../src/localmotion2transform.c
  ../src/motiondetect.c ../src/motiondetect_opt.c ../src/boxblur.c)

if(USE_MPS)
  list(APPEND TEST_SOURCES ../src/motiondetect_mps.mm)
endif()

add_executable (tests ${TEST_SOURCES})

target_link_libraries(tests m)
if(USE_OMP)
target_link_libraries(tests ${VIDSTAB_OMP_LIB})
endif()

if(USE_MPS)
  set_source_files_properties(../src/motiondetect_mps.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
  target_link_libraries(tests "-framework Metal"
                               "-framework MetalPerformanceShaders"
                               "-framework Foundation")
endif()
