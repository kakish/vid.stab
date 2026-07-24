cmake_minimum_required (VERSION 3.16)
project (vid.stab C)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules/")

include (FindSSE)
include (GNUInstallDirs)
# include (ExternalProject) # There must be a nicer way to include this
# include (sse2neon)

# AppleClang does not ship its own OpenMP runtime (unlike upstream Clang),
# so find_package(OpenMP) fails out of the box on macOS even when OpenMP
# support is perfectly usable via Homebrew's libomp. Point CMake at it
# explicitly before searching, when available -- this only takes effect
# if find_package(OpenMP) would otherwise fail, since it lets FindOpenMP's
# own compile checks confirm the flags actually work.
if(APPLE AND CMAKE_C_COMPILER_ID MATCHES "Clang")
  find_program(HOMEBREW_EXECUTABLE brew)
  if(HOMEBREW_EXECUTABLE)
    execute_process(COMMAND ${HOMEBREW_EXECUTABLE} --prefix libomp
                     OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
                     OUTPUT_STRIP_TRAILING_WHITESPACE
                     RESULT_VARIABLE HOMEBREW_LIBOMP_RESULT
                     ERROR_QUIET)
    if(HOMEBREW_LIBOMP_RESULT EQUAL 0 AND EXISTS "${HOMEBREW_LIBOMP_PREFIX}/include/omp.h")
      set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
      set(OpenMP_C_LIB_NAMES "omp")
      set(OpenMP_omp_LIBRARY "${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib")
    endif()
  endif()
endif()
find_package(OpenMP)

set(MAJOR_VERSION 1)
set(MINOR_VERSION 2)
set(PATCH_VERSION 0)
set(VIDSTAB_VERSION ${MAJOR_VERSION}.${MINOR_VERSION}${PATCH_VERSION})

# Default to release builds if no explicit build type specified.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release")
endif()

option(BUILD_SHARED_LIBS "build shared libraries instead of static libraries"
       ON)

option(USE_OMP "use parallelization use OMP" ON)

option(USE_MPS "Enable Apple Metal Performance Shaders acceleration" ON)
if(USE_MPS AND NOT APPLE)
  set(USE_MPS OFF)
endif()
if(USE_MPS)
  enable_language(OBJCXX)
  add_definitions(-DUSE_MPS)
endif()

set(CMAKE_C_STANDARD 99)

if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
  add_definitions(-Wall -Wno-pointer-sign)
elseif(MSVC)
  add_definitions(/W3)
endif()

if(NOT WIN32)
  add_definitions(-fPIC)
endif()

if(NOT WIN32)
  add_definitions(-fPIC)
endif()

if(APPLE AND NOT CMAKE_OSX_ARCHITECTURES AND NOT SSE2_FOUND)
  add_definitions(-march=armv8-a+fp+simd+crypto+crc)
endif()

# here we should check for SSE2
# our  -DUSE_SSE2_ASM code does not work with fpic
if (NOT MSVC)
  if(SSE2_FOUND)
      add_definitions( -DUSE_SSE2 -msse2 -ffast-math )
endif ()

endif()
if(MSVC AND USE_OMP)
  add_compile_options(/openmp)
endif()
if(USE_OMP AND OPENMP_FOUND)
add_definitions(${OpenMP_C_FLAGS} -DUSE_OMP)
endif()

set(SOURCES src/frameinfo.c src/transformtype.c src/libvidstab.c
  src/transform.c src/transformfixedpoint.c src/motiondetect.c
  src/motiondetect_opt.c src/serialize.c src/localmotion2transform.c
  src/boxblur.c src/vsvector.c)

set(HEADERS src/frameinfo.h src/transformtype.h src/libvidstab.h
  src/transform.h src/motiondetect.h src/serialize.h
  src/localmotion2transform.h src/boxblur.h src/vsvector.h )

if(USE_MPS)
  list(APPEND SOURCES src/motiondetect_mps.mm)
  list(APPEND HEADERS src/motiondetect_mps.h)
endif()


# Create the vidstab library
add_library (vidstab ${SOURCES})

#set version of lib
set_target_properties(vidstab PROPERTIES SOVERSION ${MAJOR_VERSION}.${MINOR_VERSION})


if(MSVC)
  target_compile_definitions(vidstab PRIVATE VIDSTAB_EXPORTS)
endif()
target_include_directories(vidstab PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)
if (NOT MSVC)
  target_link_libraries(vidstab m)
endif ()
set(PKG_EXTRA_LIBS -lm)
if(USE_OMP AND OPENMP_FOUND)
if(TARGET OpenMP::OpenMP_C)
target_link_libraries(vidstab OpenMP::OpenMP_C)
endif()
set(PKG_EXTRA_LIBS "${PKG_EXTRA_LIBS} ${OpenMP_C_FLAGS}")
endif()

if(USE_MPS)
  set_source_files_properties(src/motiondetect_mps.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
  # NOTE: plain (non-keyword) signature, matching the other
  # target_link_libraries(vidstab ...) calls above -- CMake does not allow
  # mixing plain and PRIVATE/PUBLIC/INTERFACE keyword signatures for the
  # same target within a directory scope.
  target_link_libraries(vidstab "-framework Metal"
                                 "-framework MetalPerformanceShaders"
                                 "-framework Foundation")
endif()


#if(!NOHEADERS)
FILE(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")
INSTALL(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/vid.stab)
#endif()

INSTALL(TARGETS vidstab
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

include(create_pkgconfig_file)
create_pkgconfig_file(vidstab "Vid.Stab, a library for stabilizing video clips")
