/*
 *  motiondetect_mps.mm
 *
 *  Copyright (C) Georg Martius - February 2011
 *   georg dot martius at web dot de
 *  Copyright (C) Alexey Osipov - Jule 2011
 *   simba at lerlan dot ru
 *   speed optimizations (threshold, spiral, SSE, asm)
 *
 *  This file is part of vid.stab video stabilization library
 *
 *  vid.stab is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License,
 *  as published by the Free Software Foundation; either version 2, or
 *  (at your option) any later version.
 *
 *  vid.stab is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with GNU Make; see the file COPYING.  If not, write to
 *  the Free Software Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 */
#include "motiondetect_mps.h"

#ifdef USE_MPS

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <stdlib.h>

#include "vidstabdefines.h"

// Number of threads per threadgroup used for both block-matching dispatches.
// Must match the size of the threadgroup-shared arrays in the MSL kernels
// below and must be a power of two for the tree reduction to be exact.
#define VS_MPS_THREADGROUP_SIZE 256

// Two-stage block-matching kernel source, compiled at runtime in
// vsMPSAccelCreate() via -[MTLDevice newLibraryWithSource:options:error:].
//
// One threadgroup handles one measurement field in both passes. Both share
// vsSAD(), which computes the same metric as compareSubImg_thr() on the CPU
// path (sum of |curr-prev| over a fieldSize x fieldSize block), and the
// same threadgroup-shared-memory argmin reduction (no early-exit -- that
// only helps a sequential CPU loop, not a GPU threadgroup, due to
// SIMD-group divergence).
//
// Pass 1 (vsMPSBlockMatchCoarse): evaluates a stepSize-spaced grid over
// the full [-maxShift,maxShift]^2 window -- mirrors the CPU's coarse-grid
// stage -- and writes each field's approximate winner (dx,dy) to an
// intermediate buffer. Candidate count ~ (2*maxShift/stepSize + 1)^2.
//
// Pass 2 (vsMPSBlockMatchRefine): densely (1px resolution) searches a
// small window of radius stepSize around that winner and writes the final
// (dx,dy,minSAD). Candidate count ~ (2*stepSize + 1)^2.
//
// A single fully-dense pass (candidate count ~ (2*maxShift+1)^2) does not
// scale to high resolutions: at 4K, maxShift is ~300px, making the dense
// window ~600x600 candidates per field -- measured several times SLOWER
// than the CPU path it was meant to replace. This two-stage split brings
// the candidate count back down to the same order of magnitude the CPU
// evaluates, while still evaluating every candidate in parallel.
static const char* const vs_mps_kernel_source =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"struct VSMPSFieldGPU {\n"
"  int x;\n"
"  int y;\n"
"  int size;\n"
"  int offX;\n"
"  int offY;\n"
"};\n"
"\n"
"inline uint vsSAD(device const uchar* curr, device const uchar* prev,\n"
"                   int fx, int fy, int size,\n"
"                   int currLinesize, int prevLinesize,\n"
"                   int dx, int dy) {\n"
"  int s2 = size / 2;\n"
"  device const uchar* p1 = curr + ((fx - s2) + (fy - s2) * currLinesize);\n"
"  device const uchar* p2 = prev + ((fx - s2 + dx) + (fy - s2 + dy) * prevLinesize);\n"
"  uint sum = 0;\n"
"  for (int j = 0; j < size; j++) {\n"
"    for (int k = 0; k < size; k++) {\n"
"      int v1 = (int)p1[j * currLinesize + k];\n"
"      int v2 = (int)p2[j * prevLinesize + k];\n"
"      sum += (uint)abs(v1 - v2);\n"
"    }\n"
"  }\n"
"  return sum;\n"
"}\n"
"\n"
"// shared by both kernels: reduce this thread's local best into a\n"
"// per-threadgroup argmin, tie-broken by smallest |shift| (matches the\n"
"// CPU spiral search, which visits shifts in expanding rings from zero).\n"
"inline void vsReduceArgmin(threadgroup uint* sadShared, threadgroup int* dxShared,\n"
"                            threadgroup int* dyShared, uint localIdx, uint localCount,\n"
"                            uint bestSAD, int bestDx, int bestDy,\n"
"                            thread uint& outSAD, thread int& outDx, thread int& outDy) {\n"
"  sadShared[localIdx] = bestSAD;\n"
"  dxShared[localIdx] = bestDx;\n"
"  dyShared[localIdx] = bestDy;\n"
"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  for (uint stride = localCount / 2; stride > 0; stride >>= 1) {\n"
"    if (localIdx < stride) {\n"
"      uint otherIdx = localIdx + stride;\n"
"      uint otherSAD = sadShared[otherIdx];\n"
"      uint curSAD = sadShared[localIdx];\n"
"      int odx = dxShared[otherIdx], ody = dyShared[otherIdx];\n"
"      int cdx = dxShared[localIdx], cdy = dyShared[localIdx];\n"
"      int otherDistSq = odx * odx + ody * ody;\n"
"      int curDistSq = cdx * cdx + cdy * cdy;\n"
"      if (otherSAD < curSAD || (otherSAD == curSAD && otherDistSq < curDistSq)) {\n"
"        sadShared[localIdx] = otherSAD;\n"
"        dxShared[localIdx] = odx;\n"
"        dyShared[localIdx] = ody;\n"
"      }\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  }\n"
"  outSAD = sadShared[0];\n"
"  outDx = dxShared[0];\n"
"  outDy = dyShared[0];\n"
"}\n"
"\n"
"kernel void vsMPSBlockMatchCoarse(\n"
"    device const uchar*      curr         [[buffer(0)]],\n"
"    device const uchar*      prev         [[buffer(1)]],\n"
"    constant VSMPSFieldGPU*  fields       [[buffer(2)]],\n"
"    device int*               outCoarseXY [[buffer(3)]],\n"
"    constant int&             currLinesize [[buffer(4)]],\n"
"    constant int&             prevLinesize [[buffer(5)]],\n"
"    constant int&             maxShift     [[buffer(6)]],\n"
"    constant int&             stepSize     [[buffer(7)]],\n"
"    uint                      fieldIdx     [[threadgroup_position_in_grid]],\n"
"    uint                      localIdx     [[thread_position_in_threadgroup]],\n"
"    uint                      localCount   [[threads_per_threadgroup]])\n"
"{\n"
"  VSMPSFieldGPU f = fields[fieldIdx];\n"
"  int halfWin = maxShift / stepSize;\n"
"  int window = 2 * halfWin + 1;\n"
"  int numCandidates = window * window;\n"
"\n"
"  uint bestSAD = 0xFFFFFFFFu;\n"
"  int bestDx = 0;\n"
"  int bestDy = 0;\n"
"\n"
"  for (uint c = localIdx; c < (uint)numCandidates; c += localCount) {\n"
"    int ix = (int)(c % (uint)window) - halfWin;\n"
"    int iy = (int)(c / (uint)window) - halfWin;\n"
"    int dx = ix * stepSize;\n"
"    int dy = iy * stepSize;\n"
"    uint sum = vsSAD(curr, prev, f.x, f.y, f.size, currLinesize, prevLinesize,\n"
"                      f.offX + dx, f.offY + dy);\n"
"    int distSq = dx * dx + dy * dy;\n"
"    int bestDistSq = bestDx * bestDx + bestDy * bestDy;\n"
"    if (sum < bestSAD || (sum == bestSAD && distSq < bestDistSq)) {\n"
"      bestSAD = sum; bestDx = dx; bestDy = dy;\n"
"    }\n"
"  }\n"
"\n"
"  threadgroup uint sadShared[VS_MPS_THREADGROUP_SIZE_TOKEN];\n"
"  threadgroup int  dxShared[VS_MPS_THREADGROUP_SIZE_TOKEN];\n"
"  threadgroup int  dyShared[VS_MPS_THREADGROUP_SIZE_TOKEN];\n"
"  uint redSAD; int redDx, redDy;\n"
"  vsReduceArgmin(sadShared, dxShared, dyShared, localIdx, localCount,\n"
"                 bestSAD, bestDx, bestDy, redSAD, redDx, redDy);\n"
"\n"
"  if (localIdx == 0) {\n"
"    outCoarseXY[fieldIdx * 2 + 0] = redDx;\n"
"    outCoarseXY[fieldIdx * 2 + 1] = redDy;\n"
"  }\n"
"}\n"
"\n"
"kernel void vsMPSBlockMatchRefine(\n"
"    device const uchar*      curr        [[buffer(0)]],\n"
"    device const uchar*      prev        [[buffer(1)]],\n"
"    constant VSMPSFieldGPU*  fields      [[buffer(2)]],\n"
"    device const int*        coarseXY    [[buffer(3)]],\n"
"    device int*               outXY      [[buffer(4)]],\n"
"    device uint*              outSAD     [[buffer(5)]],\n"
"    constant int&             currLinesize [[buffer(6)]],\n"
"    constant int&             prevLinesize [[buffer(7)]],\n"
"    constant int&             stepSize     [[buffer(8)]],\n"
"    uint                      fieldIdx     [[threadgroup_position_in_grid]],\n"
"    uint                      localIdx     [[thread_position_in_threadgroup]],\n"
"    uint                      localCount   [[threads_per_threadgroup]])\n"
"{\n"
"  VSMPSFieldGPU f = fields[fieldIdx];\n"
"  int coarseDx = coarseXY[fieldIdx * 2 + 0];\n"
"  int coarseDy = coarseXY[fieldIdx * 2 + 1];\n"
"  int radius = stepSize;\n"
"  int window = 2 * radius + 1;\n"
"  int numCandidates = window * window;\n"
"\n"
"  uint bestSAD = 0xFFFFFFFFu;\n"
"  int bestDx = coarseDx;\n"
"  int bestDy = coarseDy;\n"
"\n"
"  for (uint c = localIdx; c < (uint)numCandidates; c += localCount) {\n"
"    int lx = (int)(c % (uint)window) - radius;\n"
"    int ly = (int)(c / (uint)window) - radius;\n"
"    int dx = coarseDx + lx;\n"
"    int dy = coarseDy + ly;\n"
"    uint sum = vsSAD(curr, prev, f.x, f.y, f.size, currLinesize, prevLinesize,\n"
"                      f.offX + dx, f.offY + dy);\n"
"    int distSq = dx * dx + dy * dy;\n"
"    int bestDistSq = bestDx * bestDx + bestDy * bestDy;\n"
"    if (sum < bestSAD || (sum == bestSAD && distSq < bestDistSq)) {\n"
"      bestSAD = sum; bestDx = dx; bestDy = dy;\n"
"    }\n"
"  }\n"
"\n"
"  threadgroup uint sadShared[VS_MPS_THREADGROUP_SIZE_TOKEN];\n"
"  threadgroup int  dxShared[VS_MPS_THREADGROUP_SIZE_TOKEN];\n"
"  threadgroup int  dyShared[VS_MPS_THREADGROUP_SIZE_TOKEN];\n"
"  uint redSAD; int redDx, redDy;\n"
"  vsReduceArgmin(sadShared, dxShared, dyShared, localIdx, localCount,\n"
"                 bestSAD, bestDx, bestDy, redSAD, redDx, redDy);\n"
"\n"
"  if (localIdx == 0) {\n"
"    outXY[fieldIdx * 2 + 0] = redDx;\n"
"    outXY[fieldIdx * 2 + 1] = redDy;\n"
"    outSAD[fieldIdx] = redSAD;\n"
"  }\n"
"}\n";

// VSMPSFieldGPU as seen by the host side (must stay layout-compatible with
// the MSL struct above: 5 consecutive 32-bit ints, no padding on either
// side since all members share the same size/alignment).
typedef struct {
  int32_t x;
  int32_t y;
  int32_t size;
  int32_t offX;
  int32_t offY;
} VSMPSFieldGPUHost;

// One cached, already-uploaded plane buffer, tagged with the caller-supplied
// generation number it corresponds to. See vsMPSGetOrUploadPlane() below.
typedef struct {
  bool valid;
  uint64_t generation;
  size_t length;
  id<MTLBuffer> buffer;
} VSMPSPlaneCacheEntry;

// The opaque VSMPSAccel is a plain C++ struct (not an Objective-C class) so
// that the public header does not need to know about Objective-C types.
// Because this file is compiled as Objective-C++ with ARC enabled
// (-fobjc-arc), the id<...> members below are automatically retained on
// assignment and released when the struct is destroyed.
//
// The inFlight* members hold the state of a submission between
// vsMPSBatchSearchSubmit() and vsMPSBatchSearchWait() -- exactly one
// submission may be outstanding at a time (enforced by inFlightCmdBuf
// being non-nil while one is pending).
struct VSMPSAccel {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> coarsePipeline;
  id<MTLComputePipelineState> refinePipeline;

  id<MTLCommandBuffer> inFlightCmdBuf;
  id<MTLBuffer> inFlightOutXYBuf;
  id<MTLBuffer> inFlightOutSADBuf;

  // 2-slot cache of uploaded plane buffers, keyed by the caller's generation
  // tag. In steady state this frame's "prev" plane is byte-identical to
  // last frame's "curr" plane, so it hits this cache instead of being
  // re-uploaded; likewise the coarse and fine batches within one frame
  // share the same curr/prev generations and so share the same buffers.
  VSMPSPlaneCacheEntry planeCache[2];

  // exponential-moving-average per-field cost estimates, seconds/field,
  // used by vsMPSSuggestCPUShare() to adapt the CPU/GPU work split to the
  // actual relative speed of the CPU and GPU on this machine -- indexed by
  // VSMPSBatchKind, since coarse and fine have very different per-field
  // cost profiles and must not share one estimate (see VSMPSBatchKind).
  double emaCpuPerField[2];
  double emaGpuPerField[2];
  bool haveTimingData[2];
};

VSMPSAccel* vsMPSAccelCreate(void) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      // No Metal-capable GPU. This is a normal, expected fallback path
      // (e.g. running on a Mac without a Metal device, or in some CI/VM
      // environments) -- do NOT log here, the caller decides.
      return NULL;
    }

    // Build the kernel source, substituting the threadgroup size token so
    // the shared-memory array size always matches VS_MPS_THREADGROUP_SIZE.
    NSString* tokenSource = [NSString stringWithUTF8String:vs_mps_kernel_source];
    NSString* sizeStr = [NSString stringWithFormat:@"%d", VS_MPS_THREADGROUP_SIZE];
    NSString* source = [tokenSource stringByReplacingOccurrencesOfString:@"VS_MPS_THREADGROUP_SIZE_TOKEN"
                                                               withString:sizeStr];

    NSError* error = nil;
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    if (!library) {
      return NULL;
    }
    id<MTLFunction> coarseFunction = [library newFunctionWithName:@"vsMPSBlockMatchCoarse"];
    id<MTLFunction> refineFunction = [library newFunctionWithName:@"vsMPSBlockMatchRefine"];
    if (!coarseFunction || !refineFunction) {
      return NULL;
    }
    id<MTLComputePipelineState> coarsePipeline = [device newComputePipelineStateWithFunction:coarseFunction
                                                                                        error:&error];
    id<MTLComputePipelineState> refinePipeline = [device newComputePipelineStateWithFunction:refineFunction
                                                                                        error:&error];
    if (!coarsePipeline || !refinePipeline) {
      return NULL;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) {
      return NULL;
    }

    VSMPSAccel* accel = new VSMPSAccel();
    accel->device = device;
    accel->queue = queue;
    accel->coarsePipeline = coarsePipeline;
    accel->refinePipeline = refinePipeline;
    accel->inFlightCmdBuf = nil;
    accel->inFlightOutXYBuf = nil;
    accel->inFlightOutSADBuf = nil;
    accel->planeCache[0].valid = false;
    accel->planeCache[0].buffer = nil;
    accel->planeCache[1].valid = false;
    accel->planeCache[1].buffer = nil;
    for (int k = 0; k < 2; k++) {
      accel->emaCpuPerField[k] = 0.0;
      accel->emaGpuPerField[k] = 0.0;
      accel->haveTimingData[k] = false;
    }
    return accel;
  }
}

void vsMPSAccelDestroy(VSMPSAccel* accel) {
  if (accel) {
    delete accel;
  }
}

// Returns an MTLBuffer holding `length` bytes from `hostPtr`, reusing a
// cached upload if one tagged with `generation` and the same length is
// already present (see VSMPSAccel::planeCache). Evicts the
// lowest-generation entry to make room when both slots are occupied by
// different generations -- safe even if a prior command buffer is still
// reading that entry's MTLBuffer, since Metal retains resources bound to
// an in-flight command buffer for the duration of its execution regardless
// of whether this cache still references them.
static id<MTLBuffer> vsMPSGetOrUploadPlane(VSMPSAccel* accel, const uint8_t* hostPtr,
                                            size_t length, uint64_t generation) {
  for (int i = 0; i < 2; i++) {
    if (accel->planeCache[i].valid && accel->planeCache[i].generation == generation &&
        accel->planeCache[i].length == length) {
      return accel->planeCache[i].buffer;
    }
  }

  id<MTLBuffer> buf = [accel->device newBufferWithBytes:hostPtr
                                                  length:length
                                                 options:MTLResourceStorageModeShared];
  if (!buf)
    return nil;

  int slot;
  if (!accel->planeCache[0].valid) {
    slot = 0;
  } else if (!accel->planeCache[1].valid) {
    slot = 1;
  } else {
    slot = (accel->planeCache[0].generation <= accel->planeCache[1].generation) ? 0 : 1;
  }
  accel->planeCache[slot].valid = true;
  accel->planeCache[slot].generation = generation;
  accel->planeCache[slot].length = length;
  accel->planeCache[slot].buffer = buf;
  return buf;
}

int vsMPSBatchSearchSubmit(VSMPSAccel* accel,
                            const uint8_t* curr, int currLinesize,
                            const uint8_t* prev, int prevLinesize, int height,
                            const Field* fields, const Vec* offsets, int count,
                            int maxShift, int stepSize,
                            uint64_t currGeneration, uint64_t prevGeneration) {
  if (!accel || !accel->device || !curr || !prev || !fields || !offsets || count <= 0)
    return VS_ERROR;
  if (accel->inFlightCmdBuf != nil)
    return VS_ERROR;  // a submission is already outstanding -- caller bug
  if (stepSize < 1)
    stepSize = 1;
  if (stepSize > maxShift)
    stepSize = maxShift > 0 ? maxShift : 1;

  @autoreleasepool {
    id<MTLDevice> device = accel->device;

    // linesize already accounts for any row padding, so height*linesize is
    // the exact number of bytes backing the plane.
    size_t currBytes = (size_t)height * (size_t)currLinesize;
    size_t prevBytes = (size_t)height * (size_t)prevLinesize;

    id<MTLBuffer> currBuf = vsMPSGetOrUploadPlane(accel, curr, currBytes, currGeneration);
    id<MTLBuffer> prevBuf = vsMPSGetOrUploadPlane(accel, prev, prevBytes, prevGeneration);
    if (!currBuf || !prevBuf)
      return VS_ERROR;

    size_t fieldBytes = sizeof(VSMPSFieldGPUHost) * (size_t)count;
    VSMPSFieldGPUHost* fieldParams = (VSMPSFieldGPUHost*) malloc(fieldBytes);
    if (!fieldParams)
      return VS_ERROR;
    for (int i = 0; i < count; i++) {
      fieldParams[i].x    = fields[i].x;
      fieldParams[i].y    = fields[i].y;
      fieldParams[i].size = fields[i].size;
      fieldParams[i].offX = offsets[i].x;
      fieldParams[i].offY = offsets[i].y;
    }
    id<MTLBuffer> fieldBuf = [device newBufferWithBytes:fieldParams
                                                  length:fieldBytes
                                                 options:MTLResourceStorageModeShared];
    free(fieldParams);
    if (!fieldBuf)
      return VS_ERROR;

    id<MTLBuffer> coarseXYBuf = [device newBufferWithLength:sizeof(int32_t) * 2 * (size_t)count
                                                     options:MTLResourceStorageModePrivate];
    id<MTLBuffer> outXYBuf = [device newBufferWithLength:sizeof(int32_t) * 2 * (size_t)count
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> outSADBuf = [device newBufferWithLength:sizeof(uint32_t) * (size_t)count
                                                   options:MTLResourceStorageModeShared];
    if (!coarseXYBuf || !outXYBuf || !outSADBuf)
      return VS_ERROR;

    int32_t currLinesizeI = currLinesize;
    int32_t prevLinesizeI = prevLinesize;
    int32_t maxShiftI = maxShift;
    int32_t stepSizeI = stepSize;

    id<MTLCommandBuffer> cmdBuf = [accel->queue commandBuffer];

    NSUInteger threadsPerGroup = VS_MPS_THREADGROUP_SIZE;
    NSUInteger maxThreadsCoarse = accel->coarsePipeline.maxTotalThreadsPerThreadgroup;
    NSUInteger maxThreadsRefine = accel->refinePipeline.maxTotalThreadsPerThreadgroup;
    NSUInteger maxThreads = MIN(maxThreadsCoarse, maxThreadsRefine);
    while (threadsPerGroup > maxThreads && threadsPerGroup > 1) {
      threadsPerGroup >>= 1;
    }
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadsPerGroup, 1, 1);
    MTLSize threadgroupsPerGrid = MTLSizeMake((NSUInteger)count, 1, 1);

    // Pass 1: coarse, stepSize-spaced grid search -> coarseXYBuf.
    id<MTLComputeCommandEncoder> coarseEncoder = [cmdBuf computeCommandEncoder];
    [coarseEncoder setComputePipelineState:accel->coarsePipeline];
    [coarseEncoder setBuffer:currBuf offset:0 atIndex:0];
    [coarseEncoder setBuffer:prevBuf offset:0 atIndex:1];
    [coarseEncoder setBuffer:fieldBuf offset:0 atIndex:2];
    [coarseEncoder setBuffer:coarseXYBuf offset:0 atIndex:3];
    [coarseEncoder setBytes:&currLinesizeI length:sizeof(int32_t) atIndex:4];
    [coarseEncoder setBytes:&prevLinesizeI length:sizeof(int32_t) atIndex:5];
    [coarseEncoder setBytes:&maxShiftI length:sizeof(int32_t) atIndex:6];
    [coarseEncoder setBytes:&stepSizeI length:sizeof(int32_t) atIndex:7];
    [coarseEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [coarseEncoder endEncoding];

    // Pass 2: dense refine around each field's coarse winner -> outXY/outSAD.
    // Metal's automatic hazard tracking serializes this encoder's reads of
    // coarseXYBuf after pass 1's writes, since both are on the same command
    // buffer -- no manual barrier/fence needed.
    id<MTLComputeCommandEncoder> refineEncoder = [cmdBuf computeCommandEncoder];
    [refineEncoder setComputePipelineState:accel->refinePipeline];
    [refineEncoder setBuffer:currBuf offset:0 atIndex:0];
    [refineEncoder setBuffer:prevBuf offset:0 atIndex:1];
    [refineEncoder setBuffer:fieldBuf offset:0 atIndex:2];
    [refineEncoder setBuffer:coarseXYBuf offset:0 atIndex:3];
    [refineEncoder setBuffer:outXYBuf offset:0 atIndex:4];
    [refineEncoder setBuffer:outSADBuf offset:0 atIndex:5];
    [refineEncoder setBytes:&currLinesizeI length:sizeof(int32_t) atIndex:6];
    [refineEncoder setBytes:&prevLinesizeI length:sizeof(int32_t) atIndex:7];
    [refineEncoder setBytes:&stepSizeI length:sizeof(int32_t) atIndex:8];
    [refineEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [refineEncoder endEncoding];

    [cmdBuf commit];
    // Intentionally NOT waiting here -- that's the point of Submit/Wait:
    // the caller can do CPU work concurrently and call
    // vsMPSBatchSearchWait() when it actually needs the results.
    accel->inFlightCmdBuf = cmdBuf;
    accel->inFlightOutXYBuf = outXYBuf;
    accel->inFlightOutSADBuf = outSADBuf;
  }
  return VS_OK;
}

int vsMPSBatchSearchWait(VSMPSAccel* accel, VSMPSMotionResult* out, int count,
                          double* outGpuSeconds) {
  if (!accel || accel->inFlightCmdBuf == nil || !out || count <= 0)
    return VS_ERROR;

  @autoreleasepool {
    id<MTLCommandBuffer> cmdBuf = accel->inFlightCmdBuf;
    id<MTLBuffer> outXYBuf = accel->inFlightOutXYBuf;
    id<MTLBuffer> outSADBuf = accel->inFlightOutSADBuf;
    accel->inFlightCmdBuf = nil;
    accel->inFlightOutXYBuf = nil;
    accel->inFlightOutSADBuf = nil;

    [cmdBuf waitUntilCompleted];
    if (cmdBuf.status != MTLCommandBufferStatusCompleted) {
      return VS_ERROR;
    }

    if (outGpuSeconds) {
      // GPUStartTime/GPUEndTime are the GPU's own timestamps for actual
      // kernel execution -- unlike wall-clock time around submit/wait, this
      // is unaffected by however long the caller's overlapped CPU work took.
      *outGpuSeconds = cmdBuf.GPUEndTime - cmdBuf.GPUStartTime;
    }

    const int32_t* outXY = (const int32_t*) outXYBuf.contents;
    const uint32_t* outSAD = (const uint32_t*) outSADBuf.contents;
    for (int i = 0; i < count; i++) {
      out[i].x = (int16_t) outXY[i * 2 + 0];
      out[i].y = (int16_t) outXY[i * 2 + 1];
      out[i].minSAD = outSAD[i];
    }
  }
  return VS_OK;
}

double vsMPSSuggestCPUShare(VSMPSAccel* accel, VSMPSBatchKind kind, double fallbackShare) {
  int k = (int) kind;
  if (!accel || !accel->haveTimingData[k] || accel->emaCpuPerField[k] <= 0.0 ||
      accel->emaGpuPerField[k] <= 0.0) {
    return fallbackShare;
  }
  // The share of fields that, if sent to the CPU, makes the CPU share and
  // the GPU share take about the same wall-clock time -- neither side then
  // waits idle on the other. Measured data on coarse batches (large window,
  // GPU-compute-bound) showed the CPU side genuinely cheaper per-field,
  // with the ratio still trending toward more CPU share when a tighter
  // 0.75 cap was in place -- raised to 0.9 so that headroom isn't left on
  // the table. Not uncapped entirely: keeping >=10% of fields on the GPU
  // means every batch still reports a fresh gpuSeconds sample, so the
  // estimate can never get permanently stuck (gpuCount==0 would stop
  // vsMPSReportBatchTiming() from ever updating emaGpuPerField again,
  // freezing the ratio even if relative CPU/GPU speed later changes).
  double share = accel->emaGpuPerField[k] / (accel->emaCpuPerField[k] + accel->emaGpuPerField[k]);
  if (share < 0.0) share = 0.0;
  if (share > 0.9) share = 0.9;
  return share;
}

void vsMPSReportBatchTiming(VSMPSAccel* accel, VSMPSBatchKind kind,
                             double cpuSeconds, int cpuCount,
                             double gpuSeconds, int gpuCount) {
  if (!accel)
    return;
  int k = (int) kind;
  // Smoothing factor: recent frames matter more (workload/thermal state
  // drift over a long render) but a single noisy frame shouldn't swing the
  // estimate; picked empirically, not derived.
  const double alpha = 0.2;
  if (cpuCount > 0 && cpuSeconds > 0.0) {
    double perField = cpuSeconds / cpuCount;
    accel->emaCpuPerField[k] = accel->haveTimingData[k]
        ? (alpha * perField + (1.0 - alpha) * accel->emaCpuPerField[k])
        : perField;
  }
  if (gpuCount > 0 && gpuSeconds > 0.0) {
    double perField = gpuSeconds / gpuCount;
    accel->emaGpuPerField[k] = accel->haveTimingData[k]
        ? (alpha * perField + (1.0 - alpha) * accel->emaGpuPerField[k])
        : perField;
  }
  if ((cpuCount > 0 && cpuSeconds > 0.0) || (gpuCount > 0 && gpuSeconds > 0.0)) {
    accel->haveTimingData[k] = true;
  }
}

#endif // USE_MPS

/*
 * Local variables:
 *   c-file-style: "stroustrup"
 *   c-file-offsets: ((case-label . *) (statement-case-intro . *))
 *   indent-tabs-mode: nil
 *   tab-width:  2
 *   c-basic-offset: 2 t
 * End:
 *
 * vim: expandtab shiftwidth=2:
 */
