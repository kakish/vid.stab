/*
 *  motiondetect_mps.h
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

#ifndef MOTIONDETECT_MPS_H
#define MOTIONDETECT_MPS_H

#include "vidstab_api.h"

#ifdef USE_MPS

#include <stdint.h>
#include "transformtype.h"

#ifdef __cplusplus
extern "C" {
#endif

/** opaque handle to the Apple Metal / MPS acceleration context.
    the concrete definition (holding id<MTLDevice> etc.) lives in
    motiondetect_mps.mm so that this header stays plain C/C++ and can be
    included from motiondetect.c without pulling in any Objective-C types.
*/
typedef struct VSMPSAccel VSMPSAccel;

/** result of the dense block-matching search for a single measurement field */
typedef struct {
  int16_t  x;      // best matching shift x (within [-maxShift,maxShift])
  int16_t  y;      // best matching shift y (within [-maxShift,maxShift])
  uint32_t minSAD; // sum of absolute differences at (x,y)
} VSMPSMotionResult;

/** creates and initializes a Metal acceleration context, including the
 *  runtime compilation of the block-matching compute kernel.
 *  @return NULL if no Metal-capable GPU is available or the kernel could
 *  not be compiled. This is an expected, non-error fallback condition on
 *  hardware without Metal support and is therefore NOT logged here -- the
 *  caller (motiondetect.c) decides how/whether to log it, depending on
 *  whether the acceleration was explicitly requested.
 */
VS_API VSMPSAccel* vsMPSAccelCreate(void);

/** releases all resources held by the acceleration context.
 *  It is safe to pass NULL (no-op).
 */
VS_API void vsMPSAccelDestroy(VSMPSAccel* accel);

/** Submits a batched block-matching search for all given fields to the GPU
 *  (two compute passes internally, see below) and returns immediately --
 *  it does NOT wait for the GPU to finish. Exactly one submission may be
 *  in flight per VSMPSAccel at a time; call vsMPSBatchSearchWait() before
 *  submitting again. This split (Submit now, Wait later) exists so the
 *  caller can do useful CPU work (e.g. process a subset of fields on the
 *  CPU path, see calcTransFieldsMPS()) while the GPU is busy, instead of
 *  blocking on it immediately -- using both compute resources for the same
 *  frame at once rather than leaving the CPU idle during the GPU dispatch.
 *
 *  The search itself minimizes the sum of absolute differences (SAD)
 *  between curr and prev -- the same metric as compareSubImg_thr() uses on
 *  the CPU path -- via a two-stage search (added after profiling on 4K
 *  footage showed a single fully-dense pass over [-maxShift,maxShift]^2
 *  does not scale: at 4K, maxShift is ~300px, i.e. a ~600x600-candidate
 *  window per field, which made the GPU path several times SLOWER than the
 *  CPU path it was meant to replace):
 *   1. a coarse pass evaluates only a stepSize-spaced grid within
 *      [-maxShift,maxShift]^2 (mirrors the CPU's coarse-grid stage in
 *      calcFieldTransPlanar), finding a per-field approximate winner;
 *   2. a refine pass then densely (1px resolution) searches a small
 *      window of radius `stepSize` around that winner.
 *  This cuts candidate count from O(maxShift^2) to roughly
 *  O((maxShift/stepSize)^2 + stepSize^2), i.e. back to the same order of
 *  magnitude the CPU evaluates, while keeping the GPU's advantage of
 *  evaluating every one of those candidates in parallel instead of via a
 *  sequential early-exit search.
 *
 *  @param curr,currLinesize luminance plane of the current (smoothed) frame
 *  @param prev,prevLinesize luminance plane of the previous frame
 *  @param height             height (in pixels) of both planes
 *  @param fields             array of `count` measurement fields
 *  @param offsets            array of `count` per-field offsets (e.g. from
 *                            a coarse scan); added to the field position
 *                            before searching
 *  @param count              number of fields (and length of results
 *                            passed to the matching vsMPSBatchSearchWait())
 *  @param maxShift           search radius around (field+offset)
 *  @param stepSize           coarse-grid spacing (and refine-window radius);
 *                            same meaning as VSMotionDetectFields.stepSize.
 *                            Clamped to >=1 internally.
 *  @param currGeneration     opaque tag identifying the *content* currently
 *                            held in `curr` (e.g. the frame number at which
 *                            it was last (re)filled). Two calls that pass
 *                            the same generation for the same logical plane
 *                            are assumed to have byte-identical contents --
 *                            the accelerator uses this to skip re-uploading
 *                            a plane it still has cached on the GPU (see
 *                            prevGeneration below, which is the common case:
 *                            this frame's `prev` is byte-identical to last
 *                            frame's `curr`).
 *  @param prevGeneration     same, for `prev`.
 *  @return VS_OK if the work was submitted, VS_ERROR otherwise (caller
 *          falls back to CPU for this batch; no matching Wait() needed)
 */
VS_API int vsMPSBatchSearchSubmit(VSMPSAccel* accel,
                                   const uint8_t* curr, int currLinesize,
                                   const uint8_t* prev, int prevLinesize, int height,
                                   const Field* fields, const Vec* offsets, int count,
                                   int maxShift, int stepSize,
                                   uint64_t currGeneration, uint64_t prevGeneration);

/** Blocks until the submission started by vsMPSBatchSearchSubmit() has
 *  finished, then copies its `count` results into `out`. `count` must
 *  match the value passed to the corresponding Submit() call.
 *  @param outGpuSeconds if non-NULL, receives the GPU's own measured
 *         execution duration for this submission (Metal's
 *         GPUEndTime-GPUStartTime, i.e. actual compute time, not wall-clock
 *         time spent waiting) -- used by the caller to adapt the CPU/GPU
 *         work split to the actual relative speed of the hardware it runs on.
 *  @return VS_OK on success, VS_ERROR if the GPU work itself failed
 */
VS_API int vsMPSBatchSearchWait(VSMPSAccel* accel, VSMPSMotionResult* out, int count,
                                 double* outGpuSeconds);

/** The coarse and fine measurement-field sets have very different per-field
 *  cost profiles (coarse: large window, GPU-compute-bound; fine: small
 *  window, dominated by fixed per-dispatch GPU overhead rather than compute
 *  -- cheap enough that the CPU's direct spiral search is often faster
 *  per-field). vsMPSSuggestCPUShare()/vsMPSReportBatchTiming() therefore
 *  track a separate adaptive estimate per kind, selected by this enum --
 *  mixing them into one estimate would make it chase two different targets
 *  and never converge to the right ratio for either.
 */
typedef enum { VSMPSBatchCoarse = 0, VSMPSBatchFine = 1 } VSMPSBatchKind;

/** Returns a suggested fraction (0..1) of a batch's fields that should be
 *  processed on the CPU concurrently with the GPU, based on an exponential
 *  moving average of previously-reported per-field CPU/GPU costs (see
 *  vsMPSReportBatchTiming()) -- the ratio that would make the CPU share and
 *  the GPU share finish at roughly the same time, so neither side idles
 *  waiting for the other.
 *  @param kind          which field set this estimate is for (see VSMPSBatchKind)
 *  @param fallbackShare returned as-is until enough timing data has been
 *         reported for this kind (i.e. for the first such batch on a fresh
 *         accelerator).
 */
VS_API double vsMPSSuggestCPUShare(VSMPSAccel* accel, VSMPSBatchKind kind, double fallbackShare);

/** Feeds one batch's measured timings into the adaptive estimator used by
 *  vsMPSSuggestCPUShare() for the same `kind`. Call once per batch, after
 *  both the CPU share and vsMPSBatchSearchWait() have completed.
 *  @param kind        which field set this batch was (see VSMPSBatchKind)
 *  @param cpuSeconds  wall-clock time spent processing `cpuCount` fields on
 *                     the CPU (0 if cpuCount is 0)
 *  @param gpuSeconds  GPU execution time for `gpuCount` fields, as returned
 *                     by vsMPSBatchSearchWait()'s outGpuSeconds (0 if
 *                     gpuCount is 0 or unavailable)
 */
VS_API void vsMPSReportBatchTiming(VSMPSAccel* accel, VSMPSBatchKind kind,
                                    double cpuSeconds, int cpuCount,
                                    double gpuSeconds, int gpuCount);

#ifdef __cplusplus
}
#endif

#endif // USE_MPS

#endif  /* MOTIONDETECT_MPS_H */

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
