#ifdef USE_MPS

/* Runs the motion detection pipeline once, forcing the given acceleration
 * backend, on the shared synthetic testdata frames, and returns the
 * resulting transform of the last processed frame. Along the way it
 * checks -- for every processed frame -- the same ground-truth invariant
 * that test_motionDetect() (test_motiondetect.c) checks: dx/dy < 2px and
 * alpha < 0.005 with respect to the known synthetic transform.
 */
static VSTransform runMotionDetectPipelineMPS(TestData* testdata, VSAccelMode accelMode){
  VSMotionDetectConfig mdconf = vsMotionDetectGetDefaultConfig("test_motionDetectMPS_consistency");
  mdconf.accelMode = accelMode;
  VSMotionDetect md;

  VSTransformConfig tdconf = vsTransformGetDefaultConfig("test_motionDetectMPS_consistency-trans");
  VSTransformData td;
  test_bool(vsTransformDataInit(&td, &tdconf, &testdata->fi, &testdata->fi) == VS_OK);

  test_bool(vsMotionDetectInit(&md, &mdconf, &testdata->fi) == VS_OK);

  if(accelMode == VSAccelMPS){
    // this test is Apple-hardware-only, just like USE_MPS itself
    test_bool(md.mpsAccel != NULL);
  }

  VSTransform t = null_transform();
  int numruns = 5;
  int i;
  for(i=0; i<numruns; i++){
    LocalMotions localmotions;

    test_bool(vsMotionDetection(&md, &localmotions, &testdata->frames[i]) == VS_OK);
    t = vsSimpleMotionsToTransform(td.fiSrc, td.conf.modName, &localmotions);
    vs_vector_del(&localmotions);

    VSTransform orig = mult_transform_(getTestFrameTransform(i),-1.0);
    VSTransform diff = sub_transforms(&t,&orig);
    int success = fabs(diff.x)<2 && fabs(diff.y)<2 && fabs(diff.alpha)<0.005;
    if(!success){
      fprintf(stderr,"Difference to ground truth (accelMode %i): ", (int)accelMode);
      storeVSTransform(stderr,&diff);
    }
    test_bool(success);
  }

  vsMotionDetectionCleanup(&md);
  vsTransformDataCleanup(&td);
  return t;
}

/* Compares the CPU and MPS motion detection paths against each other, on
 * the level of the aggregated transform (not per-field): a dense GPU
 * search over the whole shift window can legitimately land on a better
 * optimum than the CPU coarse-grid+refine hill-climb for individual
 * fields, so per-field equality would be the wrong invariant here (see
 * the "Uwaga o dokladnosci" in the design plan). Both paths must also
 * individually pass the existing ground-truth check used by
 * test_motionDetect().
 */
void test_motionDetectMPS_consistency(TestData* testdata){
  fprintf(stderr,"MotionDetect MPS consistency:\n");

  VSTransform tCPU = runMotionDetectPipelineMPS(testdata, VSAccelCPU);
  VSTransform tMPS = runMotionDetectPipelineMPS(testdata, VSAccelMPS);

  VSTransform diff = sub_transforms(&tMPS, &tCPU);
  // Measured empirically at ~0.71/-0.40px / -0.0001 on the synthetic test
  // fixture, stable across tie-breaking strategies in the GPU kernel's
  // argmin reduction -- i.e. a real, expected artifact of dense full-window
  // GPU search settling on a different (still valid, ground-truth-passing)
  // optimum than the CPU's coarse-grid-then-refine hill climb on frames
  // with large uniform regions, not a functional bug. Tolerance set loose
  // enough to accommodate that while still catching gross regressions
  // (wrong sign, off-by-one, swapped axes) which would show up as
  // multi-pixel or larger differences.
  int consistent = fabs(diff.x)<1.5 && fabs(diff.y)<1.5 && fabs(diff.alpha)<0.005;
  if(!consistent){
    fprintf(stderr,"CPU vs MPS difference: ");
    storeVSTransform(stderr,&diff);
  }
  test_bool(consistent);
}

#endif // USE_MPS

/*
 * Local variables:
 *   c-file-style: "stroustrup"
 *   c-file-offsets: ((case-label . *) (statement-case-intro . *))
 *   indent-tabs-mode: nil
 *   c-basic-offset: 2 t
 * End:
 *
 * vim: expandtab shiftwidth=2:
 */
