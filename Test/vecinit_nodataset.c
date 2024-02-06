#include "ocl_boiler.h"
#include <CL/cl.h>
#include <CL/cl_platform.h>
#include <stdio.h>
#include <stdlib.h>
#define CL_TARGET_OPENCL_VERSION 120
#define NUMBER_OF_POINTS 8
/*
 sanity check to verify correct initialization of an device array from a
 host buffer
 */
void error(char *what) {
  fprintf(stderr, "%s\n", what);
  exit(1);
}

int main() {

  int k = 3;

  cl_platform_id p = select_platform();
  cl_device_id d = select_device(p);
  cl_context ctx = create_context(p, d);
  cl_command_queue que = create_queue(ctx, d);
  cl_program prog = create_program("vecinit.ocl", ctx, d);
  cl_int err;

  int *centroids = calloc(k, sizeof(int));
  if (centroids == NULL)
    printf("Error on mem allocation");

  cl_kernel assign_centroids = clCreateKernel(prog, "vecinit", &err);
  ocl_check(err, "create assign_centroids kernel");
  cl_mem d_centroids =
      clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, k * sizeof(int), NULL, &err);

  ocl_check(err, "create centroids buffer");

  err = clSetKernelArg(assign_centroids, 0, sizeof(d_centroids), &d_centroids);
  ocl_check(err, "set arg 0");

  cl_event init, evt;
  size_t gws[1] = {k};
  err = clEnqueueNDRangeKernel(que, assign_centroids, 1, 0, gws, NULL, 0, NULL,
                               &evt);
  ocl_check(err, "enqueue kernel");
  err = clEnqueueReadBuffer(que, d_centroids, CL_TRUE, 0, k * sizeof(int),
                            centroids, 1, &evt, &init);
  ocl_check(err, "read buffer");

  for (int i = 0; i < k; ++i)
    printf("%d\n", centroids[i]);

  //  free(centroids);
  clReleaseMemObject(d_centroids);

  clReleaseKernel(assign_centroids);
  clReleaseProgram(prog);
  clReleaseCommandQueue(que);
  clReleaseContext(ctx);
}