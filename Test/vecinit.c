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

  

  cl_float2 *dataset = malloc(NUMBER_OF_POINTS * sizeof(cl_float2));
  if (dataset == NULL)
    printf("Error on dataset allocation");

  for (int i = 0; i < NUMBER_OF_POINTS; ++i) {
    dataset[i].x = dataset[i].y = 100.0f - (float)i;
  }

  cl_mem d_dataset =
      clCreateBuffer(ctx, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                     NUMBER_OF_POINTS * sizeof(cl_float2), dataset, &err);

  ocl_check(err, "create dataset buffer");

  cl_mem d_centroids =
      clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, k * sizeof(cl_float2), NULL, &err);

  ocl_check(err, "create centroids buffer");

  cl_kernel assign_centroids = clCreateKernel(prog, "vecinit", &err);
  ocl_check(err, "create assign_centroids kernel");

  err = clSetKernelArg(assign_centroids, 0, sizeof(d_dataset), &d_dataset);
  ocl_check(err, "set arg 0");

  err = clSetKernelArg(assign_centroids, 1, sizeof(d_centroids), &d_centroids);
  ocl_check(err, "set arg 1");

  err = clSetKernelArg(assign_centroids, 2, sizeof(int), &k);
  ocl_check(err, "set arg 2");

  cl_event init, evt;
  size_t gws[1] = {256};
  err = clEnqueueNDRangeKernel(que, assign_centroids, 1, 0, gws, NULL, 0, NULL,
                               &evt);
  ocl_check(err, "enqueue kernel");
  cl_float2 *centroids = malloc(k * sizeof(cl_float2));
    if (centroids == NULL)
    printf("Error on centroids allocation");
  err = clEnqueueReadBuffer(que, d_centroids, CL_TRUE, 0, k * sizeof(int),
                            centroids, 1, &evt, &init);
  ocl_check(err, "read buffer");
  clFinish(que);

  // for (int i = 0; i < k; ++i)
  //   printf("%f %f\n", centroids[i].x, centroids[i].y);

  free(dataset);
  free(centroids);

  clReleaseKernel(assign_centroids);
  clReleaseMemObject(d_dataset);
  clReleaseMemObject(d_centroids);
  clReleaseProgram(prog);
  clReleaseCommandQueue(que);
  clReleaseContext(ctx);
}
