#include "ocl_boiler.h"
#include <CL/cl.h>
#include <CL/cl_platform.h>
#include <ctype.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TEST_DATA_MAX_NUMBER_EXCL 128.0
#define PRINT_ARRAY_TEST 0
#define N_POINTS_RANDOM_DATA 2048
#define RANDOM_DATA 0
#define FILEPATH "..//data_large.csv"
#define SUPPRESS_PRINT 0

void error(char *what) {
  fprintf(stderr, "%s\n", what);
  exit(1);
}

cl_float frand(cl_float low, cl_float high) {
  return ((cl_float)rand() * (high - low)) / (cl_float)RAND_MAX + low;
}

void test_print_dataset(cl_float2 *dataset, int points) {
  for (int i = 0; i < points; ++i)
    printf("[%d] X: %f Y: %f\n", i, dataset[i].x, dataset[i].y);
}

void test_print_centroids(cl_float2 *centroids, int k) {
  for (int i = 0; i < k; ++i)
    printf("[%d] X: %f Y: %f\n", i, centroids[i].x, centroids[i].y);
}

cl_event assign_centroids(cl_command_queue que,
                          cl_kernel assign_centroids_kernel, cl_mem *dataset,
                          cl_mem *centroids, int k) {

  size_t gws[] = {k};
  printf("assign centroids: %u\n", k);
  cl_int err;
  cl_event ret;

  err = clSetKernelArg(assign_centroids_kernel, 0, sizeof(cl_mem), dataset);
  ocl_check(err, "assign_centroids_kernel 1st arg set");
  err = clSetKernelArg(assign_centroids_kernel, 1, sizeof(cl_mem), centroids);
  ocl_check(err, "assign_centroids_kernel 2nd arg set");
  err = clSetKernelArg(assign_centroids_kernel, 2, sizeof(cl_int), &k);
  ocl_check(err, "assign_centroids_kernel 3rd arg set");

  err = clEnqueueNDRangeKernel(que, assign_centroids_kernel, 1, NULL, gws, NULL,
                               0, NULL, &ret);
  ocl_check(err, "enqueue assign_centroids");
  return ret;
}

cl_event update_points(cl_command_queue que, cl_kernel points_kernel,
                       cl_mem *dataset, cl_mem *centroids, cl_mem *assignments,
                       cl_mem *cluster_sum, cl_mem *cluster_elements,
                       int points, int k, int lws_arg) {
  size_t lws[] = {lws_arg};
  size_t gws[] = {round_mul_up(points, lws_arg)};
  // printf("update points: %u | %zu = %zu\n", points, lws[0], gws[0]);
  cl_int err;
  cl_event ret;
  err = clSetKernelArg(points_kernel, 0, sizeof(cl_mem), dataset);
  ocl_check(err, "kmeans_kernel 1st arg set");

  err = clSetKernelArg(points_kernel, 1, sizeof(cl_mem), centroids);
  ocl_check(err, "kmeans_kernel 2nd arg set");

  err = clSetKernelArg(points_kernel, 2, sizeof(cl_mem), assignments);
  ocl_check(err, "kmeans_kernel 3rd arg set");
  err = clSetKernelArg(points_kernel, 3, sizeof(cl_mem), cluster_sum);
  ocl_check(err, "kmeans_kernel 4th arg set");
  err = clSetKernelArg(points_kernel, 4, sizeof(cl_mem), cluster_elements);
  ocl_check(err, "kmeans_kernel 5th arg set");
  err = clSetKernelArg(points_kernel, 5, sizeof(cl_float2) * k, NULL);
  ocl_check(err, "kmeans_kernel 5th arg set");
  err = clSetKernelArg(points_kernel, 6, sizeof(cl_int), &points);
  ocl_check(err, "kmeans_kernel 6th arg set");
  err = clSetKernelArg(points_kernel, 7, sizeof(cl_int), &k);
  ocl_check(err, "kmeans_kernel 7th arg set");
  err =
      clEnqueueNDRangeKernel(que, points_kernel, 1, 0, gws, lws, 0, NULL, &ret);
  ocl_check(err, "enqueue points_kernel");
  return ret;
}

cl_event update_centroids(cl_command_queue que, cl_kernel centroids_kernel,
                          cl_mem *cluster_sum, cl_mem *cluster_elements,
                          cl_mem *centroids, int points, int k,
                          cl_event update_points_evt, int lws_arg) {
  size_t lws[] = {lws_arg};
  size_t gws[] = {round_mul_up(points, lws_arg)};
  // printf("update centroids: %u | %zu = %zu\n", points, lws[0], gws[0]);
  cl_int err;
  cl_event ret;
  err = clSetKernelArg(centroids_kernel, 0, sizeof(cl_mem), centroids);
  ocl_check(err, "centroids_kernel 2nd arg set");

  err = clSetKernelArg(centroids_kernel, 1, sizeof(cl_mem), cluster_sum);
  ocl_check(err, "centroids_kernel 4th arg set");
  err = clSetKernelArg(centroids_kernel, 2, sizeof(cl_mem), cluster_elements);
  ocl_check(err, "centroids_kernel 5th arg set");
  err = clSetKernelArg(centroids_kernel, 3, sizeof(cl_int), &k);
  ocl_check(err, "centroids_kernel 8th arg set");
  err = clEnqueueNDRangeKernel(que, centroids_kernel, 1, 0, gws, lws, 1,
                               &update_points_evt, &ret);
  ocl_check(err, "enqueue centroids_kernel");
  return ret;
}

int main(int argc, char **argv) {
  if (argc < 3)
    error("usage: kmeans <k> <lws>");

  cl_platform_id p = select_platform();
  cl_device_id d = select_device(p);
  cl_context ctx = create_context(p, d);
  cl_command_queue que = create_queue(ctx, d);
  cl_program prog = create_program("kmeans.ocl", ctx, d);

  cl_int err;

  srand(time(NULL));

  int k = atoi(argv[1]);
  int lws = atoi(argv[2]);
  if (k <= 1)
    error("Please pick a valid cluster number (more than one)");
  if (lws <= 0)
    error("Please pick a valid lws (positive integer)");
#if RANDOM_DATA
  int points = N_POINTS_RANDOM_DATA;
  // generate random data
  for (int i = 0; i < points; i++) {
    dataset[i].x = frand(0, TEST_DATA_MAX_NUMBER_EXCL);
    dataset[i].y = frand(0, TEST_DATA_MAX_NUMBER_EXCL);
  }
#else
  FILE *fp;
  fp = fopen(FILEPATH, "r");
  if (fp == NULL)
    error("File not found");
  int curr_row = 0;
  char line[32];
  while (fgets(line, sizeof(line), fp) != NULL) {
    ++curr_row;
  }
  rewind(fp);
  int points = curr_row;
  if (points < k) {
    printf("Cluster count (k=%d) would be smaller than dataset (n=%d). "
           "Reducing to n...\n",
           k, points);
    k = points;
  }
  curr_row = 0;
  cl_float2 *dataset = malloc(points * sizeof(cl_float2));
  int *assignments = calloc(points, sizeof(int));
  for (int i = 0; i < k; ++i)
    assignments[i] = i;
  while (fgets(line, sizeof(line), fp) != NULL) {
    sscanf(line, "%f,%f", &dataset[curr_row].x, &dataset[curr_row].y);
    ++curr_row;
  }

#endif
  cl_float2 *centroids = malloc(k * sizeof(cl_float2));

  int *cluster_elements = malloc(k * sizeof(cl_int));
  /*
  data is stored in the dataset as pairs x and y, like this:

  0 [x0][y0]
  1 [x1][y1]
  2 [x2][y2]
  ...
  */
  // create all kernels
  cl_kernel assign_centroids_kernel =
      clCreateKernel(prog, "assign_centroids", &err);
  cl_kernel update_points_kernel = clCreateKernel(prog, "update_points", &err);
  cl_kernel update_centroids_kernel =
      clCreateKernel(prog, "update_centroids", &err);

  // create all buffers
  cl_mem d_dataset =
      clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     points * sizeof(cl_float2), dataset, &err);
  cl_mem d_centroids =
      clCreateBuffer(ctx, CL_MEM_READ_WRITE, k * sizeof(cl_float2), NULL, &err);
  ocl_check(err, "create d_centroids");
  cl_mem d_assignments =
      clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     points * sizeof(cl_int), assignments, &err);
  cl_mem d_cluster_sum =
      clCreateBuffer(ctx, CL_MEM_READ_WRITE, k * sizeof(cl_float2), NULL, &err);
  cl_mem d_cluster_elements =
      clCreateBuffer(ctx, CL_MEM_READ_WRITE, k * sizeof(cl_int), NULL, &err);
  ocl_check(err, "create d_dataset");
  // write dataset data into device buffer
  err =
      clEnqueueWriteBuffer(que, d_dataset, CL_TRUE, 0,
                           sizeof(cl_float2) * points, dataset, 0, NULL, NULL);
  ocl_check(err, "enqueue write buffer");

  ocl_check(err, "create assign_centroids_kernel");
  cl_event assign_centroids_evt = assign_centroids(que, assign_centroids_kernel,
                                                   &d_dataset, &d_centroids, k);

  // clFinish(que);

  cl_event update_centroids_evt, update_points_evt;
  for (int i = 0; i < 100; ++i) {
    update_points_evt = update_points(
        que, update_points_kernel, &d_dataset, &d_centroids, &d_assignments,
        &d_cluster_sum, &d_cluster_elements, points, k, lws);

    update_centroids_evt = update_centroids(
        que, update_centroids_kernel, &d_cluster_sum, &d_cluster_elements,
        &d_centroids, points, k, update_points_evt, lws);
  }
  clFinish(que);

  clEnqueueReadBuffer(que, d_cluster_elements, CL_TRUE, 0, sizeof(cl_int) * k,
                      cluster_elements, 1, &update_centroids_evt, NULL);

  clEnqueueReadBuffer(que, d_centroids, CL_TRUE, 0, sizeof(cl_float2) * k,
                      centroids, 0, NULL, NULL);
  clEnqueueReadBuffer(que, d_assignments, CL_TRUE, 0, sizeof(cl_int) * points,
                      assignments, 0, NULL, NULL);
#if !SUPPRESS_PRINT
  for (int i = 0; i < points; i++) {
    printf("point %d (%f, %f) is in cluster %d (with centroid (%f,%f))\n", i,
           dataset[i].x, dataset[i].y, assignments[i],
           centroids[assignments[i]].x, centroids[assignments[i]].y);
  }
  for (int i = 0; i < k; ++i)
    printf("Cluster %d has %d point(s)\n", i, cluster_elements[i]);
#endif
  float t0, t1, t2;
  t0 = runtime_ms(assign_centroids_evt);
  t1 = runtime_ms(update_points_evt) * 100;
  t2 = runtime_ms(update_centroids_evt) * 100;
  // printf("assign centroids %f ms\n", runtime_ms(assign_centroids_evt));
  // printf("update points %f ms\n", runtime_ms(update_points_evt));
  // printf("update centroids %f ms\n", runtime_ms(update_centroids_evt));
  printf("%f + %f + %f = %f\n", t0, t1, t2, t0 + t1 + t2);
}
