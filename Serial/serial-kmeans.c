#include <ctype.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define TEST_DATA_MAX_NUMBER_EXCL 128.0
#define PRINT_ARRAY_TEST 0
#define N_POINTS_RANDOM_DATA 2048
#define RANDOM_DATA 0
// we must take measurement of the time ONLY if the printf's are suppressed,
// because terminal I/O time would invalidate the data
#define SUPPRESS_PRINT 1
#define FILEPATH "..//data_original.csv"

void error(char *what) {
  fprintf(stderr, "%s\n", what);
  exit(1);
}

float euclidean_distance(float x0, float y0, float x1, float y1) {
  return sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1));
}

float frand(float low, float high) {
  return ((float)rand() * (high - low)) / (float)RAND_MAX + low;
}

void test_print_dataset(float *dataset, int points) {
  for (int i = 0; i < points * 2; i += 2)
    printf("[%d] X: %f Y: %f\n", i / 2, dataset[i], dataset[i + 1]);
}

void test_print_centroids(float *centroids, int k) {
  for (int i = 0; i < k * 2; i += 2)
    printf("[%d] X: %f Y: %f\n", i / 2, centroids[i], centroids[i + 1]);
}

int main(int argc, char **argv) {
  if (argc < 2)
    error("usage: kmeans <k>");

  srand(time(NULL));
  // int points = atoi(argv[1]);
  int k = atoi(argv[1]);
  if (k < 2)
    error("Please pick a valid cluster number (more than one)");
    // printf("%d %d\n",points,k);

/*
data is stored in the dataset as interleaved x and y, like this:

0 [x0]
1 [y0]
2 [x1]
3 [y1]
...
*/
#if RANDOM_DATA
  int points = N_POINTS_RANDOM_DATA;
  // generate random data
  for (int i = 0; i < points * 2; i++)
    dataset[i] = frand(0, TEST_DATA_MAX_NUMBER_EXCL);
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

  curr_row = 0;
  float *dataset = malloc(2 * points * sizeof(float));
  int *assignments = calloc(points, sizeof(int));
  // points that are centroids are self-assigned
  for(int i=0;i<k;++i)
    assignments[i] = i;
  while (fgets(line, sizeof(line), fp) != NULL) {
    sscanf(line, "%f,%f", &dataset[curr_row * 2], &dataset[curr_row * 2 + 1]);
    ++curr_row;
  }
  float *centroids = malloc(k * 2 * sizeof(float));
  int *cluster_elements = malloc(k * sizeof(int));
  float *cluster_sum = malloc(k * 2 * sizeof(float));
#endif
    // we start doing the actual time measurement after all the I/O and
    // initialization takes place
#if SUPPRESS_PRINT
  struct timespec beg_timespec;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &beg_timespec);
#endif
  // for our centroid selection, we just pick the first elements
  for (int i = 0; i < k; ++i) {
    centroids[i * 2] = dataset[i * 2];
    centroids[i * 2 + 1] = dataset[i * 2 + 1];
  }

#if PRINT_ARRAY_TEST
  printf("\n\n\n");
  test_print_dataset(dataset, points);
  test_print_centroids(centroids, k);
#endif
  for (int iter = 0; iter < 100; ++iter) {

    // assign points to each centroid to create clusters
    for (int i = 0; i < points; ++i) {
      float distance = euclidean_distance(dataset[i * 2], dataset[i * 2 + 1],
                                          centroids[assignments[i]],
                                          centroids[assignments[i] + 1]);
      for (int j = 0; j < k; ++j) {
        float ed = euclidean_distance(dataset[i * 2], dataset[i * 2 + 1],
                                      centroids[j * 2], centroids[j * 2 + 1]);
        if (ed < distance) {
          // printf("%f < %f, assigning point %d to cluster
          // %d\n",ed,distance,i,j);
          distance = ed;
          assignments[i] = j;
        }
      }
    }
    memset(cluster_sum, 0, k * 2 * sizeof(float));
    memset(cluster_elements, 0, k * sizeof(int));
    // find new centroids by calculating the average of each coordinate
    for (int i = 0; i < points; ++i) {
      ++cluster_elements[assignments[i]];
      // printf(
      //     "Point %d belongs to cluster %d. Incrementing cluster count to
      //     %d.\n", i, assignments[i], cluster_elements[assignments[i]]);

      cluster_sum[assignments[i] * 2] += dataset[i * 2];
      cluster_sum[assignments[i] * 2 + 1] += dataset[i * 2 + 1];
    }

    for (int i = 0; i < k; ++i) {
      // printf("%d %d %f %f\n", i, cluster_elements[i],
      //        cluster_sum[i * 2] / cluster_elements[i],
      //        cluster_sum[i * 2 + 1] / cluster_elements[i]);

      // printf("%d\n", cluster_elements[i]);
      if (cluster_elements[i] != 0) {
        centroids[i * 2] = cluster_sum[i * 2] / cluster_elements[i];
        centroids[i * 2 + 1] = cluster_sum[i * 2 + 1] / cluster_elements[i];
      }
    }
  }
  // recompute cluster element count to show final result
  memset(cluster_elements, 0, k * sizeof(int));
  for (int i = 0; i < points; ++i) {
    ++cluster_elements[assignments[i]];
#if SUPPRESS_PRINT
    printf("point %d (%f, %f) is in cluster %d (with centroid (%f,%f))\n", i,
           dataset[i * 2], dataset[i * 2 + 1], assignments[i],
           centroids[assignments[i] * 2], centroids[assignments[i] * 2 + 1]);
#endif
  }
#if !SUPPRESS_PRINT
  for (int i = 0; i < k; i++) {
    printf("Cluster %d has %d point(s)\n", i, cluster_elements[i]);
  }
#else
  struct timespec end_timespec;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_timespec);
  float time_spent =
      (end_timespec.tv_sec - beg_timespec.tv_sec) +
      (end_timespec.tv_nsec - beg_timespec.tv_nsec) / 1000000000.0;
  printf("Execution time: %lf s\n", time_spent);
#endif
}
