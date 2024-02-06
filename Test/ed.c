#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float euclidean_distance(float x0, float y0, float x1, float y1) {
  return sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1));
}

void error(char * what){
    fprintf(stderr,"%s\n",what);
    exit(1);
}

int main(int argc, char **argv){
    printf("\nCalculate euclidean distance between two points in a plane\n\n");
    if(argc < 5)
        error("usage: ed <x0> <y0> <x1> <y1>");
    float x0 = atof(argv[1]);
    float y0 = atof(argv[2]);
    float x1 = atof(argv[3]);
    float y1 = atof(argv[4]);

    printf("%f\n",euclidean_distance(x0, y0, x1, y1));
    return 0;
}