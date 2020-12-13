#ifndef VECTOR_H__
#define VECTOR_H__

#include <stdio.h>
#include <stdlib.h>

int *create_random_vector(int n, int low, int high) {
    int *h = (int *)malloc(n * sizeof(int));
    if (!h) {
        perror("malloc failed");
        exit(1);
    }
    for (int i = 0; i < n; i++) {
        h[i] = (rand() % (high - low)) + low;
    }
    return h;
}

int vector_sum(const int *h, int n) {
    int res = 0;
    for (int i = 0; i < n; i++) {
        res += h[i];
    }
    return res;
}

#endif
