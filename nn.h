#ifndef NN_H_
#define NN_H_

#include <stdlib.h>

#define ARRAY_LEN(xs) sizeof((xs))/sizeof((xs[0]))

typedef struct 
{
    float b;
    float* ws;
} NN_Neuron;

float nn_randf(float min, float max);

#endif // NN_H_
#ifdef NN_IMPLEMENTATION

float nn_randf(float min, float max)
{
    float unit_random = (float)rand() / RAND_MAX;
    float range_size = max - min;
    return min + unit_random * range_size;
}

#endif // NN_IMPLEMNTATION
