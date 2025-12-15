#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define NN_IMPLEMENTATION
#include "nn.h"

float training_data[][2] = {
    { 2.f, 4.f },
    { 4.f, 8.f },
    { 3.f, 6.f }
};

int main(void)
{
    srand(18);
    float w = nn_randf(0.f, 5.f);
    float b = nn_randf(0.f, 5.f);
    float rate = 1e-2;

    for (size_t epoch = 0; epoch < 1000; ++epoch)
    {
        // calc cost
        float c = 0.f;
        size_t n = ARRAY_LEN(training_data);
        for (size_t i = 0; i < n; ++i)
        {
            float prediction = training_data[i][0] * w + b;
            float distance = prediction - training_data[i][1];
            c += distance * distance;
        }

        // gradient descent, with prediction = input * w + b
        // c            = \sum_{i=0}^n (prediction-training_data[i][0])^2
        // \partial_w c = \sum_{i=0}^n ((prediction - training_data[i][0])^2)'
        //              = \sum_{i=0}^n 2(prediction - training_data[i][0]) \partial_w prediction
        //              = \sum_{i=0}^n 2(prediction - training_data[i][0]) training_data[i][0]
        // \partial_b c = \sum_{i=0}^n 2(prediction - training_data[i][0])

        float sum = 0.f;
        for (size_t i = 0; i < n; ++i)
        {
            float prediction = training_data[i][0] * w + b;
            sum += 2.f * (prediction - training_data[i][1]) * training_data[i][0];
        }

        float dw = sum;

        sum = 0.f;
        for (size_t i = 0; i < n; ++i)
        {
            float prediction = training_data[i][0] * w + b;
            sum += 2.f * (prediction - training_data[i][1]);
        }

        float db = sum;

        w -= dw * rate;
        b -= db * rate;

        printf("epoch = %lu, c = %f\n", epoch, c);
    }

    printf("Finished training, found parameters: w = %f, b = %f\n", w, b);

    return 0;
}
