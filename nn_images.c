#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NN_IMPLEMENTATION
#include "nn.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

float training_data[][3] = {
    { 0.f, 0.f, 0.f },
    { 1.f, 0.f, 1.f },
    { 0.f, 1.f, 1.f },
    { 1.f, 1.f, 0.f },
};

char* get_next_param(int* argc, char*** argv)
{
    if ((*argc) == 0)
    {
        return NULL;
    }
    else
    {
        --(*argc);
        char* res = (*argv)[0];
        ++(*argv);
        return res;
    }
}

int main(int argc, char** argv)
{
    const char* program_name = get_next_param(&argc, &argv);
    const char* image_path = get_next_param(&argc, &argv);
    if (image_path == NULL)
    {
        fprintf(stderr, "error: image path not provided. usage: %s <image_path>\n", program_name);
        exit(EXIT_FAILURE);
    }

    int width = 0, height = 0, comp = 0; // not sure what comp is
    uint8_t* pixels = stbi_load(image_path, &width, &height, &comp, 0);

    if (pixels == NULL)
    {
        fprintf(stderr, "error: unable to read image: %s\n", image_path);
        exit(EXIT_FAILURE);
    }

    if (comp != 1)
    {
        fprintf(stderr, "error: %s is %d bits image. Only 8 bit grayscale images are supported\n", image_path, comp*8);
        exit(EXIT_FAILURE);
    }

    float* training_data = (float*) NN_MALLOC(2 * width * height * sizeof(float));

    for (size_t i = 0; i < (size_t) width*height; ++i)
    {
        training_data[2*i] = (float)i;
        training_data[2*i+1] = pixels[i] / 255.f;
    }

    size_t layer_sizes[] = { 1, 8, 8, 8, 1 };
    NN_Network nn = nn_network_init(layer_sizes, ARRAY_LEN(layer_sizes));
    NN_Network gradient = nn_network_init(layer_sizes, ARRAY_LEN(layer_sizes));
    nn_network_rand(nn);

    NN_Matrix in =
    {
        .rows = 28 * 28,
        .cols = 1,
        .stride = 2,
        .data = training_data
    };
    NN_Matrix out =
    {
        .rows = 28 * 28,
        .cols = 1,
        .stride = 2,
        .data = (training_data + 1)
    };

    NN_Layer* inputs = nn_layer_io_init_from_matrix(in);
    NN_Layer* outputs = nn_layer_io_init_from_matrix(out);

    for (size_t epoch = 0; epoch < 1000 * 1000; ++epoch)
    {
        if (epoch)
        {
            printf("epoch: %lu, cost: %f\n", epoch, nn_network_cost(nn, inputs, outputs, in.rows));
#if 0
            uint8_t* feed_forward_pixels = (uint8_t*) NN_MALLOC(width * height * sizeof(uint8_t));
            for (size_t i = 0; i < in.rows; ++i)
            {
                nn_network_set_input(nn, inputs[i]);
                nn_network_forward(nn);
                feed_forward_pixels[i] = (uint8_t)roundf(NN_OUTPUTS(nn).neurons[0].act * 255.f);
            }

            char feed_forward_path[128];
            snprintf(feed_forward_path, sizeof(feed_forward_path), "./results/epoch_%ld.png", epoch);
            stbi_write_png(feed_forward_path, width, height, comp, feed_forward_pixels, 0);
#endif
        }
        nn_network_backpropagation(nn, gradient, inputs, outputs, in.rows);
        nn_network_learn(nn, gradient, 1e-3);
    }

    return 0;
}
