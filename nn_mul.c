#define NN_IMPLEMENTATION
#include "nn.h"

float training_data[][2] = {
    { 2.f, 4.f },
    { 4.f, 8.f },
    { 3.f, 6.f }
};

int main(void)
{
    size_t layer_sizes[] = { 1, 1 };
    NN_Network nn = nn_network_init(layer_sizes, ARRAY_LEN(layer_sizes));
    NN_Network gradient = nn_network_init(layer_sizes, ARRAY_LEN(layer_sizes));
    nn_network_rand(nn);

    NN_Matrix in =
    {
        .rows = 3,
        .cols = 1,
        .stride = 2,
        .data = &training_data[0][0]
    };
    NN_Matrix out =
    {
        .rows = 3,
        .cols = 1,
        .stride = 2,
        .data = &training_data[0][1]
    };

    NN_Layer* inputs = nn_layer_io_init_from_matrix(in);
    NN_Layer* outputs = nn_layer_io_init_from_matrix(out);

    for (size_t epoch = 0; epoch < 1000 * 1000; ++epoch)
    {
        nn_network_finite_differences(nn, gradient, 1e-3, inputs, outputs, 3);
        nn_network_learn(nn, gradient, 1e-3);
    }

    float cost = nn_network_cost(nn, inputs, outputs, 3);
    printf("Finished training. Cost = %f\n", cost);
    printf("Result network: \n");
    nn_network_print(nn);
    printf("---------------\n");
    printf("Forwarding: \n");
    for (size_t i = 0; i <= 10; ++i)
    {
        float input = i;
        NN_Layer input_layer = nn_layer_io_init_from_array(&input, 1);
        nn_network_set_input(nn, input_layer);
        nn_network_forward(nn);
        printf("%lu * 2 = %f\n", i, nn.layers[nn.layers_count-1].neurons[0].act);
    }

    return 0;
}
