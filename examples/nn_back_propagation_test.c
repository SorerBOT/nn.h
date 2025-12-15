#define NN_IMPLEMENTATION
#include "../nn.h"

float training_data[][3] = {
    { 0.f, 0.f, 0.f },
    { 1.f, 0.f, 1.f },
    { 0.f, 1.f, 1.f },
    { 1.f, 1.f, 0.f },
};

int main(void)
{
    srand(69);
    size_t layer_sizes[] = { 2, 2, 1 };
    NN_Network nn = nn_network_init(layer_sizes, ARRAY_LEN(layer_sizes));
    NN_Network gradient = nn_network_init(layer_sizes, ARRAY_LEN(layer_sizes));
    nn_network_rand(nn);

    NN_Matrix in =
    {
        .rows = 4,
        .cols = 2,
        .stride = 3,
        .data = &training_data[0][0]
    };
    NN_Matrix out =
    {
        .rows = 4,
        .cols = 1,
        .stride = 3,
        .data = &training_data[0][2]
    };

    NN_Layer* inputs = nn_layer_io_init_from_matrix(in);
    NN_Layer* outputs = nn_layer_io_init_from_matrix(out);

#if 1
    nn_network_finite_differences(nn, gradient, 1e-3, inputs, outputs, in.rows);
    nn_network_print(gradient);
#else
    nn_network_backpropagation(nn, gradient, inputs, outputs, in.rows);
    nn_network_print(gradient);
#endif

    return 0;
}
