#define NN_IMPLEMENTATION
#include "nn.h"

float training_data[][2] = {
    { 2.f, 4.f },
    { 4.f, 8.f },
    { 3.f, 6.f }
};
float training_input[][1] = {
    { 2.f },
    { 4.f },
    { 3.f }
};
float training_output[] = {
    4.f,
    8.f,
    6.f,
};

int main(void)
{
    size_t layer_sizes[] = { 1, 1 };
    NN_Network nn = nn_network_init(layer_sizes, ARRAY_LEN(layer_sizes));
    printf("neurons: %lu\n", nn.layers[0].neurons_count);
    nn_network_set_input(nn, training_input[0], 1);
    nn_network_forward(nn);
    nn_network_print(nn);

    return 0;
}
