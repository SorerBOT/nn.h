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
float training_output[][1] = {
    { 4.f },
    { 8.f },
    { 6.f },
};

int main(void)
{
    size_t layer_sizes[] = { 1, 1 };
    NN_Network nn = nn_network_init(layer_sizes, ARRAY_LEN(layer_sizes));
    nn_network_rand(nn);

    NN_Layer inputs[3];
    NN_Layer outputs[3];
    for (size_t i = 0; i < 3; ++i)
    {
        inputs[i] = nn_layer_io_init_from_array(training_input[i], 1);
        outputs[i] = nn_layer_io_init_from_array(training_output[i], 1);
    }

    float cost = nn_network_cost(nn, inputs, outputs, 1);
    printf("---------------\n");
    nn_network_print(nn);
    printf("---------------\n");
    printf("Cost = %f\n", cost);

    return 0;
}
