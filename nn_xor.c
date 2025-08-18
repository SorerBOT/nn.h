#define NN_IMPLEMENTATION
#include "nn.h"

float training_data[][3] = {
    { 0.f, 0.f, 0.f },
    { 1.f, 0.f, 1.f },
    { 0.f, 1.f, 1.f },
    { 1.f, 1.f, 0.f },
};
float training_input[][2] = {
    { 0.f, 0.f },
    { 1.f, 0.f },
    { 0.f, 1.f },
    { 1.f, 1.f }
};
float training_output[][1] = {
    { 0.f },
    { 1.f },
    { 1.f },
    { 0.f }
};

int main(void)
{
    size_t layer_sizes[] = { 2, 2, 1 };
    NN_Network nn = nn_network_init(layer_sizes, ARRAY_LEN(layer_sizes));
    NN_Network gradient = nn_network_init(layer_sizes, ARRAY_LEN(layer_sizes));
    nn_network_rand(nn);

    size_t samples_count = ARRAY_LEN(training_data);
    NN_Layer* inputs = (NN_Layer*) NN_MALLOC(sizeof(NN_Layer) * samples_count);
    NN_Layer* outputs = (NN_Layer*) NN_MALLOC(sizeof(NN_Layer) * samples_count);

    for (size_t i = 0; i < samples_count; ++i)
    {
        inputs[i] = nn_layer_io_init_from_array(training_input[i], 2);
        outputs[i] = nn_layer_io_init_from_array(training_output[i], 1);
    }

    for (size_t epoch = 0; epoch < 1000 * 1000; ++epoch)
    {
        nn_network_finite_differences(nn, gradient, 1e-3, inputs, outputs, samples_count);
        nn_network_learn(nn, gradient, 1e-3);
    }
    float cost = nn_network_cost(nn, inputs, outputs, samples_count);
    printf("Finished training. Cost = %f\n", cost);
    printf("Result network: \n");
    nn_network_print(nn);
    printf("Testing the model: \n");

    for (size_t i = 0; i < samples_count; ++i)
    {
        nn_network_set_input(nn, inputs[i]);
        nn_network_forward(nn);
        printf("%f ^ %f = %f\n", inputs[i].neurons[0].act, inputs[i].neurons[1].act, outputs[i].neurons[0].act);
    }

    return 0;
}
