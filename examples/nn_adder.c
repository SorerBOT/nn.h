#define NN_IMPLEMENTATION
#include "../nn.h"

#define BITS 4
int main(void)
{
    size_t layer_sizes[] = { 8, 2, 1 };
    NN_Network nn = nn_network_init(layer_sizes, ARRAY_LEN(layer_sizes));
    NN_Network gradient = nn_network_init(layer_sizes, ARRAY_LEN(layer_sizes));
    nn_network_rand(nn);


    size_t max_value = pow(2, BITS)-1;
    size_t combinations_count = (max_value+1)*(max_value+1);

    float* training_data_alloc = (float*) NN_MALLOC(sizeof(float)*combinations_count*BITS*3); // three sets
    NN_Matrix training_data =
    {
        .stride = 3 * BITS,
        .cols = 3 * BITS,
        .rows = combinations_count,
        .data = training_data_alloc,
    };

    for (size_t i = 0; i < BITS; ++i)
    {
        for (size_t j = 0; j < BITS; ++j)
        {
            for (size_t k = 0; k < BITS; ++k)
            {
                training_data
            }

        }
    }

    for (size_t i = 0; i < 15; ++i)
    {
        for (size_t j = 0; j < 15; ++j)
        {
            size_t sum = i+j;
        }
    }

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

    for (size_t epoch = 0; epoch < 1000 * 1000; ++epoch)
    {
        nn_network_finite_differences(nn, gradient, 1e-3, inputs, outputs, in.rows);
        nn_network_learn(nn, gradient, 1e-3);
    }

    float cost = nn_network_cost(nn, inputs, outputs, in.rows);
    printf("Finished training. Cost = %f\n", cost);
    printf("Result network: \n");
    nn_network_print(nn);
    printf("Testing the model: \n");

    for (size_t i = 0; i < in.rows; ++i)
    {
        nn_network_set_input(nn, inputs[i]);
        nn_network_forward(nn);
        //printf("%f ^ %f = %f\n", inputs[i].neurons[0].act, inputs[i].neurons[1].act, NN_OUTPUTS(nn).neurons[0].act);
        printf("%f ^ %f = %f\n", inputs[i].neurons[0].act, inputs[i].neurons[1].act, outputs[i].neurons[0].act);
    }

    return 0;
}
