#ifndef NN_H_
#define NN_H_

#include <stdio.h>
#ifndef NN_SAFE_ALLOC
#define NN_SAFE_ALLOC 1
#endif

#include <stdlib.h>

#ifndef NN_MALLOC
#if NN_SAFE_ALLOC
#define NN_MALLOC(size) nn_malloc_debug((size), __FILE__, __LINE__)
#else
#define NN_MALLOC malloc
#endif
#endif

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif

#define ARRAY_LEN(xs) sizeof((xs))/sizeof((xs[0]))
#define NN_INPUTS(nn) (nn).layers[0]
#define NN_OUTPUTS(nn) (nn).layers[(nn).layers_count - 1]

typedef enum
{
    ACT_TANH,
    ACT_SIG,
    ACT_RELU
} NN_ACT;

typedef struct 
{
    float bias;
    float act;
    size_t weights_count;
    float* weights;
} NN_Neuron;

typedef struct
{
    NN_ACT act;
    size_t neurons_count;
    NN_Neuron* neurons;
} NN_Layer;

typedef struct
{
    NN_Layer* layers;
    size_t layers_count;
} NN_Network;

float nn_randf(float min, float max);

void* nn_malloc_debug(size_t size, const char* file, int line);

NN_Neuron nn_neuron_init(size_t weights_count);
void nn_neuron_rand(NN_Neuron* neuron);

NN_Layer nn_layer_init_from_array(float* activations, size_t activations_count);

NN_Network nn_network_init(size_t* architecture, size_t count);
void nn_network_rand(NN_Network nn);
void nn_network_forward(NN_Network nn);
void nn_network_print(NN_Network nn);
void nn_network_set_input(NN_Network nn, NN_Layer inputs);
float nn_network_cost(NN_Network nn, NN_Layer* inputs, NN_Layer* outputs, size_t entries_count);
void nn_network_finite_differences(NN_Network nn, NN_Network gradient, float epsilon, NN_Layer* inputs, NN_Layer* outputs, size_t entries_count);
void nn_network_learn(NN_Network nn, NN_Network gradient, float learning_rate);

static void __nn_network_zero(NN_Network nn);

#endif // NN_H_
#ifdef NN_IMPLEMENTATION

#if NN_SAFE_ALLOC
void* nn_malloc_debug(size_t size, const char* file, int line)
{
    void* ptr = malloc(size);
    if (ptr == NULL)
    {
        fprintf(stderr, "Error: failed to allocate memory in %s, %d\n", file, line);
        exit(EXIT_FAILURE);
    }
    return ptr;
}
#endif
float nn_randf(float min, float max)
{
    NN_ASSERT(min < max);
    float unit_random = (float)rand() / RAND_MAX;
    float range_size = max - min;
    return min + unit_random * range_size;
}
#define NN_RANDF() nn_randf(-5.f, 5.f)

NN_Neuron nn_neuron_init(size_t weights_count)
{
    NN_Neuron n;
    n.bias = 0.f;
    n.act = 0.f;
    n.weights_count = weights_count;
    n.weights = (float*) NN_MALLOC(sizeof(*n.weights) * n.weights_count);
    for (size_t i = 0; i < n.weights_count; ++i)
    {
        n.weights[i] = 0.f;
    }

    return n;
}
void nn_neuron_rand(NN_Neuron* neuron)
{
    neuron->bias = NN_RANDF();
    for (size_t i = 0; i < neuron->weights_count; ++i)
    {
        neuron->weights[i] = NN_RANDF();
    }
}

NN_Layer nn_layer_io_init_from_array(float* activations, size_t activations_count)
{
    NN_Layer l;
    l.neurons_count = activations_count;
    l.neurons = (NN_Neuron*) NN_MALLOC(sizeof(NN_Neuron) * activations_count);

    for (size_t i = 0; i < activations_count; ++i)
    {
        l.neurons[i].act = activations[i];
    }
    return l;
}

NN_Network nn_network_init(size_t* layer_sizes, size_t layers_count)
{
    NN_Network nn;

    nn.layers_count = layers_count;
    nn.layers = (NN_Layer*) NN_MALLOC((layers_count) * sizeof(NN_Layer));

    for (size_t i = 0; i < nn.layers_count; ++i)
    {
        NN_Layer* l = &nn.layers[i];
        l->act = ACT_SIG;
        l->neurons_count = layer_sizes[i];
        l->neurons = (NN_Neuron*) NN_MALLOC(l->neurons_count * sizeof(NN_Neuron));
        size_t neuron_weights_count = (i == 0)
            ? 0
            : nn.layers[i - 1].neurons_count;

        for (size_t j = 0; j < l->neurons_count; ++j)
        {
            l->neurons[j] = nn_neuron_init(neuron_weights_count);
        }
    }

    return nn;
}
void nn_network_rand(NN_Network nn)
{
    for (size_t i = 0; i < nn.layers_count; ++i)
    {
        NN_Layer* l = &nn.layers[i];
        for (size_t j = 0; j < l->neurons_count; ++j)
        {
            nn_neuron_rand(&l->neurons[j]);
        }
    }
}

void nn_network_set_input(NN_Network nn, NN_Layer input)
{
    NN_Layer input_layer = nn.layers[0];
    NN_ASSERT(input_layer.neurons_count == input.neurons_count);

    for (size_t i = 0; i < input_layer.neurons_count; ++i)
    {
        NN_Neuron* neuron = &input_layer.neurons[i];
        neuron->act = input.neurons[i].act;
    }
}

void nn_network_forward(NN_Network nn)
{
    __nn_network_zero(nn);
    for (size_t i = 1; i < nn.layers_count; ++i)
    {
        NN_Layer* layer = &nn.layers[i];
        NN_Layer* layer_prev = &nn.layers[i - 1];

        for (size_t j = 0; j < layer->neurons_count; ++j)
        {
            NN_Neuron* neuron = &layer->neurons[j];
            float sum = neuron->bias;
            for (size_t k = 0; k < neuron->weights_count; ++k)
            {
                NN_Neuron* neuron_prev = &layer_prev->neurons[k];
                sum += neuron_prev->act * neuron->weights[k];
            }
            neuron->act = sum;
        }
    }
}

static void __nn_network_zero(NN_Network nn)
{
    for (size_t i = 1; i < nn.layers_count; ++i)
    {
        NN_Layer* layer = &nn.layers[i];
        for (size_t j = 0; j < layer->neurons_count; ++j)
        {
            NN_Neuron* neuron = &layer->neurons[j];
            neuron->act = 0.f;
        }
    }
}


void nn_network_print(NN_Network nn)
{
    printf("Neural Network:\n");

    for (size_t i = 0; i < nn.layers_count; ++i)
    {
        printf("Layer%lu {\n", i);
        NN_Layer layer = nn.layers[i];
        for (size_t j = 0; j < layer.neurons_count; ++j)
        {
            NN_Neuron neuron = layer.neurons[j];
            printf("\tNeuron%lu {\n", j);
            printf("\t\tact = %f,\n", neuron.act);
            printf("\t\tbias = %f,\n", neuron.bias);
            printf("\t\tweights = [\n");
            for (size_t k = 0; k < neuron.weights_count; ++k)
            {
                if (k == neuron.weights_count - 1)
                    printf("\t\t\t weight%lu = %f\n", k, neuron.weights[k]);
                else
                    printf("\t\t\t weight%lu = %f,\n", k, neuron.weights[k]);
            }
            printf("\t\t]\n");
        }
        printf("}\n");
    }
}

float nn_network_cost(NN_Network nn, NN_Layer* inputs, NN_Layer* outputs, size_t entries_count)
{
    float cost = 0.f;
    for (size_t i = 0; i < entries_count; ++i)
    {
        float partial_cost = 0.f;

        NN_Layer input = inputs[i];
        NN_Layer output = outputs[i];

        NN_ASSERT(NN_INPUTS(nn).neurons_count == input.neurons_count);
        NN_ASSERT(NN_OUTPUTS(nn).neurons_count == output.neurons_count);

        nn_network_set_input(nn, input);
        nn_network_forward(nn);

        for (size_t j = 0; j < output.neurons_count; ++j)
        {
            float prediction = NN_OUTPUTS(nn).neurons[j].act;
            float expected   = output.neurons[j].act;
            float distance   = prediction - expected;
            partial_cost     += distance * distance;
        }
        cost += partial_cost; // perhaps we'd want to square it in the future
    }

    return cost / entries_count; // taking the avg sum
}

void nn_network_finite_differences(NN_Network nn, NN_Network gradient, float epsilon,
                                 NN_Layer* inputs, NN_Layer* outputs, size_t entries_count)
{
    float cost_original = nn_network_cost(nn, inputs, outputs, entries_count);

    for (size_t i = 0; i < nn.layers_count; ++i)
    {
        NN_Layer* l = &nn.layers[i];
        for (size_t j = 0; j < l->neurons_count; ++j)
        {
            NN_Neuron* neuron = &l->neurons[j];
            float temp;
            float cost_new;
            float partial_derivative;

            for (size_t k = 0; k < neuron->weights_count; ++k)
            {
                temp = neuron->weights[k];
                neuron->weights[k] += epsilon;
                cost_new = nn_network_cost(nn, inputs, outputs, entries_count);
                partial_derivative = (cost_new - cost_original) / epsilon;
                gradient.layers[i].neurons[j].weights[k] = partial_derivative;
                neuron->weights[k] = temp;
            }
            temp = neuron->bias;
            neuron->bias += epsilon;
            cost_new = nn_network_cost(nn, inputs, outputs, entries_count);
            partial_derivative = (cost_new - cost_original) / epsilon;
            gradient.layers[i].neurons[j].bias = partial_derivative;
            neuron->bias = temp;
        }
    }
}

void nn_network_learn(NN_Network nn, NN_Network gradient, float learning_rate)
{
    NN_ASSERT(nn.layers_count == gradient.layers_count);
    for (size_t i = 0; i < nn.layers_count; ++i)
    {
        NN_Layer* l = &nn.layers[i];
        NN_ASSERT(l->neurons_count == gradient.layers[i].neurons_count);
        for (size_t j = 0; j < l->neurons_count; ++j)
        {
            NN_Neuron* neuron = &l->neurons[j];
            for (size_t k = 0; k < neuron->weights_count; ++k)
            {
                neuron->weights[k] -= gradient.layers[i].neurons[j].weights[k] * learning_rate;
            }
            neuron->bias -= gradient.layers[i].neurons[j].bias * learning_rate;
        }
    }
}
#endif // NN_IMPLEMNTATION
