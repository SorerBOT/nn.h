#!/bin/bash
C_VERSION="c99"
FLAGS="-Iexternal -Wall -Wextra -gdwarf-4 -O0 -lm"
clang -std=$C_VERSION $FLAGS ./examples/nn_xor.c                    -o bin/nn_xor.o
clang -std=$C_VERSION $FLAGS ./examples/nn_mul.c                    -o bin/nn_mul.o
clang -std=$C_VERSION $FLAGS ./examples/nn_back_propagation_test.c  -o bin/nn_back_propagation_test.o
clang -std=$C_VERSION $FLAGS ./examples/nn_images.c                 -o bin/nn_images.o
