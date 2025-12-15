#!/bin/bash
C_VERSION="c99"
FLAGS="-Iexternal -Wall -Wextra -gdwarf-4 -O0 -lm"
clang -std=$C_VERSION $FLAGS nn_xor.c -o nn_xor.o
clang -std=$C_VERSION $FLAGS nn_mul.c -o nn_mul.o
clang -std=$C_VERSION $FLAGS nn_back_propagation_test.c -o nn_back_propagation_test.o
clang -std=$C_VERSION $FLAGS nn_images.c -o nn_images.o
