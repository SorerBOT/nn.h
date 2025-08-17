#!/bin/bash
C_VERSION="c99"
FLAGS="-Wall -Wextra"
clang -std=$C_VERSION $FLAGS nn_mul.c -o nn_mul.o -lm
