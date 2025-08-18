#!/bin/bash
C_VERSION="c99"
FLAGS="-Wall -Wextra"
clang -std=$C_VERSION $FLAGS nn_xor.c -o nn_xor.o -lm
