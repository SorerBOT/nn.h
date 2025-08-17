#!/bin/bash
C_VERSION="c99"
clang -std=$C_VERSION mul.c -o mul.o -lm
