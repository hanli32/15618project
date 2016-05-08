#!/bin/bash
./gpu_test 100 > result
./gpu_test 500 >> result
./gpu_test 1000 >> result
./gpu_test 5000 >> result
./gpu_test 10000 >> result
./gpu_test 50000 >> result
./gpu_test 100000 >> result
./gpu_test 500000 >> result
./gpu_test 1000000 >> result
