#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string>
#include <cassert>
#include <cstring>
#include <cstdint>
#include <float.h>

#include <mpi.h>

#include <iostream>

#include "utils.hpp"


bool VERBOSE;
char* FILE_PATH_IN;
char* FILE_PATH_OUT;
int STEP_COUNT;
double DELTA_TIME;
uint32_t N;
uint32_t NUM_PROC;

const double E0 = 1.0;

const double EPS = DBL_EPSILON;

// https://stackoverflow.com/a/34660211
float power(float base, unsigned int exp) {
    if (exp == 0)
       return 1;
    float temp = power(base, exp / 2);       
    if (exp % 2 == 0)
        return temp*temp;
    else {
        if (exp > 0)
            return base*temp*temp;
        else
            return (temp * temp) / base; // negative exponent computation 
    }
} 

double compute_V(double rij, double rik, double rkj) {
    double three = rij * rik * rkj;
    double rij2 = rij * rij;
    double rik2 = rik * rik;
    double rkj2 = rkj * rkj;
    return E0 * (1 / power(three, 3) + 3 * ((-rij2 + rik2 + rkj2) * (rij2 - rik2 + rkj2) * (rij2 + rik2 - rkj2)) 
        / 8 * power(three, 5));
}

int main(int argc, char **argv) {
    parse_args(argc, argv);

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &NUM_PROC);

    parse_input(get_input_content());
    return 0;
}
