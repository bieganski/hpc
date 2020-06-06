#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string>
#include <cassert>
#include <cstring>
#include <cstdint>
#include <float.h>

#include <iostream>
bool VERBOSE;
char* FILE_PATH_IN;
char* FILE_PATH_OUT;
int STEP_COUNT;
double DELTA_TIME;

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

int parse_args (int argc, char **argv) {
    int gflag = 0, fflag = 0;
    int c;

    opterr = 0;

    if (argc != 5 && argc != 6) {
        fprintf (stderr, "Usage: ./body3 particles_in.txt particles_out stepcount deltatime [-v]\n");
        exit(1);
    }
    FILE_PATH_IN = argv[1];
    FILE_PATH_OUT = argv[2];
    try {
        STEP_COUNT = std::stoi(std::string(argv[3]));
    } catch  (... ) {
        fprintf (stderr, "step_count must be proper unsigned int!\n");
        exit(1);
    }
    try {
        DELTA_TIME = std::stod(std::string(argv[4]));
    } catch  (... ) {
        fprintf (stderr, "delta_time must be proper double!\n");
        exit(1);
    }

    if (argc == 6) {
        if (std::strcmp(argv[5], "-v")) {
            fprintf (stderr, "last argument expected is '-v'!\n");
            exit(1);
        }
        VERBOSE = true;
    } else {
        VERBOSE = false;
    }

    return 0;
}


int main(int argc, char **argv) {
    parse_args(argc, argv);
    return 0;
}
