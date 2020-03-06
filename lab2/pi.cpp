#include <omp.h>
#include <iostream>
#include <iomanip>

//#define STEPS 1000
#define THREADS 4

double power(double x, long n) {
    if (n == 0) {
        return 1;
    }

    return x * power(x, n - 1);
}

double calcPi(long n) {
    if (n < 0) {
        return 0;
    }

    double pow_res = 1.0 / power(16, n)
                     * (4.0 / (8 * n + 1.0)
                        - 2.0 / (8 * n + 4.0)
                        - 1.0 / (8 * n + 5.0)
                        - 1.0/(8 * n + 6.0));

    return pow_res + calcPi(n - 1);
}

double powerParallelReduction(double x, long n) {
    double totalValue = 1;
#pragma omp parallel for reduction(*:totalValue) num_threads(THREADS)
    for (int i = 0; i < THREADS; i++) {
        int pow = n / THREADS;
        totalValue *= power(x, pow);
    }
    totalValue *= power(x, n % THREADS);
    return totalValue;
}



double powerParallelCritical(double x, long n) {

    double totalValue = 1.0;
#pragma omp parallel for num_threads(THREADS)
    for(int i = 0; i < n; i++) {
        #pragma omp critical
        {
            totalValue *= x;
        }
    }

    return totalValue;
}

double partOfPiCalcParallelReduction(int n) {
    return 1.0 / powerParallelReduction(16, n)
           * (4.0 / (8 * n + 1.0)
              - 2.0 / (8 * n + 4.0)
              - 1.0 / (8 * n + 5.0)
              - 1.0/(8 * n + 6.0));
}

double partOfPiCalcParallelCritical(int n) {
    return 1.0 / power(16, n)
           * (4.0 / (8 * n + 1.0)
              - 2.0 / (8 * n + 4.0)
              - 1.0 / (8 * n + 5.0)
              - 1.0/(8 * n + 6.0));
}

double calcPiParallelReduction(long n) {
    double totalValue = 0;
#pragma omp parallel for reduction(+:totalValue) num_threads(THREADS)
    for (int i = 0; i < n; i++) {
        totalValue += partOfPiCalcParallelReduction(i);
    }
    return totalValue;
}

double calcPiParallelCritical(long n) {
    double totalValue = 0;
#pragma omp parallel for num_threads(THREADS)
    for (int i = 0; i < n; i++) {
        int res = partOfPiCalcParallelCritical(i);
        #pragma omp critical
        {
            totalValue += res;
        }
//        #pragma omp atomic update
//        totalValue += partOfPiCalcParallelCritical(i);
    }
    return totalValue;
}

int main(int argc, char *argv[]) {

    if (argc < 2) {
        // Tell the user how to run the program
        std::cerr << "Wrong argument: I need STEPS, type(STEPS) = unsigned int\n";
        /* "Usage messages" are a conventional way of telling the user
         * how to run a program if they enter the command incorrectly.
         */
        return 1;
    }
    char *endptr;
    int STEPS = strtol(argv[1], &endptr, 10);

    double startSeq = omp_get_wtime();
//    std::cout << std::setprecision(10) << calcPi(STEPS) << std::endl;
    calcPi(STEPS);
    double startRed = omp_get_wtime();
//    std::cout << std::setprecision(10) << calcPiParallelReduction(STEPS) << std::endl;
    calcPiParallelReduction(STEPS);
    double startCrit = omp_get_wtime();
//    std::cout << std::setprecision(10) << calcPiParallelCritical(STEPS) << std::endl;
    calcPiParallelCritical(STEPS);
    double endCrit = omp_get_wtime();

    double seqTime = startRed - startSeq;
    double redTime = startCrit - startRed;
    double critTime = endCrit - startCrit;

//    std::cout << std::setprecision(2) << "reduction to sequential speedup: "
    std::cout << seqTime / redTime << "\n";

//    std::cout << std::setprecision(2) << "critical to sequential speedup: " << seqTime / critTime << "\n";
    std::cout << seqTime / critTime;
    return 0;
}
