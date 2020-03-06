#include <iostream>
#include <omp.h>

#include <array>
#define THREADS 4 //you can also use the OMP_NUM_THREADS environmental variable

double power(double x, long n) {
    if (n == 0) {
        return 1;
    }

    return x * power(x, n - 1);
}


using namespace std;

int sumNumbers(int* inputArray, int n) {
  int totalValue = 1;
  #pragma omp parallel for reduction(*:totalValue) num_threads(3)
  for (int i = 0; i < n; i++) {
    totalValue += inputArray[i];


  #pragma omp critical
  {
    std::cout << totalValue << "\n";
  }

  }
  return totalValue;
}

double powerParallelReduction(double x, long n) {
    double totalValue = 1;
    #pragma omp parallel for reduction(*:totalValue) num_threads(THREADS)
    for (int i = 0; i < THREADS; i++) {
        int pow = n / THREADS;
        totalValue *= power(x, pow);
    }
    totalValue *= power(x, n % THREADS); // wasn't handled yet
    return totalValue;
}

int main(int argc, char *argv[]) {
  int threadCount, threadId;

//  int arr[5] = {1,2,3,4,5};
  double res = powerParallelReduction(2.0, 5);
  std::cout << res;
  return 0;
}
