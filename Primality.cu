#include <iostream>
#include <string>
#include <cmath>
#include <chrono>
#include <assert.h>

using namespace std;
#define THREADS_PER_BLOCK 128

__global__ void parallelPrimality(int *number, bool* result) {
    int i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x + 2;
    if (*number % i == 0) {
        *result = false;
    }
}

bool sequentialPrimality(int num) {
    for(int i = 2; i*i <= num; i++) {
        if ((num % i ) == 0) {
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[]) {

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::microseconds;

    int input = stoi(argv[1]);
    bool result = true;

    auto t1 = high_resolution_clock::now();
    bool isprime = sequentialPrimality(input);
    auto t2 = high_resolution_clock::now();

    if(isprime) {
        cout << "Sequential prime" << endl;
    } else {
        cout << "Sequential Value is not prime" << endl;
    }
    auto ms_int = duration_cast<microseconds>(t2 - t1);
    std::cout << ms_int.count() << "ms\n";


    int numBlocks = (int)sqrt(input);

    int* d_input;
    bool* d_result;

    cudaMalloc((void**) &d_input, sizeof(int));
    cudaMalloc((void**) &d_result, sizeof(bool));

    cudaMemcpy(d_input, &input, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &result, sizeof(bool), cudaMemcpyHostToDevice);

    t1 = high_resolution_clock::now();
    parallelPrimality<<<numBlocks, THREADS_PER_BLOCK>>>(d_input, d_result);
    t2 = high_resolution_clock::now();
    ms_int = duration_cast<microseconds>(t2 - t1);
 

    cudaMemcpy(&result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);

    if(result){
        cout << "Parallel prime" << endl;
    } else {
        cout << "Parallel not prime" << endl;
    }
    std::cout << ms_int.count() << "ms\n";
}