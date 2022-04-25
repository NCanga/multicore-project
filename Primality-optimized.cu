#include <iostream>
#include <string>
#include <cmath>
#include <chrono>
#include <assert.h>
#include <thread>
#include <future>


using namespace std;
#define THREADS_PER_BLOCK 128

__global__ void parallelPrimality(int *number, bool* result) {
    int i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x + 2;
    if ((i*i <= *number) && (*number % i) == 0) {
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

// time is set to the execution time of the algorithm
// algo is set to true if the sequential result is used, false if the parrallel algo is used

bool optimizedPrimality(int number, int *time, bool *algo) {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::microseconds;
    using std::thread;
    using std::future;

    int input = number;
    bool result = true;
    cudaEvent_t finish;
    cudaEventCreate(&finish);
    int rootInput = (int)(sqrt(input));
	int numBlocks = (rootInput/THREADS_PER_BLOCK) + 1;

	int* d_input;
	bool* d_result;

	cudaMalloc((void**) &d_input, sizeof(int));
	cudaMalloc((void**) &d_result, sizeof(bool));

	cudaMemcpy(d_input, &input, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_result, &result, sizeof(bool), cudaMemcpyHostToDevice);

    //Launch the algorithms
    auto t1 = high_resolution_clock::now();
    // launch sequential algorithm
    future<bool> isPrime = std::async(std::launch::async, [number] {
        return sequentialPrimality(number);
    });
    // launch parrallel algorithm
    parallelPrimality<<<numBlocks, THREADS_PER_BLOCK>>>(d_input, d_result); 
    cudaEventRecord(finish);
    // wait for one of the algorithms to finish
    while (true) {
        if (isPrime.wait_for(0us) == std::future_status::ready) {
            //sequential algo finished
            *algo = true;
            break;
        }  
        if (cudaEventQuery(finish) == cudaSuccess) {
            // parrallel algo finished
            *algo = false;
            break;
        } 
    }
    auto t2 = high_resolution_clock::now();
    *time = duration_cast<microseconds>(t2 - t1).count();

    
    if (algo) {
        result = isPrime.get();
    } else {
        cudaMemcpy(&result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
    }
    return result;

}

int main(int argc, char* argv[]) {
    
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::microseconds;

   // int val = 982451653;
    //int val = 100000000;
    int numbers[11] = {982451653, 962060129, 962060125, 971296121, 971296120, 971352311, 971352313, 977953759, 977953752, 982451653, 982451654};
    for (int i = 0; i < 11; i++) {
        int val = numbers[i];
        int time;
        bool algo;
    // auto t1 = high_resolution_clock::now();
        bool isPrime = optimizedPrimality(val, &time, &algo);
    // auto t2 = high_resolution_clock::now();
        //skip 1st value 
        if (i != 0) {
            cout << "Value: " << val << endl;
            if (isPrime) {
                cout << "Result: Prime" << endl;
            } else {
                cout << "Result: Not Prime" << endl;
            }

            if (algo) {
                cout << "Algorithm: Sequential" << endl;
            } else {
                cout << "Algorithm: Parrallel" << endl;
            }
        // auto ms_int = duration_cast<microseconds>(t2 - t1);
            cout << "Execution Time: " << time << " microseconds" << endl;

            cout << endl;
        }
    }
    



} 
