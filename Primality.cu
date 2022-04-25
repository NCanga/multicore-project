#include <iostream>
#include <string>
#include <cmath>
#include <chrono>
#include <assert.h>

using namespace std;
#define THREADS_PER_BLOCK 128

__global__ void parallelPrimality(int *number, bool* result) {
    int i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x + 2;
    if ((i*i) <= *number) {
		//atomicAdd(total, 1);
		if ((*number % i) == 0) {
			atomicExch((int *) result, (int) false);
    	} 
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
    cout << "*****Prime Testing*****" << endl;
    int primes[9] = {101, 101, 1009, 10009, 100129, 1398269, 15485867, 104395303, 982451653};
    for (int j = 0; j < 9; j++) {
	    // NO PRINT STATEMENTS FOR 1st CUDA KERNEL CALL. 
	    // first call has some added latency which throws our numbers off  		
	    int input = primes[j];
	    bool result = true;
            if (j != 0) {
               cout << "Number: " << input << endl;
	    }
	    auto t1 = high_resolution_clock::now();
	    bool isprime = sequentialPrimality(input);
	    auto t2 = high_resolution_clock::now();
            if (j != 0) {
		    if(isprime) {
			cout << "Sequential Result: Prime" << endl;
		    } else {
			cout << "Sequential Result: Not Prime" << endl;
		    }
	    }
	    auto ms_int = duration_cast<microseconds>(t2 - t1);
	    if (j != 0) {
	    	std::cout << "Execution Time: " << ms_int.count() << " microseconds\n";
	    }


	    int rootInput = (int)(sqrt(input));
	    int numBlocks = (rootInput/THREADS_PER_BLOCK) + 1;

	    int* d_input;
	    bool* d_result;

	    cudaMalloc((void**) &d_input, sizeof(int));
	    cudaMalloc((void**) &d_result, sizeof(bool));


	    cudaMemcpy(d_input, &input, sizeof(int), cudaMemcpyHostToDevice);
	    cudaMemcpy(d_result, &result, sizeof(bool), cudaMemcpyHostToDevice);

	    t1 = high_resolution_clock::now();
	    parallelPrimality<<<numBlocks, THREADS_PER_BLOCK>>>(d_input, d_result); 
		cudaDeviceSynchronize();
	    t2 = high_resolution_clock::now();
	    ms_int = duration_cast<microseconds>(t2 - t1);

	    cudaMemcpy(&result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
	 
        if (j != 0) {
		    if(result){
			cout << "Parallel Result: Prime" << endl;
		    } else {
			cout << "Parallel Result: Not Prime" << endl;
		    }
	    }
        if (j != 0) {
	    	std::cout << "Execution Time: " << ms_int.count() << " microseconds\n";
			std::cout << endl;
	    }
	    cudaFree(d_input);
	    cudaFree(d_result);

	}
	
    // NOT prime numbers
    cout << "*****NOT Prime Testing*****" << endl;
    int not_primes[8] = {100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000};
    for (int j = 0; j < 8; j++) {
	    		
	    int input = not_primes[j];
	    bool result = true;
            cout << "Number: " << input << endl;
	    auto t1 = high_resolution_clock::now();
	    bool isprime = sequentialPrimality(input);
	    auto t2 = high_resolution_clock::now();

	    if(isprime) {
		cout << "Sequential Result: Prime" << endl;
	    } else {
		cout << "Sequential Result: Not Prime" << endl;
	    }
	    auto ms_int = duration_cast<microseconds>(t2 - t1);
	    std::cout << "Execution Time: " << ms_int.count() << " microseconds\n";


	    int rootInput = (int)(sqrt(input));
	    int numBlocks = (rootInput/THREADS_PER_BLOCK) + 1;

	    int* d_input;
	    bool* d_result;

	    cudaMalloc((void**) &d_input, sizeof(int));
	    cudaMalloc((void**) &d_result, sizeof(bool));

	    cudaMemcpy(d_input, &input, sizeof(int), cudaMemcpyHostToDevice);
	    cudaMemcpy(d_result, &result, sizeof(bool), cudaMemcpyHostToDevice);

	    t1 = high_resolution_clock::now();
	    parallelPrimality<<<numBlocks, THREADS_PER_BLOCK>>>(d_input, d_result); 
	    cudaDeviceSynchronize();
	    t2 = high_resolution_clock::now();
	    ms_int = duration_cast<microseconds>(t2 - t1);

	    cudaMemcpy(&result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);

	    if(result){
		cout << "Parallel Result: Prime" << endl;
	    } else {
		cout << "Parallel Result: Not Prime" << endl;
	    }
	    std::cout << "Execution Time: " << ms_int.count() << " microseconds\n";
	    std::cout << endl;
	    cudaFree(d_input);
	    cudaFree(d_result);

	}
	
}
