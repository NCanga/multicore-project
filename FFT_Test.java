import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

class FFT_Test {
    public static void main(String[] args) throws Exception{
        Random ran = new Random();
        int n = (int) Math.pow(2, 18);
        Complex[] input = new Complex[n];
        for (int i = 0; i < n; i++) {
            input[i] = new Complex(i, 0);
            input[i] = new Complex(ran.nextInt(10), 0);
        }

        ExecutorService es = Executors.newSingleThreadExecutor();
        FFT_Parallel.threadPool = Executors.newCachedThreadPool();
        FFT_Parallel.sequentialN = 15;
        Future<Complex[]> parallel_output_f = es.submit(new FFT_Parallel(input));
        long startTime = System.currentTimeMillis();
        Complex[] parallel_output = parallel_output_f.get();
        parallel_output = null;
        long endTime = System.currentTimeMillis();
        System.out.println("Total parallel execution time: " + (endTime - startTime));
        es.shutdown();
        FFT_Parallel.threadPool.shutdown();
        startTime = System.currentTimeMillis();
        Complex[] sequential_output = FFT_Sequential.FFT(input);
        endTime = System.currentTimeMillis();
        System.out.println("Total sequential execution time: " + (endTime - startTime));

        for(int i = 0; i < n; i++){
            if(!parallel_output[i].equals(sequential_output[i])){
                System.out.println("Difference between parallel and sequential");
            }
        }
    } 
}