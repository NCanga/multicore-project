import java.io.PrintWriter;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

class FFT_Test {
    public static void main(String[] args) throws Exception{
        Random ran = new Random();
        int n = (int) Math.pow(2, 25);
        Complex[] input = new Complex[n];
        for (int i = 0; i < n; i++) {
            input[i] = new Complex(i, 0);
            input[i] = new Complex(ran.nextInt(2), 0);
        }

        ExecutorService es = Executors.newSingleThreadExecutor();
        FFT_Parallel.threadPool = Executors.newCachedThreadPool();

        //Parallel FFT
        long startTime = System.currentTimeMillis();
        Future<Complex[]> parallel_output_f = es.submit(new FFT_Parallel(input));
        Complex[] parallel_output = parallel_output_f.get();
        long endTime = System.currentTimeMillis();

        System.out.println("Total parallel execution time: " + (endTime - startTime));
        es.shutdown();
        FFT_Parallel.threadPool.shutdown();

        //Sequntial FFT
        startTime = System.currentTimeMillis();
        Complex[] sequential_output = FFT_Sequential.FFT(input);
        endTime = System.currentTimeMillis();
        System.out.println("Total sequential execution time: " + (endTime - startTime));

        try
        {
            PrintWriter pr = new PrintWriter("fft_out.txt");    
        
            for (int i=0; i<n; i++)
            {
                pr.println(parallel_output[i]);
            }
            pr.close();
        }
        catch (Exception e)
        {
            e.printStackTrace();
            System.out.println("No such file exists.");
        }

        for(int i = 0; i < n; i++){
            if(!parallel_output[i].equals(sequential_output[i])){
                System.out.println("Difference between parallel and sequential outputs");
            }
        }
    } 
}