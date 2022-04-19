import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;


class FFT_Parallel implements Callable<Complex[]> {

    public static ExecutorService threadPool = Executors.newCachedThreadPool();

    Complex[] x;
    int N;

    public FFT_Parallel(Complex[] x) {
        this.x = x;
        this.N = x.length;
    }

    @Override
    public Complex[] call() throws Exception {
        if (N == 1) return  new Complex[] {x[0]};
        if (N % 2 != 0) throw new IllegalArgumentException();
        Complex[] even = new Complex[N/2];
        Complex[] odd = new Complex[N/2];

        for(int k = 0; k < N/2; k++) {
            even[k] = x[2*k];
        }
        Future<Complex[]> evenF = threadPool.submit(new FFT_Parallel(even));
        for(int k = 0; k < N/2; k++) {
            odd[k] = x[2*k + 1];
        }
        Future<Complex[]> oddF = threadPool.submit(new FFT_Parallel(odd));
        Complex[] q = evenF.get();
        Complex[] r = oddF.get();


        Complex[] y = new Complex[N];
        for (int k = 0; k < N/2; k++) {
            double kth = -2 * k * Math.PI / N;
            Complex wk = new Complex(Math.cos(kth), Math.sin(kth));
            y[k]       = q[k].plus(wk.times(r[k]));
            y[k + N/2] = q[k].minus(wk.times(r[k]));
        }
        return y;
    }
    
    public static void main(String[] args) throws InterruptedException, ExecutionException{
        Random ran = new Random();
        int n = (int) Math.pow(2, 12);
        Complex[] input = new Complex[n];
        for (int i = 0; i <n; i++) {
            input[i] = new Complex(i, 0);
            input[i] = new Complex(ran.nextInt(10), 0);
        }
        ExecutorService es = Executors.newSingleThreadExecutor();
        final long startTime = System.currentTimeMillis();
        Future<Complex[]> output = es.submit(new FFT_Parallel(input));
        final long endTime = System.currentTimeMillis();
        System.out.println("Total execution time: " + (endTime - startTime));

        //System.out.println(Arrays.toString(output.get()));
        es.shutdown();
        FFT_Parallel.threadPool.shutdown();

       // System.out.println(output.get());
    }
}