import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

class FFT_Parallel implements Callable<Complex[]> {

    public static ExecutorService threadPool;
    public static int sequentialN;
    public static volatile AtomicInteger remainingCores = new AtomicInteger(Runtime.getRuntime().availableProcessors() / 2);

    Complex[] x;
    int N;
    int cores;

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

        for(int k = 0; k < N/2; k++) {
            odd[k] = x[2*k + 1];
        }

        Complex[] q;
        Complex[] r;
        Future<Complex[]> evenF;
        Future<Complex[]> oddF;

        if(remainingCores.decrementAndGet() > 0){
            evenF = threadPool.submit(new FFT_Parallel(even));
            oddF = threadPool.submit(new FFT_Parallel(odd));
            q = evenF.get();
            r = oddF.get();
        } else {
            q = FFT_Sequential.FFT(even);
            r = FFT_Sequential.FFT(odd);
        } 


        Complex[] y = new Complex[N];
        for (int k = 0; k < N/2; k++) {
            double kth = -2 * k * Math.PI / N;
            Complex wk = new Complex(Math.cos(kth), Math.sin(kth));
            y[k]       = q[k].plus(wk.times(r[k]));
            y[k + N/2] = q[k].minus(wk.times(r[k]));
        }
        return y;
    }
}