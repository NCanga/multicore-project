import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;


class FFT implements Callable<int[]> {

    public static ExecutorService threadPool = Executors.newCachedThreadPool();
    int[] x;

    public FFT(int[] x) {
        this.x = x;    
    }
    
    public static void main(String[] args) throws InterruptedException, ExecutionException{
        int[] input = {1};
        ExecutorService es = Executors.newSingleThreadExecutor();
        Future<int[]> output = es.submit(new FFT(input));
        System.out.println(Arrays.toString(output.get()));
        es.shutdown();
        FFT.threadPool.shutdown();
    }

    @Override
    public int[] call() throws Exception {
        int N = x.length;
        if (N == 1) return x;

        return null;
    }
}