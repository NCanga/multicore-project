class FFT_Sequential {
    
    public static Complex[] FFT(Complex[] x) throws Exception {
        int N = x.length;

        if (N == 1) return  new Complex[] {x[0]};
        if (N % 2 != 0) throw new IllegalArgumentException();
        Complex[] even = new Complex[N/2];
        Complex[] odd = new Complex[N/2];

        for(int k = 0; k < N/2; k++) {
            even[k] = x[2*k];
        }
        Complex[] q = FFT(even);
        for(int k = 0; k < N/2; k++) {
            odd[k] = x[2*k + 1];
        }
        Complex[] r = FFT(odd);


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