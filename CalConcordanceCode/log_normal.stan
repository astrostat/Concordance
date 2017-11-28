data {
	int<lower=1>n;
    int<lower=1>N;
    int<lower=1>M;
	int<lower=1>I[n];
	int<lower=1>J[n];
    real y[n];
    real<lower=0> df;
	real<lower=0> beta;
    real<lower=0> tau[N];
    real b[N];
}

parameters {
    real B[N];
    real G[M];
    real<lower=0>sigma2[N];
}

transformed parameters {
    real mu[n];
	for (k in 1:n){
		mu[k] = B[I[k]] + G[J[k]] - sigma2[I[k]]/2;
	}
}

model {
    for (k in 1:n){
        y[k] ~ normal(mu[k], sqrt(sigma2[I[k]]));
	}
	for (i in 1:N){
        B[i] ~ normal(b[i], tau[i]);
		sigma2[i] ~ inv_gamma(df, beta);
    }
}
