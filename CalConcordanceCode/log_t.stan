data {
	int<lower=1>n;
    int<lower=1>N;
    int<lower=1>M;
	int<lower=1>I[n];
	int<lower=1>J[n];
    real y[n];
    real sigma;
    real nu;
    real<lower=0> tau[N];
    real b[N];
}

parameters {
    real B[N];
    real G[M];
    real<lower=0>xi[n];
}

transformed parameters {
    real mu[n];
	for (k in 1:n){
		mu[k] = B[I[k]] + G[J[k]] - sigma*sigma/(2*xi[k]);
	}
}

model {
    for (k in 1:n){
        y[k] ~ normal(mu[k], sigma/sqrt(xi[k]));
		xi[k] ~ chi_square(nu);
	}
	for (i in 1:N){
        B[i] ~ normal(b[i], tau[i]);
    }
}