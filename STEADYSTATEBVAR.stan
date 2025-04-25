functions {
  matrix kronecker_prod(matrix A, matrix B) {
  matrix[rows(A) * rows(B), cols(A) * cols(B)] C;
  int m;
  int n;
  int p;
  int q;
  m = rows(A);
  n = cols(A);
  p = rows(B);
  q = cols(B);
  for (i in 1:m) {
    for (j in 1:n) {
      int row_start;
      int row_end;
      int col_start;
      int col_end;
      row_start = (i - 1) * p + 1;
      row_end = (i - 1) * p + p;
      col_start = (j - 1) * q + 1;
      col_end = (j - 1) * q + q;
      C[row_start:row_end, col_start:col_end] = A[i, j] * B;
    }
  }
  return C;
  }
}

data {
  int<lower=0> T; //number of observations minus lag length
  int<lower=0> m; //number of variables
  int<lower=0> p; //lag length
  matrix[T, k] Y; //Matrix of responses
  matrix[T, k*p] Z; //Design matrix
  int<lower=0> H; // Forecast horizon
  
  vector[m*p*m] vec_beta_pr_mean; //vec(beta) prior mean vector
  matrix[m*p*m, m*p*m] vec_beta_pr_cov; //vec(beta) prior covariance matrix
  
  vector[m] mu_pr_mean; //steady-state prior mean vector
  matrix[m, m] mu_pr_cov; //steady-state prior covariance matrix
  
  int<lower=0> gamma; //Sigma prior degrees of freedom
  matrix[m, m] Xi_Sigma; //Sigma prior scale matrix
}

transformed data {
    vector[p] jota = rep_vector(1, p); //Column vector of ones
    matrix[p, p] I = diag_matrix(rep_vector(1, p)); //Identity matrix
}

parameters {
  matrix[m*p, m] Beta;
  matrix[m] mu; //steady-state
  cov_matrix[m] Sigma;
}

model {
  //Likelihood
  for(i in 1:(T)){
      Y[i] ~ multi_normal(mu' + Z[i]*Beta - jota' * kronecker_prod(I,mu') * Beta, Sigma);
  }
  //Priors
  to_vector(Beta) ~ multi_normal(vec_beta_pr_mean, vec_beta_pr_cov);
  mu ~ multi_normal(mu_pr_mean, mu_pr_cov);
  Sigma ~ inv_wishart(gamma, Xi_Sigma);
}

generated quantities {
  
  matrix[m, m] phi[p];
  for (i in 1:p) {
    phi[i] = Beta[((i - 1) * m + 1):(i * m), :];  //Extract phi_1 , ... , phi_p
  }
  
  matrix[H, m] Y_hat; //forecasts
  
  for (h in 1:H) {
    vector[m] e_t = multi_normal_rng(rep_vector(0, m), Sigma);
    vector[m] mu_t = mu; //y_t = mu_t + e_t

    if (h > 1) {
      for (i in 1:min(h-1, p)) {
        mu_t += to_vector((Y_hat[h-i] - mu') * phi[i]);
      }
    }

    if (h <= p) {
      for (i in h:p) {
        mu_t += to_vector((Y[T + h - i] - mu') * phi[i]);
      }
    }
    Y_hat[h] = (mu_t + e_t)';
  }
}
