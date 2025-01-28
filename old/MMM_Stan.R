###
#
# Market Mix Modelling with Stan
#
###
library(rstan)
library(ggplot2)
library(tidyr)
library(cowplot)
data <-  read.csv("data.csv")
data = rbind(data,data, data)

data$wk_strt_dt2 <- as.Date(data$wk_strt_dt)

p1 <- data %>% ggplot(aes(wk_strt_dt2, y= sales/10^6)) + 
  geom_point() + 
  geom_smooth(color = "black") + 
  xlab("Time") + 
  ylab("Sales in Millions") + 
  theme_minimal() + 
  ylim(0, 400) +
  geom_hline(yintercept = 0, linetype="dashed")
p2 <- data %>% ggplot(aes(wk_in_yr_nbr, y= sales/10^6)) +
  geom_point() + 
  geom_smooth(color = "black") + 
  xlab("Week of the year") + 
  ylab("Sales in Millions") + 
  theme_minimal()  + 
  ylim(0, 400) +
  geom_hline(yintercept = 0, linetype="dashed")

plot_grid(p1, p2)



# Media variables
mdip <- data[names(data)[grep("mdip_", names(data))]] # chanels
names(mdip) <- c("Direct Mail", "Insert", "Newspaper", "Digital Audio", "Radio", "TV", "Digital Video" , "Social", "Online Display",
                 "Email",  "SMS", "Affiliates", "SEM")




data.frame(time = data$wk_strt_dt2 , mdip) %>%
  pivot_longer(!time, names_to = "Chanel", values_to = "spending") %>%
  ggplot(aes(x = time, y = spending/10^6)) + geom_point() + geom_smooth(span = 0.2) + facet_grid(Chanel ~ ., scales = "free") + 
  ylab("Spending in Million")


porportions = data.frame(time = data$wk_strt_dt2 , mdip) %>%
  pivot_longer(!time, names_to = "Chanel", values_to = "spending") %>% mutate(total_spending = sum(spending)) %>% 
  group_by(Chanel) %>% summarise(prop = sum(spending)/total_spending[1], spend =  round(sum(spending)/10^6))


porportions %>% ggplot(aes(x=reorder(Chanel, prop), y = prop)) + geom_bar(stat="identity") + coord_flip() + theme_minimal() + 
  ylab("Proportion spend on chanel \n and total amount spend in million") + xlab("Chanel") + geom_text(aes(label = spend), hjust=-0.1) + ylim(0,0.3)

data.frame(sales =data$sales , mdip) %>%
  pivot_longer(!sales, names_to = "Chanel", values_to = "spending") %>% ggplot(aes(x=spending/10^6, y=sales/10^6)) + geom_point() +
  facet_wrap(. ~ Chanel, scales = "free") + ylim(0,400) + geom_smooth(method= "lm") + xlab("Spending in Million") + xlab("Sales in Million")

summary(lm(sales ~ +  ., data = data.frame(sales =data$sales , mdip) ))


hldy <- data[names(data)[grep("hldy_", names(data))]] #holiday 
mrkdn <- data[names(data)[grep("mrkdn_", names(data))]] #discount 
seas <- data[names(data)[grep("seas_", names(data))]] #seas 
me <- data[names(data)[grep("me_", names(data))]] #macro economics 

control_variables <-  cbind(hldy, mrkdn, seas, me)

control_variables_centered <- scale(control_variables, center = TRUE, scale = FALSE)



stan_dat <- list(N = dim(control_variables_centered)[1],
            N_pred = dim(control_variables_centered)[2],
            X = control_variables_centered,
            sales = data$sales)


simple_control_model = "
data {
  int N; // number of observations
  int N_pred; // number of predictors

  matrix[N, N_pred] X;
  vector[N] sales; 
}

parameters {
  vector[N_pred] beta; // regression coefficients 
  real<lower=0> intercept; // intercept
  real<lower=0> sigma; // residual variance
}

model {
  // Define the priors
  beta ~ normal(0, 1); 
  sigma ~ inv_gamma(1, 1);
  // The likelihood
  sales ~ normal(X *beta + intercept, sigma);
}

generated quantities {
  vector[N] log_lik;
  for (n in 1:N) log_lik[n] = normal_lpdf(sales[n] | dot_product(X[n,], beta) + intercept, sigma);
}
"



m_control <- stan( model_code  = simple_control_model, data=stan_dat , chains=4, cores=4, iter = 2000, refresh = 10)       
stan_dens(m_control, pars = "beta")

rstan::loo(m_control, pars = "log_lik")

library(dplyr)
library(purrr)
data.frame(decay = c(0.5,0.3,0.7), max_lag = 10) %>% mutate(lag_w = map(decay, ~ .x^seq_len(max_lag)/sum(.x^seq_len(max_lag))),
                                                            time = map(decay, ~ seq_len(max_lag))) %>% unnest() %>%
  ggplot(aes(time, lag_w, color= as.factor(decay), group= decay)) + geom_line()



m2 = "
functions {
  // the Hill function
  real Hill(real t, real ec, real slope) {
    return 1 / (1 + (t / ec)^(-slope));
  }
  // the adstock transformation with a vector of weights
  real Adstock(vector t, row_vector weights) {
    return dot_product(t, weights) / sum(weights);
  }
}

data {
  
  int<lower=1> N;
  
  real y[N]; // the vector of sales
  // the maximum duration of lag effect, in weeks
  int<lower=1> max_lag;
  // the number of media channels
  int<lower=1> num_media;
  // matrix of media variables
  matrix[N + max_lag -1, num_media] X_media;
  // vector of media variables’ mean
  real mu_mdip[num_media];
  // the number of other control variables
  int<lower=1> num_ctrl;
  // a matrix of control variables
  matrix[N, num_ctrl] X_ctrl;
}

parameters {
  // residual variance
  real<lower=0> noise_var;
  // the intercept
  real intercept;
  // the coefficients for media variables and base sales
  vector<lower=0>[num_media] beta_media;
  vector[num_ctrl] beta_ctrl;
  
  // the decay  parameter for the adstock transformation of
  // each media
  vector<lower=0,upper=1>[num_media] decay;
}

transformed parameters {
  // the cumulative media effect after adstock
  real cum_effect;
  // matrix of media variables after adstock
  matrix[N, num_media] X_media_adstocked;
  // matrix of all predictors
  //matrix[N, num_media+num_ctrl] X;
  
  // adstock, mean-center, log1p transformation
  row_vector[max_lag] lag_weights;
  for (nn in 1:N) {
    for (media in 1 : num_media) {
      for (lag in 1 : max_lag) {
        lag_weights[max_lag-lag+1] <- pow(decay[media], (lag - 1) ^ 2);
      }
      cum_effect <- Adstock(sub_col(X_media, nn, media, max_lag), lag_weights);
      X_media_adstocked[nn, media] <- cum_effect; //log1p(cum_effect/mu_mdip[media]);
    }
    //X <- append_col(X_media_adstocked, X_ctrl);
  } 
}
model {
  decay ~ beta(3,3);
  intercept ~ normal(0, 5);
  beta_media ~ normal(0, 1);
  beta_ctrl ~ normal(0, 1);
  noise_var ~ inv_gamma(0.05, 0.05 * 0.01);
  
   for (i in 1 : num_ctrl) {
 beta_ctrl[i] ~ normal(0, 1);
   }
  for (i in 1 : num_media) {
 beta_media[i] ~ normal(0, 1);
 }
  
  y ~ normal(intercept + X_ctrl * beta_ctrl + X_media_adstocked * beta_media, sqrt(noise_var));
}
"

max_lag = 10

stan_dat <- list(N = dim(control_variables_centered)[1],
                 num_ctrl = dim(control_variables_centered)[2],
                 X_ctrl = control_variables_centered,
                 y = data$sales,
                 max_lag = max_lag,
                 num_media = dim(mdip)[2],
                 X_media = rbind(as.matrix(mdip), matrix(0, nrow = max_lag-1, ncol= dim(mdip)[2] )),
                 mu_mdip = attributes(scale(mdip))$`scaled:center`)


summary(lm(log(data$sales)~ c(data$mdip_on/10^6)))
lm(log(data$sales)~ c(data$mdip_sms/10^6))
lm((data$sales)~ c(data$mdip_audtr))
lm(data$sales~ (data$mdip_sms))

m_log <- stan( model_code  = m2, data=stan_dat , chains = 4, cores = 4, iter = 1000, refresh = 10)       


stan_trace(m_log, pars = "decay")
stan_trace(m_log, pars = "beta_media") 



summary(m_log, pars = "decay")$summary
pairs(m_log, pars = "decay")
pairs(m_log, pars = "beta_media")
stan_dens(m_log, pars = "decay") 
stan_dens(m_log, pars = "beta_media")

stan_diag(m_control)




m3 = "
functions {
  // the Hill function
  real Hill(real t, real ec, real slope) {
    return 1 / (1 + (t / ec)^(-slope));
  }
  // the adstock transformation with a vector of weights
  real Adstock(vector t, row_vector weights) {
    return dot_product(t, weights) / sum(weights);
  }
}

data {
  
  int<lower=1> N;
  
  real y[N]; // the vector of sales
  // the maximum duration of lag effect, in weeks
  int<lower=1> max_lag;
  // the number of media channels
  int<lower=1> num_media;
  // matrix of media variables
  matrix[N + max_lag -1, num_media] X_media;
  // vector of media variables’ mean
  real mu_mdip[num_media];
  // the number of other control variables
  int<lower=1> num_ctrl;
  // a matrix of control variables
  matrix[N, num_ctrl] X_ctrl;
}

parameters {
  // residual variance
  real<lower=0> noise_var;
  // the intercept
  real intercept;
  // the coefficients for media variables and base sales
  vector<lower=0>[num_media] beta_media;
  vector[num_ctrl] beta_ctrl;
  
  // the decay  parameter for the adstock transformation of
  // each media
  vector<lower=0,upper=1>[num_media] decay;
  
  // hill
  vector<lower=0>[num_media] ec;
  vector<lower=0>[num_media] slope;

}

transformed parameters {
  // the cumulative media effect after adstock
  real cum_effect;
  // matrix of all predictors
  //matrix[N, num_media+num_ctrl] X;
  
  // adstock, mean-center, log1p transformation
  row_vector[max_lag] lag_weights;
  //hill
  matrix[N, num_media] cum_effects_hill;
  real mu[N];
  for (nn in 1:N) {
    for (media in 1 : num_media) {
      for (lag in 1 : max_lag) {
        lag_weights[max_lag-lag+1] <- pow(decay[media], (lag - 1) ^ 2);
      }
      cum_effect <- Adstock(sub_col(X_media, nn, media, max_lag), lag_weights);
      cum_effects_hill[nn, media] <- Hill(cum_effect, ec[media], slope[media]);

    }
    mu[nn] <- intercept + dot_product(cum_effects_hill[nn], beta_media) +
    dot_product(X_ctrl[nn], beta_ctrl);
  } 
}
model {
  decay ~ beta(3,3);
  intercept ~ normal(0, 5);
  beta_media ~ normal(0, 1);
  beta_ctrl ~ normal(0, 1);
  noise_var ~ inv_gamma(0.05, 0.05 * 0.01);
  
   for (i in 1 : num_ctrl) {
 beta_ctrl[i] ~ normal(0, 1);
   }
  for (i in 1 : num_media) {
 beta_media[i] ~ normal(0, 1);
 }
  // hill
  slope ~ gamma(3, 1);
  ec ~ gamma(3, 1);
  y ~ normal(mu, sqrt(noise_var));
}
"


m_hill <- stan( model_code  = m3, data=stan_dat , chains = 4, cores = 4, iter = 1000, refresh = 10)       


stan_dens(m_hill, pars = "decay") 
stan_dens(m_hill, pars = "beta_media")

stan_dens(m_hill, pars = "slope") 
stan_dens(m_hill, pars = "ec")















# Define the Hill function
hill_function <- function(x, ec, slope) {
  return( 1 / (1 + (x / ec)^(-slope)))
}

# Define a sequence of ligand concentrations
x_values <- seq(0, 10, length.out = 100)

# Define different Hill coefficients and dissociation constants
ec <- c(1, 5)
slope <- c(0.1,0.5, 1, 2, 5)

# Create a data frame to store the results
results <- data.frame()

# Calculate the Hill function for each combination of parameters
for (n in ec) {
  for (Kd in slope) {
    y_values <- hill_function(x_values, n, Kd)
    temp_data <- data.frame(x = x_values, y = y_values, n = n, Kd = Kd)
    results <- rbind(results, temp_data)
  }
}

# Plot the results using ggplot2
ggplot(results, aes(x = x, y = y, color = factor(n), linetype = factor(Kd))) +
  geom_line() +
  labs(title = "Hill Function Realizations",
       x = "Ligand Concentration",
       y = "Fraction Bound",
       color = "Hill Coefficient (n)",
       linetype = "Dissociation Constant (Kd)") +
  theme_minimal()

