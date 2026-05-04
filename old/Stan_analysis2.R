
# Market Mix Modeling with Stan
# This script demonstrates how to fit a Bayesian Media Mix Model (MMM) using RStan.
# The model estimates the contribution of different media channels to sales, incorporating carryover and shape effects.

# Load necessary libraries
library(rstan)
library(ggplot2)
library(tidyr)
library(dplyr)
library(cowplot)
library(lubridate)
library(purrr)
library(ggExtra)
library(viridis) 
library(bayesplot)



# Load data from the specified CSV file
data <- read.csv("mmm.csv")

# Extract month and year from the date column
data$week_dt2 <- as.Date(data$week_dt)

data$month <- month(data$week_dt2)
data$year <- year(data$week_dt2)

# Calculate weeks past from the earliest date
data$weeks_past <- as.numeric(difftime( data$week_dt2,min(data$week_dt2), units = "weeks"))

# Select control variables and center them
control_variables <- data %>% select(weeks_past, year, month)
control_variables_centered <- scale(control_variables, center = TRUE, scale = FALSE)

# Stan model code
m_hill = "
functions {
  // Hill function for diminishing returns
  real Hill(real t, real ec, real slope) {
    return 1 / (1 + (t / ec)^(-slope));
  }
  // Adstock transformation for carryover effects
  real Adstock(vector t, row_vector weights) {
    return dot_product(t, weights) / sum(weights);
  }
}

data {
  int<lower=1> N; // Number of observations
  real y[N]; // Sales vector
  int<lower=1> max_lag; // Maximum lag duration
  int<lower=1> num_media; // Number of media channels
  matrix[N + max_lag -1, num_media] X_media; // Media variables matrix
  int<lower=1> num_ctrl; // Number of control variables
  matrix[N, num_ctrl] X_ctrl; // Control variables matrix
}

parameters {
  real<lower=0> sigma; // Residual variance
  real intercept; // Intercept
  vector[num_media] beta_media; // Coefficients for media variables
  vector[num_ctrl] beta_ctrl; // Coefficients for control variables
  vector<lower=0,upper=1>[num_media] decay; // Decay parameter for adstock
  vector<lower=0>[num_media] ec; // Hill function parameter
  vector<lower=0>[num_media] slope; // Hill function slope
}

transformed parameters {
  real cum_effect; // Cumulative media effect
  row_vector[max_lag] lag_weights; // Lag weights
  matrix[N, num_media] cum_effects_hill; // Cumulative effects after Hill transformation
  real mu[N]; // Predicted sales

  for (nn in 1:N) {
    for (media in 1:num_media) {
      for (lag in 1:max_lag) {
        lag_weights[lag] <- pow(decay[media], (lag) ^ 2);
      }
      cum_effect <- Adstock(sub_col(X_media, nn, media, max_lag), lag_weights);
      cum_effects_hill[nn, media] <- Hill(cum_effect, ec[media], slope[media]);
    }
    mu[nn] <- intercept + dot_product(cum_effects_hill[nn], beta_media) + dot_product(X_ctrl[nn], beta_ctrl);
  }
}

model {
  decay ~ beta(3,10);
  intercept ~ normal(0, 5);
  beta_media ~ normal(0, 5);
  beta_ctrl ~ normal(0, 5);
  sigma ~ normal(0, 5);
  slope ~ normal(1,5);
  ec ~ gamma(4, 0.05);
  y ~ normal(mu, sigma);
}

generated quantities {
  array[N] real y_rep;

  // Log-likelihood for LOO-CV model comparison
  vector[N] log_lik;
  
  vector[N] tot;        
  vector[N] base_contr; 
  matrix[N, num_media] media_contr; 
  for (nn in 1:N) {

     tot[nn] = intercept 
              + dot_product(cum_effects_hill[nn], beta_media) 
              + dot_product(X_ctrl[nn], beta_ctrl);

      base_contr[nn] = intercept + dot_product(X_ctrl[nn], beta_ctrl);

    for (media in 1:num_media) {
      vector[num_media] effects_without;
      effects_without = cum_effects_hill[nn]';  
      effects_without[media] = 0;            
      
      media_contr[nn, media] = intercept 
                               + dot_product(effects_without, beta_media) 
                               + dot_product(X_ctrl[nn], beta_ctrl);
    }

    // Posterior predictives
    y_rep[nn] = normal_rng(tot[nn], sigma);

    // likelihood
    log_lik[nn] = normal_lpdf(y[nn] | tot[nn], sigma);
  }
}
"

# Data for Stan
max_lag = 10
mdip = data %>% select(TV, radio, newspaper)

stan_dat <- list(N = dim(control_variables_centered)[1],
                 num_ctrl = dim(control_variables_centered)[2],
                 X_ctrl = control_variables_centered,
                 y = data$sales,
                 max_lag = max_lag,
                 num_media = dim(mdip)[2],
                 X_media = rbind(as.matrix(mdip), matrix(0, nrow = max_lag-1, ncol= dim(mdip)[2] )),
                 mu_mdip = attributes(scale(mdip))$`scaled:center`)



# Fit the Stan model
m_hill <- stan(model_code = m_hill, data = stan_dat, chains = 4, cores = 4, iter = 2000, refresh = 200)

# Plot results
stan_dens(m_hill, pars = "decay")
stan_dens(m_hill, pars = "beta_media")
stan_dens(m_hill, pars = "slope")
stan_dens(m_hill, pars = "ec")
stan_dens(m_hill, pars = "sigma")
stan_trace(m_hill, pars = "slope")
stan_trace(m_hill, pars = "ec")

samples <- rstan::extract(m_hill)




# Posterior predictive checks

y_rep <-  samples$y_rep

# Density 
ppc_dens_overlay(stan_dat$y, y_rep[1:50, ])

# time series with intervals
 posterior_preds <- ppc_intervals(stan_dat$y, y_rep, x = as.numeric(data$week_dt2)) +
  scale_x_continuous(
    breaks = as.numeric(pretty(data$week_dt2)),
    labels = format(pretty(data$week_dt2), "%b %Y")
  ) +
  theme_minimal() +
  xlab("Time") + ylab("Sales")




# Summary statistics
 posterior_preds_full <- plot_grid(posterior_preds,plot_grid(
  ppc_stat(stan_dat$y, y_rep, stat = "mean"),
  ppc_stat(stan_dat$y, y_rep, stat = "sd"),
  ppc_stat(stan_dat$y, y_rep, stat = "max"),
  ppc_stat(stan_dat$y, y_rep, stat = "min")
 ),nrow=2)

ggsave("plots/post_pred.png", posterior_preds_full, width = 8, height = 8)

# Plot lag weights
plag <- data.frame(decay = apply(samples$decay, 2, mean), max_lag = 10, group = c("TV", "Radio", "News Paper")) %>%
  mutate(lag_w = map(decay, ~ .x^seq(0, 10, by = 0.2) / sum(.x^seq(0, 10, by = 0.2))),
         time = map(decay, ~ seq(0, 10, by = 0.2))) %>%
  unnest() %>%
  ggplot(aes(time, lag_w, color = group, group = group)) + geom_line(size = 1.2) + theme_minimal() + ylab("Weight") + xlab("Lag in Weeks") +
  theme(legend.position = c(0.8, 0.8), legend.title = element_blank())
ggsave("plots/lag_mmm_data.png", plag, width = 3, height = 3)

# Plot sales contribution
p_prop <- data.frame(time = data$week_dt2,
                     prop_tv = apply(1 - (samples$media_contr[,,1] / samples$tot), 2, mean),
                     prop_radio = apply(1 - (samples$media_contr[,,2] / samples$tot), 2, mean),
                     prop_np = apply(1 - (samples$media_contr[,,3] / samples$tot), 2, mean),
                     prop_rest = apply((samples$contr / samples$tot), 2, mean)) %>%
  pivot_longer(cols = starts_with("prop"), names_to = "group", values_to = "prop") %>%
  mutate(group = factor(group, levels = c('prop_rest', 'prop_tv', 'prop_radio', 'prop_np'))) %>%
  ggplot(aes(x = time, y = prop, fill = group)) +
  geom_area(alpha = 0.6, size = .5, colour = "white") +
  viridis::scale_fill_viridis(discrete = T, name = "", labels = c("Other", "TV", "Radio", "News Paper")) +
  ggtitle("Sales contribution") + theme_minimal() + geom_smooth(aes(color = group), se = F) +
  viridis::scale_color_viridis(discrete = T, name = "", labels = c("Other", "TV", "Radio", "News Paper")) +
  ylab("Relative contribution to sales")
ggsave("plots/contribution_mmm_data.png", p_prop, width = 6, height = 3)













# Define the Hill function
hill_function <- function(x, ec, slope) {
  return( 1 / (1 + (x / ec)^(-slope)))
}

hill_data = data %>% mutate(TV_func =  mean(samples$beta_media[,1]) * hill_function(TV,mean(samples$ec[,1]),mean(samples$slope[,1])),
                            Radio_func =   mean(samples$beta_media[,2]) * hill_function(radio,mean(samples$ec[,2]),mean(samples$slope[,2])),
                            NP_func =  mean(samples$beta_media[,3]) * hill_function(newspaper,mean(samples$ec[,3]),mean(samples$slope[,3])),
                            hill_TV = hill_function(TV,mean(samples$ec[,1]),mean(samples$slope[,1])),
                            hill_radio = hill_function(radio,mean(samples$ec[,2]),mean(samples$slope[,2])),
                            hill_newspaper =  hill_function(newspaper,mean(samples$ec[,3]),mean(samples$slope[,3]))) 



df1 = hill_data %>%
  pivot_longer(
    cols = c(TV, radio, newspaper),
    names_to = "grouping_variable",
    values_to = "values1"
  )


df3 <- hill_data %>%
  pivot_longer(
    cols = ends_with("_func"),
    names_to = "group",
    values_to = "values2"
  ) 

# Plot Hill function
p_hill <- df1 %>% select(values1, grouping_variable) %>%
  cbind(df3 ) %>% ggplot(aes(x = values1, y=values2, color=grouping_variable)) + geom_line(size=1.2) + geom_point(alpha=0) +
  theme_minimal() + theme(legend.position = c(0.8, 0.2), legend.title = element_blank()) + ylab("Return in sales") + xlab("Investment")
#facet_wrap(vars(grouping_variable),scales="free_x") 
p_hill <- ggMarginal(p_hill, type = "density", groupFill = TRUE, margins = "x")
ggsave("plots/p_hill_mmm_data.png",p_hill, width = 3, height = 3)


# Plot ROI
p2 <- df1 %>% select(values1, grouping_variable) %>%
  cbind(df3 ) %>% ggplot(aes(x = values1, y=values2/values1, color=grouping_variable)) + geom_line(size=1.2) + 
  theme_minimal() + theme(legend.position = c(0.8, 0.8), legend.title = element_blank()) + 
  ylab("Return in Investment") +
  xlab("Investment in chanel") +
  facet_wrap(vars(grouping_variable),scales="free_x") 
p2
ggsave("plots/roi_mmm_data.png",p2, width = 5, height = 3)







# Plot sales
data$week_dt2 <- as.Date(data$week_dt)
p1 <- data %>% ggplot(aes(week_dt2, y= sales)) + 
  geom_point() + 
  geom_smooth(color = "black") + 
  xlab("Time") + 
  ylab("Sales") + 
  theme_minimal() + 
  ylim(0, 30) +
  geom_hline(yintercept = 0, linetype="dashed")


plot_grid(p1)
ggsave("plots/sales_mmm_data.png",p1, width = 5, height = 3)


# Plot investments
p2 <- data.frame(time = data$week_dt2, TV = data$TV, radio  = data$radio, newspaper = data$newspaper) %>%
  pivot_longer(!time, names_to = "Chanel", values_to = "spending") %>%
  ggplot(aes(x = time, y = spending)) + geom_point() + geom_smooth(span = 0.2) + facet_grid(Chanel ~ ., scales = "free") + 
  ylab("Spending") + theme_minimal()
ggsave("plots/investment_mmm_data.png",p2, width = 5, height = 7)

      




###
#
# Optimization
#
###

n_sim = 1000 # number of posterior samples
posterior_samples <- rstan::extract(m_hill)

# Get relevant posterior parameters
beta_media_posterior  <- posterior_samples$beta_media[1:n_sim, ]
ec_posterior          <- posterior_samples$ec[1:n_sim, ]
slope_posterior       <- posterior_samples$slope[1:n_sim, ]
intercept_posterior   <- posterior_samples$intercept[1:n_sim]
decay_posterior       <- posterior_samples$decay[1:n_sim, ]
beta_ctrl_posterior   <- posterior_samples$beta_ctrl[1:n_sim, ]

# Total budget constraint: I'll keep the average
total_budget <- mean(data$TV) + mean(data$newspaper) + mean(data$radio)

# Mean control variables for simulation ( I use  mean for simplicity)
mean_ctrl <- colMeans(control_variables_centered)

# Softmax to ensure allocations sum to 1 and stay positive
softmax <- function(x) exp(x) / sum(exp(x))

# Adstock function 
adstock_function <- function(t, decay, max_lag) {
  weights <- decay^((1:max_lag)^2)
  if (length(t) < max_lag) {
    t <- c(rep(0, max_lag - length(t)), t)
  } else if (length(t) > max_lag) {
    t <- tail(t, max_lag)
  }
  return(sum(t * weights) / sum(weights))
}

# Expected incremental sales for a given budget allocation
# props: unconstrained 3-vector (will be softmax-transformed)
expected_incremental_sales <- function(props) {
  alloc <- softmax(props) * total_budget
  tv_spend        <- alloc[1]
  newspaper_spend <- alloc[2]
  radio_spend     <- alloc[3]
  
  incremental <- numeric(n_sim)
  
  for (s in 1:n_sim) {
    # Adstock + Hill transformation per channel
    tv_effect <- hill_function(
      adstock_function(rep(tv_spend, max_lag), decay_posterior[s, 1], max_lag),
      ec_posterior[s, 1], slope_posterior[s, 1]
    )
    newspaper_effect <- hill_function(
      adstock_function(rep(newspaper_spend, max_lag), decay_posterior[s, 2], max_lag),
      ec_posterior[s, 2], slope_posterior[s, 2]
    )
    radio_effect <- hill_function(
      adstock_function(rep(radio_spend, max_lag), decay_posterior[s, 3], max_lag),
      ec_posterior[s, 3], slope_posterior[s, 3]
    )
    
    # Total predicted sales
    predicted_sales <- intercept_posterior[s] +
      beta_media_posterior[s, 1] * tv_effect +
      beta_media_posterior[s, 2] * radio_effect +
      beta_media_posterior[s, 3] * newspaper_effect +
      sum(beta_ctrl_posterior[s, ] * mean_ctrl)
    
    # Baseline: predicted sales with zero media spend
    baseline_sales <- intercept_posterior[s] +
      sum(beta_ctrl_posterior[s, ] * mean_ctrl)
    
    incremental[s] <- predicted_sales - baseline_sales
  }
  
  return(mean(incremental))
}

# Objective: negative incremental sales (optim minimizes)
objective <- function(props) -expected_incremental_sales(props)

# Run optimization from multiple starting points to avoid local optima
set.seed(42)
n_starts <- 10
results <- vector("list", n_starts)

for (i in 1:n_starts) {
  init <- rnorm(3) # random unconstrained starting point
  results[[i]] <- optim(
    par    = init,
    fn     = objective,
    method = "BFGS",
    control = list(maxit = 500)
  )
}

# Pick best result across starting points
best <- results[[which.min(sapply(results, function(r) r$value))]]
optimal_alloc <- softmax(best$par) * total_budget

cat("Optimal budget allocation:\n")
cat(sprintf("  TV:        %.2f (%.1f%%)\n", optimal_alloc[1], 100 * optimal_alloc[1] / total_budget))
cat(sprintf("  Newspaper: %.2f (%.1f%%)\n", optimal_alloc[2], 100 * optimal_alloc[2] / total_budget))
cat(sprintf("  Radio:     %.2f (%.1f%%)\n", optimal_alloc[3], 100 * optimal_alloc[3] / total_budget))
cat(sprintf("Expected incremental sales: %.2f\n", -best$value))

# --- Uncertainty around optimal allocation ---
# Re-run expected incremental sales per posterior sample at optimal allocation
optimal_props <- best$par
alloc <- softmax(optimal_props) * total_budget

incremental_posterior <- numeric(n_sim)

for (s in 1:n_sim) {
  tv_effect <- hill_function(
    adstock_function(rep(alloc[1], max_lag), decay_posterior[s, 1], max_lag),
    ec_posterior[s, 1], slope_posterior[s, 1]
  )
  newspaper_effect <- hill_function(
    adstock_function(rep(alloc[2], max_lag), decay_posterior[s, 2], max_lag),
    ec_posterior[s, 2], slope_posterior[s, 2]
  )
  radio_effect <- hill_function(
    adstock_function(rep(alloc[3], max_lag), decay_posterior[s, 3], max_lag),
    ec_posterior[s, 3], slope_posterior[s, 3]
  )
  
  predicted_sales <- intercept_posterior[s] +
    beta_media_posterior[s, 1] * tv_effect +
    beta_media_posterior[s, 2] * radio_effect +
    beta_media_posterior[s, 3] * newspaper_effect +
    sum(beta_ctrl_posterior[s, ] * mean_ctrl)
  
  baseline_sales <- intercept_posterior[s] +
    sum(beta_ctrl_posterior[s, ] * mean_ctrl)
  
  incremental_posterior[s] <- predicted_sales - baseline_sales
}

cat(sprintf("\n90%% credible interval for incremental sales at optimal allocation:\n"))
cat(sprintf("  [%.2f, %.2f]\n", quantile(incremental_posterior, 0.05), quantile(incremental_posterior, 0.95)))
cat(sprintf("Interquartile range: [%.2f, %.2f]\n", quantile(incremental_posterior, 0.25), quantile(incremental_posterior, 0.75)))

# --- Plot: posterior distribution of incremental sales at optimum ---
iq_low  <- quantile(incremental_posterior, 0.25)
iq_high <- quantile(incremental_posterior, 0.75)

print(mean(incremental_posterior>10))

p_opt <- ggplot(data.frame(incremental = incremental_posterior), aes(x = incremental)) +
  geom_density(fill = "steelblue", alpha = 0.5) +
  geom_vline(xintercept = mean(incremental_posterior), linetype = "dashed") +
  annotate("rect", xmin = iq_low, xmax = iq_high, ymin = -Inf, ymax = Inf,
           alpha = 0.15, fill = "orange") +
  annotate("text", x = mean(c(iq_low, iq_high)), y = Inf,
           label = "IQR", vjust = 1.5, size = 3.5, color = "darkorange") +
  theme_minimal() + xlim(0,35) +
  xlab("Incremental Sales") +
  ylab("Density") +
  ggtitle("Posterior distribution of sales improvement at optimal budget allocation")

ggsave("plots/optimal_incremental_sales.png", p_opt, width = 5, height = 3)


p_alloc <- ggplot(alloc_comparison, aes(x = Channel, y = Allocation, fill = Scenario)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
  theme_minimal() +
  ylab("Budget") +
  ggtitle("Current vs. optimal budget allocation") +
  scale_fill_manual(values = c("Current" = "grey60", "Optimal" = "steelblue"))

ggsave("plots/allocation_comparison.png", p_alloc, width = 5, height = 3)
