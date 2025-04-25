
# Market Mix Modeling with Stan
# This script demonstrates how to fit a Bayesian Media Mix Model (MMM) using RStan.
# The model estimates the contribution of different media channels to sales, incorporating carryover and shape effects.

# Load necessary libraries
library(rstan)
library(ggplot2)
library(tidyr)
library(cowplot)
library(lubridate)
library(purrr)
library(ggExtra)
library(viridis) 



# Load data from the specified CSV file
data <- read.csv("mmm.csv")

# Extract month and year from the date column
data$week_dt2 <- as.Date(data$week_dt)

data$month <- month(data$week_dt2)
data$year <- year(data$week_dt2)

# Calculate weeks past from the maximum date
data$weeks_past <- as.numeric(difftime(max(data$week_dt2), data$week_dt2, units = "weeks"))

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
  real<lower=0> noise_var; // Residual variance
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
  beta_media ~ normal(0, 1);
  beta_ctrl ~ normal(0, 1);
  noise_var ~ inv_gamma(0.05, 0.05 * 0.01);
  slope ~ normal(1,0.3);
  ec ~ gamma(4, 0.1);
  y ~ normal(mu, sqrt(noise_var));
}

generated quantities {
  real cum_effect2;
  row_vector[max_lag] lag_weights2;
  matrix[N, num_media] cum_effects_hill2;
  matrix[N, num_media] media_contr;
  real tot[N];
  real contr[N];

  for (nn in 1:N) {
    for (media in 1:num_media) {
      for (lag in 1:max_lag) {
        lag_weights2[lag] <- pow(decay[media], (lag) ^ 2);
      }
      cum_effect2 <- Adstock(sub_col(X_media, nn, media, max_lag), lag_weights2);
      cum_effects_hill2[nn, media] <- Hill(cum_effect2, ec[media], slope[media]);
    }
    tot[nn] <- intercept + dot_product(cum_effects_hill2[nn], beta_media) + dot_product(X_ctrl[nn], beta_ctrl);
    contr[nn] <- intercept + dot_product(X_ctrl[nn], beta_ctrl);
    media_contr[nn, 1] <- intercept + dot_product([0, cum_effects_hill2[nn,2],cum_effects_hill2[nn,3]], beta_media) + dot_product(X_ctrl[nn], beta_ctrl);
    media_contr[nn, 2] <- intercept + dot_product([cum_effects_hill2[nn,1],0,cum_effects_hill2[nn,3]], beta_media) + dot_product(X_ctrl[nn], beta_ctrl);
    media_contr[nn, 3] <- intercept + dot_product([cum_effects_hill2[nn,1],cum_effects_hill2[nn,2],0], beta_media) + dot_product(X_ctrl[nn], beta_ctrl);
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
m_hill <- stan(model_code = m_hill, data = stan_dat, chains = 4, cores = 4, iter = 4000, refresh = 10)

# Plot results
stan_dens(m_hill, pars = "decay")
stan_dens(m_hill, pars = "beta_media")
stan_dens(m_hill, pars = "slope")
stan_dens(m_hill, pars = "ec")
stan_trace(m_hill, pars = "slope")
stan_trace(m_hill, pars = "ec")

samples <- rstan::extract(m_hill)

# Plot lag weights
plag <- data.frame(decay = apply(samples$decay, 2, mean), max_lag = 10, group = c("TV", "Radio", "News Paper")) %>%
  mutate(lag_w = map(decay, ~ .x^seq(0, 10, by = 0.2) / sum(.x^seq(0, 10, by = 0.2))),
         time = map(decay, ~ seq(0, 10, by = 0.2))) %>%
  unnest() %>%
  ggplot(aes(time, lag_w, color = group, group = group)) + geom_line(size = 1.2) + theme_minimal() + ylab("Weight") + xlab("Lag") +
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
n_sim =250 # number of posterior samples
posterior_samples <- rstan::extract(m_hill)


# Get relevant posterior parameters
beta_media_posterior <- posterior_samples$beta_media[1:n_sim, ]
ec_posterior <- posterior_samples$ec[1:n_sim, ]
slope_posterior <- posterior_samples$slope[1:n_sim, ]
intercept_posterior <- posterior_samples$intercept[1:n_sim]
beta_ctrl_posterior <- posterior_samples$beta_ctrl[1:n_sim, ] # If you had control variables
decay_posterior <- posterior_samples$decay




adstock_function <- function(t, decay, max_lag) {
  weights <- decay^((1:max_lag) ^ 2)
  if (length(t) < max_lag) {
    t <- c(rep(0, max_lag - length(t)), t) # Pad with zeros for shorter histories
  } else if (length(t) > max_lag) {
    t <- tail(t, max_lag) # Take the last max_lag periods
  }
  return(sum(t * weights) / sum(weights))
}


# Define budget
# Lets say I don't want to change my average budget
total_budget <- mean(data$TV) + mean(data$newspaper) + mean(data$radio)


# Generates a grid of points uniformly distributed on a 3-dimensional simplex
generate_simplex_grid <- function(n_grid = 7) {
 
  u_vals <- seq(0.1, 1, length.out = n_grid)
  v_vals <- seq(0, 1, length.out = n_grid)
  
  simplex_grid <- expand.grid(u = u_vals, v = v_vals) %>%
    mutate(
      p1 = 1 - u,
      p2 = u * (1 - v),
      p3 = u * v
    ) %>%
    select(p1, p2, p3)
  
  return(simplex_grid)
}

simplex_grid = generate_simplex_grid(20)

# Create different budget allocation scenarios (proportions)
budget_scenarios <- data.frame(
  tv_prop = simplex_grid$p1,
  newspaper_prop = simplex_grid$p2,
  radio_prop = simplex_grid$p3
)


# Ensure proportions sum to 1 and use reasonable proportions
budget_scenarios <- budget_scenarios %>%
  mutate(total_prop = tv_prop + newspaper_prop + radio_prop) %>%
  filter(abs(total_prop - 1) < 1e-6) %>%
  filter( newspaper_prop <= 0.5) %>% # more is unrealistic
  filter( radio_prop <= 0.5) %>%
  select(-total_prop)

# Simulate sales and ROI for each scenario
simulation_results <- list()

for (i in 1:nrow(budget_scenarios)) {
  scenario <- budget_scenarios[i, ]
  scenario_sales <- numeric(n_sim)
  scenario_investment <- total_budget
  
  for (s in 1:n_sim) {
    # Apply the budget allocation to get channel-specific spending
    tv_spend <- scenario$tv_prop * scenario_investment
    newspaper_spend <- scenario$newspaper_prop * scenario_investment
    radio_spend <- scenario$radio_prop * scenario_investment
    
    # Assume a constant recent spending for simplicity in this illustration
    # In a real application, you'd simulate future spending patterns
    recent_tv <- rep(tv_spend, length(data$week_dt))
    recent_newspaper <- rep(newspaper_spend, length(data$week_dt))
    recent_radio <- rep(radio_spend, length(data$week_dt))
    
    # Calculate the cumulative effects for this simulation
    tv_adstocked <- hill_function(adstock_function(recent_tv, decay_posterior[s, 1], max_lag), ec_posterior[s, 1], slope_posterior[s, 1])
    newspaper_adstocked <- hill_function(adstock_function(recent_newspaper, decay_posterior[s, 2], max_lag), ec_posterior[s, 2], slope_posterior[s, 2])
    radio_adstocked <- hill_function(adstock_function(recent_radio, decay_posterior[s, 3], max_lag), ec_posterior[s, 3], slope_posterior[s, 3])
    
    # Simulate sales
    # Assuming no change in control variables for simplicity 
      simulated_sales <- intercept_posterior[s] +
      beta_media_posterior[s, 1] * tv_adstocked +
      beta_media_posterior[s, 3] * newspaper_adstocked +
      beta_media_posterior[s, 2] * radio_adstocked 
    
    scenario_sales[s] <- simulated_sales
  }
  
  # Calculate ROI (Incremental Sales / Investment)
  baseline_sales_posterior <- intercept_posterior 
  incremental_sales_posterior <- scenario_sales - mean(baseline_sales_posterior) # Using mean for simplicity
  
  scenario_roi_posterior <- incremental_sales_posterior / scenario_investment
  
  simulation_results[[i]] <- data.frame(
    tv_prop = scenario$tv_prop,
    newspaper_prop = scenario$newspaper_prop,
    radio_prop = scenario$radio_prop,
    roi = scenario_roi_posterior
  )
}

# Combine results
roi_comparison <- bind_rows(simulation_results)





roi_opt <- ggplot(roi_comparison, aes(x = (tv_prop), y = roi, color = ((radio_prop)))) +
  geom_point(alpha = 0.04) +
  geom_smooth(color = "red") +
  facet_grid(. ~ round(newspaper_prop, 1)) + # Create facet grid based on rounded newspaper_prop
  scale_color_viridis(option = "D") + # Use the "D" option from viridis
  labs(
    title = "Posterior Distribution of ROI \nNewspaper Proportion",
    x = "Proportion of Budget allocated to TV",
    y = "Return on Investment",
    color = "Radio Proportion"
  ) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) # Adjust x-axis labels if needed

ggsave("plots/ROI_optimum.png",roi_opt, width = 8, height = 5)


average_roi_by_allocation <- roi_comparison %>%
  group_by(tv_prop, newspaper_prop, radio_prop) %>%
  summarise(average_roi = mean(roi), low = quantile(roi, prob = 0.025), high = quantile(roi, prob = 0.975), .groups = 'drop')

# Find the maximum average ROI
max_average_roi <- average_roi_by_allocation %>%
  slice_max(average_roi, n = 1)

print("Maximum Average ROI:")
print(max_average_roi)
