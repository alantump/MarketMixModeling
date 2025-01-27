###
#
# Market Mix Modelling with Stan
#
###
library(rstan)
library(ggplot2)
library(tidyr)
library(cowplot)
#from https://github.com/leopoldavezac/BayesianMMM/tree/main/data
data <-  read.csv("mmm.csv")

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
###
#simple check
summary(lm(sales ~ TV + radio + newspaper, data = data))




p2 <- data.frame(time = data$week_dt2, TV = data$TV, radio  = data$radio, newspaper = data$newspaper) %>%
  pivot_longer(!time, names_to = "Chanel", values_to = "spending") %>%
  ggplot(aes(x = time, y = spending)) + geom_point() + geom_smooth(span = 0.2) + facet_grid(Chanel ~ ., scales = "free") + 
  ylab("Spending") + theme_minimal()
ggsave("plots/investment_mmm_data.png",p2, width = 5, height = 7)


porportions = data.frame(time = data$week_dt2, TV = data$TV, radio  = data$radio, newspaper = data$newspaper) %>%
  pivot_longer(!time, names_to = "Chanel", values_to = "spending") %>% mutate(total_spending = sum(spending)) %>% 
  group_by(Chanel) %>% summarise(prop = sum(spending)/total_spending[1], spend =  round(sum(spending)))


porportions %>% ggplot(aes(x=reorder(Chanel, prop), y = prop)) + geom_bar(stat="identity") + coord_flip() + theme_minimal() + 
  ylab("Proportion spend on chanel \n and total amount spend") + xlab("Chanel") + geom_text(aes(label = spend), hjust=-0.1) + ylim(0,1)

data.frame(sales = data$sales, TV = data$TV, radio  = data$radio, newspaper = data$newspaper) %>%
  pivot_longer(!sales, names_to = "Chanel", values_to = "spending") %>% ggplot(aes(x=spending, y=sales)) + geom_point() +
  facet_wrap(. ~ Chanel, scales = "free") + ylim(0,30) + geom_smooth(method= "lm") + xlab("Spending in Million") + xlab("Sales in Million")

library(lubridate)
data$month <- month(data$week_dt2)
data$year <- year(data$week_dt2)
data$weeks_past <- as.numeric(difftime(max(data$week_dt2), data$week_dt2, units = "weeks"))

control_variables <-  data %>% select(weeks_past, year, month)

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
data.frame(decay = c(0.5,0.001,0.7), max_lag = 10) %>% mutate(lag_w = map(decay, ~ .x^seq_len(max_lag)/sum(.x^seq_len(max_lag))),
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
        //lag_weights[max_lag-lag+1] <- pow(decay[media], (lag - 1) ^ 2);
        lag_weights[lag] <- pow(decay[media], (lag) ^ 2);
      }
      cum_effect <- Adstock(sub_col(X_media, nn, media, max_lag), lag_weights);
      X_media_adstocked[nn, media] <- cum_effect; //log1p(cum_effect/mu_mdip[media]);


    }
  } 
}
model {
  decay ~ beta(3,10);
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
Adstock <- function(t, lag_weights){
  return (sum(t * lag_weights)/sum(lag_weights))
  }

lag_weights = rep(0,max_lag)
N = dim(control_variables_centered)[1]
non_add <- c(data$TV,rep(0,max_lag-1))
with_add =NULL
for (nn in 1:N) {
  for (lag in 1 : max_lag) {
    lag_weights[lag] <- 0.2^((lag) ^ 2)
  }
    with_add[nn] <- Adstock((non_add[nn:(nn+max_lag-1)]), lag_weights);
}
plot(with_add[1:N],non_add[1:N])

mdip = data %>% select(TV, radio, newspaper)

max_lag = 10

stan_dat <- list(N = dim(control_variables_centered)[1],
                 num_ctrl = dim(control_variables_centered)[2],
                 X_ctrl = control_variables_centered,
                 y = data$sales,
                 max_lag = max_lag,
                 num_media = dim(mdip)[2],
                 X_media = rbind(as.matrix(mdip), matrix(0, nrow = max_lag-1, ncol= dim(mdip)[2] )),
                 mu_mdip = attributes(scale(mdip))$`scaled:center`)



m_decay <- stan( model_code  = m2, data=stan_dat , chains = 4, cores = 4, iter = 1000, refresh = 10)       


stan_trace(m_decay, pars = "decay")
stan_trace(m_decay, pars = "beta_media") 



summary(m_decay, pars = "decay")$summary
pairs(m_decay, pars = "decay")
pairs(m_decay, pars = "beta_media")
stan_dens(m_decay, pars = "decay") 
stan_dens(m_decay, pars = "beta_media")

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
  vector[num_media] beta_media;
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
  

  // adstock, mean-center, log1p transformation
  row_vector[max_lag] lag_weights;
  
  //hill
  matrix[N, num_media] cum_effects_hill;
  
  real mu[N];
  for (nn in 1:N) {
    for (media in 1 : num_media) {
      for (lag in 1 : max_lag) {
        lag_weights[lag] <- pow(decay[media], (lag) ^ 2);
      }
      cum_effect <- Adstock(sub_col(X_media, nn, media, max_lag), lag_weights);
      cum_effects_hill[nn, media] <- Hill(cum_effect, ec[media], slope[media]);

    }
    mu[nn] <- intercept + dot_product(cum_effects_hill[nn], beta_media) +
    dot_product(X_ctrl[nn], beta_ctrl);
  } 
}
model {
  decay ~ beta(3,10);
  intercept ~ normal(0, 5);
  beta_media ~ normal(0, 1);
  beta_ctrl ~ normal(0, 1);
  noise_var ~ inv_gamma(0.05, 0.05 * 0.01);
  

  // hill
  slope ~ normal(1,0.3);
  ec ~ gamma(4, 0.1);
  y ~ normal(mu, sqrt(noise_var));
}

generated quantities {
  // for calculating the marginal effects
  real cum_effect2;
  row_vector[max_lag] lag_weights2;
  matrix[N, num_media] cum_effects_hill2;
  matrix[N, num_media] media_contr;
  real tot[N];
  real contr[N];
  
  for (nn in 1:N) {
    for (media in 1 : num_media) {
      for (lag in 1 : max_lag) {
        lag_weights2[lag] <- pow(decay[media], (lag) ^ 2);
      }
      cum_effect2 <- Adstock(sub_col(X_media, nn, media, max_lag), lag_weights2);
      cum_effects_hill2[nn, media] <- Hill(cum_effect2, ec[media], slope[media]);

    }
    // predicted sales
    tot[nn] <- intercept + dot_product(cum_effects_hill2[nn], beta_media) +
    dot_product(X_ctrl[nn], beta_ctrl);
    
    // predicted sales without media
    contr[nn] <- intercept +
    dot_product(X_ctrl[nn], beta_ctrl);
    
    // predicted sales without first media
    media_contr[nn, 1] <- intercept + dot_product([0, cum_effects_hill2[nn,2],cum_effects_hill2[nn,3]], beta_media) +
    dot_product(X_ctrl[nn], beta_ctrl);
    
    // predicted sales without second media
    media_contr[nn, 2] <- intercept + dot_product([cum_effects_hill2[nn,1],0,cum_effects_hill2[nn,3]], beta_media) +
    dot_product(X_ctrl[nn], beta_ctrl);
    
    // predicted sales without third media
    media_contr[nn, 3] <- intercept + dot_product([ cum_effects_hill2[nn,1],cum_effects_hill2[nn,2],0], beta_media) +
    dot_product(X_ctrl[nn], beta_ctrl);
    

  } 

}
"


m_hill <- stan( model_code  = m3, data=stan_dat , chains = 4, cores = 4, iter = 2000, refresh = 10)       


stan_dens(m_hill, pars = "decay") 
stan_dens(m_hill, pars = "beta_media")

stan_dens(m_hill, pars = "slope") 
stan_dens(m_hill, pars = "ec")


stan_trace(m_hill, pars = "slope")
stan_trace(m_hill, pars = "ec")
pairs(m_hill, pars = c("ec","beta_media"))


stan_dens(m_hill, pars = "mu")




samples <- rstan::extract(m_hill)

hist(apply(samples$contr[1]/samples$tot,2, mean))
hist(apply(1-(samples$media_contr[,,3]/samples$tot),2, mean))
hist(apply(1-(samples$media_contr[,,2]/samples$tot),2, mean))
hist(apply(1-(samples$media_contr[,,1]/samples$tot),2, mean))


data.frame(time = data$week_dt2,
           prop_tv = apply(1-(samples$media_contr[,,1]/samples$tot),2, mean),
           prop_radio = apply(1-(samples$media_contr[,,2]/samples$tot),2, mean),
           prop_np = apply(1-(samples$media_contr[,,3]/samples$tot),2, mean),
           prop_rest =apply((samples$contr/samples$tot),2, mean)) %>%
  pivot_longer(cols = starts_with("prop"), names_to = "group", values_to = "prop") %>% 
  mutate(group = factor(group, levels=c('prop_rest', 'prop_tv', 'prop_radio', 'prop_np'))) %>% 
  ggplot(aes(x=time, y=prop, fill=group)) +
  geom_area(alpha=0.6 , size=.5, colour="white") +
  viridis::scale_fill_viridis(discrete = T, name = "", labels = c("Other", "TV", "Radio", "News Paper")) + 
  ggtitle("Sales contribution") +theme_minimal() + geom_smooth(aes(color=group), se = F) + 
  viridis::scale_color_viridis(discrete = T, name = "", labels = c("Other", "TV", "Radio", "News Paper")) +
  ylab("Proportion")




data.frame(time = data$week_dt2,
           tot_tv = apply(samples$tot-(samples$media_contr[,,1]),2, mean),
           tot_radio = apply((samples$tot-samples$media_contr[,,2]),2, mean),
           tot_np = apply((samples$tot-samples$media_contr[,,3]),2, mean),
           tot_rest =apply((samples$contr),2, mean)) %>%
  pivot_longer(cols = starts_with("tot"), names_to = "group", values_to = "tot") %>% 
  mutate(group = factor(group, levels=c('tot_rest', 'tot_tv', 'tot_radio', 'tot_np'))) %>% 
  ggplot(aes(x=time, y=tot, fill=group)) +
  geom_area(alpha=0.6 , size=.5, colour="white") +
  viridis::scale_fill_viridis(discrete = T, name = "", labels = c("Other", "TV", "Radio", "News Paper")) + 
  ggtitle("Sales") +theme_minimal() + geom_smooth(aes(color=group), se = F) + 
  viridis::scale_color_viridis(discrete = T, name = "", labels = c("Other", "TV", "Radio", "News Paper")) +
  ylab("Proportion")



data.frame(time = data$week_dt2, prediction = apply(samples$tot,2, mean), sales = data$sales) %>% 
  pivot_longer(cols=prediction:sales) %>% ggplot(aes(x=time, y =value, group = name, color=name))+
  geom_line() + theme_minimal() + ylab("Sales") 






data.frame(time = data$week_dt2,
           tot_tv = apply(samples$tot-(samples$media_contr[,,1]),2, mean)/data$TV,
           tot_radio = apply((samples$tot-samples$media_contr[,,2]),2, mean)/data$radio,
           tot_np = apply((samples$tot-samples$media_contr[,,3]),2, mean)/data$newspaper) %>%
  pivot_longer(cols = starts_with("tot"), names_to = "group", values_to = "tot") %>% 
  #mutate(group = factor(group, levels=c('tot_rest', 'tot_tv', 'tot_radio', 'tot_np'))) %>% 
  ggplot(aes(x=tot)) +
  geom_histogram()+ facet_wrap(vars(group))
  viridis::scale_fill_viridis(discrete = T, name = "", labels = c("Other", "TV", "Radio", "News Paper")) + 
  ggtitle("Sales contribution") +theme_minimal() + geom_smooth(aes(color=group), se = F) + 
  viridis::scale_color_viridis(discrete = T, name = "", labels = c("Other", "TV", "Radio", "News Paper")) +
  ylab("Proportion")




hill_data = data %>% mutate(TV_func =  mean(samples$beta_media[,1]) * hill_function(TV,mean(samples$ec[,1]),mean(samples$slope[,1])),
                            Radio_func =   mean(samples$beta_media[,2]) * hill_function(radio,mean(samples$ec[,2]),mean(samples$slope[,2])),
                            NP_func =  mean(samples$beta_media[,3]) * hill_function(newspaper,mean(samples$ec[,3]),mean(samples$slope[,3])),
                            hill_TV = hill_function(TV,mean(samples$ec[,1]),mean(samples$slope[,1])),
                            hill_radio = hill_function(radio,mean(samples$ec[,2]),mean(samples$slope[,2])),
                            hill_newspaper =  hill_function(newspaper,mean(samples$ec[,3]),mean(samples$slope[,3]))) 


hist(hill_data$TV_func/hill_data$TV)
hist(hill_data$Radio_func/hill_data$radio)
hist(hill_data$NP_func/hill_data$newspaper)

plot(hill_data$TV_func/hill_data$TV ~ hill_data$TV )
plot(hill_data$Radio_func/hill_data$radio ~ hill_data$radio )
plot(hill_data$NP_func/hill_data$newspaper ~ hill_data$newspaper )


mean(hill_data$TV_func[hill_data$TV!=0]/hill_data$TV[hill_data$TV!=0])
mean(hill_data$Radio_func[hill_data$radio!=0]/hill_data$radio[hill_data$radio!=0])
mean(hill_data$NP_func[hill_data$newspaper!=0]/hill_data$newspaper[hill_data$newspaper!=0])

hill_data %>% ggplot(aes(x=TV, y=TV_func))  + geom_line()
hill_data %>% ggplot(aes(x=radio, y=Radio_func))  + geom_line()
hill_data %>% ggplot(aes(x=newspaper, y=NP_func))  + geom_line()


df1 = hill_data %>%
  pivot_longer(
    cols = c(TV, radio, newspaper),
    names_to = "grouping_variable",
    values_to = "values1"
  )


df2 <- hill_data %>%
  pivot_longer(
    cols = starts_with("hill_"),
    names_to = "hill_grouping_variable",
    values_to = "values2"
  ) %>%
  mutate(hill_grouping_variable = sub("hill_", "", hill_grouping_variable)) %>% 
  select(values2, hill_grouping_variable)

df3 <- hill_data %>%
  pivot_longer(
    cols = ends_with("_func"),
    names_to = "group",
    values_to = "values2"
  ) 


p <- df1 %>% select(values1, grouping_variable) %>%
  cbind(df3 ) %>% ggplot(aes(x = values1, y=values2, color=grouping_variable)) + geom_line(size=1.2) + geom_point(alpha=0) +
  theme_minimal() + theme(legend.position = c(0.8, 0.2), legend.title = element_blank()) + ylab("Return in sales") + xlab("Investment")
   #facet_wrap(vars(grouping_variable),scales="free_x") 
p_hill <- ggMarginal(p, type = "density", groupFill = TRUE, margins = "x")

ggsave("plots/p_hill_mmm_data.png",p_hill, width = 3, height = 3)



p2 <- df1 %>% select(values1, grouping_variable) %>%
  cbind(df3 ) %>% ggplot(aes(x = values1, y=values2/values1, color=grouping_variable)) + geom_line(size=1.2) + 
  theme_minimal() + theme(legend.position = c(0.8, 0.8), legend.title = element_blank()) + 
  ylab("Return in Investment") +
  xlab("Investment in chanel") +
facet_wrap(vars(grouping_variable),scales="free_x") 
p2
ggsave("plots/roi_mmm_data.png",p_hill, width = 3, height = 3)



hill_data %>% ggplot(aes(x=TV, y= hill_TV)) + geom_line()
hill_data %>% ggplot(aes(x=radio, y= hill_radio)) + geom_line()
hill_data %>% ggplot(aes(x=newspaper, y= hill_np)) + geom_line()

hill_data %>%  ggplot(aes(x=TV, y=sales)) + geom_point() + geom_line(aes(x=TV, y=TV_func))

hill_data %>% ggplot(aes(x=radio, y=sales)) + geom_point() + geom_line(aes(x=radio, y=Radio_func))

hill_data %>% ggplot(aes(x=newspaper, y=sales)) + geom_point() + geom_line(aes(x=newspaper, y=NP_func))










# Define the Hill function
hill_function <- function(x, ec, slope) {
  return( 1 / (1 + (x / ec)^(-slope)))
}

# Define a sequence of ligand concentrations
x_values <- seq(0, 300, length.out = 100)

# Define different Hill coefficients and dissociation constants
ec <- c(1, 100)
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

