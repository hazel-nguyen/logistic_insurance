######### STA 463 Final Project ########################
# Binary Logistic Regression Model
# Dataset: Travel Insurance
# Date: 12/4/2021
# Group 1 Members: Bich Ha Nguyen, Jacob Akubire, Jeremy Hampton, Lucy Cobble
########################################################


# Load packages
# install.packages("glmulti")
library(dlookr)
library(caTools)
library(MPV)
library(ggfortify)
library(faraway)
library(car)
library(caret)
library(leaps)
library(glmulti)
library(kableExtra)
library(GGally)
library(purrr)
library(tidyr)
library(ggplot2)
library(tidyverse)

# Set directory
setwd("C:/Users/77 thaiha/Pictures/2021F/STA 463/Final Project")

# Load data
raw_data = read_csv("TravelInsurancePrediction.csv", col_select=2:10)

# Transform numerical variables to factor variables
raw_data$TravelInsurance <- as.factor(raw_data$TravelInsurance)
raw_data$ChronicDiseases <- as.factor(raw_data$ChronicDiseases)


# Part 1: EDA

# Generate automatic report of all variables 
raw_data %>%
  eda_web_report(target = "TravelInsurance", subtitle = "Travel Insurance", 
                 output_dir = "./", output_file = "EDA.html", theme = "blue")

# Plot distributions of numerical predictors 
raw_data %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()

# Examine correlation between variables and potential interaction with response variable
ggpairs(raw_data)


#### Comments: ####
# The numerical variables have not too skewed distributions -> no transformation needed
# The categorical variables have fairly good proportions
# The response variable has  ~ 30% of observations as 1 -> enough observations to make predictions


# Part 2: Split data into training and testing sets - 80/20
set.seed(101) 
sample = sample.split(raw_data$TravelInsurance, SplitRatio = .80)
train = subset(raw_data, sample == TRUE)
test  = subset(raw_data, sample == FALSE) 

# Part 3: Full Fitted model
# M0: Null
null_fit <- glm(TravelInsurance ~ 1, data = train, family = binomial (link = logit))
# M1: Full main effects
full_fit <- glm(TravelInsurance ~ ., data=train, family=binomial(link=logit))

# Usefulness of Model Test - M1
anova(null_fit, full_fit, test="LRT")

# VIF
vif(full_fit)

# M1 Summary
summary(full_fit)
# Transformed Coefficients
exp(full_fit$coefficients) - 1

# M1: Diagnostics
# Linearity
plot(full_fit, which=1)
# bin the residuals: 100 bins
train %>% 
  mutate(residuals = residuals(full_fit), linpred=predict(full_fit)) %>%
  group_by(cut(linpred, breaks = unique(quantile(linpred, (1:100)/101)))) %>%
  summarise(residuals=mean(residuals), linpred = mean(linpred)) %>%
  ggplot() +
  geom_point(aes(x = linpred, y=residuals)) +
  labs(x="linear predictor", title = "Binned Residuals vs linear predicted values")
# halfnormal plot
halfnorm(hatvalues(full_fit))
# influential points
plot(full_fit, which=4)
train[c(705, 1333, 1556),] %>% kbl() %>% kable_styling()


## Part 4: Variable Selection for Main effect model

# AIC Backward
back_fit_AIC <- stats::step(full_fit, direction = "backward", trace=0, plot=T)
summary(back_fit_AIC)

# Age + Annual Income + Family + Chronic + Frequent + EverTravel

# BIC Backward
back_fit_BIC <- stats::step(full_fit, direction = "backward", k= log(nrow(train)), trace=0)
summary(back_fit_BIC)
# Age + Annual Income + Family + Frequent Flyer + Travel Abroad

# AIC Forward
forw_fit_AIC=step(null_fit, direction="forward", scope=list(upper=full_fit), trace = 0)
summary(forw_fit_AIC)
# same as backward

# BIC Forward
forw_fit_BIC=step(null_fit, direction="forward", k= log(nrow(train)), scope=list(upper=full_fit), trace = 0)
summary(forw_fit_BIC)
# same as backward

# Stepwise Selection
stepwise_AIC=step(null_fit, direction="both", scope=list(upper=full_fit),trace=0)
summary(stepwise_AIC)
# same as backward

stepwise_BIC=step(null_fit, direction="both", scope=list(upper=full_fit),k=log(nrow(train)), trace=0)
summary(stepwise_BIC)
# same as backward

## Compare the best main effects model

# M2: Age + Annual Income + Family + Chronic + Frequent + EverTravel

# M3: Age + Annual Income + Family + Frequent Flyer + Travel Abroad
AICvec=c(AIC(back_fit_AIC),AIC(back_fit_BIC)) # AIC citerion
BICvec=c(BIC(back_fit_AIC),BIC(back_fit_BIC)) # BIC criterion
PRESSvec=c(PRESS(back_fit_AIC),PRESS(back_fit_BIC)) # PRESS Criterion
pvec=c((summary(back_fit_AIC)$df[1]-1),(summary(back_fit_BIC)$df[1]-1))

data=cbind(pvec,PRESSvec,AICvec,BICvec)
data %>% kbl() %>% kable_styling()

# Comment: Not really a huge improvement with model including the Chronic Disease
# Furthermore, the T-test result shows that Chronic Disease is not meaningful so we decide to exclude


## Best main effect model - BIC_back_fit model
# Summary table
summary(back_fit_BIC)
# Diagnostics
plot(back_fit_BIC, which = 1)
# bin the residuals: 100 bins
train %>% 
  mutate(residuals = residuals(back_fit_BIC), linpred=predict(back_fit_BIC)) %>%
  group_by(cut(linpred, breaks = unique(quantile(linpred, (1:100)/101)))) %>%
  summarise(residuals=mean(residuals), linpred = mean(linpred)) %>%
  ggplot() +
  geom_point(aes(x = linpred, y=residuals)) +
  labs(x="linear predictor", title = "Binned Residuals vs linear predicted values")
# half normal
halfnorm(hatvalues(back_fit_BIC))
# influential
plot(back_fit_BIC, which=4)
# VIF
vif(back_fit_BIC)

# Coefficient Interpretation
exp(back_fit_BIC$coefficients) - 1 # percent change


## Part 5: Fit the main interaction model
glmulti.logistic.out <- do.call("glmulti",
        list(TravelInsurance ~ AnnualIncome + EverTravelledAbroad + Age + FamilyMembers + FrequentFlyer,
          data = train,
          level = 2,               # 2 way interaction considered
          method = "h",            # Exhaustive approach
          crit = "aic",            # AIC as criteria
          confsetsize = 5,         # Keep 5 best models
          plotty = T, report = T,  
          fitfunction = "glm",
          family = binomial))      # lm function

## Show 5 best models (Use @ instead of $ for an S4 object)
glmulti.logistic.out@formulas

## Model with the least number of predictors
summary(glmulti.logistic.out@objects[[2]])

## Results from console #####################
# FamilyMembers                       -2.702e+00
# FrequentFlyerYes                    -3.154e+00
# EverTravelledAbroadNo:AnnualIncome   6.224e-07
# EverTravelledAbroadYes:AnnualIncome  4.454e-06
# EverTravelledAbroadNo:Age           -3.464e-01
# EverTravelledAbroadYes:Age          -4.434e-01
# FamilyMembers:Age                    9.585e-02
# FrequentFlyerYes:AnnualIncome        2.719e-06
# FamilyMembers:FrequentFlyerYes
#############################################

# Using the summary, fit the interaction model
# M4: best interaction model based on AIC 
best_inter <- glm(TravelInsurance ~ FamilyMembers + FrequentFlyer + EverTravelledAbroad:AnnualIncome + 
                    Age:EverTravelledAbroad + FamilyMembers:Age + FrequentFlyer:AnnualIncome + 
                    FrequentFlyer:FamilyMembers, data = train, family = binomial (link = logit))
# Diagnostics
# Linearity
plot(best_inter, which =1)
halfnorm(hatvalues(best_inter))
# Influential
plot(best_inter, which = 4)
# Summary
summary(best_inter)
# Coefficient Interpretation
exp(best_inter$coefficients) # odds
# Comment: The coefficients change compared to the main effect model

## Part 5: Model Performance on Testing data

# Interaction model
test_trial <- test %>%
  mutate(inter_prob = predict(best_inter, newdata=test,
                              type="response"),
         InteractionPredictBuy = inter_prob >= 0.5,
         BIC_prob = predict(back_fit_BIC, newdata=test,
                            type="response"),
         MainPredictBuy = BIC_prob >= 0.5)

# Confusion Matrix
# NOTE: True = Predict Buy, False = Predict Not Buy, TravelInsurance: 0 = Actually Not Buy, 1 = Actually Buy

# Interaction Model: 
xtabs(~InteractionPredictBuy + TravelInsurance, data=test_trial)
# Main Effect Model
xtabs(~MainPredictBuy + TravelInsurance, data=test_trial) 

