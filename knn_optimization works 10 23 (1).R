#Load the libraries
library(tidyverse)
library(caret)
library(mlbench)
suppressMessages(library("tidyverse"))
library(reshape2)
library(readxl)
library(rpart) 
library(partykit) 
library(randomForest)
library(class)
library (rminer)
library(plyr)

dat <- read_excel("Absenteeism_at_work.xls")
col <- c("ID", "Reason for absence", "Month of absence", "Day of the week", "Seasons", "Disciplinary failure", "Education", "Social drinker",   "Social smoker")
#dat[col] <- lapply(dat[col], as.factor)
colnames(dat) <- c("ID", "Reason", "Month", "Day", "Seasons", "Transportation_expense", "Distance", "Service_time", "Age", "Work_load", "Hit_target", "Disciplinary_failure", "Education", "Children", "Social_drinker", "Social_smoker", "Pet", "Weight", "Height", "BMI", "Absent_time")


nums <- unlist(lapply(dat, is.numeric))  
dat.num <- dat[ , nums]

#change variable represent missed time one day or greater
dat <- dat %>% mutate(Absent_time= ifelse(dat$Absent_time <=8,0,1))
str(dat)
dat$Absent_time <- as.factor(dat$Absent_time)
#Transforming to Data Frame
dat <- as.data.frame(dat)

str(dat)

###Optimizing the KNN

#For the tunning of the KNN model, we are going to create another traning/test data sets.

#scaling the data:
dat_v <- dat #we are going to use dat_v for the manipulation

dat_v[c(2:20)] <- lapply(dat_v[c(2: 20)], function(x) c(scale(x)))
str(dat_v)

#predicting class:
AB_class <- dat_v[, 21]
names(AB_class) <- c(1:nrow(dat_v))
dat_v$ID <- c(1:nrow(dat_v))

dat_v <- dat_v[1:737,]
nrow(dat_v)
rand_permute <- sample(x = nrow(dat_v), size = nrow(dat_v))

all_id_random <- dat_v[rand_permute, "ID"]
dat_v <- dat_v[,-1] #remove ID

#random samples for training test
validate_id <- as.character(all_id_random[1:248])
training_id <- as.character(all_id_random[249:737])

dat_v_train <- dat_v[training_id, ]
dat_v_val <- dat_v[validate_id, ]
AB_class_train <- AB_class[training_id]
AB_class_val <- AB_class[validate_id]
table(AB_class_train)

#Study significance of the variables
exp_vars <- names(dat_v_train)
exp_var_fstat <- as.numeric(rep(NA, times = 20))
names(exp_var_fstat) <- exp_vars

for (j in 1:length(exp_vars)) {
  exp_var_fstat[exp_vars[j]] <- summary(lm(as.formula(paste(exp_vars[j], " ~ AB_class_train")), 
                                           data = dat_v_train))$fstatistic[1]
}

exp_var_fstat

exp_var_fstat2 <- sapply(exp_vars, function(x) {
  summary(lm(as.formula(paste(x, " ~ AB_class_train")), data = dat_v_train))$fstatistic[1]
})
exp_var_fstat2

names(exp_var_fstat2) <- exp_vars

#plyr version of the fit

wbcd_df_L <- lapply(exp_vars, function(x) {
  df <- data.frame(sample = rownames(dat_v_train), 
                   variable = x, value = dat_v_train[, x], 
                   class = AB_class_train)
  df
})
head(wbcd_df_L[[1]])

names(wbcd_df_L) <- exp_vars

library(plyr)
var_sig_fstats <- laply(wbcd_df_L, function(df) {
  fit <- lm(value ~ class, data = df)
  f <- summary(fit)$fstatistic[1]
  f
})

names(var_sig_fstats) <- names(wbcd_df_L)
var_sig_fstats[1:3]

#Conclusions about significance of the variables

most_sig_stats <- sort(var_sig_fstats, decreasing = T)
most_sig_stats[1:5]

#As per 'most_sig_stats' the 5 most significant variables for the prediction are: 
#'Seasons', 'Reason', 'Service_time', 'Month' and 'work_load'

#Re ordering variables by significance:

dat_v_train_ord <- dat_v_train[, names(most_sig_stats)]
str(dat_v_train_ord)

  
dat_v_val_ord <- dat_v_val[, names(dat_v_train_ord)]
str(dat_v_val_ord)

#######################
#######################

#Monte Carlo Validation:

size <- length(training_id)
(2/3) * length(training_id)

training_family_L <- lapply(1:500, function(j) {
  perm <- sample(1:size, size = size, replace = F)
  shuffle <- training_id[perm]
  trn <- shuffle[1:326]
  trn
})

validation_family_L <- lapply(training_family_L, 
                              function(x) setdiff(training_id, x))

#Finding an optimal set of variables and optimal k

N <- seq(from = 2, to = 19, by = 1)
sqrt(length(training_family_L[[1]]))
K <- seq(from = 1, to = 15, by = 1)
times <- 500 * length(N) * length(K)

#Execution of the test with loops

paramter_errors_df <- data.frame(mc_index = as.integer(rep(NA, times = times)), 
                                 var_num = as.integer(rep(NA, times = times)), 
                                 k = as.integer(rep(NA, times = times)), 
                                 error = as.numeric(rep(NA, times = times)))

#Core knn_model:
# j = index, n = length of range of variables, k=k

core_knn <- function(j, n, k) {
  set.seed(101)
  knn_predict <- knn(train = dat_v_train_ord[training_family_L[[j]], 1:n], 
                     test = dat_v_train_ord[validation_family_L[[j]], 1:n], 
                     cl = AB_class_train[training_family_L[[j]]], 
                     k = k)
  tbl <- table(knn_predict, AB_class_train[validation_family_L[[j]]])
  err <- (tbl[1, 2] + tbl[2, 1])/(tbl[1, 2] + tbl[2, 1]+tbl[1, 1] + tbl[2, 2])
  err
}


param_df1 <- merge(data.frame(mc_index = 1:500), data.frame(var_num = N))
param_df <- merge(param_df1, data.frame(k = K))


knn_err_est_df <- ddply(param_df[1:times, ], .(mc_index, var_num, k), function(df) {
  set.seed(101)
  err <- core_knn(df$mc_index[1], df$var_num[1], df$k[1])
  err
})

head(knn_err_est_df)
names(knn_err_est_df)[4] <- "error"

mean_errs_df <- ddply(knn_err_est_df, .(var_num, k), function(df) mean(df$error))
head(mean_errs_df)
names(mean_errs_df)[3] <- "mean_error"

library(ggplot2)
ggplot(data = mean_errs_df, aes(x = var_num, y = k, color = mean_error)) + geom_point(size = 5) + 
  theme_bw()

#This is the model that produces the lowest mean error var_num = 6 and k = 1:
mean_errs_df[which.min(mean_errs_df$mean_error), ]

mean_errs_df %>% arrange(mean_error)



#load files from previous analysis
#load( file='errmatrix.RData')
#load( file='sensmatrix.RData')
#load( file='fmeasmatrix.RData')
#load( file='gmeanmatrix.RData')

#eventually run old to compare with new. 
#We see that although error lower, other metrics hurt. We care about identifying >8 hours  so modify

#Repeat with sensitivity

core_knn_sen <- function(j, n, k) {
  set.seed(1876)
  knn_predict <- knn(train = dat_v_train_ord[training_family_L[[j]], 1:n], 
                     test = dat_v_train_ord[validation_family_L[[j]], 1:n], 
                     cl = AB_class_train[training_family_L[[j]]], 
                     k = k)

  tbl <- table(knn_predict, AB_class_train[validation_family_L[[j]]])
  
  #generate confusion matrix ( the 1 tells the model we care about that output)
  cm_KNN <-  confusionMatrix(data = tbl, reference =AB_class_train[validation_family_L[[j]]], positive = "1")
  
  sen <- cm_KNN$byClass[1]
  sen
}


param_df1_2 <- merge(data.frame(mc_index = 1:500), data.frame(var_num = N))
param_df_2 <- merge(param_df1_2, data.frame(k = K))

set.seed(1876)
knn_err_est_df_2 <- ddply(param_df_2[1:times, ], .(mc_index, var_num, k), function(df) {
  set.seed(1876)
  sen <- core_knn_sen(df$mc_index[1], df$var_num[1], df$k[1])
  sen
})

head(knn_err_est_df_2)
names(knn_err_est_df_2)[4] <- "Sensitivity"

mean_sens_df <- ddply(knn_err_est_df_2, .(var_num, k), function(df) mean(df$Sensitivity))
head(mean_sens_df)
names(mean_sens_df)[3] <- "mean_sensitivity"

library(ggplot2)
ggplot(data = mean_sens_df, aes(x = var_num, y = k, color = mean_sensitivity)) + geom_point(size = 5) + 
  theme_bw()

#This is the model that produces the best value of sensitivity:
mean_sens_df[which.max(mean_sens_df$mean_sensitivity), ]

mean_sens_df %>% arrange(desc(mean_sensitivity))

#Best KNN:
set.seed(1876)
KNN_5_3 <- knn(train = dat_v_train_ord[, 1:5], 
               dat_v_val_ord[, 1:5], AB_class_train, 
               k = 3)

tbl_bm_val <- table(KNN_5_3, AB_class_val)
tbl_bm_val

cm_KNN_opt <-  confusionMatrix(data = tbl_bm_val, reference = dat_v_val_ord[, 1:4], positive = "1")
cm_KNN_opt
###########################

R <- 50 # replications

# create the matrix to store values 1 row per model
err_matrix_opt <- matrix(0, ncol=1, nrow=R)

sensitivity_matrix_opt <- matrix(0, ncol=1, nrow=R)

fmeasure_matrix_opt <- matrix(0, ncol=1, nrow=R)

gmean_matrix_opt <- matrix(0, ncol=1, nrow=R)

# these are optional but I like to see how the model did each run so I can check other output
KNNcm <- matrix(0, ncol=4, nrow=R)

dat_smaller <- dat[, names(dat_v_train_ord)]
dat_smaller[,20] <- dat$Absent_time


dat_smaller <- dat_smaller[1:737,] # remove lines with non-meaningful data

scale <- sapply(dat_smaller, is.numeric)
dat_smaller[scale] <- lapply(dat_smaller[scale],scale)
head(dat_smaller)

set.seed(1876)


for (r in 1:R){
  
  # subsetting data to training and testing data
  p <- .6 # proportion of data for training
  w <- sample(1:nrow(dat_smaller), nrow(dat_smaller)*p, replace=F)
  data_train <-dat_smaller[w,] 
  data_test <- dat_smaller[-w,]
  
  ################################################################ knn
  
  #Running the classifier
  
  knn <- knn(data_train[,1:2],
             test = data_test[,1:2],
             cl=data_train[,20], k=5)
  
  #predict doesn't work with KNN for factors
  knntable <- table(knn, data_test[,20])
  
  #generate confusion matrix ( the 1 tells the model we care about that output)
  cm_KNN <-  confusionMatrix(data = knntable, reference = data_test[,1:2], positive = "1")
  
  KNNcm [[r,1]] <-  cm_KNN$table[1,1]
  KNNcm [[r,2]] <-  cm_KNN$table[1,2]
  KNNcm [[r,3]] <-  cm_KNN$table[2,1]
  KNNcm [[r,4]] <-  cm_KNN$table[2,2]
  
  err_matrix_opt [[r,1]] <-  (cm_KNN$table[1,2]+cm_KNN$table[2,1])/nrow( data_test)
  
  # store the errors (change the 1 to whichever model you have)   
  
  sensitivity_matrix_opt[[r, 1]] <- cm_KNN$byClass[1]
  
  fmeasure_matrix_opt [[r, 1]] <- cm_KNN$byClass[7]
  
  gmean_matrix_opt [[r, 1]] <- sqrt(cm_KNN$byClass[1]* cm_KNN$byClass[2])
  
  cat("Finished Rep",r, "\n")
}

colnames(sensitivity_matrix_opt)<- c("KNN")
graph_sens <- melt(sensitivity_matrix_opt)

graph <- ggplot(graph_sens,aes(x=Var2, y=value) )+ geom_boxplot()
graph


colnames(err_matrix_opt)<- c("KNN")
graph_err <- melt(err_matrix_opt)

graph <- ggplot(graph_err,aes(x=Var2, y=value) )+ geom_boxplot()
graph

