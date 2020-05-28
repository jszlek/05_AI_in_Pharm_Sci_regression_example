library(groupdata2)
library(dplyr)
library(neuralnet)
library(randomForest)
library(scales)
library(ggplot2)
library(gridExtra)
library(grid)
library(MLmetrics)

my_data <- read.csv('dataset/PLGA_300in_SR_BAZA.csv', sep = '\t')

my_data$Formulation_no <- as.factor(my_data$Formulation_no)


# 1) Feature selection

my_data_num <- my_data[,-1] # get data without "Formulation_no" column
fit_rf = randomForest(Q_perc~., data=my_data_num, ntree = 500) # develop RF model
importance(fit_rf) # create an importance based on mean decreasing gini

scaled_imp <- scales::rescale(importance(fit_rf), to = c(0,100)) # scale from 0 to 100
scaled_imp <- scaled_imp[order(scaled_imp, decreasing =TRUE),] # reorder scaled data frame

# plot the variable importance
pdf(file=paste("res_feature_selection",".pdf", sep=""), height = 8, width = 12)
op <- par(mar = c(10,4,4,2) + 0.1)
barplot(scaled_imp[1:25], main="Variable importance",
        names.arg=names(scaled_imp[1:25]), cex.names=0.8, las=2)
par(op)
dev.off()

# select only 20 first features + "Formulation_no" + "Q_perc"
selected_features <- c("Formulation_no", names(scaled_imp[1:20]),"Q_perc") # create names vector
my_data <- my_data[,(c(selected_features))] # chose columns from vector names



set.seed(1234) # set.seed for splitting data

# 2) Quick data visualisation
# Plot formulation dissolution profiles
pdf(file=paste("obs_cum",".pdf", sep=""), height = 12, width = 18)

plots_obs <- list()

  plots_obs <- ggplot(my_data, aes(Time_Days, Q_perc, color="blue")) +
    geom_line(aes(group = Formulation_no)) +
    scale_color_manual(labels = c("obs"), values = c("blue")) +
    facet_wrap(~Formulation_no)

print(plots_obs)  

dev.off()

# 3) Data splitting
# create column with data partition
df_folded <- fold(my_data, 10,
                  id_col = "Formulation_no",
                  method = "n_dist")

# create empty lists to store training, testing and result
test <- list()
train <- list()
results <- list()


for(i in 1:10){
  test[[i]] = df_folded %>% filter(.folds==i)
  train[[i]] = df_folded %>% filter(.folds!=i)
}

drops <- c("Formulation_no", ".folds")

for(i in 1:10){
  tmp_test = as.data.frame(test[[i]])
  tmp_train = as.data.frame(train[[i]])
  
  tmp_test = tmp_test[, !(names(tmp_test) %in% drops)]
  tmp_train = tmp_train[, !(names(tmp_train) %in% drops)]
  
  tmp_test = tmp_test[,c(2:20,1,21)]
  tmp_train = tmp_train[,c(2:20,1,21)]
  
  write.table(tmp_test, file=paste("FS_data/t-PLGA_",(ncol(tmp_train)-1),"in_no_",i,".txt",sep=""), quote = FALSE, row.names = FALSE)
  write.table(tmp_train, file=paste("FS_data/PLGA_",(ncol(tmp_test)-1),"in_no_",i,".txt",sep=""), quote = FALSE, row.names = FALSE)
}



# 4) Modeling - Random Forest
# here we apply 10-fold cv scheme
for(i in 1:length(test)){
  
  tmp_test = as.data.frame(test[[i]])
  tmp_train = as.data.frame(train[[i]])
  
  tmp_test = tmp_test[, !(names(tmp_test) %in% drops)]
  tmp_train = tmp_train[, !(names(tmp_train) %in% drops)]
  
  tmp_rf = randomForest(Q_perc~., data=tmp_train, ntree=500, mtry=10) # training of RF models 
  tmp_pred <- predict(tmp_rf, tmp_test) # testing RF model
  
  results[[i]] <- tmp_pred # saving in results
  
}

# 5) Modeling - Random Forest - Results

rf_r2 <- list()
rf_rmse <- list()

for(i in 1:10){
  test[[i]] <- cbind(test[[i]],pred=results[[i]])
  rf_r2[[i]] <- MLmetrics::R2_Score(as.matrix(test[[i]]['pred']),as.matrix(test[[i]]['Q_perc']))
  rf_rmse[[i]] <- MLmetrics::RMSE(as.matrix(test[[i]]['pred']), as.matrix(test[[i]]['Q_perc']))
}

mean_rf_r2 <- mean(unlist(rf_r2))
mean_rf_rmse <- mean(unlist(rf_rmse))

cat("Mean 10-fold cv RMSE: ", mean_rf_rmse, "\n", sep="")
cat("Mean 10-fold cv R2: ", mean_rf_r2, "\n", sep="")


# 6) Modeling - Random Forest - Plots
pdf(file=paste("res_cum_rf",".pdf", sep=""), height = 27, width = 27)

plots_res <- list()

for(i in 1:length(test)){
  
  plots_res[[i]] <- ggplot(test[[i]], aes(Time_Days, Q_perc, color="blue")) +
    geom_line(aes(group = Formulation_no)) +
    geom_line(aes(Time_Days, pred, color="red")) +
    geom_line(aes(group = Formulation_no)) + 
    scale_color_manual(labels = c("obs", "pred"), values = c("blue", "red")) +
    facet_wrap(~Formulation_no)
  
}

do.call('grid.arrange',c(plots_res, ncol = 2))

dev.off()



# 7) Neural networks with neuralnet package 

# Scaling
maxs <- apply(my_data[2:ncol(my_data)], 2, max) 
mins <- apply(my_data[2:ncol(my_data)], 2, min)

scaled <- as.data.frame(
                        cbind(
                              Formulation_no=my_data$Formulation_no,
                              scale(my_data[2:ncol(my_data)], center = mins, scale = maxs - mins)
                              )
                        )

scaled$Formulation_no <- as.factor(scaled$Formulation_no)

# Data splitting
set.seed(1234)
# create column with data partition
df_folded_nn <- fold(scaled, 10,
                  id_col = "Formulation_no",
                  method = "n_dist")

# create empty lists to store training, testing and result
test_nn <- list()
train_nn <- list()
results_nn <- list()


for(i in 1:10){
  test_nn[[i]] = df_folded_nn %>% filter(.folds==i)
  train_nn[[i]] = df_folded_nn %>% filter(.folds!=i)
}

drops <- c("Formulation_no", ".folds")

# 8) Modeling - neural networks
# here we apply 10-fold cv scheme
for(i in 1:length(test_nn)){
  
  tmp_test_nn = as.data.frame(test_nn[[i]])
  tmp_train_nn = as.data.frame(train_nn[[i]])
  
  tmp_test_nn = tmp_test_nn[, !(names(tmp_test_nn) %in% drops)]
  tmp_train_nn = tmp_train_nn[, !(names(tmp_train_nn) %in% drops)]
  
  
  tmp_nn <- neuralnet(Q_perc ~ . , data=tmp_train_nn,
                      hidden=c(9,6,3),
                      linear.output=TRUE,
                      stepmax = 100000,
                      threshold=0.1)
  
  tmp_nn$result.matrix
  
  # plot(tmp_nn)
  
  tmp_nn$weights
  
  tmp_pred_nn <- predict(tmp_nn, tmp_test_nn) # testing RF model
  
  results_nn[[i]] <- tmp_pred_nn # saving in results
  
}

# plotting structure of ANN
pdf(file=paste("nn_structure",".pdf", sep=""), height = 9, width = 15)
print(plot(tmp_nn, rep="best", radius = 0.1, arrow.length = 0.1, intercept = TRUE,
           intercept.factor = 0.2, information = TRUE, information.pos = 0.05))
dev.off()


# 9) Neural networks - results
nn_r2 <- list()
nn_rmse <- list()

for(i in 1:10){
  
  results_nn[[i]] <- results_nn[[i]]*(max(my_data$Q_perc)-min(my_data$Q_perc))+min(my_data$Q_perc)
  test_nn[[i]]['Time_Days'] <- test_nn[[i]]['Time_Days']*(max(my_data$Time_Days)-min(my_data$Time_Days))+min(my_data$Time_Days)
  test_nn[[i]]['Q_perc'] <- test_nn[[i]]['Q_perc']*(max(my_data$Q_perc)-min(my_data$Q_perc))+min(my_data$Q_perc)
  test_nn[[i]] <- cbind(test_nn[[i]],pred=results_nn[[i]])
  nn_r2[[i]] <- MLmetrics::R2_Score(as.matrix(test_nn[[i]]['pred']),as.matrix(test_nn[[i]]['Q_perc']))
  nn_rmse[[i]] <- MLmetrics::RMSE(as.matrix(test_nn[[i]]['pred']), as.matrix(test_nn[[i]]['Q_perc']))
  
}

mean_nn_r2 <- mean(unlist(nn_r2))
mean_nn_rmse <- mean(unlist(nn_rmse))

cat("Mean 10-fold cv RMSE: ", mean_nn_rmse, sep="")
cat("Mean 10-fold cv R2: ", mean_nn_r2, sep="")


# 10) Neural networks - visualisation

pdf(file=paste("res_cum_nn",".pdf", sep=""), height = 27, width = 27)

plots_res_nn <- list()

for(i in 1:length(test_nn)){
  
  plots_res_nn[[i]] <- ggplot(test_nn[[i]], aes(Time_Days, Q_perc, color="blue")) +
    geom_line(aes(group = Formulation_no)) +
    geom_line(aes(Time_Days, pred, color="red")) +
    geom_line(aes(group = Formulation_no)) + 
    scale_color_manual(labels = c("obs", "pred"), values = c("blue", "red")) +
    facet_wrap(~Formulation_no)
  
}

do.call('grid.arrange',c(plots_res_nn, ncol = 2))

dev.off()
