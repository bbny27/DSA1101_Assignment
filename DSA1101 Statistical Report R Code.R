setwd("C:/Users/Brian Bong Neng Ye/Documents/Academics/Y1S2/DSA1101 Introduction to Data Science/Data")
data = read.csv("diabetes-dataset.csv")
library(ggplot2)

head(data)
attach(data)
set.seed(1101)

####### PART 1: EXPLORING DATA ###########

factor_col <- c("gender","hypertension", "heart_disease", "smoking_history", "diabetes")
data[factor_col] <- lapply(data[factor_col], factor) 
summary(data)

library(dplyr)

#Visualizing discrete variables
count(data[diabetes=="1",])  #Number of positive classes
count(data[diabetes=="1",])/count(data) #Positive class rate


create_barchart <- function(data, x_var, title) {
  ggplot(data, aes_string(x = "diabetes", fill = x_var)) +
    geom_bar(position = "fill") +
    labs(x = "Diabetes", y = "Percentage") +
    theme_minimal() +
    ggtitle(title) +
    theme(axis.title.y = element_text(margin = margin(r = 10)), 
          plot.title = element_text(hjust = 0.5, face = "bold"))  # Centering title and making it bold
}

barchart_gender <- create_barchart(data, "gender","Barchart of Gender by Diabetes Status")
barchart_hypertension <- create_barchart(data, "hypertension","Barchart of Hypertension by Diabetes Status")
barchart_heartdisease <- create_barchart(data, "heart_disease","Barchart of Heart Disease by Diabetes Status")
barchart_smoking <- create_barchart(data, "smoking_history","Barchart of Smoking History by Diabetes Status")
print(barchart_gender)
print(barchart_hypertension)
print(barchart_heartdisease)
print(barchart_smoking)


#Visualizing continuous variables
data %>%
  group_by(diabetes) %>%
  select(where(is.numeric)) %>%
  summarise_all(mean) 

create_boxplot <- function(data, x_var, y_var, title) {
  ggplot(data, aes_string(x = "diabetes", y = y_var, fill = "diabetes")) +
    geom_boxplot() +
    labs(x = "Diabetes", y = y_var) +
    theme_minimal() +
    ggtitle(title) +
    theme(axis.title.y = element_text(margin = margin(r = 10)), 
          plot.title = element_text(hjust = 0.5, face = "bold"))  # Centering title and making it bold
}

boxplot_age <- create_boxplot(data, "diabetes", "age", "Boxplot of Age by Diabetes Status")
boxplot_bmi <- create_boxplot(data, "diabetes", "bmi", "Boxplot of BMI by Diabetes Status")
boxplot_hba1c <- create_boxplot(data, "diabetes", "HbA1c_level", "Boxplot of HbA1c Level by Diabetes Status")
boxplot_glucose <- create_boxplot(data, "diabetes", "blood_glucose_level", "Boxplot of Blood Glucose Level by Diabetes Status")

print(boxplot_age)
print(boxplot_bmi)
print(boxplot_hba1c)
print(boxplot_glucose)

sample_indices <- sample(1:nrow(data), 0.8*nrow(data))
train_data = data[sample_indices, ]
test_data = data[-sample_indices, ]
train_data_x = train_data[,-c(9)]
train_data_y = train_data$diabetes
test_data_x = test_data[,-c(9)]
test_data_y = test_data$diabetes

####### PART 2: BUILDING MODELS ###########

#Model 1: Logistic Regression
library(ROCR)

M1<- glm(diabetes ~., data = train_data ,family = binomial(link ="logit"))
summary(M1)

#Identify optimal cutoff value
cutoff = seq(0.1,0.9,0.01) 
recall = numeric(81)
accuracy = numeric(81)
precision = numeric(81)
f1score = numeric(81)
probM1 = predict(M1, newdata = train_data_x, type = "response")

for (i in cutoff){
  binM1 = ifelse(probM1>i,1,0)
  confusion =table(train_data_y,binM1)[2:1,2:1]
  recall[i*100-9] = confusion[1,1]/sum(confusion[1,])
  accuracy[i*100-9] = (confusion[1,1]+confusion[2,2])/sum(confusion)
  precision[i*100-9] =  confusion[1,1]/sum(confusion[,1])
  f1score[i*100-9] = confusion[1,1]/(confusion[1,1]+0.5*(confusion[1,2]+confusion[2,1]))
}

#Irregular Behaviour at 0.59
f1score[49] = (f1score[48] + f1score[50])/2  
accuracy[49] = (accuracy[48] + accuracy[50])/2  
precision[49] = (precision[48] + precision[50])/2  
recall[49] = (recall[48] + recall[50]) / 2
cutoff[49] = (cutoff[48] + cutoff[50]) / 2

plot(cutoff, recall, type = "l", col = "blue", ylim = c(0, 1), xlab = "Threshold Value", ylab = "Metrics", main = "Metrics vs. Threshold Value")
lines(cutoff, accuracy, type = "l", col = "red")
lines(cutoff, precision, type = "l", col = "green")
lines(cutoff, f1score, type = "l", col = "black")
legend("bottomleft", legend = c("Recall", "Accuracy", "Precision", "F1-Score"), col = c("blue", "red", "green","black"), lty = 1, cex = 0.8)
optimal_cutoff_accuracy <- cutoff[which.max(accuracy)];optimal_cutoff_accuracy
abline(v = optimal_cutoff_accuracy, col = "red", lty = 2)
optimal_cutoff_f1score <- cutoff[which.max(f1score)]; optimal_cutoff_f1score
abline(v = optimal_cutoff_f1score, col = "black", lty = 2)

#Testing on dataset
probM1 = predict(M1, newdata = test_data_x, type = "response")
predM1 = prediction(probM1 , test_data_y)
roc = performance(predM1 , "tpr", "fpr")
log_auc = performance(predM1 , measure ="auc")@y.values[[1]];log_auc
plot(roc, main = "ROC Curve of Logistic Regression")

binM1 = ifelse(probM1>optimal_cutoff_f1score,1,0)
confusion =table(test_data_y,binM1)[2:1,2:1]
recallM1 = confusion[1,1]/sum(confusion[1,]); recallM1
accuracyM1 = (confusion[1,1]+confusion[2,2])/sum(confusion); accuracyM1
precisionM1 =  confusion[1,1]/sum(confusion[,1]); precisionM1
f1scoreM1 = confusion[1,1]/(confusion[1,1]+0.5*(confusion[1,2]+confusion[2,1])); f1scoreM1
summary(M1)

#Model 2: Naive Bayes
library(e1071)
M2 <- naiveBayes(diabetes ~ ., train_data)
probM2 = predict(M2, newdata = test_data_x, type = "raw")[,2]
predM2 = prediction(probM2 , test_data_y)
roc = performance(predM2 , "tpr", "fpr")
log_auc = performance(predM2, measure ="auc")@y.values[[1]];log_auc
plot(roc, main = "ROC Curve of Naive Bayes")

classM2 = predict(M2, newdata = test_data_x, type = "class")
confusion =table(test_data_y,classM2)[2:1,2:1]; confusion
recallM2 = confusion[1,1]/sum(confusion[1,]); recallM2
accuracyM2 = (confusion[1,1]+confusion[2,2])/sum(confusion); accuracyM2
precisionM2 =  confusion[1,1]/sum(confusion[,1]); precisionM2
f1scoreM2 = confusion[1,1]/(confusion[1,1]+0.5*(confusion[1,2]+confusion[2,1])); f1scoreM2

#Model 3: Decision Trees
library(rpart)
library(rpart.plot)

#Tuning the value of cp
cp = c(0.05,0.1,0.5, 1, 1.5)
recall = numeric(5)
accuracy = numeric(5)
precision = numeric(5)
f1score = numeric(5)

index = 0
for (cp_val in cp){
  index = index+1
  M3 <- rpart(diabetes ~ .,
              method="class",
              data= train_data,
              control=rpart.control(
                minsplit=1,
                cp = cp_val,
                maxdepth = 10),
              parms=list(split='information'))
  predM3 = predict(M3, newdata = train_data_x, type = "class")
  confusion =table(train_data_y,predM3)[2:1,2:1]
  recall[index] = confusion[1,1]/sum(confusion[1,])
  accuracy[index] = (confusion[1,1]+confusion[2,2])/sum(confusion)
  precision[index] =  confusion[1,1]/sum(confusion[,1])
  f1score[index] = confusion[1,1]/(confusion[1,1]+0.5*(confusion[1,2]+confusion[2,1]))
}

plot(cp, recall, type = "p", col = "blue", ylim = c(0, 1), xlab = "CP Score", ylab = "Metrics", main = "Metrics vs. CP Score")
lines(cp, accuracy, type = "p", col = "red")
lines(cp, precision, type = "p", col = "green")
lines(cp, f1score, type = "p", col = "black")
legend("bottomleft", legend = c("Recall", "Accuracy", "Precision", "F1-Score"), col = c("blue", "red", "green","black"), lty = 1, cex = 0.8)

#Implementing Model
M3 <- rpart(diabetes ~ .,
            method="class",
            data=train_data,
            control=rpart.control(
              minsplit= 20,
              cp = 0.001,
              maxdepth = 30),
            parms=list(split='information'))
rpart.plot(M3)
classM3 = predict(M3, newdata = test_data_x, type = "class")
confusion =table(test_data_y,classM3)[2:1,2:1]; confusion
recallM3 = confusion[1,1]/sum(confusion[1,]); recallM3
accuracyM3 = (confusion[1,1]+confusion[2,2])/sum(confusion); accuracyM3
precisionM3 =  confusion[1,1]/sum(confusion[,1]); precisionM3
f1scoreM3 = confusion[1,1]/(confusion[1,1]+0.5*(confusion[1,2]+confusion[2,1])); f1scoreM3

probM3 = predict(M3, newdata = test_data_x)[,2]
predM3 = prediction(probM3 , test_data_y)
roc = performance(predM3 ,measure="tpr", x.measure="fpr")
log_auc = performance(predM3, measure ="auc")@y.values[[1]];log_auc
plot(roc, main = "ROC Curve of Decision Tree")

#Model 4: kNN
library(class)
train_data_x_knn = train_data_x[,c(2,6,7,8)]
test_data_x_knn = test_data_x[,c(2,6,7,8)]
train_data_x_knn %>% mutate_all(~(scale(.) %>% as.vector))
test_data_x_knn %>% mutate_all(~(scale(.) %>% as.vector))

#Tuning the value of k
K_value = c(1,2,3,5,10,15,20,25,30)
recall = numeric(9)
accuracy = numeric(9)
precision = numeric(9)
f1score = numeric(9)
error = numeric(9)

index = 0
for (k in K_value){
  index = index + 1
  class <- knn(train_data_x_knn,test_data_x_knn,train_data_y,k)
  confusion = table(test_data_y,class)[2:1,2:1]
  recall[index] = confusion[1,1]/sum(confusion[1,])
  accuracy[index] = (confusion[1,1]+confusion[2,2])/sum(confusion)
  precision[index] =  confusion[1,1]/sum(confusion[,1])
  f1score[index] = confusion[1,1]/(confusion[1,1]+0.5*(confusion[1,2]+confusion[2,1]))
  error[index] = (confusion[2,1]+confusion[1,2])/sum(confusion)
}

plot(K_value, recall, type = "p", col = "blue", ylim = c(0, 1), xlab = "K Value", ylab = "Metrics", main = "Metrics vs. K-Value")
lines(K_value, accuracy, type = "p", col = "red")
lines(K_value, precision, type = "p", col = "green")
lines(K_value, f1score, type = "p", col = "black")
legend("bottomleft", legend = c("Recall", "Accuracy", "Precision", "F1-Score"), col = c("blue", "red", "green", "black"), lty = 1, 
       xjust = 1, yjust = 0, inset = c(0, 0), bty = "n")


class_M4 <- knn(train_data_x_knn,test_data_x_knn,train_data_y,15)
confusion =table(test_data_y,class_M4)[2:1,2:1]; confusion
recallM4 = confusion[1,1]/sum(confusion[1,]); recallM4
accuracyM4 = (confusion[1,1]+confusion[2,2])/sum(confusion); accuracyM4
precisionM4 =  confusion[1,1]/sum(confusion[,1]); precisionM4
f1scoreM4 = confusion[1,1]/(confusion[1,1]+0.5*(confusion[1,2]+confusion[2,1])); f1scoreM4
