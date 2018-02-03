###############################################################
#                                                             #
#            Ensemble for Regression                          #
#                                                             #
###############################################################
#                                                             #
# Credit: Gursahib Singh                                      #
# Email : gursahib.kvm@gmail.com                              #
#                                                             #
###############################################################
#                                                             #
#                                                             #
# This script do the following:                               #
# 1. Load the Data                                            #
# 2. Partition the data into Train/Test set                   #
# 3. Train the NeuralNetwork Model                            #
# 4. Test                                                     #
# 5. Evaluate on : Correlation, Regression, RMSE, Accuracy.   # 
# 6. Finally Saving the results.                              #
#                                                             #
###############################################################


#--------------------------------------------------------------
# Step 0: Start; Getting the starting time
#--------------------------------------------------------------
cat("\nSTART\n")
startTime = proc.time()[3]
startTime

modelName="ensemble4"


#--------------------------------------------------------------
# Step 1: Library Inclusion
#--------------------------------------------------------------

library(ggplot2)
library(RRF)

library(survival)
library(plotmo)
library(plotrix)
library(TeachingDemos)
library(lattice)
library(splines)
library(parallel)
library(pls)
library(earth)

library(party)
library(mboost)
library(grid)
library(mvtnorm)
library(modeltools)
library(stats4)

library(grid)
library(mvtnorm)
library(modeltools)
library(stats4)
library(strucchange)
library(zoo)
library(party)

library(splines)
library(foreach)
library(gam)

library(rpart)
library(Cubist)
library(hmeasure)

library(party)

library(party)
library(MASS)
library(doParallel)
library(iterators)
library(randomGLM)

library(kernlab)

library(plsdof)
library(MASS)
library(plsRglm)
library(RWeka)

#--------------------------------------------------------------
# Step 2: Input File
#--------------------------------------------------------------
InputDataFileName="regressionDataSet.csv"
InputDataFileName

training = 70      # Defining Training Percentage; Testing = 100 - Training



#--------------------------------------------------------------
# Step 3: Data Loading
#--------------------------------------------------------------
cat("\nStep 3: Data Loading")
dataset <- read.csv(InputDataFileName)      # Read the datafile
dataset <- dataset[sample(nrow(dataset)),]  # Shuffle the data row wise.

head(dataset)   # Show Top 6 records
nrow(dataset)   # Show number of records
names(dataset)  # Show fields names or columns names



#--------------------------------------------------------------
# Step 4: Count total number of observations/rows.
#--------------------------------------------------------------
cat("\nStep 4: Counting dataset")
totalDataset <- nrow(dataset)
totalDataset



#--------------------------------------------------------------
# Step 5: Choose Target variable
#--------------------------------------------------------------
cat("\nStep 5: Choose Target Variable")
target  <- names(dataset)[1]

target
#--------------------------------------------------------------
# Step 6: Choose inputs Variables
#--------------------------------------------------------------
cat("\nStep 6: Choose Inputs Variable")
inputs <- setdiff(names(dataset),target)
inputs
length(inputs)

#Feature Selection
#n=4
#inputs <-sample(inputs, n)



#--------------------------------------------------------------
# Step 7: Select Training Data Set
#--------------------------------------------------------------
cat("\nStep 7: Select training dataset")
trainDataset <- dataset[1:(totalDataset * training/100),c(inputs, target)]
head(trainDataset)    # Show Top 6 records
nrow(trainDataset)    # Show number of train Dataset



#--------------------------------------------------------------
# Step 8: Select Testing Data Set
#--------------------------------------------------------------
cat("\nStep 8: Select testing dataset")
testDataset <- dataset[(totalDataset * training/100):totalDataset,c(inputs, target)]
head(testDataset)
nrow(testDataset)



#--------------------------------------------------------------
# Step 9: Model Building (Training)
#--------------------------------------------------------------
formula <- as.formula(paste(target, "~", paste(c(inputs), collapse = "+")))
model1   <- RRF(formula,trainDataset)

formula2 <- as.formula(paste(target, "~", paste(c(inputs), collapse = "+")))
model2   <- earth(Class~.,data=trainDataset)

formula3 <- as.formula(paste(target, "~", paste(c(inputs), collapse = "+")))
model3   <- blackboost(formula,data=trainDataset)

formula4 <- as.formula(paste(target, "~", paste(c(inputs), collapse = "+")))
model4 <- ctree(Class ~ ., data = trainDataset,controls = ctree_control(maxsurrogate = 3))

formula5 <- as.formula(paste(target, "~", paste(c(inputs), collapse = "+")))
model5   <- gam(formula,family=gaussian(),trainDataset)

formula6 <- as.formula(paste(target, "~", paste(c(inputs), collapse = "+")))
model6 <- cubist(x=trainDataset[,-1],y=trainDataset$Class,committees = 10)

formula7 <- as.formula(paste(target, "~", paste(c(inputs), collapse = "+")))
model7   <- glmboost(formula,data=trainDataset)

formula8 <- as.formula(paste(target, "~", paste(c(inputs), collapse = "+")))
model8   <- randomGLM(trainDataset,trainDataset$Class)

formula9 <- as.formula(paste(target, "~", paste(c(inputs), collapse = "+")))
model9   <- ksvm(formula,data=trainDataset)

formula10 <- as.formula(paste(target, "~", paste(c(inputs), collapse = "+")))
model10   <- M5P(formula,data=trainDataset)

#--------------------------------------------------------------
# Step 10: Extracting Predicted
#--------------------------------------------------------------
Predicted1 <- predict(model1, testDataset)
head(Predicted1)

Predicted2 <- predict(model2, testDataset)
head(Predicted2)

Predicted3 <- predict(model3, testDataset)
head(Predicted3)

Predicted4 <- predict(model4, testDataset)
head(Predicted4)

Predicted5 <- predict(model5, testDataset)
head(Predicted5)

Predicted6 <- predict(model6, testDataset)
head(Predicted6)

Predicted7 <- predict(model7, testDataset)
head(Predicted7)

Predicted8 <- predict(model8, testDataset)
head(Predicted8)

Predicted9 <- predict(model9, testDataset)
head(Predicted9)

Predicted10 <- predict(model10, testDataset)
head(Predicted10)

Predicted <- predict(model1, testDataset)

head(Predicted)
for(i in 1:3581){
  Predicted[i]=(Predicted1[i]+Predicted3[i]+Predicted5[i]+Predicted7[i])/4
}

head(Predicted)

#--------------------------------------------------------------
# Step 11: Extracting Actual
#--------------------------------------------------------------
cat("\nStep 11: Extracting Actual")
Actual <- as.double(unlist(testDataset[target]))
head(Actual)

#--------------------------------------------------------------
# Step 12: Model Evaluation
#--------------------------------------------------------------
cat("\nStep 12: Model Evaluation")

# Step 12.1: Correlation
r <- cor(Actual,Predicted )
r <- round(r,2)
r

# Step 12.2: RSquare
R <- r * r 
R <- round(R,2)
R

# Step 12.3: MAE (Mean Absolute)
mae <- mean(abs(Actual-Predicted))
mae <- round(mae,2)
mae

# Step 12.4: Accuracy
#accuracy <- mean(abs(Actual-Predicted) <=1)     #acceptable error=100
#accuracy <- round(accuracy,4) *100
#accuracy
count=0
error=abs(Actual-Predicted)
for(i in 1:3581){
  if(error[i]<100){
    count=count+1
  }
}
n=nrow(testDataset)
accuracy<-(count/n)*100
accuracy



# Step 12.6: Total Time
totalTime = proc.time()[3] - startTime
totalTime

# Step 12.7: Save evaluation resut 
result <- data.frame(modelName,r,R,mae,accuracy, totalTime)[1:1,]
result



#--------------------------------------------------------------
# Step 13: Writing to file
#--------------------------------------------------------------
cat("\nStep 13: Writing to file")

# Step 13.1: Writing to file (evaluation result)
write.csv(result, file=paste(modelName,"-Evaluation-Result.csv",sep=''), row.names=FALSE)

# Step 13.2: Writing to file (Actual and Predicted)
write.csv(data.frame(Actual,Predicted), file=paste(modelName,"-ActualPredicted-Result.csv",sep=''), row.names=FALSE)



#--------------------------------------------------------------
#                           END 
#--------------------------------------------------------------

