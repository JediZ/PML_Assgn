Machine learning: How well do you lift your barbell?
========================================================

### Introduction
In this project, my goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the [website
here](http://groupware.les.inf.puc-rio.br/har). The data for this project come from [this source](http://groupware.les.inf.puc-rio.br/har).


```r
setwd("F:/mygits/PML_Assgn")
library(caret);library(kernlab)

# download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv","pml-training.csv")
# download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv","pml-testing.csv")
toTrain <- read.csv("pml-training.csv")
toTest <- read.csv("pml-testing.csv")
```
### Cleaning up data

I choose to model the motion with a set of 6 variables, 2 each for arm, belt, and dumbell, respectively: roll_arm,pitch_arm,roll_belt,pitch_belt,roll_dumbbell,pitch_dumbbell. So I subset both sets so that only these columns are left.

```r
vars <- c("roll_arm","pitch_arm","roll_belt","pitch_belt","roll_dumbbell","pitch_dumbbell")
keep <- c(vars,"problem_id")
toTest <- toTest[,keep]

keep <- c(vars,"classe")
toTrain <- toTrain[,keep]
## now both sets are cleaned up.
save(toTest,file="pmltest.RData")
save(toTrain,file="pmltrain.RData")
```
### Partitioning the data
Now I subset the original training set (toTrain) into training set (training) and a local testing set (testing) and a validation set (validation). 


```r
setwd("F:/mygits/PML_Assgn");library(caret);library(kernlab)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
load("pmltest.RData");load("pmltrain.RData")
set.seed(54321)

inBuild <- createDataPartition(toTrain$classe, p = 0.7, list=FALSE)
validation <- toTrain[-inBuild,];buildData <- toTrain[inBuild,]
inTrain <- createDataPartition(buildData$classe, p = 0.7, list=FALSE)
testing <- buildData[-inTrain,];training <- buildData[inTrain,]
rm(toTrain);rm(buildData);rm(inBuild);rm(inTrain)
```



### Machine Learning
Next I first build two models (gbm and rf), and then combine them together as final model.In the rf method I will use 3-fold cross validation.


```r
set.seed(12345)

mod1 <- train(classe~., method="gbm",data=training,verbose=FALSE,
              trControl=trainControl(method="repeatedcv",number=5,repeats=1))
save(mod1,file="mod1.RData")
mod2 <- train(classe~., method="rf", data=training,
              trControl=trainControl(method="cv",number=3))
save(mod2,file="mod2.RData")
```

```r
load("mod1.RData");load("mod2.RData")
pred1 <- predict(mod1,testing)
```

```
## Loading required package: gbm
## Loading required package: survival
## Loading required package: splines
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: parallel
## Loaded gbm 2.1
## Loading required package: plyr
```

```r
pred2 <- predict(mod2,testing)
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
predDF <-data.frame(pred1,pred2,classe=testing$classe)
combModFit <- train(classe~., method="rf",data=predDF)
combPred <- predict(combModFit,predDF)
```
Then I calculate the accuracies of the models:

```r
cm1<-confusionMatrix(pred1,testing$classe)
cm2<-confusionMatrix(pred2,testing$classe)
cmx <-confusionMatrix(combPred,testing$classe)
accuracy1 <- cm1$overall[1]
accuracy2 <- cm2$overall[1]
accuracyX <- cmx$overall[1]
```
The accuracies for gbm model is 0.8317, for rf model is 0.9109, and for mixed model is 0.9109. The confusion matrix for the mixed model is:

```r
cmx
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1093   31   36   27   18
##          B   13  710   29    6    6
##          C   27   31  617   30    7
##          D   31   22   29  610    5
##          E    7    3    7    2  721
## 
## Overall Statistics
##                                         
##                Accuracy : 0.911         
##                  95% CI : (0.902, 0.919)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : < 2e-16       
##                                         
##                   Kappa : 0.887         
##  Mcnemar's Test P-Value : 0.00484       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.933    0.891    0.859    0.904    0.952
## Specificity             0.962    0.984    0.972    0.975    0.994
## Pos Pred Value          0.907    0.929    0.867    0.875    0.974
## Neg Pred Value          0.973    0.974    0.970    0.981    0.989
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.265    0.172    0.150    0.148    0.175
## Detection Prevalence    0.293    0.186    0.173    0.169    0.180
## Balanced Accuracy       0.948    0.937    0.916    0.939    0.973
```
Next I predict on the validation dataset

```r
pred1v <- predict(mod1,validation)
pred2v <- predict(mod2,validation)
predvDF <- data.frame(pred1=pred1v,pred2=pred2v)
combPredv <- predict(combModFit,predvDF)
```
Now I evaluate on the validation

```r
cm1v<-confusionMatrix(pred1v,validation$classe)
cm2v<-confusionMatrix(pred2v,validation$classe)
cmxv <-confusionMatrix(combPredv,validation$classe)
accuracy1v <- cm1v$overall[1]
accuracy2v <- cm2v$overall[1]
accuracyXv <- cmxv$overall[1]
```
On the validation set, the accuracies for gbm model is 0.8243, for rf model is 0.9028, and for mixed model is 0.9028. The confusion matrix for the mixed model is:

```r
cmxv
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1544   49   51   49   30
##          B   23 1006   31   16   17
##          C   42   62  894   44    9
##          D   55   17   46  854   11
##          E   10    5    4    1 1015
## 
## Overall Statistics
##                                        
##                Accuracy : 0.903        
##                  95% CI : (0.895, 0.91)
##     No Information Rate : 0.284        
##     P-Value [Acc > NIR] : < 2e-16      
##                                        
##                   Kappa : 0.877        
##  Mcnemar's Test P-Value : 6.71e-07     
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.922    0.883    0.871    0.886    0.938
## Specificity             0.957    0.982    0.968    0.974    0.996
## Pos Pred Value          0.896    0.920    0.851    0.869    0.981
## Neg Pred Value          0.969    0.972    0.973    0.978    0.986
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.262    0.171    0.152    0.145    0.172
## Detection Prevalence    0.293    0.186    0.179    0.167    0.176
## Balanced Accuracy       0.940    0.932    0.920    0.930    0.967
```
Based on this result using just six variables, I expect about 10% error rate for the testing, which is a little bit higher than the 9% error rate during the training.
Finally I make predictions on the real testing sets (toTest)

```r
pred1v <- predict(mod1,toTest[,1:6])
pred2v <- predict(mod2,toTest[,1:6])
predvDF <- data.frame(pred1=pred1v,pred2=pred2v)
combPredv <- predict(combModFit,predvDF)
result <- data.frame(problem_id = toTest$problem_id,
                     prediction = combPredv)
result
```

```
##    problem_id prediction
## 1           1          B
## 2           2          C
## 3           3          B
## 4           4          A
## 5           5          A
## 6           6          E
## 7           7          C
## 8           8          D
## 9           9          A
## 10         10          A
## 11         11          B
## 12         12          C
## 13         13          B
## 14         14          A
## 15         15          E
## 16         16          E
## 17         17          A
## 18         18          B
## 19         19          B
## 20         20          B
```



