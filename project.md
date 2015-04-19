Predicting How Well Exercise is Performed
=========================================

Author: Harry J. Braviner


This document is my project writeup for the Coursera Data Science course Practical Machine Learning.


The dataset used for this model is described [here](http://groupware.les.inf.puc-rio.br/har).

## Cleaning up the dataset

Set the seed to aid reproducibility.

```r
set.seed(1357)
```

We begin by reading in our training dataset.

```r
training <- read.csv('pml-training.csv')
nrow(training)
```

```
## [1] 19622
```
A glance at the data reveals several variables in which almost all entries as `NA` (e.g. `max_roll_dumbell`) and several that have missing values masked by their conversion to factor variables (e.g. `kurtosis_roll_belt`).

```r
cleanCols <- !(apply(training, 2, function(x) {any(is.na(x) || x == "" || x == "#DIV/0!")}))
numCleanCols <- sum(cleanCols)
```
Removing these columns with poor quality data, we are left with 60 variables, and we build a reduced training set which we call `train` for brevity.
I also remove the `user_name` field, since I don't want to train to individual users,
the timestamp fields, since I suspect these don't matter (a real user may be more tired at certain times, but these subjects are being told to make deliberate mistakes),
and the `X`, `new_window` and `num_window` fields, since I don't really know what these are, which is reason enough to remove them.

```r
train <- training[,cleanCols]
train <- train[,8:ncol(train)]
```
We're left with the following variables (note that `classe` is what we want to predict):

```r
names(train)
```

```
##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [52] "magnet_forearm_z"     "classe"
```

We'll use cross-validation to assess the error of our model, but as a final check it would be good to have a *verification* dateset.
Since we have 19622 observations, setting aside 10% for verification should leave us with plenty of data to build the model on.

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
partition1 <- createDataPartition(y=train$classe, times=1, p=0.9, list= FALSE)
train1 <- train[ partition1, ]
train2 <- train[-partition1, ]
```

## Building the model

52 variables is still a very large amount to fit a model to.
We'd likely overfit to the noise, and building the model would take a very long time.
We'll use **principal component analysis** to select variables explaining 80% of the variation.

```r
pp <- preProcess(train1[,-53], method = "pca", thresh = 0.8)
pp
```

```
## 
## Call:
## preProcess.default(x = train1[, -53], method = "pca", thresh = 0.8)
## 
## Created from 17662 samples and 52 variables
## Pre-processing: principal component signal extraction, scaled, centered 
## 
## PCA needed 12 components to capture 80 percent of the variance
```
This has reduced us to 12 variables (that are linear combinations of the original 52).
I'm pretty comfortable with using this process, since I suspect that angles of different joints in the subject's body could easilly be linearly related during an exercise.

We must not assess the accuracy of our model on the test set and then use that to refine the model.
We should instead use a form of **cross-validation**, in which we build the model using part of the training set, and assess its accuracy on another part, and then repeat for different choices of subsets.
The form of cross-validation I shall use is **20-fold subsampling**.
This is set via the a `trainControl` object that shall be passed to the `train` function later.

```r
TC <- trainControl(method = "cv", number = 10, p=1.0)
```
Now we pre-process the training variables into the principal components, and build our model.
We shall build a **random forrest**, in which `ntree` classification trees are built, and a 'vote' over these is taken when making a prediction.

```r
train1PC <- predict(pp, train1[,-53])
## Note that trainPC contains only 12 variables
str(train1PC)
```

```
## 'data.frame':	17662 obs. of  12 variables:
##  $ PC1 : num  4.41 4.38 4.39 4.38 4.39 ...
##  $ PC2 : num  1.59 1.61 1.6 1.66 1.63 ...
##  $ PC3 : num  -2.77 -2.77 -2.76 -2.73 -2.77 ...
##  $ PC4 : num  0.849 0.846 0.845 0.835 0.843 ...
##  $ PC5 : num  -1.33 -1.26 -1.27 -1.28 -1.27 ...
##  $ PC6 : num  2.14 2.07 2.11 2.15 2.09 ...
##  $ PC7 : num  -0.166 -0.148 -0.176 -0.19 -0.157 ...
##  $ PC8 : num  -2.72 -2.75 -2.72 -2.72 -2.75 ...
##  $ PC9 : num  -0.01106 -0.01242 0.01059 -0.03623 -0.00891 ...
##  $ PC10: num  -0.325 -0.315 -0.324 -0.344 -0.296 ...
##  $ PC11: num  0.645 0.653 0.661 0.604 0.641 ...
##  $ PC12: num  -0.791 -0.811 -0.837 -0.8 -0.847 ...
```

```r
startTime <- proc.time()
model1 <- train(train1$classe ~ ., data = train1PC, method = "rf")
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
modelBuildTime <- (proc.time() - startTime)[3]
FM <- model1$finalModel
```
It took 24 minutes to build the model.

## The error rate

The final model consists of the random forest *rebuilt using all of the training data*.
Attempting to extract an accuracy from this same data set gives 100%:

```r
sum(predict(FM, newdata = train1PC) == train1$classe) / nrow(train1)
```

```
## [1] 1
```
This is deceptive and should *not* be taken to be the accuracy of the model.


The **out-of-bag** estimate for the error rate is more informative.

```r
FM
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 3.23%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4925   26   42   21    8  0.01931501
## B   68 3270   59   13    8  0.04330018
## C   16   46 2981   30    7  0.03214286
## D   19   12  123 2737    4  0.05457686
## E    4   24   19   21 3179  0.02094241
```
We see that the out-of-bag error rate is estimated to be 3%.
The `train` function estimates this by cross-validation.
Trees are built using a subset of the training data (90% of it, since we're using 10-fold validation) and then the remaining 10% of the data is tested against this tree.
This is repeated for each of the 10 folds to generate this estimate of the error rate.

Let's see how accurately this model makes predictions on our validation set.

```r
train2PC <- predict(pp, train2[,-53])
valErrRate <- sum(predict(FM, newdata = train2PC) == train2$classe) / nrow(train2)
valErrRate
```

```
## [1] 0.9729592
```
So we have an error rate of 2.7% on our validation set.
This is reassuringly close to our out-of-bag error rate.
