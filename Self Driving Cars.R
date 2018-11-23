getwd()
setwd("C:/Users/Metika/Desktop/R direct")

library("caret")
library("rpart")
library("rpart.plot")
library("ggplot2")
library("randomForest")

s <- read.csv("selfdrivingcars_images.csv")

#EDA
dim(s) 
head(s)
tail(s)
sum(is.na(s)) #no missing values as sum is zero
names(s)
str(s)

#removing the first coloumn id
s <- s[-1]

#models on single train-test split************************************************************

set.seed(24) #setting seed

rows <- sample(nrow(s)) #shuffling rows
head(rows)

s <- s[rows, ] #shuffling the data set

split <- round(rows*0.75) #the number at which the data set has to be split

train <- s[1:split, ]
test <- s[(split + 1):nrow(s), ]

#knn________________________________________
model1 <-  knn3(sign_type~., train, k=5)
pred1 <- predict(model1,test, type = "class")
#install.packages('e1071', dependencies=TRUE) #for confusion matrix

confusionMatrix(test$sign_type,pred1, mode = "prec_recall") #accuracy = 0.869  
#precission by class- pedestrian(0.7037), speed(0.9583), stop(0.9394)

#decison tree________________________________________
tree1 <- rpart(sign_type~., data = train, method = "class")
rpart.plot(tree1, extra = 4)
pred2 <- predict(tree1, test, type = "class")
confusionMatrix(test$sign_type, pred2, mode = "prec_recall") #accuracy = 0.8095
#precission by class- pedestrian(0.8519), speed(0.7500), stop(0.8182)

#random forest________________________________________
rf1 <- randomForest(sign_type~., data = train, method = "class")
pred5 <- predict(rf1, test, type = "class")
confusionMatrix(test$sign_type, pred5, mode = "prec_recall") #accuracy = 0.9881
#precission by class- pedestrian(1.0000), speed(0.9583), stop(1.0000)

#models with k-fold cross validation*************************************************************
trControl <- trainControl(method = "cv", number = 5) #number of folds created is 5

#knn________________________________________
set.seed(5)
model2 <- train(sign_type~., method = "knn", data = train, trControl = trControl) 
model2 #the model chosen is with k=7 as that has the maximum accuracy

pred <- predict.train(model2,newdata = test, type = "raw")
pred

confusionMatrix(test$sign_type,pred, mode = "prec_recall")  #accuracy =  0.9405   
#precission by class- pedestrian(0.8333), speed(0.9667), stop(1.0000)

#plotting the outcomes of train object
plot.train(model2) #in general, as the number of neighbours increases, the accuracy increases 

#decission tree________________________________________
set.seed(31)
tree2 <- train(sign_type~., method = "rpart", data = train, trControl = trControl)
tree2 #the model chosen is with cp (complexity parameter) = 0.04938272 as that has maximum accuracy

pred3 <- predict.train(tree2, newdata = test, type = "raw")
pred3

confusionMatrix(test$sign_type,pred3, mode = "prec_recall") #accuracy = 0.9048 
#precission by class- pedestrian(0.7917), speed(0.9667), stop(0.9333)

plot.train(tree2) #as the complexity parameter increases, accuracy decreaes

#random forest________________________________________
set.seed(124)
rf2 <- train(sign_type~., method = "rf", data = train, trControl = trControl)
rf2 #the model was chosen with mtry = 2 as that has the maximum accuracy

pred4 <- predict.train(rf2,newdata = test, type = "raw")
pred4

confusionMatrix(test$sign_type,pred4,mode = "prec_recall")#accuracy = 0.9831
#precission by class- pedestrian( 1.0000), speed(1.0000), stop(0.9600)

plot.train(rf2) #as the number of randomly selected predictors increases, accuracy decreases
