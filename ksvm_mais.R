mais <- read.csv(file = "mais_train.csv", header = TRUE,na.strings="?",row.names=NULL)
head(mais)
mais <- subset(mais, select = -X )
#mais <- as.data.frame(scale(mais))
library(e1071)
#install.packages("kernlab")
library(kernlab)
K <- 10
folds = sample(1:K,nrow(mais),replace=TRUE)
CV <- matrix(data=0,nrow = 10,ncol = 9)

kernel<-c("rbfdot","polydot","vanilladot","tanhdot","laplacedot","besseldot","anovadot","splinedot")

I<-9
for(i in (1:I)){
  for(k in (1:K) ){
    reg.kcross <- ksvm(yield_anomaly ~ . , mais[folds!=k,],kernel=kernel[1])
    pred.kcross <- predict(reg.kcross,newdata = as.data.frame(mais[folds==k,]))
    CV[k,1]<- mean((mais$yield_anomaly[folds==k]-pred.kcross)^2)
  }
}

boxplot(CV)
