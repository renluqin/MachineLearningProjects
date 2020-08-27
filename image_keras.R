library(OpenImageR) # HOG_apply
library(imager) # load.image, resize
library(kernlab)

# get images data
resize.images<-function(path){
  pages <- list.files(path = path,full.names = TRUE)
  for(x in 1:length(pages)) {
    file <- load.image(pages[x])
    resized <- resize(file, 200, 200)
    save.image(resized, file = gsub("JPG", "jpg", paste(pages[x])))
  }
}
path <- file.path(getwd(), 'car/')
car<-HOG_apply(path, cells = 3, orientations = 6, threads = 1)$hog
car<-as.data.frame(car)
class<-rep(0, nrow(car))
car<-cbind(car,class)

path <- file.path(getwd(), 'cat/')
resize.images(path)
cat<-HOG_apply(path, cells = 3, orientations = 6, threads = 1)$hog
cat<-as.data.frame(cat)
class<-rep(1, nrow(cat))
cat<-cbind(cat,class)

path <- file.path(getwd(), 'flower/')
resize.images(path)
flower<-HOG_apply(path, cells = 3, orientations = 6, threads = 1)$hog
flower<-as.data.frame(flower)
class<-rep(2, nrow(flower))
flower<-cbind(flower,class)

data <- rbind(car, cat, flower)

# Remove pixels with constant value
# s<-apply(data[,names(data)!='class'],2,sd)
# ii<-which(s>1)
# data<-data[,ii]

# PCA
x<-scale(data[,names(data)!='class'])
pca<-prcomp(x)
lambda<-pca$sdev^2
pairs(pca$x[,1:5],col=data[,'class'],pch=as.numeric(data[,'class']))
plot(cumsum(lambda)/sum(lambda),type="l",xlab="q",ylab="proportion of explained variance")
abline(h=0.9,col="blue",lwd=2)
abline(v=25,col="red",lwd=2)
q<-25
x.pca<-scale(pca$x[,1:q])

y<-as.factor(data[,'class'])
n<-nrow(x.pca)

CC<-c(0.001,0.01,0.1,1,10,100,1000,10e4)
N<-length(CC)
M<-10 # nombre de r¨¦p¨¦titions de la validation crois¨¦e
err<-matrix(0,N,M)
for(k in 1:M){
  for(i in 1:N){
    err[i,k]<-cross(ksvm(x=x.pca,y=y,kernel="rbfdot",C=CC[i],cross=5))
  }
}
Err<-rowMeans(err)
plot(CC,Err,type="b",log="x",xlab="C",ylab="CV error")
C<-CC[which.min(Err)]
set.seed(1234)
err<-cross(ksvm(x=x.pca,y=y,kernel="rbfdot",C=C,cross=10))

#
library(keras)
#install_keras()
path <- file.path(getwd(), 'car','/')
pages <- list.files(path = path,full.names = TRUE)
file <- readImage(pages[1])
#resized <- resize(file, 200, 200)
for(x in 2:length(pages)) {
  file <- readImage(pages[x])
  resized <- rbind(file,resized)
}

path <- file.path(getwd(), 'cat','/')
pages <- list.files(path = path,full.names = TRUE)
for(x in 1:length(pages)) {
  file <- load.image(pages[x])
  resized <- rbind(resized,resize(file, 32, 32))
}

path <- file.path(getwd(), 'flower','/')
pages <- list.files(path = path,full.names = TRUE)
for(x in 1:length(pages)) {
  file <- load.image(pages[x])
  resized <- rbind(resized,resize(file, 32, 32))
}

car<-rep(0,485)
cat<-rep(1,590)
flower<-rep(2,521)
y.keras <- c(car, cat, flower)
y.keras <- to_categorical(y.keras, 3)
dim(resized)<-c(1596,32,32,3)

n <- nrow(resized)
set.seed(1234)
train <- sample(1:n,round(2*n/3))
x_train <- resized[train,,,]
y_train <- y.keras[train,]
x_test <- resized[-train,,,]
y_test <- y.keras[-train,]


#vgg16
conv_base <- application_vgg16(weights ="imagenet",include_top = FALSE,input_shape = c(32,32,3))
summary(conv_base)
freeze_weights(conv_base)
model <- keras_model_sequential()%>%
  conv_base%>%
  layer_flatten()%>%
  layer_dense(256, activation ="relu")%>%
  layer_dense(units = 3,"softmax")
summary(model)
freeze_weights(conv_base)
length(model$trainable_weights)
model %>% compile(
  loss ="categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 2e-5),
  metrics = c("accuracy"))
history <- model %>% fit(
  x_train, y_train,
  batch_size = 32,
  epochs = 30,
  shuffle = TRUE,
  validation_split = 0.2
)
plot(history)
model %>% evaluate(x_test, y_test) 



model1<-keras_model_sequential()
#configuring the Model
model1 %>%  
  #defining a 2-D convolution layer
  
  layer_conv_2d(
    filter = 32, kernel_size = c(3,3), padding = "same",
    input_shape = c(32, 32, 3)
  ) %>%
  layer_activation("relu") %>%
  
  layer_conv_2d(filter = 32, kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%
  
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same") %>%
  layer_activation("relu") %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%
  
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  layer_flatten() %>%
  layer_dense(512) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%
  
  layer_dense(3) %>%
  layer_activation("softmax")
summary(model1)
opt <- optimizer_rmsprop(lr = 0.0001, decay = 1e-6)
model1 %>% compile(
  loss = "categorical_crossentropy",
  optimizer = opt,
  metrics = "accuracy"
)
ptm <- proc.time()
history <- model1 %>% fit(
  x_train, y_train,
  batch_size = 32,
  epochs = 60,
  shuffle = TRUE,
  validation_split = 0.2
)


plot(history)
model1 %>% evaluate(x_test, y_test) 
# $`loss`
# [1] 0.176883
# 
# $accuracy
# [1] 0.943609

predictions<-model1 %>% predict_classes(x_test)
model1 %>% save_model_tf("model")
list.files("model")

serialize_model(model, include_optimizer = TRUE)
unserialize_model(model, custom_objects = NULL, compile = TRUE)

