using Printf
using Statistics

# Load X and y variable
using JLD
dataName = "citiesBig1.jld"
X = load(dataName,"X")
y = load(dataName,"y")
# Xtest = load(dataName,"Xtest")
# ytest = load(dataName,"ytest")
Xtest=[-123.6 49.15]
# Fit a KNN classifierfloor
k = 1
include("knn.jl")
model =  cknn(X,y,k)
#modell=  knn(X,y,k)
# Evaluate training error
yhat =@time  model.predict(X)
trainError = mean(yhat .!= y)
@printf("Train Error with %d-nearest neighbours: %.3f\n",k,trainError)

#yhatl =@time  modell.predict(X)



yhat = model.predict(Xtest)
