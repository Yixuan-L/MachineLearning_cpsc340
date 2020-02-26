using Printf
using Statistics

# Load X and y variable
using JLD
dataName = "citiesBig2.jld"
X = load(dataName,"X")
y = load(dataName,"y")
Xtest = load(dataName,"Xtest")
ytest = load(dataName,"ytest")

# Fit a KNN classifierfloor
k = 1
include("knn.jl")
model = cknn(X,y,k)

# Evaluate training error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@printf("Train Error with %d-nearest neighbours: %.3f\n",k,trainError)

yhat = model.predict(Xtest)
trainError = mean(yhat .!= ytest)
@printf("Train Error with %d-nearest neighbours: %.3f\n",k,trainError)
