using Printf
using Statistics

# Load data
using JLD
fileName = "vowel.jld"
X = load(fileName,"X")
y = load(fileName,"y")
Xtest = load(fileName,"Xtest")
ytest = load(fileName,"ytest")
(n,d) = size(X)
(t,d) = size(Xtest)
# Fit a decision tree classifier
include("decisionTree_infoGain.jl")
depth = Inf
model = decisionTree_infoGain(X,y,depth)

# Evaluate training error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@printf("Train Error with depth-%d decision tree: %.3f\n",depth,trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean(yhat .!= ytest)
@printf("Test Error with depth-%d decision tree: %.3f\n",depth,testError)

# Fit a random tree classifier
include("randomTree.jl")
depth = Inf
model = randomTree(X,y,depth)

# Evaluate training error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@printf("Train Error with depth-%d random tree: %.3f\n",depth,trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean(yhat .!= ytest)
@printf("Test Error with depth-%d random tree: %.3f\n",depth,testError)




k=50

modelforest=randomForest(X,y,depth,k)

prehat=zeros(n,k)
yhatt=zeros(n)
for i in 1:k
 prehat[:,i]=modelforest[i].predict(X)
end
for j in 1:n
    yhatt[j]=mode(prehat[j,:])
end
trainError=mean(yhatt .!=y)
@printf("Train Error with depth-%d random forest: %.3f\n",depth,trainError)

prehattest=zeros(t,k)
yhattest=zeros(n)
 for i in 1:k
     prehattest[:,i]=modelforest[i].predict(Xtest)
end
for j in 1:t
    yhattest[j]=mode(prehattest[j,:])
end

testError = mean(yhattest .!= ytest)
@printf("Test Error with depth-%d random forest: %.3f\n",depth,testError)
