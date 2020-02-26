using Printf
using Random

# Load X and y variable
using JLD
data = load("basisData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Data is sorted, so *randomly* split into train and validation:
n = size(X,1)
perm = randperm(n)

stop=floor(Int,(n/10))
@show(stop)
validNdx=zeros(10,stop)
trainNdx=zeros(10,n-stop)
zeros(10,n-stop)
for i in 1:10
	validEnd=stop*i
	validStart=validEnd-stop+1
	validNdx[i,:] = perm[validStart:validEnd] # Indices of validation examples
    trainNdx[i,:] = perm[setdiff(1:n,validStart:validEnd)] # Indices of training examples
end


# Find best value of RBF variance parameter,
#	training on the train set and validating on the test set
include("leastSquares.jl")
minErr = Inf
bestSigma = []
for sigma in 2.0.^(-15:15)
	sumErr=0
for i in 1:10
	Xtrain = X[trainNdx[i,:],:]
	ytrain = y[trainNdx[i,:]]
	Xvalid = X[validNdx[i,:],:]
	yvalid = y[validNdx[i,:]]
	# Train on the training set
	model = leastSquaresRBF(Xtrain,ytrain,sigma,10^(-12))

	# Compute the error on the validation set
	yhat = model.predict(Xvalid)
	validError = sum((yhat - yvalid).^2)/(n/2)
	@printf("With sigma = %.3f, validError = %.2f\n",sigma,validError)
    sumErr+=validError
end
	# Keep track of the lowest validation error
	aveErr=sumErr/10
	if aveErr < minErr
		global minErr = aveErr
		global bestSigma = sigma
	end

end

# Now fit the model based on the full dataset
model = leastSquaresRBF(X,y,bestSigma,10^(-12))

# Report the error on the test set
t = size(Xtest,1)
yhat = model.predict(Xtest)
testError = sum((yhat - ytest).^2)/t
@printf("With best sigma of %.3f, testError = %.2f\n",bestSigma,testError)

# Plot model
using PyPlot
figure()
plot(X,y,"b.")
plot(Xtest,ytest,"g.")
Xhat = minimum(X):.1:maximum(X)
Xhat = reshape(Xhat,length(Xhat),1) # Make into an n by 1 matrix
yhat = model.predict(Xhat)
plot(Xhat,yhat,"r")
ylim((-300,400))
