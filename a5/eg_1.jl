using Printf
using Statistics

# Load X and y variable
using JLD
data = load("basisData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Fit a least squares model
include("leastSquares.jl")
#lambda = 1
lambda=10^(-6)
p=3
sigma=1
#model = leastSquaresBiasL2(X,y,lambda)
#model=leastSquaresKernelBasis(X,y,p,lambda)
 model=leastSquaresRBFBasis(X,y,lambda,sigma)
# Evaluate training error
yhat = model.predict(X)
trainError = mean((yhat - y).^2)
@printf("Squared train Error with least squares: %.3f\n",trainError)

# Evaluate test error
yhattest = model.predict(Xtest)
testError = mean((yhattest - ytest).^2)
@printf("Squared test Error with least squares: %.3f\n",testError)

# Plot model
using PyPlot
figure(2)
plot(X,y,"b.")
Xhat = minimum(X):.1:maximum(X)
yhat = model.predict(Xhat[:,:])
plot(Xhat,yhat,"g")
gcf()
