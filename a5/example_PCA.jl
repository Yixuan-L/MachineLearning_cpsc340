using DelimitedFiles

# Load data
dataTable = readdlm("animals.csv",',')
X = float(real(dataTable[2:end,2:end]))
(n,d) = size(X)

# Standardize columns
include("misc.jl")
(X,mu,sigma) = standardizeCols(X)
include("PCA.jl")
model=PCA(X,2)
Z=model.compress(X)
# Plot matrix as image
using PyPlot
figure(2)
clf()
plot(Z[:,1],Z[:,2],".")
for i in 1:n
	annotate(dataTable[i+1,1],xy=[Z[i,1],Z[i,2]],xycoords="data")
end

# imshow(X)
#
# # Show scatterplot of 2 random features
# j1 = rand(1:d)
# j2 = rand(1:d)
# figure(2)
# clf()
# plot(X[:,j1],X[:,j2],".")
# for i in rand(1:n,10)
#     annotate(dataTable[i+1,1],
# 	xy=[X[i,j1],X[i,j2]],
# 	xycoords="data")
# end
w=model.W
ww=zeros(85)

for i in 1:85
	ww[i]=abs(w[2,i])
end
@show(findmax(ww))
