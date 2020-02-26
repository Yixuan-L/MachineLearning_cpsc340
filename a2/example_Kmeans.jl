# Load data
using JLD
X = load("clusterData.jld","X")

# K-means clustering
k = 2
include("kMeans.jl")
#gcfferror=zeros(i,2)
#for i in 1:50F5TRTGFTR55R5RTR5R5T5TR45TTR54YTT5T5RT5T5T55TR4

model = kMeans(X,k,doPlot=true)
y = model.predict(X)
# ferror[i,1]=model.f
# ferror[i,2]=model.W
# end
# errororder=sortperm(ferror[:,1])
# ww=ferror[errororder,2]

include("clustering2Dplot.jl")
clustering2Dplot(X,y,model.W)
gcf()
