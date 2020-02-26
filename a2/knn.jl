include("misc.jl") # Includes GenericModel typedef

function knn_predict(Xhat,X,y,k)
  (n,d) = size(X)
  (t,d) = size(Xhat)
  k = min(n,k) # To save you some debuggin

  DisM=distancesSquared(X,Xhat)
#  @show(DisM
  p=zeros(n,t)
  yhat=zeros(t)
tobemode=zeros(t,k)
 for i in 1:t
   p[:,i]=sortperm(DisM[:,i])

   yhat[i]=mode(y[p[1:k,i].|>Int])
  #yhat[i]=mode(y[p[i,1:k]].|>Int)

end
@show(size(DisM))



  return yhat
end

function knn(X,y,k)
	# Implementation of k-nearest neighbour classifier
  predict(Xhat) = knn_predict(Xhat,X,y,k)
  return GenericModel(predict)
end

function cknn(X,y,k)
	# Implementation of condensed k-nearest neighbour classifier
	(n,d) = size(X)
	Xcond = X[1,:]'
	ycond = [y[1]]
	for i in 2:n
    		yhat = knn_predict(X[i,:]',Xcond,ycond,k)
    		if y[i] != yhat[1]
			Xcond = vcat(Xcond,X[i,:]')  #ADD
			push!(ycond,y[i])
    		end
	end
@show(size(Xcond))
	predict(Xhat) = knn_predict(Xhat,Xcond,ycond,k)
	return GenericModel(predict)
end
