
X = load("citiesSmall.jld","X")
y = load("citiesSmall.jld","y")

if(X[:,2] .> 37.669)

 if(X[:,1] .> -96.090)
  yhat=1
 else
  yhat=2
 end

elseif(X[:,1] .> -115.578)
  yhat=2
     else
  yhat=1
 end
end
