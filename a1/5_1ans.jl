using DelimitedFiles
dataTable=readdlm("fluTrends.csv",',')

using Statistics

X=real(dataTable[2:end,:]);
@show X;
maxx=maximum(X);
minx=minimum(X);
avgx=mean(X);
mediax=median(X);

function mode(x)
	# Returns mode of x
	# if there are multiple modes, returns the smallest
	x = sort(x[:]);

	commonVal = [];
	commonFreq = 0;
	x_prev = NaN;
	freq = 0;
	for i in 1:length(x)
		if(x[i] == x_prev)
			freq += 1;
		else
			freq = 1;
		end
		if(freq > commonFreq)
			commonFreq = freq;
			commonVal = x[i];
		end
		x_prev = x[i];
	end

	return commonVal
end
modex=mode(X);
using PyPlot
plot(1:52,X[:,1])
