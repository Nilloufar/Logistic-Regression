function [yhat] = predict(sampleRow,w)
yhat =1./(1+exp(-sampleRow*w));
end

