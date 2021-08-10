function p = gaussan(features,mu,sigma)
%   GAUSSAN PROBABILITY DISTRIBUTION
%   p = gaussan(features,mu,sigma)
    p = exp(-(features-mu)*inv(sigma)*(features-mu)'/2)/((2*pi)*sqrt(det(sigma)));
end