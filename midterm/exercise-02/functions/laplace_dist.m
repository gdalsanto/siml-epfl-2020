function p = laplace_dist(features,mu,b)
%   LAPLACE PROBABILITY DISTRIBUTION
% p = laplace_dist(features,mu,b)
    p = exp(-abs(features-mu)./b')/2./b';
end