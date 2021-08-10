function theta = gradientAscentLR(X,y,alpha,numIter,theta)
    for k = 1 : numIter
    for i = 1 : size(X,1)
        h = 1/(1+exp(-X(i,:)*theta));
        for j = 1 : size(theta,1)
            theta(j) = theta(j) + alpha*(y(i)-h)*X(i,j);
        end
    end
%     h = 1/(1+exp(-X*theta));
%     for j = 1 : size(theta,1)
%         theta(j) = theta(j) + alpha*(y-h)*X(:,j);
%     end
%     end
end