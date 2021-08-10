function sim = gaussianKernel(x1, x2, sigma)
    if length(x1) ~= length(x2)
        error('length of vextors x1 and x2 must be the same');
    end
    sim = 0;
    sim = exp(-sum((x1-x2).^2)/(2*sigma^2));
end

