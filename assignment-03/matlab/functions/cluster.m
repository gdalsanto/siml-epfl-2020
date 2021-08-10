function cluster = cluster( T, n_c, niter )
% cluster = cluster( T, n_c, niter )
% T: matrix of word counts (bag of words)
% n_c: number of clusters (topics)
% niter: number of iteration 

n_d = size(T,1);    % total nuber of documents

% mixture parameters
mu_jc = rand(size(T,2),n_c); 
mu_jc = mu_jc./repmat(sum(mu_jc,2),1,n_c);
pi_c=rand(n_c,1); 
pi_c=pi_c/sum(pi_c,1);

% distribution of the latent variable
gamma=ones(size(T,1),n_c); 


mu_jc_updated=ones(size(T,2),n_c);
pi_c_updated=ones(n_c,1);
iter=0;
while(~(isequal(mu_jc,mu_jc_updated) && isequal(pi_c,pi_c_updated)) && iter<niter)
    
    iter=iter+1;
    
    mu_jc_updated=mu_jc;
    pi_c_updated=pi_c;
    
   % E-step 
    for i = 1 : size(T,1)
        for c=1:n_c
            gamma(i,c) = pi_c(c)*prod(mu_jc(:,c)'.^(T(i,:)));
        end
        gamma(i,:) = gamma(i,:)/sum(gamma(i,:));
    end
    
    

	% M-step
    pi_c = sum(gamma,1)'/n_d;

    for j = 1 : size(T,2)
        for c = 1 : n_c
            mu_jc(j,c) = sum(gamma(:,c).*T(:,j))/sum(gamma(:,c).*(sum(T,2)));
        end
    end
        
end

cluster = zeros(size(T,1),1);

for i=1:size(T,1)
    [~,I]=max(gamma(i,:));
    cluster(i)=I;
end

end

