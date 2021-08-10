function acc = accuracy(label,cluster)
% acc = accuracy(labels,idx)
% labels:   true labels from the dataset
% cluster:  vector of estimated clusters (labels)

k = max(label);
n_d = length(label);  % number of documents

% reordening of the labels 
for i = 1 : k
    temp1 = find(label==i);
    temp2 = find(cluster==i);
    indx_labels{i} = temp1;
    indx_clusters{i} = temp2;
end

% computes all possible combination 
p = perms((1:k));
n_p = length(p);

com=zeros(1,k);
for i = 1 : n_p
    for j=1:k
       com(i,j) = sum(ismember(indx_clusters{p(i,j)},indx_labels{j}));
    end 
end 
 
% find the combination that best represents the order of the labels
[max_val,~] = max(sum(com,2));

acc=max_val/n_d*100; 
