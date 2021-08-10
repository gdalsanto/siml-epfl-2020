% SIML HW03 ex2: EM algorithm
clear variables; close all; clc
addpath ./data ./functions

load data.csv
T = data(:,1:100);
label = data(:,101);

niter = 1000;       % number of iterations
n_c = 4;            % number of clusters
topic = cluster(T,n_c,niter);

acc = accuracy(label,topic);
fprintf('Accuracy %.2f %%\n',acc);

