% SIML HW03 ex1: PCA

clear; clc; close all;
addpath ./data ./functions

%% Data loading
load 'leaf.csv';

X = leaf(:,(3:16)); % we take off the 2 first column

%% PCA

% Data-centering + covariance
Mu = mean(X); % computes the mean of the data
Sigma = std(X-Mu); % computes the standard deviation of the data
X_cent = (X - Mu)./Sigma; % centers and scales the data

C = cov(X_cent); % covariance matrix
% C = X_cent'*X_cent/size(X_cent,1)

% Eigenvectors
[V,D] = eig(C); % computes eigenvectors and eigenvalues matrices
[EigenValues,ind] = sort(diag(D),'descend'); % extracts + orders eigvalues
EigenVectors = V(:,ind); % extracts + orders eigvectors

% Projection
k = 2; % nb of eigenvectors to cutoff (2 is given from the assignment)
Ap = (EigenVectors(:,(1:k)))'; % executes the cutoff
Yproj = Ap*(X_cent)'; % projection of the data

% Reconstruction + error
[N,M] = size(X_cent); % matrix size
X_tilde = Yproj' * Ap; % computes the recontructed data
MSE = sum(sum((X_cent-X_tilde).^2)) / (N*M);% computes the reconstruction error

%% Plots

% Plotting the eigenvalues
[eigval_asc,ind] = sort(diag(D),'ascend');

plot(eigval_asc, 'LineWidth', 2)
title('Eigenvalues')
ylabel('Eigenvalues')
xlabel('Eigenvector index')

set(gca,'xtick',1:length(eigval_asc));
set(gca,'xlim',[1,length(eigval_asc)]);
grid on

% Plotting the explained Variance from EigenValues
% This helps us to visualize the necessary number of eigenvectors to reach 
% the desired amount of kept information
ExpVar = EigenValues/sum(EigenValues); % normalization
CumVar = cumsum(ExpVar); % variance is here cumulated

figure
plot(CumVar, '--r', 'LineWidth', 2) ; hold on;
bar(ExpVar, 0.2, 'b');
title('Explained Variance from EigenValues')
ylabel('% Cumulative Variance Explained')
xlabel('Eigenvector index')

set(gca,'xtick',1:length(ExpVar));
set(gca,'xlim',[0.5,length(ExpVar)]);
set(gca,'ylim',[0,1]);
grid on

% Plotting the projection
figure
plot(Yproj(1,:), Yproj(2,:),'b.','MarkerSize', 10)
title('Projected data')
ylabel('Eigenvector \alpha_2')
xlabel('Eigenvector \alpha_1')


