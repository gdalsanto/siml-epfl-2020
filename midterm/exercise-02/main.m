% MGT-448 Statistical Inference and Machine Learning - Midterm project
% EX-2 Generative models
clear all; close all; clc
addpath ./data ./functions

load('train.mat');      % training set
load('test.mat');       % testing set

N = length(label);
% figure('Name','Training data'); plotData(features,label);
% title('Training data');
% figure('Name','Test data');; plotData(features_test,label_test);
% title('Test data');

%% GDA with different covariance matricies 

% vectors with label values where '1' indicates that the label that it
% represents (0 or 1) is verified
label_1 = label';
label_0 = 1 - label';   % [0 1] -> [1 0]

% maximum likelihood estimation
% Bernoulli parameter
phi = sum(label_1)/N;   
% mean
mu1 = label_1'*features/sum(label_1);
mu0 = label_0'*features/sum(label_0);
% covariance matricies
p0 = 0; p1 = 0;     
for i = 1 : length(label)
    if label(i) == 1
        p1 = p1 + (features(i,:)' - mu1')*(features(i,:)' - mu1')';
    else
        p0 = p0 + (features(i,:)' - mu0')*(features(i,:)' - mu0')';
    end
end
sigma1 = p1/sum(label_1);
sigma0 = p0/sum(label_0);

% argmax TRAIN
y_est_1 = zeros(N,1);   % predicted labels
for i = 1 : N
   if gaussan(features(i,:),mu1,sigma1)*phi > gaussan(features(i,:),mu0,sigma0)*(1-phi)
        y_est_1(i) = 1;
   else
        y_est_1(i) = 0;
   end
end
err = sum(label'~=y_est_1);
fprintf('GDA model with different covariance matrix - classification error: \ntraining set %.2f %%\n', err/N*100)

% argmax TEST
for i = 1 : N
   if gaussan(features_test(i,:),mu1,sigma1)*phi > gaussan(features_test(i,:),mu0,sigma0)*(1-phi)
        y_est_1(i) = 1;
   else
        y_est_1(i) = 0;
   end
end
err = sum(label_test'~=y_est_1);
fprintf('testing set %.2f %%\n', err/N*100)


%% GDA with same covariance matricies

% maximum likelihood estimation
% mean and bernoulli parameter are the same as before
% covariance matrix
mu_y = zeros(1000,2);
for i = 1 : length(label)
    if label(i) == 1
        mu_y(i,:) = mu1;
    else
        mu_y(i,:) = mu0;
    end
end
sigma = (features - mu_y)'*(features - mu_y)/N;

% argmax TRAIN
y_est_2 = zeros(N,1);
for i = 1 : N
   if gaussan(features(i,:),mu1,sigma)*phi > gaussan(features(i,:),mu0,sigma)*(1-phi)
        y_est_2(i) = 1;
   else
        y_est_2(i) = 0;
   end
end
err = sum(label'~=y_est_2);
fprintf('GDA model with constant covariance matrix - classification error: \ntraining set %.2f %%\n', err/N*100)

% argmax TEST
for i = 1 : N
   if gaussan(features_test(i,:),mu1,sigma)*phi > gaussan(features_test(i,:),mu0,sigma)*(1-phi)
        y_est_2(i) = 1;
   else
        y_est_2(i) = 0;
   end
end
err = sum(label_test'~=y_est_2);
fprintf('testing set %.2f %%\n', err/N*100)

%% Laplace model 

% separating the features according to the value of their label
indx = find(label);
features0 = features;
features0(indx,:) = [];

indx = find(label==0);
features1 = features;
features1(indx,:) = [];

% maximum likelihood estimation
% mean
mu0 = median(features0);
mu1 = median(features1);
% diversity
b0 = zeros(2,1); 
b1 = zeros(2,1);

b0(1) = sum(label_0.*abs(features(:,1)-mu0(1)))/sum(label_0);
b0(2) = sum(label_0.*abs(features(:,2)-mu0(2)))/sum(label_0);

b1(1) = sum(label_1.*abs(features(:,1)-mu1(1)))/sum(label_1);
b1(2) = sum(label_1.*abs(features(:,2)-mu1(2)))/sum(label_1);

% argmax TRAIN
y_est_3 = zeros(N,1);
for i = 1 : N
   if laplace_dist(features(i,1),mu1(1),b1(1))*laplace_dist(features(i,2),mu1(2),b1(2))*phi >...
           laplace_dist(features(i,1),mu0(1),b0(1))* laplace_dist(features(i,2),mu0(2),b0(2))*(1-phi)
        y_est_3(i) = 1;
   else
        y_est_3(i) = 0;
   end
end

err = sum(label'~=y_est_3);
fprintf('Laplace model - classification error: \ntraining set %.2f %%\n', err/N*100)

% argmax TEST
for i = 1 : N
   if laplace_dist(features_test(i,1),mu1(1),b1(1))*laplace_dist(features_test(i,2),mu1(2),b1(2))*phi >...
           laplace_dist(features_test(i,1),mu0(1),b0(1))* laplace_dist(features_test(i,2),mu0(2),b0(2))*(1-phi)
        y_est_3(i) = 1;
   else
        y_est_3(i) = 0;
   end
end
err = sum(label_test'~=y_est_3);
fprintf('testing set %.2f %%\n', err/N*100)

%% Logistic regression
% log-likelihood maximization though stochastic gradient ascent
X = [ones(N,1) features];
theta = zeros(size(X,2),1);
alpha = 0.001;    % learning rate
numIter = 1000;
theta = gradientAscentLR(X,label,alpha,numIter,theta);

% classification TRAIN
y_est_LR = zeros(N,1);
for i = 1 : N
    h = 1/(1-exp(-X(i,:)*theta));
    if h >= 0.5
        y_est_LR(i)=1;
    else
        y_est_LR(i)=0;
    end        
end
err = sum(label'~=y_est_LR);
fprintf('Logistic Regression - classification error: \ntraining set %.2f %%\n', err/N*100)

% classification TEST
X_test = [ones(N,1) features_test];
for i = 1 : N
    h = 1/(1-exp(-X_test(i,:)*theta));
    if h >= 0.5
        y_est_LR(i)=1;
    else
        y_est_LR(i)=0;
    end   
end
err = sum(label_test'~=y_est_LR);
fprintf('testing set %.2f %%\n', err/N*100)




