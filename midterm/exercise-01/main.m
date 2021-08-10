% MGT-448 Statistical Inference and Machine Learning - Midterm project
% EX-1 SVM in Spam Classification
close all; clear all; clc
addpath ./data ./functions
load('data1.mat'); 

%% SVM with linear kernel

C = 1;
figure('Name','Data set')
plotData(X,y);
hold on

model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
figure('Name','Linear kernel')
visualizeBoundaryLinear(X, y, model);

%% SVM with non linear kernel
clear variables; clc
C = 10; sigma = 0.1;
load('data2.mat');
model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
figure('Name','Gaussian kernel')
visualizeBoundary(X, y, model);

%% Spam Classification
clear variables; clc

fileContents = readFile('emailSample1.txt');
wordIndices = processEmail(fileContents);
x = emailFeatures(wordIndices);

%% Training SVM for spam classification
clear variables; clc

load('spamTrain.mat');
load('spamTest.mat');

C = 0.1;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);

p = svmPredict(model, X);
fprintf('Accuracy: %.2f %%\n', sum(p==y)/length(y) * 100);

p = svmPredict(model, Xtest);
fprintf('Accuracy: %.2f %%\n', sum(p==ytest)/length(ytest) * 100);




