% SIML HW03 ex3: MCMC algorithm
clear all; close all; clc
addpath ./data ./functions

rng('default')
s = rng;
% randomness
% seed = 369;
% randn('state',seed);
% rand('state',seed);

% open and read the text file
fileID = fopen('data2.txt','r');
formatSpec = '%f';

x = fscanf(fileID,formatSpec);
x = x(:);   % make sure it's a column
K = 2;
N = length(x);
n_iter = 1e3;          % number of iterations 

% distribution parameters
delta = ones(1,K);      % Dirichlet
a = rand(1,K);          % Gamma - shape  
b = rand(1,K);          % Gamma - rate
alpha = rand(1,K);      % Normal - precision
m = rand(1,K);          % Normal - mean

%% 1. set initial values rho, mu, phi

rho = 0.5*ones(1,K) + [-0.1, 0.1]; 
phi = [1/var(x), 1/var(x)];                   
mu = [mean(x)-0.1, mean(x)+0.1];   

%% Updating the parameters

for i = 1 : n_iter
    % 2. Update z
    [z,nk] = p_z(x,rho,mu,phi,K,N);
    % 3. Update rho
    delta_conj = delta + nk;
    temp = rand(K,1);
    temp = temp/sum(temp);
    rho = dirpdf(delta_conj,1);
    % 4. Update phi
    a_conj = a + nk;
    temp = [sum((x(z==1)-mu(1)).^2),...
        sum((x(z==2)-mu(2)).^2)];
    b_conj = b + temp;
    phi = [gamrnd(a_conj(1)/2,2/b_conj(1)),...
        gamrnd(a_conj(2)/2,2/b_conj(2))];
    % 5. Update mu 
    alpha_conj = alpha + nk;
    m_conj = [(alpha(1)*m(1)+nk(1)*mean(x(z==1)))/(alpha(1)+nk(1)),...
        (alpha(2)*m(2)+nk(2)*mean(x(z==2)))/(alpha(2)+nk(2))];
    mu = [normrnd(m_conj(1),1/(alpha_conj(1)*phi(1))),...
        normrnd(m_conj(2),1/(alpha_conj(2)*phi(2)))];
end

%% visualize the posterior distribution of your unknown parameters 
rho_p = dirpdf(delta_conj, N);
phi_p = [gamrnd(a_conj(1)/2,2/b_conj(1),N,1),...
        gamrnd(a_conj(2)/2,2/b_conj(2),N,1)];
mu_p = [normrnd(m_conj(1),1/(alpha_conj(1)*phi(1)),N,1),...
        normrnd(m_conj(2),1/(alpha_conj(2)*phi(2)),N,1)];
figure('name','parameter posterior distribution','DefaultAxesFontSize',14);
subplot(1,3,1);
histogram(rho_p(:,1)); hold on
histogram(rho_p(:,2))
grid on
legend('$\rho_1$','$\rho_2$','interpreter','latex','FontSize',20);
title('$\rho$','interpreter','latex','FontSize',20);
subplot(1,3,2);
histogram(phi_p(:,1)); hold on
histogram(phi_p(:,2));
grid on
legend('$\phi_1$','$\phi_2$','interpreter','latex','FontSize',20);
title('$\phi$','interpreter','latex','FontSize',20);
subplot(1,3,3);
histogram(mu_p(:,1)); hold on
histogram(mu_p(:,2));
grid on
legend('$\mu_1$','$\mu_2$','interpreter','latex','FontSize',20);
title('$\mu$','interpreter','latex','FontSize',20);

%% compute the posterior mean of your unknown parameters
rho_mean = mean(rho_p);
phi_mean = mean(phi_p);
mu_mean = mean(mu_p); 
fprintf('Posterior mean of the parameters:\n rho -> [%.2f %.2f] \n phi -> [%.2f %.2f] \n mu -> [%.2f %.2f]\n',...
    rho_mean,phi_mean,mu_mean)

%% Generate 1000 samples \tilde{x} from your estimated model
sigma(:,:,1) = 1/phi_mean(1);
sigma(:,:,2) = 1/phi_mean(2);
gm = gmdistribution(mu_mean',sigma,rho_mean');
x_tilde = random(gm,1e3);
figure('name','posterior mean estimation','DefaultAxesFontSize',14);
histogram(x); hold on
histogram(x_tilde);
grid on
legend('$x$','$\tilde{x}$','interpreter','latex','FontSize',20);
%% probability density functions 

% 2. z posterior pdf
function [z,nk] = p_z(x,rho,mu,phi,K,N)
    z = zeros(N,1);
    nk = zeros(K,1);
    temp = rho.*sqrt(phi).*exp(-0.5*phi.*(x-mu).^2);
    p_zi = temp./sum(temp,2);
    indx_z1 = find(p_zi(:,1)>=p_zi(:,2));
    indx_z2 = find(p_zi(:,1)<p_zi(:,2));
    z(indx_z1) = 1;
    z(indx_z2) = 2;
    nk = [sum(z==1),sum(z==2)];
end

% 3. random sampe from dirichlet pdf
function rho = dirpdf(delta,N)
    K = length(delta);
    rho = gamrnd(repmat(delta,N,1),1,N,K);
    rho = rho ./ repmat(sum(rho,2),1,K);
end
