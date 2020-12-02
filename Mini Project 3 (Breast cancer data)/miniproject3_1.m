% Mini project 3
% Ray Lee, Kevin Mao
% Using Gaussian Generative Model and Logistic Regression
% as two methods to classify sets of binary feature data
% Models are then used to classify external data

clc;
clear all;
close all;

% load data
load('mlData.mat');

% make data variables
xcircles = circles.x;
xunimodal = unimodal.x;
ycircles = circles.y;
yunimodal = unimodal.y;

%% GAUSSIAN GENERATIVE
% Plot Circle and Unimodal Data
figure;
subplot(1,2,1);
scatter(xcircles(1:200,1),xcircles(1:200,2), 'b');
hold on;
scatter(xcircles(201:400,1),xcircles(201:400,2), 'r');
title('Circles Dataset');
xlim([-0.2, 2.2]);
ylim([-0.2, 2.2]);
axis square;

subplot(1,2,2);
scatter(xunimodal(1:200,1),xunimodal(1:200,2), 'b');
hold on;
scatter(xunimodal(201:400,1),xunimodal(201:400,2), 'r');
title('Unimodal Dataset');
xlim([-4, 5])
ylim([-4, 5])
axis square;

% Unimodal 
uni_N1 = 200; % # of class 1 data points
uni_N2 = 200; % # of class 2 data points
uni_N = uni_N1 + uni_N2; % # total data points

uni_PC1 = uni_N1/uni_N; % Eq 4.73
uni_sigma = cov(xunimodal); % covariance of x values

% Circles
circ_N1 = 200; % # of class 1 data points
circ_N2 = 200; % # of class 2 data points
circ_N = circ_N1 + circ_N2; % # total data points

circ_PC1 = circ_N1/circ_N; % Eq 4.73

% Add 3rd basis function for circles
xcircles = [xcircles, xcircles(:,1).^2 + xcircles(:,2).^2]; 
circ_sigma = cov(xcircles); % covariance of x values

% Calculate means of 
uni_mu1 = 1/uni_N1*xunimodal'*yunimodal; % Eq 4.75
uni_mu2 = 1/uni_N2*xunimodal'*(1-yunimodal); % Eq 4.76

circ_mu1 = 1/circ_N1*xcircles'*ycircles;
circ_mu2 = 1/circ_N2*xcircles'*(1-ycircles);

% weights
uni_w = uni_sigma^-1*(uni_mu1 - uni_mu2); % Eq 4.66
uni_w0 = -1/2*uni_mu1'*uni_sigma^-1*uni_mu1 + 1/2*uni_mu2.'*uni_sigma^-1*uni_mu2 + log(circ_PC1/(1-circ_PC1)); % Eq 4.67

circ_w = circ_sigma^-1*(circ_mu1 - circ_mu2);
circ_w0 = -1/2*circ_mu1'*circ_sigma^-1*circ_mu1 + 1/2*circ_mu2.'*circ_sigma^-1*circ_mu2 + log(circ_PC1/(1-circ_PC1));

% Get decision boundary of unimodal data
[uni_x, uni_y] = meshgrid(linspace(-4, 5, 100), linspace(-8, 8, 100));
uni_xbound = [uni_x(:), uni_y(:)];
uni_pred = (uni_w.'*uni_xbound.'+uni_w0); % Eq. 4.65
uni_sigmoid = 1 ./ (1 + exp(-uni_pred)); % Probabiliy determined by sigmoid function

% Get decision boundary of circles data
[circ_x, circ_y] = meshgrid(linspace(-0.5, 2.5, 100), linspace(-0.5, 2.5, 100));
circ_xbound = [circ_x(:), circ_y(:), circ_x(:).^2 + circ_y(:).^2];
circ_pred = (circ_w.'*circ_xbound.'+circ_w0);
circ_sigmoid = 1 ./ (1 + exp(-circ_pred));

% Threshold to determine ROC using P_d and P_f
ROC_threshold = 1:-0.01:0;
a = (uni_w.' * xunimodal.' + uni_w0).';
uni_pred_y = 1 ./ (1 + exp(-a)) > ROC_threshold;

acirc = (circ_w.' * xcircles.' + circ_w0).';
circ_pred_y_lin = 1 ./ (1 + exp(-acirc)) > ROC_threshold;

uni_correct_lin = 1 - sum(abs(yunimodal - uni_pred_y(:,51)))./(uni_N1 + uni_N2);
% Prob detection
uni_P_D_lin = mean(yunimodal(uni_N1+1:end) & uni_pred_y(uni_N1+1:end,:),1);
% Prob false alarm
uni_P_F_lin = mean(~yunimodal(1:uni_N1) & uni_pred_y(1:uni_N1,:),1);


circ_correct_lin = 1 - sum(abs(ycircles - uni_pred_y(:,51)))./(circ_N1 + circ_N2);
% Prob detection
circ_P_D_lin = mean(ycircles(circ_N1+1:end) & uni_pred_y(circ_N1+1:end,:),1);
% Prob false alarm
circ_P_F_lin = mean(~ycircles(1:circ_N1) & uni_pred_y(1:circ_N1,:),1);

% Contour Plot For Boundary Condition
figure;
subplot(1,2,1);
scatter(xunimodal(1:200,1), xunimodal(1:200,2), 'b');
hold on;
scatter(xunimodal(201:400,1), xunimodal(201:400,2), 'r');
hold on;
contour(uni_x, uni_y, reshape(uni_sigmoid, [100,100]), [0.5 0.5]);
title('Gaussian Generative On Unimodal Data');
legend('Class 1','Class 2');
xlim([-4, 5])
ylim([-4, 5])
axis square;

subplot(1,2,2);
scatter(xcircles(1:200,1), xcircles(1:200,2), 'b');
hold on;
scatter(xcircles(201:400,1), xcircles(201:400,2), 'r');
hold on;
contour(circ_x, circ_y, reshape(circ_sigmoid, [100,100]), [0.5 0.5]);
title('Gaussian Generative On Circles Data');
legend('Class 1','Class 2');
xlim([-0.5, 2.5]);
ylim([-0.5, 2.5]);
axis square;

%% LOGISTIC REGRESSION
% Use logistic regression on unimodal data
uni_iota = xunimodal;
uni_w_old = [0.3; 0.2]; %Initial weights with random values

% IRLS
for i = 1:100
    y = uni_w_old'*uni_iota'; % posterior target
    R = (y.*(1-y)).*eye(size(uni_iota,1)); % Eq 4.98
    R_inv = 1./diag(R).*eye(size(uni_iota,1));
    Z = uni_iota*uni_w_old - R_inv*(y' - yunimodal); % Eq  4.100
    uni_w_new = (uni_iota'*R*uni_iota)\uni_iota'*R*Z; % Newton-Raption update formula for logestic regression model
    uni_w_old = uni_w_new;
end

[X, Y] = meshgrid(linspace(-4, 5, 500), linspace(-8, 8, 500));

uni_xbound = [X(:), Y(:)]; % boundary conditions
uni_pred = (uni_w_new'*uni_xbound').'; % Eq 4.65
uni_sigmoid = 1./(1+exp(-uni_pred)); % Probabiliy determined by sigmoid function

uni_a= (uni_w_new.'*uni_iota.').';
threshold = 1:-0.01:0;
uni_pred_y = 1./(1+exp(-uni_a)) > threshold;

%  Midpoint threshold where it equals  0.5 and in this matrix, medium is
%  row 51
uni_correct_log = 1  - sum(abs(yunimodal - uni_pred_y(:,51)))./uni_N;

% Calculate P_D and P_F to show ROC
uni_P_D_log =  mean(yunimodal(uni_N1+1:end) & uni_pred_y(uni_N1+1:end,:),1);
uni_P_F_log = mean(~yunimodal(1:uni_N1) & uni_pred_y(1:uni_N1,:),1);

figure;
subplot(1,2,1);
scatter(xunimodal(1:200,1),xunimodal(1:200,2), 'b')
hold on
scatter(xunimodal(200:400,1),xunimodal(200:400,2), 'r')

contour(X, Y, reshape(uni_sigmoid,  [500, 500]), [0.5 0.5]);
title('Logistic Regression on Unimodal Data')
xlabel('x_1')
ylabel('x_2')
legend('Class 1', 'Class 2')
xlim([-4, 5])
ylim([-4, 5])
axis square
hold off

% Use logistic regression on circles data
circ_iota = [xcircles ones(size(xcircles,1),1)]; % iota matrix
circ_w_old = (circ_iota'*circ_iota)\circ_iota'*ycircles;  % initiial weight matrix, optimized to reduce steps needed in IRLS

% Newton-Raption Update with 100 steps
for i = 1:100
    y = circ_w_old'*circ_iota'; % posterior target
    R = (y.*(1-y)).*eye(size(circ_iota,1)); % Eq. 4.98
    R_inv = 1./diag(R).*eye(size(circ_iota,1)); 
    Z = circ_iota*circ_w_old - R_inv*(y' - ycircles); % Eq 4.100
    
    % Newton-Raption update formula for logestic regression model
    circ_w_new = (circ_iota'*R*circ_iota)\circ_iota'*R*Z;
    circ_w_old = circ_w_new; % update old weights
end

[X, Y] = meshgrid(linspace(-0.5, 2.5, 500), linspace(-0.5, 2.5, 500));
circ_xbound = [X(:), Y(:), X(:).^2+Y(:).^2, ones(500^2, 1)];
circ_pred = (circ_w_new.' * circ_xbound.');
circ_sigmoid = 1./(1+exp(-circ_pred)); %sigmoid for probablity

circ_a= (circ_w_new.' * circ_iota.').';
circ_pred_y = 1./(1+exp(-circ_a)) > threshold;

circ_correct_log = 1  - sum(abs(ycircles - circ_pred_y(:,51)))./circ_N;

circ_P_D_log =  mean(ycircles(circ_N1+1:end) & circ_pred_y(circ_N1+1:end,:),1);

circ_P_F_log = mean(~ycircles(1:circ_N1) & circ_pred_y(1:circ_N1,:),1);

% Plot boundary for circles data
subplot(1,2,2);
scatter(xcircles(1:200,1),xcircles(1:200,2), 'b')
hold on
scatter(xcircles(200:400,1),xcircles(200:400,2), 'r')

contour(X, Y, reshape(circ_sigmoid,  [500, 500]), [0.5 0.5]);
title('Logistic Regression on Circles Data')
xlabel('x_1')
ylabel('x_2')
legend('Class 1', 'Class 2')
xlim([-0.5, 2.5]);
ylim([-0.5, 2.5]);
axis square
hold off

%% Plot ROC of both methods for circles and unimodal data
figure;
subplot(2,1,1);
plot(uni_P_F_lin, uni_P_D_lin);
hold on
plot(uni_P_F_log, uni_P_D_log);
title('Unimodal Data ROC')
xlabel('P_F');
ylabel('P_D');
legend('Gaussian Generative','Logistic Regression');
hold off

subplot(2,1,2);
plot(circ_P_F_lin, circ_P_D_lin);
hold on
plot(circ_P_F_log, circ_P_D_log);
title('Circles Data ROC')
xlabel('P_F');
ylabel('P_D');
hold off
legend('Gaussian Generative','Logistic Regression');

%% Kaggle Dataset Section

% Patients with Liver disease have been continuously increasing because of 
% excessive consumption of alcohol, inhale of harmful gases, intake of 
% contaminated food, pickles and drugs. This dataset was used to evaluate 
% prediction algorithms in an effort to reduce burden on doctors.

T = readtable('BreastCancer.csv');

thickness = T.Cl_thickness;
thickness = str2double(thickness);

size = T.Cell_size;
size = str2double(size);

shape = T.Cell_shape;
shape = str2double(shape);

adhesion = T.Marg_adhesion;
adhesion = str2double(adhesion); 

c_size = T.Epith_c_size;
c_size = str2double(c_size);

cromatin = T.Bl_cromatin;
cromatin = str2double(cromatin);

nucleoli = T.Normal_nucleoli;
nucleoli = str2double(nucleoli);

mitoses = T.Mitoses;
mitoses = str2double(mitoses);

labels = T.Class;

data = [thickness, size, shape, adhesion, c_size, cromatin, nucleoli, mitoses, labels];
data = sortrows(data,9);

x = data(:,1:8);
y = data(:,9);

N1 = nnz(~labels); % classified as 0
N2 = nnz(labels); % classified as 1
N = N1 + N2;

%% Gaussian Generative
PI = N1/N;
Sigma = cov(x);

% calculate mean
mu1 = 1/N1*x'*y; % Eq 4.75
mu2 = 1/N2*x'*(1-y); % Eq 4.76

w = Sigma^-1*(mu1-mu2);
w0 = -1/2*mu1'*Sigma^-1*mu1 + 1/2*mu2.'*Sigma^-1*mu2 + log(PI/(1-PI)); % Eq 4.67

% Threshold to determine ROC using P_d and P_f
ROC_threshold = 1:-0.01:0;
a = (w.' * x.' + w0).';
pred_lin = 1 ./ (1 + exp(-a)) > ROC_threshold;

correct_lin = 1 - sum(abs(y - pred_lin(:,51)))./N;

P_D_lin = mean(y(N1+1:end) & pred_lin(N1+1:end,:),1);
% Prob false alarm
P_F_lin = mean(~y(1:N1) & pred_lin(1:N1,:),1);


%% Logistic Regression
iota = x; % iota matrix
w_old = (iota'*iota)\iota'*y;  % initiial weight matrix, optimized to reduce steps needed in IRLS

% Newton-Raption Update with 300 steps
for i = 1:300
    t = w_old'*iota'; % posterior target
    R = (t.*(1-t)).*eye(N); % Eq. 4.98
    R_inv = 1./diag(R).*eye(N); 
    Z = iota*w_old - R_inv*(t' - y); % Eq 4.100
    
    % Newton-Raption update formula for logestic regression model
    w_new = (iota'*R*iota)\iota'*R*Z;
    w_old = w_new; % update old weights
end

a= (w_new.' * iota.').';
pred_log = 1./(1+exp(-a)) > threshold;

correct_log = 1  - sum(abs(y - pred_log(:,51)))./N;

P_D_log =  mean(y(N1+1:end) & pred_log(N1+1:end,:),1);

P_F_log = mean(~y(1:N1) & pred_log(1:N1,:),1);

figure;
plot(P_F_lin, P_D_lin);
hold on
plot(P_F_log, P_D_log);
title('Breast Cancer Data ROC')
xlabel('P_F');
ylabel('P_D');
hold off
legend('Gaussian Generative','Logistic Regression');

correctness = [correct_lin correct_log];
xlabel = categorical({'Gaussian Generative' 'Logistic Regression'});
xlabel = reordercats(xlabel, {'Gaussian Generative' 'Logistic Regression'});
figure;
bar(xlabel,correctness)
title('% Correctness')