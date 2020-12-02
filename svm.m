% Ray Lee & Kevin Mao
% Classification using SVM
clear all;
close all;

% Load data
circles = load('mlData.mat', 'circles');
xcircles =  circles.circles.x;
x1 = xcircles(:,1);
x2  = xcircles(:,2);
ycircles = circles.circles.y;

%% Using SVM to classify circle data

N1 = 200; % # of data point in class 1
N2 = 200; % # of data point in class 2
N = N1+N2; %  Total
p = randperm(N, N); % Create array of N unqiue random numbers
ycircles(1:N1) = -1; % Define class 1 as -1 for Kernel

xcircles = [x1(p) x2(p)];
ycircles = ycircles(p);

% Create subset of training, test and validation data for features
train_x = xcircles(1:N*0.8, :); % Subset of training data
test_x = xcircles(N*0.8+1:N*0.9, :); % Subset of test data
validation_x = xcircles(N*0.9+1:end, :); %Subset of validation data

% Create subset of training, test and validation data for targets
train_y = ycircles(1:N*0.8, :); % Subset of training data
test_y = ycircles(N*0.8+1:N*0.9, :); % Subset of test data
validation_y = ycircles(N*0.9+1:end, :); %Subset of validation data

% Matlab SVM
Mdl = fitcsvm(train_x, train_y, 'KernelFunction', 'RBF', 'BoxConstraint', Inf, 'ClassNames', [-1, 1]);

[X, Y] = meshgrid(linspace(-0.5, 2.5, 500), linspace(-0.5, 2.5, 500));
circ_xbound = [X(:), Y(:)];

% A vector of predicted classification scores with boundary
[label, score, cost] = predict(Mdl, circ_xbound);
[v_label, v_score, v_cost] = predict(Mdl, validation_x);

% Plot data and classification
figure;
scatter(x1(1:N1), x2(1:N1), 'b');
hold on
scatter(x1(N1+1:end), x2(N1+1:end), 'r');
contour(X, Y, reshape(score(:, 2), size(X)), [0 0]);
hold off

% Posterior
ScoreSVMModel = fitPosterior(Mdl, train_x, train_y);

% Predicted classes using test data
[y_label, y_score] = predict(Mdl, test_x);

% Threshold to determine ROC using P_d and P_f
ROC_threshold = -3:0.1:3;
circ_pred_y = y_score(:,2) > ROC_threshold;

% Sort data and rewrite class 1 back to value of 0
sorted = sortrows([test_y, circ_pred_y]);
test_y = sorted(:, 1);
N1 = length(find(test_y - 1));
test_y(1:N1) = 0;
circ_pred_y = sorted(:, 2:end);

% Prob detection
circ_P_F = mean(~test_y(1:N1) & circ_pred_y(1:N1,:),1);

% Prob false alarm
circ_P_D = mean(test_y(N1+1:end) & circ_pred_y(N1+1:end,:),1);

% Plot P_F vs. P_D
figure;
plot(circ_P_F, circ_P_D);
title('Circles Data ROC')
xlabel('P_F');
ylabel('P_D');
xlim([-0.1 1])
ylim([0 1.1])

%% Breast Cancer Data, using SVM
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
labels(labels == 0) = -1;

data = [size, nucleoli, labels];
x = data(:,1:2);
y = data(:,3);

N1 = nnz(~labels); % classified as 0
N2 = nnz(labels); % classified as 1
N = N1 + N2;

train_x = x(1:N*0.8, :);
test_x = x(N*0.8+1:N*0.9, :);
validation_x = x(N*0.9+1:end, :);

train_y = y(1:N*0.8, :);
test_y = y(N*0.8+1:N*0.9, :);
validation_y = y(N*0.9+1:end, :);

Mdl = fitcsvm(train_x, train_y, 'KernelFunction', 'RBF', 'BoxConstraint', Inf, 'ClassNames', [-1, 1]);

[X, Y] = meshgrid(linspace(-0.5, 2.5, 500), linspace(-0.5, 2.5, 500));
circ_xbound = [X(:), Y(:)];

[label, score, cost] = predict(Mdl, circ_xbound);
[v_label, v_score, v_cost] = predict(Mdl, validation_x);

ScoreSVMModel = fitPosterior(Mdl, train_x, train_y);

[y_label, y_score] = predict(Mdl, test_x);

pred_y = y_score(:,2) > ROC_threshold;

sorted = sortrows([test_y, pred_y]);
test_y = sorted(:, 1);
N1 = length(find(test_y - 1));
test_y(1:N1) = 0;
pred_y = sorted(:, 2:end);

P_F = mean(~test_y(1:N1) & pred_y(1:N1,:),1);
P_D = mean(test_y(N1+1:end) & pred_y(N1+1:end,:),1);

figure;
plot(P_F, P_D);
title('Breast Cancer Data ROC')
xlabel('P_F');
ylabel('P_D');
xlim([-0.1 1])
ylim([0 1.1])

