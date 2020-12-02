%Ali Rahman and Aziza Almanakly
%10.30.19
clear all;
close all;

% Loading Data
circles= load('mlData.mat', 'circles');
unimodal= load('mlData.mat', 'unimodal');
% Separating Circle Data
xcircles = circles.circles.x;
x1circles = xcircles(:,1);
x2circles = xcircles(:,2);
ycircles = circles.circles.y;
% Separating Unimodal Data
xunimodal = unimodal.unimodal.x;
x1unimodal = xunimodal(:,1);
x2unimodal = xunimodal(:,2);
yunimodal = unimodal.unimodal.y;
% Plotting Unimodal Data With Label Identifier
N1 = 200;
N2 = 200;
figure(1)
scatter(x1unimodal(1:N1), x2unimodal(1:N1))
hold on
scatter(x1unimodal(N1 + 1:end), x2unimodal(N1 + 1:end))
% Generative Linear - Unimodal
PI = N1/(N1 + N2); % EQ. 4.73
Sigma = cov(xunimodal);
% Mean Calculations for Each Dataset
mu1 = 1/N1*xunimodal'*yunimodal; % EQ. 4.75
mu2 = 1/N2*xunimodal'*(1 - yunimodal); % EQ. 4.76
% Weight Calculations for Features
w = Sigma^-1*(mu1 - mu2); % EQ. 4.66
w0 = -1/2*mu1'*Sigma^-1*mu1 + 1/2 * mu2.' * Sigma^-1 * mu2 + log(PI/(1-PI)); % EQ. 4.67
% Using All Points to Determine Boundary Condition
[xplot, yplot] = meshgrid(linspace(-4, 5, 500), linspace(-8, 8, 500));
xBound = [xplot(:), yplot(:)];
pred = (w.' * xBound.' + w0).'; % EQ. 4.65
sigmoid = 1 ./ (1 + exp(-pred)); % Sigmoid to Determine Probability
% Shifting Threshold to Determine P_F and P_D for ROC
a = (w.' * xunimodal.' + w0).';
threshold = 1:-0.01:0;
y_pred_uni_linear = 1 ./ (1 + exp(-a)) > threshold;
% Correct Amount at Threshold of 0.5 threshold(51)
correct_uni_linear = 1 - sum(abs(yunimodal - y_pred_uni_linear(:,51)))./(N1 + N2);
p_D_uni_linear = mean(yunimodal(N1+1:end) & y_pred_uni_linear(N1+1:end,:),1); % Detection
p_F_uni_linear = mean(~yunimodal(1:N1) & y_pred_uni_linear(1:N1,:),1); % False Alarm
% Contour Plot For Boundary Condition
contour(xplot, yplot, reshape(sigmoid, [500,500]), [0.5 0.5]);
title('Gaussian Generative - Unimodal Data')
xlabel('x_1');
ylabel('x_2');
legend('Class 1','Class 2')
hold off
% Plotting Circle Data With Label Identifier
N1 = 200;
N2 = 200;
figure(2)
scatter(x1circles(1:N1), x2circles(1:N1))
hold on
scatter(x1circles(N1+1:end), x2circles(N1+1:end))
% Generative Linear - Circles
PI = N1/(N1 + N2); % EQ. 4.73
xcircles = [xcircles x1circles.^2 + x2circles.^2]; % Third Basis Function
Sigma = cov(xcircles);
% Mean Calculations for Each Dataset
mu1 = 1/N1*xcircles'*ycircles; % EQ. 4.75
mu2 = 1/N2*xcircles'*(1 - ycircles); % EQ. 4.76
% Weight Calculations for Features
w = Sigma^-1*(mu1 - mu2); % EQ. 4.66
w0 = -1/2*mu1'*Sigma^-1*mu1 + 1/2 * mu2.' * Sigma^-1 * mu2 + log(PI/(1-PI)); % EQ. 4.67
% Using All Points to Determine Boundary Condition
[xplot, yplot] = meshgrid(linspace(-0.5, 2.5, 500), linspace(-0.5,25, 500));
xBound = [xplot(:), yplot(:), xplot(:).^2 + yplot(:).^ 2];
pred = (w.' * xBound.' + w0).'; % EQ. 4.65
sigmoid = 1 ./ (1 + exp(-pred)); % Sigmoid to Determine Probability
% Shifting Threshold to Determine P_F and P_D for ROC
a = (w.' * xcircles.' + w0).';
y_pred_circles_linear = 1 ./ (1 + exp(-a)) > threshold;
% Correct Amount at Threshold of 0.5 threshold(51)
correct_circle_linear = 1 - sum(abs(ycircles - y_pred_circles_linear(:,51)))./(N1 + N2);
p_D_circles_linear = mean(ycircles(N1+1:end) & y_pred_circles_linear(N1+1:end,:),1);
p_F_circles_linear = mean(~ycircles(1:N1) & y_pred_circles_linear(1:N1,:),1);
% Contour Plot For Boundary Condition of 0.5
contour(xplot, yplot, reshape(sigmoid, [500,500]), [0.5 0.5]);
title('Gaussian Generative - Circle Data')
xlabel('x_1');
ylabel('x_2');
legend('Class 1','Class 2')
% Logistic Regression - Unimodal
Iota = xunimodal; % Design Matrix
wold = (Iota'*Iota)\Iota'*yunimodal; % Initial Value (Can be Random)
% We chose this value for wold since it was the least squares solutionin
% order to help the solution converge faster (tested with random aswell)
% IRLS Algorithm
for i = 1:100
y = wold'*Iota'; % Predicted Label
R = (y.*(1-y)).*eye(size(Iota,1)); % EQ. 4.98
Rinv = 1./diag(R).*eye(size(Iota,1)); % Better Inversion
z = Iota*wold - Rinv*(y' - ycircles); % EQ. 4.100
wnew = (Iota'*R*Iota)\Iota'*R*z; % EQ. 4.99
wold = wnew; % Update for Next Iteration
end
% Using All Points to Determine Boundary Condition
[xplot, yplot] = meshgrid(linspace(-4, 5, 500), linspace(-8, 8, 500));
xBound = [xplot(:), yplot(:)];
pred = (wnew.' * xBound.').'; % EQ. 4.65
sigmoid = 1 ./ (1 + exp(-pred)); % Sigmoid to Determine Probability
% Shifting Threshold to Determine P_F and P_D for ROC
a = (wnew.' * Iota.').';
y_pred_uni_logi = 1 ./ (1 + exp(-a)) > threshold;
% Correct Amount at Threshold of 0.5 threshold(51)
correct_uni_logi = 1 - sum(abs(yunimodal - y_pred_uni_logi(:,51)))./ (N1 + N2);
p_D_uni_logi = mean(yunimodal(N1+1:end) & y_pred_uni_logi(N1+1:end,:),1);
p_F_uni_logi = mean(~yunimodal(1:N1) & y_pred_uni_logi(1:N1,:),1);
% Plotting Unimodal Data With Correct Label
figure(3)
scatter(x1unimodal(1:N1), x2unimodal(1:N1))
hold on
scatter(x1unimodal(N1+1:end), x2unimodal(N1+1:end))
% Boundary Condition Where Threshold = 0.5
contour(xplot, yplot, reshape(sigmoid, [500,500]), [0.5 0.5]);
title('Logistic Regression - Unimodal Data')
xlabel('x_1');
ylabel('x_2');
legend('Class 1','Class 2')
hold off
% Logistic Regression - Circles
Iota = [xcircles ones(size(xcircles,1), 1)]; % Design Matrix
wold = (Iota'*Iota)\Iota'*ycircles; % Initial Value
% We chose this value for wold since it was the least squares solutionin
% order to help the solution converge faster (tested with random aswell)
% IRLS Algorithm
for i = 1:100
y = wold'*Iota'; % Predicted Label
R = (y.*(1-y)).*eye(size(Iota,1)); % EQ. 4.98
Rinv = 1./diag(R).*eye(size(Iota,1)); % Better Inversion
z = Iota*wold - Rinv*(y' - ycircles); % EQ. 4.100
wnew = (Iota'*R*Iota)\Iota'*R*z; % EQ. 4.99
wold = wnew; % Update for Next Iteration
end
% Using All Points to Determine Boundary Condition
[xplot, yplot] = meshgrid(linspace(-0.5, 2.5, 500), linspace(-0.5,2.5, 500));
xBound = [xplot(:), yplot(:), xplot(:).^2 + yplot(:).^ 2, ones(500^2,1)];
pred = (wnew.' * xBound.'); % EQ. 4.65
sigmoid = 1 ./ (1 + exp(-pred)); % Sigmoid to Determine Probability

% Shifting Threshold to Determine P_F and P_D for ROC
a = (wnew.' * Iota.').';
y_pred_circle_logi = 1 ./ (1 + exp(-a)) > threshold;
% Correct Amount at Threshold of 0.5 threshold(51)
correct_circle_logi = 1 - sum(abs(ycircles - y_pred_circle_logi(:,51)))./(N1 + N2);
p_D_circles_logi = mean(ycircles(N1+1:end) & y_pred_circle_logi(N1+1:end,:),1);
p_F_circles_logi = mean(~ycircles(1:N1) & y_pred_circle_logi(1:N1,:),1);
% Plotting Circle Data With Correct Labels
figure(4)
scatter(x1circles(1:N1), x2circles(1:N1))
hold on
scatter(x1circles(N1 + 1:end), x2circles(N1 + 1:end))
% Boundary Condition Where Threshold = 0.5
contour(xplot, yplot, reshape(sigmoid, [500,500]), [0.5 0.5]);
title('Logistic Regression - Circle Data')
xlabel('x_1');
ylabel('x_2');
legend('Class 1','Class 2')
hold off
% ROC Plot for Unimodal Data
figure(5)
subplot(2,1,1)
plot(p_F_uni_linear, p_D_uni_linear);
hold on
plot(p_F_uni_logi, p_D_uni_logi);
title('Unimodal Data ROC')
xlabel('P_F');
ylabel('P_D');
legend('Linear','Logistic')
% ROC Plot for Circles Data
subplot(2,1,2)
plot(p_F_circles_linear, p_D_circles_linear);
hold on
plot(p_F_circles_logi, p_D_circles_logi);
title('Circle Data ROC')
xlabel('P_F');
ylabel('P_D');
legend('Linear','Logistic')