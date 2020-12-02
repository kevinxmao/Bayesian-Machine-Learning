clear all;
close all;

circles = load('mlData.mat', 'circles');
unimodal = load('mlData.mat', 'unimodal');

xcircles = circles.circles.x;
x1circles = xcircles(:,1);
x2circles = xcircles(:,2);
ycircles = circles.circles.y;

xunimodal = unimodal.unimodal.x;
x1unimodal = xunimodal(:,1);
x2unimodal = xunimodal(:,2);
yunimodal  = unimodal.unimodal.y;

figure;
subplot(1,2,1);
scatter(xunimodal(1:200,1),xunimodal(1:200,2), 'b');
hold on;
scatter(xunimodal(201:400,1),xunimodal(201:400,2), 'r');
title('Unimodal Dataset');
xlim([-4, 5])
ylim([-4, 5])
axis square;

subplot(1,2,2);
scatter(xcircles(1:200,1),xcircles(1:200,2), 'b');
hold on;
scatter(xcircles(201:400,1),xcircles(201:400,2), 'r');
title('Circles Dataset');
xlim([-0.2, 2.2]);
ylim([-0.2, 2.2]);
axis square;

N1 = 200;
N = 400;
N2 = N - N1;

% Use logistic regression on unimodal data
uni_iota = xunimodal;

% uni_w_old = (uni_iota'*uni_iota)\uni_iota'*yunimodal;
q = rand(2,1);
uni_w_old = q;

% IRLS
for i = 1:100
    y = uni_w_old'*uni_iota';
    R = (y.*(1-y)).*eye(size(uni_iota,1));
    R_inv = 1./diag(R).*eye(size(uni_iota,1));
    Z = uni_iota*uni_w_old - R_inv*(y' - yunimodal);
    uni_w_new = (uni_iota'*R*uni_iota)\uni_iota'*R*Z;
    uni_w_old = uni_w_new;
end

[X, Y] = meshgrid(linspace(-4, 5, 500), linspace(-8, 8, 500));
uni_xbound = [X(:), Y(:)];
uni_pred = (uni_w_new'*uni_xbound').';
uni_sigmoid = 1./(1+exp(-uni_pred));

uni_a= (uni_w_new.'*uni_iota.').';
threshold = 1:-0.01:0;
uni_pred_y = 1./(1+exp(-uni_a)) > threshold;

uni_correct = 1  - sum(abs(yunimodal - uni_pred_y(:,51)))./N;

uni_p_D_log =  mean(yunimodal(N1+1:end) & uni_pred_y(N1+1:end,:),1);
uni_p_F_log = mean(~yunimodal(1:N1) & uni_pred_y(1:N1,:),1);

figure;
subplot(1,2,1);
scatter(x1unimodal(1:N1), x2unimodal(1:N1), 'b')
hold on
scatter(x1unimodal(N1+1:end), x2unimodal(N1+1:end), 'r')

contour(X, Y, reshape(uni_sigmoid,  [500, 500]), [0.5 0.5]);
title('Logistic Regression on Unimodal Data')
xlabel('x_1')
ylabel('x_2')
legend('Class 1', 'Class 2')
hold off

% Use logistic regression on circles data
xcircles = [xcircles x1circles.^2 + x2circles.^2]; % add 3rd basis function
circ_iota = [xcircles ones(size(xcircles,1),1)];
circ_w_old = (circ_iota'*circ_iota)\circ_iota'*ycircles;

for i = 1:100
    y = circ_w_old'*circ_iota';
    R = (y.*(1-y)).*eye(size(circ_iota,1));
    R_inv = 1./diag(R).*eye(size(circ_iota,1));
    Z = circ_iota*circ_w_old - R_inv*(y' - ycircles);
    circ_w_new = (circ_iota'*R*circ_iota)\circ_iota'*R*Z;
    circ_w_old = circ_w_new;
end

[X, Y] = meshgrid(linspace(-0.5, 2.5, 500), linspace(-0.5, 2.5, 500));
circ_xbound = [X(:), Y(:), X(:).^2+Y(:).^2, ones(500^2, 1)];
circ_pred = (circ_w_new.' * circ_xbound.');
circ_sigmoid = 1./(1+exp(-circ_pred));

circ_a= (circ_w_new.' * circ_iota.').';
circ_pred_y = 1./(1+exp(-circ_a)) > threshold;

circ_correct = 1  - sum(abs(ycircles - circ_pred_y(:,51)))./N;

circ_p_D_log =  mean(ycircles(N1+1:end) & circ_pred_y(N1+1:end,:),1);

circ_p_F_log = mean(~ycircles(1:N1) & circ_pred_y(1:N1,:),1);

subplot(1,2,2);
scatter(x1circles(1:N1), x2circles(1:N1), 'b')
hold on
scatter(x1circles(N1+1:end), x2circles(N1+1:end), 'r')

contour(X, Y, reshape(circ_sigmoid,  [500, 500]), [0.5 0.5]);
title('Logistic Regression on Circles Data')
xlabel('x_1')
ylabel('x_2')
legend('Class 1', 'Class 2')
hold off

figure;
subplot(2,1,1);
plot(uni_p_F_log, uni_p_D_log);
title('Unimodal Data ROC')
xlabel('P_F');
ylabel('P_D');
