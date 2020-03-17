%% HW 3

%% Load all camera data
clear all; close all; clc

load('cam1_1.mat') %1
load('cam1_2.mat') %2
load('cam1_3.mat') %3
load('cam1_4.mat') %4
load('cam2_1.mat') %5
load('cam2_2.mat') %6
load('cam2_3.mat') %7
load('cam2_4.mat') %8
load('cam3_1.mat') %9
load('cam3_2.mat') %10
load('cam3_3.mat') %11
load('cam3_4.mat') %12

%% Manually determine length of video needed

% 226 frames
vidFrames2_1 = vidFrames2_1(:,:,:,[59:284]);
vidFrames3_1 = vidFrames3_1(:,:,:,[7:232]);

% 208 frames
vidFrames1_2 = vidFrames1_2(:,:,:,[70:277]);
vidFrames2_2 = vidFrames2_2(:,:,:,[95:302]);
vidFrames3_2 = vidFrames3_2(:,:,:,[1:226]);

% 226 frames
vidFrames1_3 = vidFrames1_3(:,:,:,[1:226]);
vidFrames2_3 = vidFrames2_3(:,:,:,[20:245]);
vidFrames3_3 = vidFrames3_3(:,:,:,[30:237]);

% 226 frames
vidFrames1_4 = vidFrames1_4(:,:,:,[90:315]);
vidFrames2_4 = vidFrames2_4(:,:,:,[93:318]);
vidFrames3_4 = vidFrames3_4(:,:,:,[90:315]);

% play movie result
implay(vidFrames3_3)

%% Record frames as double (also take averages for potential averaging filters)

numFrames = size(vidFrames1_1, 4);
X_avg = 0;
for j = 1:numFrames
    X_1_4(:,:,:,j) = im2double(vidFrames1_1(:,:,:,j));
    X_avg = X_avg + X_1_4(:,:,:,j);
end
X_avg = X_avg/numFrames;

%% No modifications to original data

imshow(X_1_4(:,:,:,1))
[x,y] = ginput(1);

imshow(X_1_4(:,:,:,numFrames))
[x1,y1] = ginput(1);

%% Removing average frame

imshow(X_1_4(:,:,:,1))
[x,y] = ginput(1);
for j = 1:numFrames
    X_1_4(:,:,:,j) = X_1_4(:,:,:,j) - X_avg;
    %imshow(X_1_2(:,:,:,j)); drawnow
end
imshow(X_1_4(:,:,:,j))
[x1,y1] = ginput(1);

%% using frame subtracting

imshow(X_1_4(:,:,:,1))
[x,y] = ginput(1);
for j = 1:numFrames - 1
    X_1_4(:,:,:,j) = X_1_4(:,:,:,j + 1) - X_1_4(:,:,:,j);
    %imshow(X_2_1(:,:,:,j)); drawnow
end
imshow(X_1_4(:,:,:,j-1))
[x1,y1] = ginput(1);

%% Manual input 

imshow(X_1_4(:,:,:,1))
[x,y] = ginput(1);
prev_frame = 1; 

for j = 6:3:numFrames
    imshow(X_1_4(:,:,:,j))
    [x1,y1] = ginput(1);
    x_space = linspace(x, x1, j - prev_frame);
    y_space = linspace(y, y1, j - prev_frame);
    x = x1; y = y1;
    max_y(prev_frame:j) = [x_space x1];
    max_x(prev_frame:j) = [y_space y1];
    prev_frame = j + 1;
end

max_y = round(max_y);
max_x = round(max_x);


%% Determining maximum

filter = zeros(480,640,3);
filter(round(y):round(y1), round(x1):round(x), :) = 1;

for j = 1:numFrames
    X_1_4(:,:,:,j) = X_1_4(:,:,:,j).*filter;
    totalmax = X_1_4(:,:,1,j) + X_1_4(:,:,2,j)+ X_1_4(:,:,3,j);
    [~, INDEX] = max(totalmax(:));
    [ix, iy] = ind2sub(size(totalmax), INDEX);
    max_x(j) = round(mean(ix));
    max_y(j) = round(mean(iy));
%     imshow(X_1_4(:,:,:,j)); 
%     plot(max_y,max_x); drawnow
end

%%
imshow(X_1_4(:,:,:,j)); hold on
plot(max_y, max_x); 
%%
for j = 1:numFrames
    imshow(X_1_4(:,:,:,j)); hold on
    plot(max_y(1:j),max_x(1:j));
    drawnow
end
%%
max_x = [max_x max_x(225)];
max_y = [max_y max_y(225)];

%%
max_x = max_x(1:226); max_y = max_y(1:226);
% for j = 1:numFrames
%     if max_y(j) > 433 
%         max_y2(j) = 433 - (max_y(j) - 433);
%     else 
%         max_y2(j) = max_y(j);
%     end
% end
% for j = 1:numFrames
%     imshow(X_1_4(:,:,:,j)); hold on
%     plot(max_y2(1:j),max_x(1:j));
%     drawnow
% 
% end
% 




%%
all_x(1,:,10) = max_x;
all_y(1,:,10) = max_y;

%% need different matrix size for test 3 - A3 saved separately
A3(1:2,:) = [all_x(1,1:end-18,2); all_y(1,1:end-18,2)];
A3(3:4,:) = [all_x(1,1:end-18,6); all_y(1,1:end-18,6)];
A3(5:6,:) = [max_x; max_y];

%%
A3_mean = mean(A3, 2);
A3 = A3 - A3_mean;

%%
val = 1;
for j = 1:2:5
    A1(j:j+1,:) = [all_x(1,:,val); all_y(1,:,val)];
    A2(j:j+1,:) = [all_x(1,:,val + 1); all_y(1,:,val + 1)];
    A4(j:j+1,:) = [all_x(1,:,val + 3); all_y(1,:,val + 3)];
    
    A1_mean(j:j+1,:) = mean(A1(j:j+1,:), 2);
    
    A2_mean(j:j+1,:) = mean(A2(j:j+1,:), 2);
    
    A4_mean(j:j+1,:) = mean(A4(j:j+1,:), 2);
    
    val = val + 4;
end

A1 = A1 - A1_mean;
A2 = A2 - A2_mean;
A4 = A4 - A4_mean;

%% SVD Calculation
[U1,S1,V1] = svd(A1,'econ');
% lambda = diag(S).^2;
% Y=U'*A4;
[U2,S2,V2] = svd(A2,'econ');
[U3,S3,V3] = svd(A3,'econ');
[U4,S4,V4] = svd(A4,'econ');

sig = diag(S);
energy1 = sig(1)^2/sum(sig.^2);

figure(2)
subplot(2,4,1)
plot(diag(S1),'ko','Linewidth',2) %
axis([0 15 0 2000])
title('Test 1')
ylabel('\sigma')
set(gca,'Fontsize',16,'Xtick',0:5:25)
subplot(2,4,2)
plot(diag(S2),'ko','Linewidth',2) %
axis([0 15 0 2000])
title('Test 2')
ylabel('\sigma')
set(gca,'Fontsize',16,'Xtick',0:5:25)
subplot(2,4,3)
plot(diag(S3),'ko','Linewidth',2) %
axis([0 15 0 2000])
title('Test 3')
ylabel('\sigma')
set(gca,'Fontsize',16,'Xtick',0:5:25)
subplot(2,4,4)
plot(diag(S4),'ko','Linewidth',2) %
axis([0 15 0 2000])
title('Test 4')
ylabel('\sigma')
set(gca,'Fontsize',16,'Xtick',0:5:25)

subplot(2,4,5)
plot(diag(S1).^2/sum(diag(S1).^2),'ko','Linewidth',2)
axis([0 25 0 1])
ylabel('Energy')
set(gca,'Fontsize',16,'Xtick',0:5:25)
subplot(2,4,6)
plot(diag(S2).^2/sum(diag(S2).^2),'ko','Linewidth',2)
axis([0 25 0 1])
ylabel('Energy')
set(gca,'Fontsize',16,'Xtick',0:5:25)
subplot(2,4,7)
plot(diag(S3).^2/sum(diag(S3).^2),'ko','Linewidth',2)
axis([0 25 0 1])
ylabel('Energy')
set(gca,'Fontsize',16,'Xtick',0:5:25)
subplot(2,4,8)
plot(diag(S4).^2/sum(diag(S4).^2),'ko','Linewidth',2)
axis([0 25 0 1])
ylabel('Energy')
set(gca,'Fontsize',16,'Xtick',0:5:25)



%% Plot sigma_1*u_1 (scaled appropriately)
n = 226;
y1 = S(1,1)/sqrt(n-1)*U(:,1);
c = compass(y1(1),y1(2));
set(c,'Linewidth',2);

%% Plot sigma_2*u_2 (scaled appropriately)
figure(2)
y2 = S(2,2)/sqrt(n-1)*U(:,2);
c = compass(y2(1),y2(2));
set(c,'Linewidth',2);

%%
% figure(3)
% plot(U(:,1)+A4_mean(1), 'k.')
% axis equal
% figure(2)
% plot(Y(1,:),Y(2,:),'k.');
% hold on
% y1 = U'*y1;
% y2 = U'*y2;
% c = compass(y1(1),y1(2));
% set(c,'Linewidth',2);
% c = compass(y2(1),y2(2));
% set(c,'Linewidth',2);








