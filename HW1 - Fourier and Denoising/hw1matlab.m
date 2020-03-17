%% An Ultrasound Problem

clear; close all; clc;
load Testdata

% Initializing spatial and frequency dimensions, domains
L = 15; % spatial domain
n = 64; % Fourier modes

% Template for 3D grid coordinates
x2 = linspace(-L,L,n+1); 
x = x2(1:n); 
y = x; 
z = x;
% creating the freq domain
k = (2*pi/(2*L))*[0:(n/2-1) -n/2:-1]; 
ks = fftshift(k);

% Create 3D grid coordinates for spatial and freq domain
[X,Y,Z] = meshgrid(x,y,z);
[Kx,Ky,Kz] = meshgrid(ks,ks,ks);

%% Determining Frequency Signature of Marble

% Initialize variable ave for FFT in loop, fullave for total average across
% 20 time points
ave = zeros(n, n, n);
fullave = zeros(n, n, n);

% Loop for summing together all FFTs for each time
for j = 1:20
    ave(:, :, :) = reshape(Undata(j,:),n,n,n);
    ave = fftn(ave);
    fullave = fullave + ave;
end

% Take average of all FFTs, shifting to match freq domain created with ks 
fullave = fftshift(abs(fullave)/20);

% Determine max signal in fullave (aka freq signature of marble), also find
% location of freq signature in 3D grid coordinate
[M, I] = max(fullave(:));
[x1, y1, z1] = ind2sub(size(fullave), I);

% Use "location" of freq signature in 3D grid coordinate to find value of 
% freq in Kx, Ky, Kz
xfreq = Kx(x1, y1, z1);
yfreq = Ky(x1, y1, z1);
zfreq = Kz(x1, y1, z1);

%% Determining Marble Path

% Initialize filter and its parameters
tau = 0.2;
filter = exp(-tau*((Kx - xfreq).^2 + (Ky - yfreq).^2 + (Kz - zfreq).^2));

% Initialize the 20 locations in x, y, and z of marble
xpath = zeros(20, 1);
ypath = zeros(20, 1);
zpath = zeros(20, 1);

close all; 
% Loop through each time point
for j = 1:20
    % Filtering our signal data with Gaussian filter
        signal(:, :, :) = reshape(Undata(j,:),n,n,n);
        signal = fftshift(fftn(signal));
        f_signal = signal.*filter;
    % Use ifftn to revert back to signal in spatial coordinates
        f_signal_path = ifftn(f_signal);
    % plot location of marble using f_signal_path
        figure(1)
        isosurface(X,Y,Z,abs(f_signal_path),0.4)
        axis([-20 20 -20 20 -20 20]), grid on, drawnow
    % Find location of signal 
        [val, index] = max(f_signal_path(:));
        [x1, y1, z1] = ind2sub(size(f_signal_path), index);
        xpath(j) = X(x1, y1, z1);
        ypath(j) = Y(x1, y1, z1);
        zpath(j) = Z(x1, y1, z1);
end

% Plot marble path using signal location at each point in time 
hold on
plot3(xpath, ypath, zpath, 'k')
axis([-20 20 -20 20 -20 20])
xlabel('distance (x)')
ylabel('distance (y)')
zlabel('distance (z)')

%% Location of Marble at Final Time

x_final = xpath(20);
y_final = ypath(20);
z_final = zpath(20);













