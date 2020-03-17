%% HW 2 - Audio Signals and Time-Frequency Analysis
%% Part 1 Initialization of Variables
clear all; close all; clc

load handel

handel_signal = y';
n = length(handel_signal);
t = (1:length(handel_signal))/Fs;
L = t(end) - t(1);
k_odd = (2*pi/L)*[0:(n-1)/2 -(n-1)/2:-1]; 
k_even = (2*pi/L)*[0:n/2-1 -n/2:-1]; % compare k for odd and even vals of n
ks = fftshift(k_odd);

%% Construct Gabor window and add to time domain plot
tau = 4;
a = [0.5, 1, 3, 50];
filter = exp(-a(2)*(t-tau).^2);

filtered_signal = filter.*handel_signal;
signal_fft = fft(filtered_signal);

% See filter used
close all;
subplot(3,1,1) 
plot(t,handel_signal,'k','Linewidth',2) 
hold on 
plot(t,filter,'m','Linewidth',2)
set(gca,'Fontsize',16), xlabel('Time (t)'), ylabel('S(t)')

subplot(3,1,2) 
plot(t,filtered_signal,'k','Linewidth',2) 
set(gca,'Fontsize',16), xlabel('Time (t)'), ylabel('Sg(t)')

subplot(3,1,3) 
plot(ks,abs(fftshift(signal_fft)),'r','Linewidth',2); %axis([-50 50 0 1])
set(gca,'Fontsize',16)
xlabel('frequency (\omega)'), ylabel('FFT(Sg)')

%% See filter sliding to determine test ranges for spectogram
tslide = 0:1:9;


close all;
for j = 1:length(tslide)
    filter = exp(-a(3)*(t-tslide(j)).^2);
    filtered_signal = filter.*handel_signal;
    signal_fft = fft(filtered_signal);
    
    subplot(3,1,1) 
    plot(t,handel_signal,'k','Linewidth',2) 
    hold on 
    plot(t,filter,'m','Linewidth',2)
    hold off
    set(gca,'Fontsize',16), xlabel('Time (t)'), ylabel('S(t)')

    subplot(3,1,2) 
    plot(t,filtered_signal,'k','Linewidth',2) 
    set(gca,'Fontsize',16), xlabel('Time (t)'), ylabel('Sg(t)')

    subplot(3,1,3) 
    plot(ks,abs(fftshift(signal_fft)),'r','Linewidth',2); %axis([-50 50 0 1])
    set(gca,'Fontsize',16)
    xlabel('frequency (\omega)'), ylabel('FFT(Sg)')
    drawnow
    pause(0.1)
end

%% Plot Spectrogram Matrix

% Initialize Gabor width, spacing between each Gabor, and position within
% spectrogram matrix
a = [0.3, 1, 3];
spacing = [0.05, 0.1, 0.5, 1, 2];
position = 0;

% Loop over all widths and spacings to create 15 spectrograms with
% different parameter combination
close all;
for q = 1:length(a)
    for r = 1:length(spacing)
        
        % Take Gabor transforms for every spacing, saving them to use in
        % plot for spectrogram
        tslide = 0:spacing(r):9;
        signal_spec = zeros(length(tslide), n);
        for j = 1:length(tslide)
            filter = exp(-a(q)*(t-tslide(j)).^2);
            filtered_signal = filter.*handel_signal;
            signal_fft = fft(filtered_signal);
            signal_spec(j, :) = fftshift(abs(signal_fft));
        end
        
        % Plot spectrogram in next position using subplot
        position = position + 1;
        subplot(length(a), length(spacing), position)
        pcolor(tslide, ks, signal_spec.') 
        shading interp
        colormap(pink)
        drawnow
    end
end



%% Apply filter and take fft
Sg = filter.*S;
Sgt = fft(Sg);

figure(2)
subplot(3,1,1) 
plot(t,S,'k','Linewidth',2) 
hold on 
plot(t,filter,'m','Linewidth',2)
set(gca,'Fontsize',16), xlabel('Time (t)'), ylabel('S(t)')

subplot(3,1,2) 
plot(t,Sg,'k','Linewidth',2) 
set(gca,'Fontsize',16), xlabel('Time (t)'), ylabel('Sg(t)')

subplot(3,1,3) 
plot(ks,abs(fftshift(Sgt))/max(abs(Sgt)),'r','Linewidth',2); axis([-50 50 0 1])
set(gca,'Fontsize',16)
xlabel('frequency (\omega)'), ylabel('FFT(Sg)')


%% play back 
p8 = audioplayer(handel_signal,Fs);
playblocking(p8);

%% Part 2: Generalized code for piano or recorder, filtered or unfiltered overtones
clear all; close all; clc

% Read the music into Matlab for analysis (can change between piano and
% recorder)
[y_piano, Fs_piano] = audioread('music1.wav');
tr_piano = length(y_piano)/Fs_piano;  % record time in seconds
y_piano = y_piano';

% Initialize variables for use when taking Gabor transforms
n = length(y_piano);
t = (1:length(y_piano))/Fs_piano;
L = tr_piano;
k = (1/L)*[0:n/2-1 -n/2:-1]; 
ks = fftshift(k);
tslide = 0:0.1:16;

% Gaussian for Gabor, initialize width 
tau = 0.85;
a = 90;
a_overtone = 0.05;

% Initialize variables to look at maximum frequencies (record), and plot
% spectrograms (signal_spec and signal_unfilter)
record = zeros(length(tslide), 2);
signal_spec = zeros(length(tslide), n);
signal_unfilter = zeros(length(tslide), n);

% Loop to build spectrograms 
for j = 1:length(tslide)
    
    % Builds spectrogram with unfiltered overtones
    filter = exp(-a*(t-tslide(j)).^2);
    filtered_signal = filter.*y_piano;
    signal_fft = fft(filtered_signal);
    signal_spec(j, :) = fftshift(abs(signal_fft));
    [MAX, INDEX] = max(fftshift(abs(signal_fft)));
    
    %Builds spectrogram with filtered overtones
    filter_overtones = exp(-a_overtone*(ks-ks(INDEX)).^2);
    filtered_over_sig = filter_overtones.*fftshift(signal_fft);
    signal_unfilter(j, :) = abs(filtered_over_sig);

    record(j, 1) = MAX;
    record(j, 2) = ks(INDEX);
    
end
%% Plot Spectrogram, choosing signal_spec or signal_unfilter

figure(1)
pcolor(tslide, ks, signal_spec.') 
shading interp
set(gca,'Ylim',[-500 500],'Fontsize',16)
xlabel('Time (seconds)'), ylabel('Frequency (Hertz)')
title('Spectrogram for Mary had a Little Lamb on Piano')
colormap(pink)

%%
clear all; close all; clc

[y,Fs] = audioread('music2.wav');
tr_rec = length(y)/Fs;  % record time in seconds
plot((1:length(y))/Fs,y);
xlabel('Time [sec]'); ylabel('Amplitude');
title('Mary had a little lamb (recorder)');
% p8 = audioplayer(y,Fs); playblocking(p8);
%%
clear all; close all; clc

[y_piano, Fs_piano] = audioread('music2.wav');
tr_piano = length(y_piano)/Fs_piano;  % record time in seconds
y_piano = y_piano';

n = length(y_piano);
t = (1:length(y_piano))/Fs_piano;
L = tr_piano;
k = (1/L)*[0:n/2-1 -n/2:-1]; 
ks = fftshift(k);

tau = 0.85;
a = 90;
a_overtone = 0.05;

tslide = 0:0.1:14.5;
filter = exp(-a*(t-tau).^2);
filtered_signal = filter.*y_piano;
signal_fft = fft(filtered_signal);
maxsig = max(abs(signal_fft));

record = zeros(length(tslide), 2);
signal_spec = zeros(length(tslide), n);

for j = 1:length(tslide)
    filter = exp(-a*(t-tslide(j)).^2);
    filtered_signal = filter.*y_piano;
    signal_fft = fft(filtered_signal);
    signal_spec(j, :) = fftshift(abs(signal_fft));
    [MAX, INDEX] = max(fftshift(abs(signal_fft)));
%     filter_overtones = exp(-a_overtone*(ks-ks(INDEX)).^2);
%     filtered_over_sig = filter_overtones.*fftshift(signal_fft);
%     signal_spec(j, :) = abs(filtered_over_sig);
    record(j, 1) = MAX;
    record(j, 2) = ks(INDEX);
    
%     subplot(3,1,1)
%     plot(t,y_piano,'k','Linewidth',2) 
%     hold on 
%     plot(t,filter,'m','Linewidth',2)
%     hold off
%     set(gca,'Fontsize',16), xlabel('Time (t)'), ylabel('S(t)')
% 
%     subplot(3,1,2) 
%     plot(t,filtered_signal,'k','Linewidth',2) 
%     set(gca,'Fontsize',16), xlabel('Time (t)'), ylabel('Sg(t)')
% 
%     subplot(3,1,3) 
%     plot(ks,abs(filtered_over_sig),'r','Linewidth',2); axis([-500 0 0 300])
%     set(gca,'Fontsize',16)
%     xlabel('frequency (\omega)'), ylabel('FFT(Sg)')
%     drawnow
%     pause(0.5)
end

%%
figure(2)
pcolor(tslide, ks, signal_spec.')
set(gca,'Ylim',[-1700 1700],'Fontsize',16) 
xlabel('Time (seconds)'), ylabel('Frequency (Hertz)')
title('Spectrogram for Mary had a Little Lamb on Recorder (Unfiltered Overtone)')
shading interp
colormap(pink)













