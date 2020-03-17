%% HW 4 - Music Clasification
%% Test 1, 2, 3
clear all; close all; clc

%% Test 1
[y_bts, Fs] = audioread('wavfiles/bts.wav'); %20 samples bts music
[y_chopin, ~] = audioread('wavfiles/mus2.wav'); %20 samples bpsical music
[y_cole, ~] = audioread('wavfiles/mus4.wav'); %20 samples exo music
[y_test1, ~] = audioread('wavfiles/test1.wav'); %36 samples of all genres

%% Test 2 (keep y_bts from Test 1)
[y_exo, Fs] = audioread('wavfiles/mus1.wav'); %20 samples bts artist 3
[y_bp, ~] = audioread('wavfiles/blackpink.wav'); %20 samples bts artist 2
[y_test2, ~] = audioread('wavfiles/test2.wav'); %15 samples of bts

%% Test 3 
[y_pop, Fs] = audioread('wavfiles/pop.wav'); %20 samples bts
[y_rap, ~] = audioread('wavfiles/rap.wav'); %20 samples exo
[y_clas, ~] = audioread('wavfiles/classical.wav'); %20 samples bpsical
[y_test3, ~] = audioread('wavfiles/test3.wav'); %36 samples of all genres


%% Create Spectrograms from file
[spec_bts] = spectrogram_matrix(y_pop, Fs, 5, 3);
[spec_exo] = spectrogram_matrix(y_rap, Fs, 5, 3);
[spec_bp] = spectrogram_matrix(y_clas, Fs, 5, 3);

[U, w, threshold_min, threshold_max, index_max, index_mid, index_min, ...
    sort_bts, sort_exo, sort_bp, D] = ...
    trainer(spec_bts, spec_exo, spec_bp, 50);

%% Create correct labels and run test data through model
% bts = 1, exo = 2, bp = 3
correct_labels = [2*ones(1, 12) 3*ones(1, 12) ones(1, 12)];

[num_correct, num_wrong, pval, label_error] = tester(U, w, y_test3, Fs, 5, 3, ...
    index_max, index_mid, index_min, threshold_min, threshold_max, correct_labels);

close all;
figure(1)
plot(sort_bts, zeros(size(spec_bts,2)),'ob','Linewidth',2)
hold on
plot(sort_exo, ones(size(spec_exo,2)),'or','Linewidth',2)
plot(sort_bp, 2*ones(size(spec_bp,2)),'og','Linewidth',2) 
title('Test 3: Clasification during Training')
% legend('trained bts', 'trained exo', 'trained bp')
ylim([0 3])

%% Making labels for plotting
vals = [pval; label_error(2,:)];
errors = label_error(1,:) - label_error(2,:);
error_position = (errors == 0);
pval_error = pval;
pval_error(error_position) = NaN;
pval_error = pval_error(~isnan(pval_error));

bp_testlabels = (vals(2,:) == 3)*2.5;
bp_correctlabels = (correct_labels(1,:) == 3)*2.25;
exo_testlabels = (vals(2,:) == 2)*1.5;
exo_correctlabels = (correct_labels(1,:) == 2)*1.25;
bts_testlabels = (vals(2,:) == 1)*0.5;
bts_correctlabels = (correct_labels(1,:) == 1)*0.25;

testpval_bp = pval;
testpval_bp(~bp_testlabels) = NaN;
correctpval_bp = pval;
correctpval_bp(~bp_correctlabels) = NaN;

testpval_exo = pval;
testpval_exo(~exo_testlabels) = NaN;
correctpval_exo = pval;
correctpval_exo(~exo_correctlabels) = NaN;

testpval_bts = pval;
testpval_bts(~bts_testlabels) = NaN;
correctpval_bts = pval;
correctpval_bts(~bts_correctlabels) = NaN;

%% Plotting Test and Correct Labels 
figure(2)
plot(testpval_bp, bp_testlabels,'ob','Linewidth',2)
hold on
plot(correctpval_bp, bp_correctlabels,'dk','Linewidth',2)

plot(testpval_exo, exo_testlabels,'or','Linewidth',2)
plot(correctpval_exo, exo_correctlabels,'ok','Linewidth',2)

plot(testpval_bts, bts_testlabels,'og','Linewidth',2)
plot(correctpval_bts, bts_correctlabels,'*k','Linewidth',2)

xline(threshold_min, 'Linewidth',2)
xline(threshold_max, 'Linewidth',2)
for i = 1:length(pval_error)
    xline(pval_error(i))
end
legend('test clas', 'correct clas', 'test rap', 'correct rap', 'test pop', 'correct pop')
title('Test 3: Model Predictions of Three Genres')
ylim([0 3])

% Function to create spectrogram of each song vector from file
function [spec] = spectrogram_matrix(y_music, Fs, song_length, sample_rate)
    y_music = y_music(:,1)'; % all pan to left speaker
    
    num_songs = size(y_music,2)/(Fs*song_length);
    song_sample = Fs*song_length;
    
    % Break down songs into rows of the matrix
    full_song_matrix = zeros(num_songs, Fs*song_length);
    full_song_matrix(1,:) = y_music(1:song_sample);
    
    for i = 1:num_songs-1
        full_song_matrix(i+1,:) = y_music(song_sample*i+1 : song_sample*(i+1));
    end
    
    % Take every sample_rate samples
    song_matrix = zeros(num_songs, size(full_song_matrix,2)/sample_rate);
    for k = sample_rate:sample_rate:size(full_song_matrix,2)
        song_matrix(:, k/sample_rate) = full_song_matrix(:, k);
    end
    
    % Initialize and construct Gabor window and Spectrograms
    n = length(song_matrix); %
    L = song_length; %length of song
    t = (1:n)/(n/L); %linspace from 0 to 5 seconds

    a = 25;
    tslide = 0:0.2:5;
    close all;
    
    % Create spectrogram
    song_matrix = song_matrix';
    for m = 1:num_songs
        for p = 1:length(tslide)
            filter = exp(-a*(t-tslide(p)).^2);
            filtered_signal = filter'.*song_matrix(:,m);
            signal_fft = fft(filtered_signal');
            signal_spec(:, p) = fftshift(abs(signal_fft));
        end
        signal_spec_vec = signal_spec(:); % convert spectrogram matrix into vector
        spec(:,m) = signal_spec_vec;
    end

end

% Principal Componenet Analysis (PCA) and Linear Discriminant Analysis (LDA) 
function [U, w, threshold_min, threshold_max, index_max, index_mid, index_min, sort_1, sort_2, sort_3, D] = ...
    trainer(song_set1, song_set2, song_set3, feature)
    
    % number of songs in each set
    num_1 = size(song_set1,2); 
    num_2 = size(song_set2,2);
    num_3 = size(song_set3,2);
    
    [U,S,V] = svd([song_set1 song_set2 song_set3], 'econ');
    
    % projection onto principal components
    songs = S*V';
    U = U(:, 1:feature);
    set1 = songs(1:feature, 1:num_1);
    set2 = songs(1:feature, num_1+1:num_1+num_2);
    set3 = songs(1:feature, num_1+num_2+1:num_1+num_2+num_3);

    % calculating bps variance
    mean1 = mean(set1, 2);
    mean2 = mean(set2, 2);
    mean3 = mean(set3, 2);
    mean_songs = mean([set1 set2 set3], 2);
    
    Sw = 0; % within bps variances
    for k=1:num_1
        Sw = Sw + (set1(:,k)-mean1)*(set1(:,k)-mean1)';
    end
    for k=1:num_2
        Sw = Sw + (set2(:,k)-mean2)*(set2(:,k)-mean2)';
    end
    for k=1:num_3
        Sw = Sw + (set3(:,k)-mean3)*(set3(:,k)-mean3)';
    end

    Sb = (mean1-mean_songs)*(mean1-mean_songs)' + ...
        (mean2-mean_songs)*(mean2-mean_songs)' + ...
        (mean3-mean_songs)*(mean3-mean_songs)'; % between bps 
    
    [V2,D] = eig(Sb,Sw); % LDA
    [~,ind] = max(abs(diag(D)));
    w = V2(:,ind); w = w/norm(w,2);

    v_1 = w'*set1; 
    v_2 = w'*set2;
    v_3 = w'*set3;

    % find index of max, mid, min of average projection
    [~,index_max] = max([mean(v_1) mean(v_2) mean(v_3)]);
    [~,index_min] = min([mean(v_1) mean(v_2) mean(v_3)]);
    index_mid = 6 - index_max - index_min;
    
    % sort each set
    sort_1 = sort(v_1);
    sort_2 = sort(v_2);
    sort_3 = sort(v_3);

    sort_matrix = [sort_1; sort_2; sort_3];
    
    smallest = sort_matrix(index_min, :);
    middle = sort_matrix(index_mid, :);
    largest = sort_matrix(index_max, :);
    
    % find threshold
    t1 = length(smallest);
    t2 = 1;
    while smallest(t1) > middle(t2)
        t1 = t1 - 1;
        t2 = t2 + 1;
    end
    threshold_min = (smallest(t1) + middle(t2))/2;

    t3 = length(middle);
    t4 = 1;
    while middle(t3) > largest(t4)
        t3 = t3 - 1;
        t4 = t4 + 1;
    end
    threshold_max = (middle(t3) + largest(t4))/2;
    
end

function [num_correct, num_wrong, pval, label_error] = tester(U, w, y_music, Fs, song_length, sample_rate, ...
    index_max, index_mid, index_min, threshold_min, threshold_max, correct_labels)

    [spec_test] = spectrogram_matrix(y_music, Fs, song_length, sample_rate); %spectrogram
    test_matrix = U'*spec_test; %PCA
    pval = w'*test_matrix; %LDA

    test_labels = zeros(1,length(pval));
    for i = 1:length(pval)
        if pval(i) < threshold_min
            test_labels(i) = index_min;
        elseif pval(i) > threshold_max
            test_labels(i) = index_max;
        else
            test_labels(i) = index_mid;
        end
    end
    
    num_wrong = nnz(correct_labels - test_labels);
    num_correct = length(pval) - num_wrong;
    
    label_error = [correct_labels; test_labels];
end














