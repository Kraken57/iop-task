
% Define parameters
num_samples = 5000; % Number of data points
seq_length = 1000; % Increased resolution for high-frequency switching
switching_freq = 60e3; % 60 kHz switching frequency
fundamental_freq = 50; % Assuming 50 Hz fundamental frequency

modulation_indices = rand(num_samples, 1) * 1.2; % Random values from 0 to 1.2
spwm_data = zeros(num_samples, seq_length);

% Time vector adjusted for high switching frequency
t = linspace(0, 1/fundamental_freq, seq_length); 

for i = 1:num_samples
    M = modulation_indices(i);
    sin_wave = M * sin(2 * pi * fundamental_freq * t); % Reference sine wave
    carrier_wave = 2 * abs(mod(switching_freq * t, 1) - 0.5); % Custom 60 kHz triangular wave
    spwm_wave = double(sin_wave > carrier_wave); % Generate SPWM (0s and 1s)
    spwm_data(i, :) = spwm_wave;
end

% Save dataset to CSV (update path if needed)
csvwrite('C:\Users\kasaa\spwm_dataset_60kHz.csv', [modulation_indices, spwm_data]);

disp('SPWM dataset saved to Desktop as spwm_dataset_60kHz.csv');

% ---- 🔹 Plot Two Random SPWM Waves ----
figure;
rand_indices = randperm(num_samples, 2); % Select two random indices
subplot(2,1,1);
plot(t, spwm_data(rand_indices(1), :), 'b'); % First SPWM wave
title(['SPWM Wave for Modulation Index = ', num2str(modulation_indices(rand_indices(1)))]);
xlabel('Time (s)');
ylabel('SPWM Signal');
ylim([-0.1, 1.1]);

subplot(2,1,2);
plot(t, spwm_data(rand_indices(2), :), 'r'); % Second SPWM wave
title(['SPWM Wave for Modulation Index = ', num2str(modulation_indices(rand_indices(2)))]);
xlabel('Time (s)');
ylabel('SPWM Signal');
ylim([-0.1, 1.1]);

disp('Two random SPWM waves plotted.');