clear; clc;

load("average_spike.mat")
noise_alpha = 0.5;
half_sec = round(Fs/2);
total_traces = 1000;
traces_per_increment = 1000;

for j = 1:total_traces/traces_per_increment

spikes = [];
ripples = [];


for i = 1:traces_per_increment

    fprintf('Generating Trace %i of %i for increment %i \n', i, traces_per_increment, j)
    spike = make_sims("Pink+Spike", 2);
    [ripple, ~, ~, hann]  = make_sims("Pink+Spike+HFO", 2);
    
    spike_noise_int = randi([10, half_sec - 10]);
    noise_spike_front = make_pink_noise(noise_alpha, 1/Fs: 1/Fs:spike_noise_int/Fs);
    noise_spike_back = make_pink_noise(noise_alpha, 1/Fs: 1/Fs:(half_sec - spike_noise_int)/Fs);
    spikes(i, :) = [noise_spike_front, spike(1:uint32(Fs)), noise_spike_back];

    ripple_noise_int = randi([10, half_sec - 10]);
    noise_ripple_front = make_pink_noise(noise_alpha, 1/Fs: 1/Fs:ripple_noise_int/Fs);
    noise_ripple_back = make_pink_noise(noise_alpha, 1/Fs: 1/Fs:(half_sec - ripple_noise_int)/Fs);
    ripples(i, :) = [noise_ripple_front, ripple(1:uint32(Fs)), noise_ripple_back];
    ripple_labels(i,:) = [zeros(1, length(noise_ripple_front)), hann(1:uint32(Fs)), zeros(1,length(noise_ripple_back))];


end

% writematrix(spikes, strcat('spikes_' , num2str(j - 1) , '.csv'))
% writematrix(ripples, strcat('ripples_' , num2str(j - 1) , '.csv'))

writematrix(spikes, 'spikes_net.csv')
writematrix(ripples, 'ripples_net.csv')
writematrix(ripple_labels, 'labels_net.csv')

end


function [x1new] = make_pink_noise(alpha, t)
% Makes pink noise based on a vector of time points t

dt = diff(t(1:2));
L = 2 * ceil(numel(t)/2);

x1 = randn(1, L);
xf1 = fft(x1);
A = abs(xf1);
phase = angle(xf1);

df = 1.0 / (dt * L);
faxis = (0:L/2)*df;
faxis = [faxis, faxis(end-1:-1:2)];
oneOverf = 1.0 ./ faxis.^alpha;
oneOverf(1)=0.0;

Anew = A.*oneOverf;
xf1new = Anew .* exp(1i*phase);
x1new = real(ifft(xf1new));

if numel(t) < L, x1new(end) = []; end
if size(t, 1) > 1, x1new = x1new'; end
  
end



