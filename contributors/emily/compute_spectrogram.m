function [S, f, t, s] = compute_spectrogram(trace, Fs, freq)
% Based on Jess's function to compute spectrograms: 
% apply a hanning taper, then compute the spectrum on 200 ms intervals,
% every 5 ms. `S` is the spectrogram power (in decibels); `s` is the
% short-time fourier transform.

if nargin < 3 || isempty(freq), freq = [30 250]; end  % use a default range

if size(trace, 1) == 1, trace = trace'; end  % transpose if horizontal

x = normalize(trace, 'center', 'mean'); % .* hann(numel(trace));
window = hann(round(.2 * Fs/2)*2);  % 200 ms windows
step = round(.005 * Fs/2)*2;  % 5 ms steps
noverlap = length(window) - step; % ... spectrogram function takes overlap as input instead of step

if numel(freq) == 2
    nfft = 2^ceil(log2(numel(window)));  % this is the default
    [s, f, t] = spectrogram(x, window, noverlap, nfft, Fs);
    mask = f >= freq(1) & f <= freq(2);
    s = s(mask, :);
    f = f(mask);
else
    [s, f, t] = spectrogram(x, window, noverlap, freq, Fs);
end

% ... convert the STFT to power
S = abs(s).^2;


% ... smooth along the time dimension (25 ms gaussian kernel)
S = smoothdata(S, 2, 'gaussian', .025*1/diff(t(1:2)));
S = smoothdata(S, 1, 'gaussian', 5);

% win = gausswin(round(.025/diff(t(1:2))) * 4, 2)' ...  % sigma ≈ 25 ms in time
%     .* gausswin(5, 2);  % ... df Hz in frequencies
% win = win./sum(win, 'all');
% S = conv2(S, win, 'same');

% ... convert the STFT to power in decibels
S = pow2db(S);


end
