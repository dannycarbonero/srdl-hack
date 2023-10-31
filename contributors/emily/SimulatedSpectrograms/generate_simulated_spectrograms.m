

N_spikeRipple = 200; 
N_spike = 100;
N_pink = 100;

times = readmatrix('Spike_Ripple_Training_Data/spectrograms/times.csv');
freq = readmatrix('Spike_Ripple_Training_Data/spectrograms/frequencies.csv');

Fs = 2035;

basename = 'contributors/emily/SimulatedSpectrograms';
% rm -rf contributors/emily/SimulatedSpectrograms/*/
outname_ = @(label, simtype, spiketime) fullfile( ...
    basename ...
    , validatestring(label, ["yes", "no"]) ...
    , sprintf('%s%05.0f.jpg', simtype, spiketime*1e4) ...
    );

% Create directory structure
for label = ["yes" "no"]
    mkdir(fileparts(outname_(label, '', 1)));
end

% Generate spike-ripple sims
disp('Creating spike-ripple sims')
l = 0;
for ii = 1:N_spikeRipple

    % Pick a random time between .1 and .85;
    spiketime = rand * .75 + .1;

    % ... generate a simulated 1 s spike-ripple and compute the spectrogram
    sim = make_sims("pink+spike+hfo", 1, Fs, 'spiketimes', spiketime);
    s = compute_spectrogram(sim, Fs, freq);

    % ... save the spectrogram as an image
    im = spectrogram2image(s, outname_('yes', 'spikeripple', spiketime));
    
    % ... report status
    if ~mod(ii, 10), l = fprintf([repmat('\b', 1, l) '%d/%d'], ii, N_spikeRipple); end

end
fprintf('\n')

% Generate spike only sims
disp('Creating spike sims')
l = 0;
for ii = 1:N_spike

    % Pick a random time between .1 and .85;
    spiketime = rand * .75 + .1;

    % ... generate a simulated 1 s spike and compute the spectrogram
    sim = make_sims("pink+spike", 1, Fs, 'spiketimes', spiketime);
    s = compute_spectrogram(sim, Fs, freq);

    % ... save the spectrogram as an image
    im = spectrogram2image(s, outname_('no', 'spike', spiketime));

    % ... report status
    if ~mod(ii, 10), l = fprintf([repmat('\b', 1, l) '%d/%d'], ii, N_spike); end
    
end
fprintf('\n')


% Generate pink noise sims
disp('Creating pink noise sims')
l = 0;
for ii = 1:N_pink

    spiketime = ii/1e3; 

    % ... generate a simulated 1 s pink noise and compute the spectrogram
    sim = make_sims("pink", 1, Fs);
    s = compute_spectrogram(sim, Fs, freq);

    % ... save the spectrogram as an image
    im = spectrogram2image(s, outname_('no', 'pink', spiketime));

    % ... report status
    if ~mod(ii, 10), l = fprintf([repmat('\b', 1, l) '%d/%d'], ii, N_pink); end
    
end
fprintf('\n');

