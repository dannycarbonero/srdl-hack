% Create the simulated data in Figure 2 of [Chu et al, J Neurosci Methods, 2017]

function [sig, t, spikeTrain, hannTrain, hfoTrain] = make_sims(simulation, T, Fs, everyNth, varargin)
% [sig, t, spikeTrain, hannTrain, hfoTrain] = make_sims(simulation, T, Fs, everyNth, varargin)
% Creates a train of spikes or spike-ripples on a background of pink noise.
% Input args:
%   simulation (default: "Pink+Spike+HFO"): Options are 
%       "Pink": pink noise
%       "Pink+Pulse": pink noise with triangular pulses
%       "Pink+Spike": pink noise with spikes based on a template 
%       "Pink+Spike+HFO": pink noise with spike-ripples (spikes with high
%           frequency oscillations)
%   T (default: 600): duration of the signal in seconds
%   Fs (default: 2035): sampling frequency of the signal in Hz
%   everyNth (default: 1): add HFOs to every n-th spike if simulation type is Pink+Spike
%       + HFO 
%
% Optional name-value pair arguments:
%   'SpikeSNR', 20: signal to noise ratio of spikes (or pulses)
%   'HfoSNR', 2: signal to noise ratio of the HFOs
%   'HfoOnset', .0934: onset time of the HFOs relative to the spike onset
%       (this is highly dependent on the template waveform)
%   'HfoPassband', [110 120]: frequency of HFOs (filtered from a white
%       noise signal)
%   'HfoDuration', .05: duration (s) of the HFO component of the spike-ripple
%       (width of the Hann window used to isolate the ripple component of
%       thte signals)
%   'PulseWidth', .05: width of the triangular pulse (in seconds) for
%       "Pink+Pulse" simulations
%   'SpikeTimes', (.5:1:T-1) : times (in seconds) of spikes or pulses
%
% Outputs:
%   sig: the simulated signal (pink + spikeTrain + hfoTrain)
%   t: time
%   spikeTrain: the spike component of the simulated signal
%   hannTrain: the hann-window component of the simulated signal
%   hfoTrain: the ripple component of the simulated signal


%% Parse inputs
simulation = validatestring(simulation, [ ...
    "Pink" ...  % pink noise
    , "Pink+Pulse" ...  % Pink noise with triangular pulses
    , "Pink+Spike" ...  % Pink noise plus spikes (from template in average_spike.mat)
    , "Pink+Spike+HFO" ...  % Pink noise plus spikes with HFOs
    ]);

% See if Fs or everyNth are characters or strings and if so, move them to
% varargin
if nargin > 2 && ismember(class(Fs), ["string" "char"])
    varargin = [Fs, everyNth, varargin]; 
    Fs = []; 
    everyNth = []; 
end
if nargin > 3 && ismember(class(everyNth), ["string", "char"])
    varargin = [everyNth, varargin]; 
    everyNth = [];
end

if nargin < 1 || isempty(simulation), simulation = "Pink+Spike+HFO"; end  % spike-ripple
if nargin < 2 || isempty(T), T = 60*10; end  % duration of simulation (10 minutes)
if nargin < 3 || isempty(Fs), Fs = 2035; end  % Sampling rate
if nargin < 4 || isempty(everyNth), everyNth = 1; end  % Add an HFO to every n-th spike

P = struct( ...
    'SpikeSNR', 20 ...  % 20 * SD(noise)
    , 'HfoSNR', 2 ...  % 1 * SD(noise)
    , 'HfoOnset', .0934 ... % 93.4 ms after spike onset
    , 'HfoPassband', [110 120] ...  % 110-120 Hz
    , 'HfoDuration', 0.050 ... % 50 ms
    , 'PulseWidth', 0.05 ...  % 50 ms
    , 'SpikeTimes', [] ...  % defaults to 1:T-1
    );


% Parse name-value pairs
for ii = 1:2:numel(varargin)
    arg = validatestring(varargin{ii}, fields(P));
    val = varargin{ii + 1};
    P.(arg) = val;
end

% Set default spike times and remove invalid times if given as argument
if isempty(P.SpikeTimes)
    spikeTimes = .5:T - 1;
else
    spikeTimes = P.SpikeTimes;
    spikeTimes(spikeTimes > T || spikeTimes < 0) = [];
end

% ... make arguments into ints
Fs = round(Fs); T = ceil(T); everyNth = round(everyNth);


%% Main

% Create pink noise time series
t = 1/Fs : 1/Fs : T; 
pink = make_pink_noise(0.5, t);
stdNoise = std(pink);

% ... initialize output variables
if nargout > 2
    [spikeTrain, hannTrain, hfoTrain] = deal([]);
end


switch simulation
    case "Pink"
        sig = pink;

    case "Pink+Pulse"
        % Generate a signal with triangular pulses at indicated times
        spikeTrain = pulstran(t, spikeTimes, 'tripuls', P.PulseWidth, 0);  % skew=0

        % ... rescale pulses
        spikeTrain = rescale(spikeTrain, 0, P.SpikeSNR * stdNoise); 
        
        % ... add pulses to pink noise
        sig = pink + spikeTrain;

    case {"Pink+Spike", "Pink+Spike+HFO"}
        % Load the spike template and scale it
        spike = load_spike_template(Fs) * P.SpikeSNR * stdNoise;

        % ... find where the spike has its max and align this to requested
        % spike times
        [~, offset] = max(spike);

        % ... initialize empty spike and hann window trains
        [spikeTrain, hannTrain] = deal(zeros(size(pink)));

        % ... generate a signal with template spikes
        for tt = spikeTimes
            % ... inject the spike into spikeTrain
            inds = find(t >= tt, 1) - offset + (0 : length(spike)-1);
            if inds(end) > numel(spikeTrain) || inds(1) < 1
                valid = inds > 0 & inds <= numel(spikeTrain);
                spikeTrain(inds(valid)) = spike(valid);
            else
                spikeTrain(inds) = spike;
            end
        end

        % ... add the spike train and pink noise signals together
        sig = pink + spikeTrain;

        % ... add HFOs to the waveform (if requested)
        if simulation == "Pink+Spike+HFO"  
            % Create a pulse train of scaled Hann windows
            hannWin = hann(round(P.HfoDuration * Fs)) * P.HfoSNR * stdNoise;
            for tt = spikeTimes(1:everyNth:end) + P.HfoOnset
                if tt > t(end), continue; end
                inds = find(t >= tt, 1) - offset + (0: length(hannWin) - 1);
                if inds(1) < 1 || inds(end) > numel(hannTrain)
                    valid = inds > 0 & inds <= numel(hannTrain);
                    hannTrain(inds(valid)) = hannWin(valid);
                else
                    hannTrain(inds) = hannWin;
                end
                
            end

            % ... multiply by a persistent HFO signal and scale
            HFO = create_HFO(t, P.HfoPassband);
            hfoTrain = hannTrain .* HFO;

            % ... add the signals together
            sig = sig + hfoTrain;
        end


    otherwise
        error('Simulation %s not recognized', simulation)

end

  
end


function HFO = create_HFO(t, passband)
% HFO = create_HFO(Fs, passband=[110, 120], duration=0.05)
% Create a `duration` s HFO in frequency range `passband`.

if nargin < 2 || isempty(passband), passband = [110, 120]; end
assert(numel(passband) == 2, 'Argument passband should be a 2-element vector of the form [low, high].')
if nargin < 3 || isempty(duration), duration = .05; end  % 50 ms

Fs = round(1/diff(t(1:2)));  % get the frequency
whiteNoise = randn(size(t));  % 1 s normal random
Wn = passband/(Fs/2);	%...set the passband,
n  = 100;					%...and filter order,
b  = fir1(n,Wn);			%...build bandpass filter.
HFO = filtfilt(b, 1, whiteNoise);	%...and apply filter.
HFO = normalize(HFO);  % z-score (center and standardize)

end


function spike = load_spike_template(Fs)
% Load the template spike waveform from average_spike.mat, and resample to
% frequency Fs if necessary

spikeTemplate = load('average_spike.mat', 'Fs', 'spike');
spikeTemplate.Fs = round(spikeTemplate.Fs);
if spikeTemplate.Fs ~= Fs  % resample to frequency Fs
    warning('Resampling spike template from %d Hz to %d Hz', spikeTemplate.Fs, Fs);
    spike = resample(spikeTemplate.spike, Fs, spikeTemplate.Fs);
else
    spike = spikeTemplate.spike;
end
spike = spike - mean(spike);
spike = spike / max(spike);
spike = spike .* hann(length(spike));
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


