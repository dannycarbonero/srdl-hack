% Make sims using Mark's simulation code

% Create the simulated data in Figure 2 of [Chu et al, J Neurosci Methods, 2017]
% Downloaded from https://github.com/Mark-Kramer/Spike-Ripple-Detector-Method.git, simulations branch

function [d0,t0] = make_sims(simulation, Fs, T)
% make_sims(simulation, Fs [Hz], T [s])

simulation = validatestring(simulation, ...
    ["Pink", "Pink+Pulse", "Pink+Spike", ...
        "Pink+Spike+HFO", "Pink+Spike+HFO+30%"] ...
    );

% Sampling rate
if nargin < 2 || isempty(Fs), Fs = 2035; end
% duration to simulation
if nargin < 3 || isempty(T), T = 60*10; end

% Say things...
fprintf('Running %s simulation ... \n', simulation)

% Create pink noise time series
d0 = make_pink_noise(0.5,T*Fs,1/Fs);

% 

switch simulation
    case "Pink"
        % do nothing
    case "Pink+Pulse"
        fprintf(['Running Pink+Pulse simulation ... \n'])
        t = 1/Fs : 1/Fs : T;
        d = 1/Fs : 1    : T;
        y = pulstran(t,d,'tripuls',0.05,0);
        y = y / max(y);
        [~, i0] = findpeaks(y);
        for i=1:length(i0)
          y(i0(i)-100:i0(i)+100) = 20*std(d0)*y(i0(i)-100:i0(i)+100);
        end
        d0 = d0 + y';

    case ["Pink+Spike", "Pink+Spike+HFO"]
          load('average_spike.mat', 'Fs', 'spike')
          Fs = round(Fs);
          spike = spike - mean(spike);
          spike = spike / max(spike);
          spikeH = spike .*hann(length(spike));
          spike0 = spikeH;
          std0 = std(d0);
          for i=1:T
              spike = spike0;

              if simulation == "Pink+Spike+HFO"
                  % ... add an HFO to the spike waveform

                  HFO = randn(Fs,1);  % 1 s normal random
                  Wn = [110, 120]/(Fs/2);	%...set the passband,
                  n  = 100;					%...and filter order,
                  b  = fir1(n,Wn);			%...build bandpass filter.
                  HFO = filtfilt(b,1,HFO);	%...and apply filter.
        
                  % filtered HFO
                  HFO = HFO(1000:1000+round(0.05*Fs)-1);
                  HFO = hann(length(HFO)).*HFO;
                  HFO = HFO/std(HFO);
                  HFO = 0.05*std(spike)*HFO;

                  istart = 190;
                  spike(istart:istart+length(HFO)-1) = spike(istart:istart+length(HFO)-1) + 1*HFO;
              end

              i0 = round(Fs/2) + round((i-1)*Fs);
              d0(i0:i0+length(spike)-1) = d0(i0:i0+length(spike)-1) + 20*std0*spike;
          end
    case "Pink+Spike+HFO+30%"

end

  
  %%%%%%%%%% PINK+SPIKE+HFO+30% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if strcmp(simulation, 'Pink+Spike+HFO+30%')
      
      
      load(['average_spike.mat']);
      Fs = round(Fs);
      spike = spike - mean(spike);
      spike = spike / max(spike);
      spike = spike .*hann(length(spike));
      spike0 = spike;
      d0 = make_pink_noise(0.5,T*Fs,1/Fs);
      std0 = std(d0);
      
      for i=1:600
          spike = spike0;
          i0 = round(Fs/2) + round((i-1)*Fs);
          
          if mod(i,3)==0
              HFO = randn(Fs,1);
              Wn = [110, 120]/(Fs/2);			%...set the passband,
              n  = 100;                         %...and filter order,
              b  = fir1(n,Wn);                  %...build bandpass filter.
              HFO = filtfilt(b,1,HFO);          %...and apply filter.
              HFO = HFO(1000:1000+round(0.05*Fs)-1);
              HFO = hann(length(HFO)).*HFO;
              HFO = HFO/std(HFO);
              HFO = 0.05*std(spike)*HFO;
              istart = 190;
              spike(istart:istart+length(HFO)-1) = spike(istart:istart+length(HFO)-1) + 1*HFO;
          end
          
          d0(i0:i0+length(spike)-1) = d0(i0:i0+length(spike)-1) + 20*std0*spike;
      end
  end
  
  t0 = (1:length(d0))/Fs;
  
end


