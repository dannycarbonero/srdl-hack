% Example
% clear; clf

% Load an example time series & plot it.
base_path           = fullfile(pwd, 'Spike_Ripple_Training_Data');  % path to the directory containing folder "silver"
subject             = 'pBECTS003';
electrode           = 'C5';
data_path           = 'silver/Time_Series_for_all_time/';
classification_path = 'silver/Spectra_for_all_time/';
background_path     = 'silver/expert_marked_new_no_spectra/';

load( ...
    fullfile( ...
        base_path, data_path ...  % path
        , sprintf('%s_%s.mat', subject, electrode) ...  % file
        ) ...
    , 'dEDF', 'electrode_label', 'tEDF' ...  % variables
    );
plot(tEDF, dEDF)
Fs = round(1/diff(tEDF(1:2)));

% Load classifications for this subject
load( ...
    fullfile( ...
        base_path, classification_path ...  % path
        , sprintf('updated_%s_%s.mat', subject, electrode)) ...  % file
    , 'S' ...  % variables
    );

% Convert S to a table
data = struct2table(cat(1, S{:}));
data.label = string(data.label);
% ...put frequency on the first axis
if size(data.pow{1}, 1) == numel(times)
    data.pow = cellfun(@(x) {x'}, data.pow); 
end
data.label(data.label == "n") = "no";
data.label(data.label == "y") = "yes";
head(data)

% Check that times are all the same length and dt and that frequencies are
% all the same
L = cellfun(@numel, data.times);
assert(all(L == L(1)))

dt = cellfun(@(x) round(mean(diff(x)) * 1e4), data.times);
assert(all(dt == dt(1)));

freq = data.freq(1, :);
f = arrayfun(@(ii) all(data.freq(ii, :) == freq), 1:height(data));
assert(all(f));


% Write times and frequencies for all spectrograms to files
times = data.times{1}; times = times - times(1);
writematrix(times, fullfile(base_path, 'spectrograms', 'times.csv'));
writematrix(freq, fullfile(base_path, 'spectrograms', 'frequencies.csv'));


% Write the power matrices to files in a directory structure organized by
% labels
nameFun_ = @(subject, electrode, t0) ...
    sprintf('%s_%s_%010.0f.jpg', subject, electrode, t0 * 1e4);
for ii = 1:height(data)
    label = validatestring(data.label(ii), ["yes", "no", "u", "new"]);
    if ~ismember(label, ["yes", "no"]), continue; end

    % ... name files according to start time
    fname = fullfile(base_path, 'spectrograms', 'silver', label ...
        , nameFun_(subject, electrode, data.times{ii}(1)) ...
        );

    % ... write the matrices
    try
        spectrogram2image(pow2db(data.pow{ii}), fname);
%         writematrix(data.pow{ii}, fname);
    catch ME  % ... make directories if necessary
        if ~exist(fileparts(fname), 'dir')
            mkdir(fileparts(fname));
            spectrogram2image(data.pow{ii}, fname);
%             writematrix(data.pow{ii}, fname);
        else
            rethrow(ME)
        end
    end
end


% All data for this patient should be written...
% Next, see if you can read this in as a datastore and use 

