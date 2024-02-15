clear; clc;
load_path_prefix ='/home/warehaus/Neural Data/Spike Ripples/silver/Schlafly/priors/'; %'/home/SSD 1/Neural Data/Spike Ripples/silver/Schlafly/priors/';
save_path = '/home/warehaus/Neural Data/Spike Ripples/silver/priors_csvs/'; %'/home/SSD 1/Neural Data/Spike Ripples/silver/priors_csvs/';


file_names = {'yes', 'no', 'artifact'};

for i = 1:length(file_names)
    
    load(strcat(load_path_prefix,file_names{i},'.mat'));
    writetable(rippleparams, strcat(save_path, file_names{i},'_rippleparams.csv'));
    writetable(rippletimes, strcat(save_path, file_names{i},'_rippletimes.csv'));
    writetable(spikeparams, strcat(save_path, file_names{i},'_spikeparams.csv'));
    writetable(spiketimes, strcat(save_path, file_names{i},'_spiketimes.csv'));
    writematrix(sims', strcat(save_path, file_names{i},'_series.csv'));
    writematrix(t, strcat(save_path, file_names{i},'_time.csv'));


end

clear rippleparams rippletimes sims spikeparams spiketimes t
