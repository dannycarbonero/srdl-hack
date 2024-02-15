clear; clc;
load_path_prefix ='/home/warehaus/Naive/'; %'/home/warehaus/Neural Data/Spike Ripples/silver/Schlafly/naive/'; %'/home/SSD 1/Neural Data/Spike Ripples/silver/Schlafly/naive/';
save_path = load_path_prefix; %'/home/warehaus/Neural Data/Spike Ripples/silver/naive_csvs/'; %'/home/SSD 1/Neural Data/Spike Ripples/silver/naive_csvs/';


num_files = 5;


master_data = [];
for i =1:num_files
    load(strcat(load_path_prefix, '0', num2str(i), '.mat'));
    master_data = [master_data, data];
end

class_table = struct2cell(master_data);
class_table = squeeze(class_table)';
 
for i = 1:length(master_data)
    series(i,:) = master_data(i).d;
    time(i,:) = master_data(i).t;
    time_start(i) = master_data(i).tstart;
    time_stop(i) = master_data(i).tstop;

end

if size(class_table, 1) ~= size(series, 1) || size(series, 1) ~= size(time, 1) || length(time_start) ~= length(time_stop)
    error('Mismatched number of rows among variables. Aborting save operation.');
end

writecell(class_table(:,1:3), strcat(save_path,'subject_event_frame.csv'));
writematrix(series, strcat(save_path,'series.csv'));
writematrix(time, strcat(save_path,'time.csv'));
writematrix([time_start;time_stop]', strcat(save_path,'event_times.csv'))

clear class_table time series time_start time_stop data master_data;

