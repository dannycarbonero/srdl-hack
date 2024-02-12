clear; clc;
load_path_prefix = '/home/warehaus/Neural Data/Spike Ripples/silver/Schlafly/augmented/'; %'/home/SSD 1/Neural Data/Spike Ripples/silver/Schlafly/augmented/';
save_path_prefix = '/home/warehaus/Neural Data/Spike Ripples/silver/augmented_csvs/'; %'/home/SSD 1/Neural Data/Spike Ripples/silver/augmented_csvs/'
subject_prefix = 'pBECTS0';
subject_nums = {'03', '07', '11', '15', '33', '43'};

num_files = 5;


for j = 1:length(subject_nums)
    
    fprintf('Writing files for subject %i of %i \n',j,length(subject_nums))
    subject = strcat(subject_prefix, subject_nums{j});
    
    save_path = strcat(save_path_prefix, subject,'/');
    if isfolder(save_path) == 0
        mkdir(save_path);
    end

    load_path = strcat(load_path_prefix, subject, '/');
    
    master_data = [];
    for i =1:num_files
        load(strcat(load_path_prefix, subject, '/0', num2str(i), '.mat'));
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

    clear class_table time series time_start time_stop data master_data

end