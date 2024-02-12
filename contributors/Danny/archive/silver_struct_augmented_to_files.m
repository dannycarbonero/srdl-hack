path = '/home/SSD 1/Neural Data/Spike Ripples/silver/Schlafly/augmented/';
subject = 'All'; %'pBECTS043';
num_files = 5;
master_data = [];
for i =1:num_files
    load(strcat(path, subject, '/0', num2str(i), '.mat'));
    master_data = [master_data, data];
end

class_table = struct2cell(master_data);
class_table = squeeze(class_table)';

save_path = strcat('/home/SSD 1/Neural Data/Spike Ripples/silver/augmented_csvs/', subject,'/');

writecell(class_table(:,1:3), strcat(save_path,'subject_event_frame.csv'));


for i = 1:length(master_data)
    series(i,:) = master_data(i).d;
    time(i,:) = master_data(i).t;
    time_start(i) = master_data(i).tstart;
    time_stop(i) = master_data(i).tstop;

end

writematrix(series, strcat(save_path,'series.csv'));
writematrix(time, strcat(save_path,'time.csv'));
writematrix([time_start;time_stop]', strcat(save_path,'event_times.csv'))
