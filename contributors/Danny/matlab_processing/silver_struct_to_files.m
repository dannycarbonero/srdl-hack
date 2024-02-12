% data = load('silver_data_structure_interval_2.mat')
data_path = '/home/SSD 1/Neural Data/Spike Ripples/silver/Schlafly/';
load(strcat(data_path, 'silver_data_structure_interval_2.mat'))
save_path = '/home/SSD 1/Neural Data/Spike Ripples/silver/data_csvs/';
class_table = struct2cell(data);
class_table = squeeze(class_table)';

writecell(class_table(:,1:3), strcat(save_path,'subject_event_frame.csv'));


for i = 1:length(data)
    series(i,:) = data(i).d;
    time(i,:) = data(i).t;
    % logY(i,:) = data(i).isY;
    % logN(i,:) = data(i).isN;
    % logBk(i,:) = data(i).isBk;
    time_start(i) = data(i).tstart;
    time_stop(i) = data(i).tstop;

end

writematrix(series, strcat(save_path,'series.csv'));
writematrix(time, strcat(save_path,'time.csv'));
% writematrix(logY, 'logical_Y.csv');
% writematrix(logN, 'logical_N.csv');
% writematrix(logBk, 'logical_Bk.csv');
writematrix([time_start;time_stop]', strcat(save_path,'event_times.csv'))

% subjects = unique(class_table(:,1))
% class_frame = {}
%
% % Determine the maximum number of columns needed
% maxColumns = 7;
% 
% % Fill class_frame
% for i = 1:length(subjects)
%     indices = find(strcmp(data(:,1), subjects{i}));
%     working_frame = data(indices,:);
%     class_data = working_frame(:,3);
%     [subject_classes, ~, idx] = unique(class_data);
%     counts = accumarray(idx, 1);
% 
%     % Initialize row with empty cells
%     class_frame_row = cell(1, maxColumns);
%     class_frame_row{1, 1} = subjects{i}; % First column is subject
% 
%     % Fill in the classes and counts
%     for j = 1:length(subject_classes)
%         class_frame_row{1, 2*j} = subject_classes{j}; % Class
%         class_frame_row{1, 2*j+1} = counts(j); % Count
%     end
% 
%     % Add the row to class_frame
%     class_frame = [class_frame; class_frame_row];
% end