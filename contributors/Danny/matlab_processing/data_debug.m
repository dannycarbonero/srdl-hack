data = squeeze(struct2cell(data))';
subjects = unique(data(:,1));
for i=1:length(subjects)
    subject_indices = find(strcmp(subjects{i},data(:,1)))
    subject_data = data(subject_indices,:);
    yes_indices = find(strcmp('y', subject_data(:,3)));
        
    if isempty(yes_indices) == 0
       subject_yes_data = subject_data(yes_indices,:)
       for j=1:size(subject_yes_data)
           plot(subject_yes_data{,4});
           hold on
       end
       title(subjects{i});
       saveas(gcf, strcat(subjects{i},'.jpg'));
       hold off
    end
end