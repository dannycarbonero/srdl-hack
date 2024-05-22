import numpy as np
import pandas as pd
import pickle as pkl
from directory_handling import get_parent_path

# file_prefixes = ['yes', 'no', 'artifact']
# file_prefixes = ['yes_1000_2024-02-16', 'no_1000_2024-02-16'];

classification_legend = ['y', 'n', 'a']

ripple_params = []
ripple_times = []
spike_params = []
spike_times = []
series = []
working_time = []
time = []
classifications = []

data_path = get_parent_path('data', 'Spike Ripples/silver/priors_csvs')
save_path = get_parent_path('data', 'Spike Ripples/silver')
for prefix in file_prefixes:

    ripple_params.append(pd.read_csv(data_path + prefix + '_rippleparams.csv'))
    ripple_times.append(pd.read_csv(data_path + prefix + '_rippletimes.csv'))
    spike_params.append(pd.read_csv(data_path + prefix + '_spikeparams.csv'))
    spike_times.append(pd.read_csv(data_path + prefix + '_spiketimes.csv'))
    series.append(pd.read_csv(data_path + prefix + '_series.csv', header = None))
    working_time.append(pd.read_csv(data_path+ prefix + '_time.csv', header = None))

for i in range(len(file_prefixes)):
    classifications.extend(classification_legend[i] * series[i].shape[0])
    time.append(np.repeat(working_time[i], series[i].shape[0], axis = 1).T)

ripple_params = pd.concat(ripple_params)
ripple_times = pd.concat(ripple_times)
spike_params = pd.concat(spike_params)
spike_times = pd.concat(spike_times)
series = np.vstack(series)
time = np.vstack(time)
ripple_times.columns = ['ripple_start','spike_stop']
spike_times.columns = ['spike_start', 'spike_stop']

master_frame = pd.concat([ripple_params, ripple_times, spike_params, spike_times], axis = 1)
master_frame['classification'] = classifications
master_frame['time'] = time.tolist()
master_frame['series'] = series.tolist()


with open(save_path + 'silver_priors_val_data_frame.pkl', 'wb') as file:
    pkl.dump(master_frame, file)