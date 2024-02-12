import numpy as np
import pandas as pd
import pickle as pkl
from directory_handling import get_parent_path

#%%
load_path = get_parent_path('data', subdirectory = 'Spike Ripples/silver/data_csvs')
save_path = get_parent_path('data', subdirectory = 'Spike Ripples/silver')

#%%
subject_frame = pd.read_csv(load_path + 'subject_event_frame.csv', header = None)
time = list(np.array(pd.read_csv(load_path + 'time.csv', header = None)))
series = list(np.array(pd.read_csv(load_path + 'series.csv', header = None)))
# log_Y = list(np.array(pd.read_csv(data_path + 'logical_Y.csv', header = None)))
# log_N = list(np.array(pd.read_csv(data_path + 'logical_N.csv', header = None)))
# log_Bk = list(np.array(pd.read_csv(data_path + 'logical_Bk.csv', header = None)))
event_times = np.array(pd.read_csv(load_path + 'event_times.csv', header = None))

#%%
master_data_frame = pd.DataFrame({'subject': list(subject_frame.iloc[:,0]), 'electrode': list(subject_frame.iloc[:,1]), 'classification': list(subject_frame.iloc[:,2]), 'time_start': event_times[:,0], 'time_stop': event_times[:,1], 'time': time, 'series': series}) #, 'log_Y': log_Y, 'log_N': log_N, 'log_Bk': log_Bk})

#%%
with open(save_path + 'silver_data_frame.pkl', 'wb') as file:
    pkl.dump(master_data_frame, file)