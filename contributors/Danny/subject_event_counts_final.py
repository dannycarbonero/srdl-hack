from directory_handling import get_parent_path
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import seaborn as sns
from utilities import generate_LOO_subjects

#%%
data_directory = get_parent_path('data', 'Spike Ripples/silver')
with open(data_directory + 'silver_data_frame.pkl', 'rb') as file:
    data = pickle.load(file)

unique_subjects = data['subject'].nunique()
print("Unique subjects:", unique_subjects)

y_counts = []
LOO_subjects = generate_LOO_subjects()
for subject in LOO_subjects:
    y_counts.append(len(data[(data['subject'] == subject) & (data['classification'] == 'y')]))



#%%
bk_frame = data[data['classification'] == 'bk']
bk_counts = len(bk_frame)
working_bk = []
for subject in LOO_subjects:
    working_bk.append(len(bk_frame[bk_frame['subject'] != subject]))


#%%
data['classification'] = data['classification'].replace('bk', 'background')
data['classification'] = data['classification'].replace('y', 'spike ripple')
data['classification'] = data['classification'].replace('n', 'spike')


#%%
data['subject'] = data['subject'].apply(lambda x: x if x in LOO_subjects else 'Others Agg.')
LOO_subjects = [subject[-3:] for subject in LOO_subjects]
labels = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']

plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 14})
fig = plt.figure(dpi = 600)
ax = fig.add_subplot(111)
ax = sns.histplot(data = data, x = 'subject', hue = 'classification', multiple = 'stack')
ax.tick_params(axis='x', rotation=45)
legend_entries = ['Others Agg.']
legend_entries.extend(labels)
ax.set_xticklabels(labels)
ax.spines[['right', 'top']].set_visible(False)
ax.set_xlabel('Subject')
ax.yaxis.set_major_locator(FixedLocator([0, 350, 1100]))
fig.tight_layout()
# fig.savefig('figures/subject_counts.png')
# fig.savefig('figures/subject_counts.svg')
# fig.savefig('figures/subject_counts.tif')
fig.savefig('figures/Figure 3.pdf')
fig.show()

#%%
# LOO_subjects = generate_LOO_subjects()
# LOO_subjects.insert(0, 'Others Agg.')
# value_counts = []
# for subject in LOO_subjects:
#     value_counts.append(data[data['subject'] == subject]['classification'].value_counts())
#
# # number of ripples
# ripple_counts = []
# for i in range(1,len(LOO_subjects)):
#     ripple_counts.append(value_counts[i]['spike ripple'])
# print(f"The minimum value is {np.min(ripple_counts)}, the maximum value is {np.max(ripple_counts)}, and the mean is {np.mean(ripple_counts):.2f}.")
#
# # ripples per LOO set
# ripple_counts_all = []
# for i in range(len(LOO_subjects)):
#     ripple_counts_all.append(value_counts[i]['spike ripple'])
# LOO_sets = [sum(ripple_counts_all) - ripple_counts_all[i] for i in range(1,len(ripple_counts_all))]
# print(f"The minimum value is {np.min(LOO_sets)}, the maximum value is {np.max(LOO_sets)}, and the mean is {np.mean(LOO_sets):.2f}.")
#
# print(f"The total number of background events were {len(data[data['classification'] == 'background']):.2f}.")
#
# print(value_counts[0])