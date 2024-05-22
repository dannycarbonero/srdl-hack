from directory_handling import get_parent_path
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
fig = plt.figure(dpi = 300)
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
fig.savefig('subject_counts.png')
fig.savefig('subject_counts.svg')
fig.savefig('subject_counts.tif')
fig.show()

#%%
