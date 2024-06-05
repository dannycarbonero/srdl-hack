import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import seaborn as sns
from matplotlib.animation import FuncAnimation

from directory_handling import get_parent_path
from utilities import generate_LOO_subjects

# Load data
data_directory = get_parent_path('data', 'Spike Ripples/silver')
with open(data_directory + 'silver_data_frame.pkl', 'rb') as file:
    data = pickle.load(file)

# Unique subjects
unique_subjects = data['subject'].nunique()
print("Unique subjects:", unique_subjects)

# Generate LOO subjects and y_counts
y_counts = []
LOO_subjects = generate_LOO_subjects()
for subject in LOO_subjects:
    y_counts.append(len(data[(data['subject'] == subject) & (data['classification'] == 'y')]))


# bk_frame and working_bk
bk_frame = data[data['classification'] == 'bk']
bk_counts = len(bk_frame)
working_bk = []
for subject in LOO_subjects:
    working_bk.append(len(bk_frame[bk_frame['subject'] != subject]))

# Update classifications
data['classification'] = data['classification'].replace('bk', 'background')
data['classification'] = data['classification'].replace('y', 'spike ripple')
data['classification'] = data['classification'].replace('n', 'spike')

# Update subjects
data['subject'] = data['subject'].apply(lambda x: x if x in LOO_subjects else 'Others Agg.')
LOO_subjects = [subject[-3:] for subject in LOO_subjects]
labels = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']

# Plot setup
plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 14})
fig, ax = plt.subplots(dpi=300)
hist = sns.histplot(data=data, x='subject', hue='classification', multiple='stack', ax=ax)
ax.tick_params(axis='x', rotation=45)
legend_entries = ['Others Agg.']
legend_entries.extend(labels)
ax.set_xticklabels(labels)
ax.spines[['right', 'top']].set_visible(False)
ax.set_xlabel('Subject')
ax.yaxis.set_major_locator(FixedLocator([0, 350, 1100]))
fig.tight_layout()

# Create a mapping from subjects to patches
subject_patches = {label: [] for label in labels}
for patch in ax.patches:
    label = patch.get_x()
    if label < len(labels):  # Ensure index is within bounds
        subject_patches[labels[int(np.ceil(label))]].append(patch)

# Animation function
def update(frame):
    if frame == 0:
        # Show all bars without fading for the first frame
        for label in labels:
            for patch in subject_patches[label]:
                patch.set_alpha(1.0)
    else:
        current_label = labels[frame]  # Use frame directly to index labels
        for label in labels:
            for patch in subject_patches[label]:
                if label == current_label:
                    patch.set_alpha(1.0)
                else:
                    patch.set_alpha(0.3)

# Create animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 7), interval=1000, repeat=True)

# Save animation
ani.save('subject_counts_animation.gif', writer='pillow')

plt.show()
