import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import joblib
import os


data_pickle_path = '' # add data path here


plot_data = joblib.load(data_pickle_path)

plt.style.use('ggplot')

DUMP_DIR = os.path.dirname(data_pickle_path)


extent = plot_data['extent']
kl_pre_post_grid = plot_data['kl']
model_std_grid = plot_data['std']
FONTSIZE_CAPTION = 13

FONTSIZE_AXIS_LABEL = 11

plt.rc('xtick', labelsize=FONTSIZE_AXIS_LABEL)    # fontsize of the tick labels
plt.rc('ytick', labelsize=FONTSIZE_AXIS_LABEL)    # fontsize of the tick labels


img_filename = os.path.join(DUMP_DIR, 'policy_plasticity')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 6.5))
fig.tight_layout(pad=2.5, w_pad=0, h_pad=2.5, rect=[0, 0, 1, 1])
#fig.tight_layout(pad=4., w_pad=1.5, h_pad=3.0, )

ax1.set_title('Model ensemble standard deviation', fontsize=FONTSIZE_CAPTION)
ax1.set_ylabel('y', fontsize=FONTSIZE_AXIS_LABEL)
ax1.set_xlabel('x', fontsize=FONTSIZE_AXIS_LABEL)
im1 = ax1.imshow(model_std_grid, extent=extent)
fig.colorbar(ax=ax1, mappable=im1, shrink=0.8, orientation='vertical')
ax1.grid(False)

ax2.set_title('KL-div. pre-/post-update policy', fontsize=FONTSIZE_CAPTION)
ax2.set_ylabel('y', fontsize=FONTSIZE_AXIS_LABEL)
ax2.set_xlabel('x', fontsize=FONTSIZE_AXIS_LABEL)
im2 = ax2.imshow(kl_pre_post_grid, extent=extent)
fig.colorbar(ax=ax2, mappable=im2, shrink=0.8, orientation='vertical')
ax2.grid(False)



# save plot
fig.savefig(img_filename)
