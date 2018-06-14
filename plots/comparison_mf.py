from rllab.viskit import frontend
from rllab.viskit import core
import matplotlib.pyplot as plt
from plots.plot_utils_mf import *

plt.style.use('ggplot')
#plt.rc('font', family='Times New Roman')
import matplotlib
matplotlib.use('TkAgg')
#matplotlib.font_manager._rebuild()


SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 26
LINEWIDTH = 3

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

data_path = '/home/ignasi/Desktop/mf-comparison'
exps_data = core.load_exps_data([data_path], False)

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
COLORS = dict(ours=colors.pop(0))
LEGEND_ORDER=dict(ours=0, acktr=1, trpo=2, ppo=3, ddpg=4)
def sorting_legend(label):
    return LEGEND_ORDER[label]


def get_color(label):
    if label not in COLORS.keys():
        COLORS[label] = colors.pop(0)
    return COLORS[label]


def plot_from_exps(exp_data,
                   filters={},
                   split_figures_by=None,
                   split_plots_by=None,
                   x_key='n_timesteps',
                   y_key=None,
                   sup_y_key=None,
                   plot_name='./bad-models.png',
                   subfigure_titles=None,
                   plot_labels=None,
                   x_label=None,
                   y_label=None,
                   fontsize=20,
                   num_rows=1,
                   x_limits=None,
                   y_limits=None,
                   report_max_performance=False,
                   log_scale=False,
                   round_x=None,
                   ):

    exp_data = filter(exp_data, filters=filters)
    exps_per_plot = group_by(exp_data, group_by_key=split_figures_by)
    num_columns = len(exps_per_plot.keys())
    assert num_columns % num_rows == 0
    num_columns = num_columns // num_rows
    fig, axarr = plt.subplots(num_rows, num_columns, figsize=(20, 8))
    fig.tight_layout(pad=4.0, w_pad=1.5, h_pad=3, rect=[0, 0, 1, 1])
    # iterate over subfigures
    for i, (default_plot_title, plot_exps) in enumerate(sorted(exps_per_plot.items())):
        plots_in_figure_exps = group_by(plot_exps, split_plots_by)
        subfigure_title = subfigure_titles[i] if subfigure_titles else default_plot_title
        r, c = i//num_columns, i%num_columns
        axarr[r, c].set_title(subfigure_title)
        axarr[r, c].xaxis.set_major_locator(plt.MaxNLocator(5))

        # iterate over plots in figure
        y_max_mean = -1e10
        for j, default_label in enumerate(sorted(plots_in_figure_exps, key=sorting_legend)):
            exps = plots_in_figure_exps[default_label]
            if default_label != 'ours':
                _round_x = round_x
            else:
                _round_x = None
            x, y_mean, y_std = prepare_data_for_plot(exps, x_key=x_key, y_key=y_key, sup_y_key=sup_y_key, round_x=_round_x)

            label = plot_labels[j] if plot_labels else default_label
            _label = label if i == 0 else "__nolabel__"
            if log_scale:
                axarr[r, c].semilogx(x, y_mean, label=_label, linewidth=LINEWIDTH, color=get_color(label))
            else:
                axarr[r, c].plot(x, y_mean, label=_label, linewidth=LINEWIDTH, color=get_color(label))

            axarr[r, c].fill_between(x, y_mean + y_std, y_mean - y_std, alpha=0.2, color=get_color(label))

            # axis labels
            axarr[r, c].set_xlabel(x_label if x_label else x_key)
            axarr[r, c].set_ylabel(y_label if y_label else y_key)
            if x_limits is not None:
                axarr[r, c].set_xlim(*x_limits)
            if y_limits is not None:
                axarr[r, c].set_xlim(*y_limits)
            if max(y_mean) > y_max_mean:
                y_max_mean = max(y_mean)
        if report_max_performance:
            label = 'max' if i == 0 else "__nolabel__"
            axarr[r, c].plot(axarr[r, c].get_xlim(), [y_max_mean]*2, 'k--', label=label)

    fig.legend(loc='lower center', ncol=6, bbox_transform=plt.gcf().transFigure)
    fig.savefig(plot_name)


filter_dict = {'fast_lr': 0.001}

exps_data_filtered = filter(exps_data, filter_dict)


plot_from_exps(exps_data,
               split_figures_by='env.$class',
               split_plots_by='exp_name',
               y_key='AverageReturn',
               filters=filter_dict,
               sup_y_key=['PrePolicy-AverageReturn', 'EnvTrajs-AverageReturn', 'AverageReturn'],
               # subfigure_titles=['HalfCheetah - output_bias_range [0.0, 0.1]',
               #                  'HalfCheetah - output_bias_range [0.0, 0.5]',
               #                  'HalfCheetah - output_bias_range [0.0, 1.0]'],
               # plot_labels=['ME-MPG', 'ME-TRPO'],
               x_label='Time steps',
               y_label='Average return',
               plot_name='./mf_comparison.png',
               num_rows=2,
               x_limits=[0, 1e7],
               report_max_performance=True,
               log_scale=True,
               round_x=10000,
               )