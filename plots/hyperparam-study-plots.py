from rllab.viskit import frontend
from rllab.viskit import core
import matplotlib.pyplot as plt
from plots.plot_utils import *

plt.style.use('ggplot')
#plt.rc('font', family='Times New Roman')
import matplotlib
# matplotlib.use('TkAgg')
#matplotlib.font_manager._rebuild()

data_path_fast_lr = '/home/ignasi/Desktop/hyperparams/hyperparam-study-fast-lr'
data_path_num_models = '/home/ignasi/Desktop/hyperparams/hyperparam-study-num-models'
data_path_maml_iter = '/home/ignasi/Desktop/hyperparams/hyperparam-study-maml-iter'

exps_data_fast_lr = core.load_exps_data([data_path_fast_lr], False)
exps_data_num_models = core.load_exps_data([data_path_num_models], False)
exps_data_maml_iter = core.load_exps_data([data_path_maml_iter], False)

SMALL_SIZE = 30
MEDIUM_SIZE = 32
BIGGER_SIZE = 36
LINEWIDTH = 4

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
matplotlib.rcParams['lines.linewidth'] = LINEWIDTH
def plot_from_exps(exps_data_fast_lr,
                   exps_data_num_models,
                   exps_data_maml_iter,
                   filters={},
                   split_figures_by=None,
                   split_plots_by=None,
                   x_key='n_timesteps',
                   y_key=None,
                   plot_name='./hyperparam-study',
                   subfigure_titles=None,
                   plot_labels=None,
                   x_label=None,
                   y_label=None
                   ):

    #exp_data = filter(exp_data, filters=filters)
    #exps_per_plot = group_by(exp_data, group_by_key=split_figures_by)
    fig, axarr = plt.subplots(1, 3, figsize=(30, 8))
    fig.tight_layout(pad=4., w_pad=1.5, h_pad=3.0, rect=[0, 0, 1, 1])

    # x axis formatter
    xfmt = matplotlib.ticker.ScalarFormatter()
    xfmt.set_powerlimits((3, 3))

    # inner learning rate
    i = 0
    plots_in_figure_exps = group_by(exps_data_fast_lr, group_by_key='fast_lr')
    subfigure_title = r'Inner Learning Rate $\alpha$'
    axarr[i].set_title(subfigure_title)
    axarr[i].xaxis.set_major_formatter(xfmt)
    axarr[i].xaxis.set_major_locator(plt.MaxNLocator(5))

    # iterate over plots in figure
    for j, (default_label, exps) in enumerate(sorted(plots_in_figure_exps.items())):
        x, y_mean, y_std = prepare_data_for_plot(exps, x_key=x_key, y_key=y_key)

        label = default_label
        axarr[i].plot(x, y_mean, label=label)
        axarr[i].fill_between(x, y_mean + y_std, y_mean - y_std, alpha=0.2)

        # axis labels
        axarr[i].set_xlabel(x_label if x_label else x_key)
        axarr[i].set_ylabel(y_label if y_label else y_key)

        axarr[i].set_xlim(0, 200000)
        axarr[i].legend(loc='lower right', ncol=2, bbox_transform = plt.gcf().transFigure)

    # Number of models
    i = 1
    plots_in_figure_exps = group_by(exps_data_num_models, group_by_key='num_models')
    subfigure_title = 'Number of Models'
    axarr[i].set_title(subfigure_title)
    axarr[i].xaxis.set_major_formatter(xfmt)
    axarr[i].xaxis.set_major_locator(plt.MaxNLocator(5))

    # iterate over plots in figure

    for j, (default_label, exps) in enumerate(sorted([(int(k), v) for k, v in plots_in_figure_exps.items()])):
        x, y_mean, y_std = prepare_data_for_plot(exps, x_key=x_key, y_key=y_key)

        label = default_label
        axarr[i].plot(x, y_mean, label=label)
        axarr[i].fill_between(x, y_mean + y_std, y_mean - y_std, alpha=0.2)

        # axis labels
        axarr[i].set_xlabel(x_label if x_label else x_key)
        axarr[i].set_ylabel(y_label if y_label else y_key)

        axarr[i].set_xlim(0, 200000)
        axarr[i].legend(loc='lower right', ncol=2, bbox_transform = plt.gcf().transFigure)


    # Number of maml steps per iter
    i = 2
    plots_in_figure_exps = group_by(exps_data_maml_iter, group_by_key='num_maml_steps_per_iter')
    subfigure_title = 'Meta Gradient Steps per Iteration'
    axarr[i].set_title(subfigure_title)
    axarr[i].xaxis.set_major_formatter(xfmt)
    axarr[i].xaxis.set_major_locator(plt.MaxNLocator(5))

    # iterate over plots in figure

    for j, (default_label, exps) in enumerate(sorted([(int(k), v) for k, v in plots_in_figure_exps.items()])):
        x, y_mean, y_std = prepare_data_for_plot(exps, x_key=x_key, y_key=y_key)

        label = default_label
        axarr[i].plot(x, y_mean, label=label)
        axarr[i].fill_between(x, y_mean + y_std, y_mean - y_std, alpha=0.2)

        # axis labels
        axarr[i].set_xlabel(x_label if x_label else x_key)
        axarr[i].set_ylabel(y_label if y_label else y_key)

        axarr[i].set_xlim(0, 200000)
        axarr[i].legend(loc='lower right', ncol=2, bbox_transform=plt.gcf().transFigure)

    fig.savefig(plot_name + '.png')
    fig.savefig(plot_name + '.pdf')



plot_from_exps(exps_data_fast_lr,
               exps_data_num_models,
               exps_data_maml_iter,
               split_plots_by='fast_lr',
               y_key='EnvTrajs-AverageReturn',
               subfigure_titles=['HalfCheetah - output_bias_range [0.0, 0.1]',
                                'HalfCheetah - output_bias_range [0.0, 0.5]',
                                'HalfCheetah - output_bias_range [0.0, 1.0]'],
               x_label='Time steps',
               y_label='Average return',
               )