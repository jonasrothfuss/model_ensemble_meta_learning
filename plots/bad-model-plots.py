from rllab.viskit import frontend
from rllab.viskit import core
import matplotlib.pyplot as plt
from plots.plot_utils import *

plt.style.use('ggplot')
#plt.rc('font', family='Times New Roman')
import matplotlib
matplotlib.use('TkAgg')
#matplotlib.font_manager._rebuild()

data_path = '/home/jonasrothfuss/Dropbox/Eigene_Dateien/UC_Berkley/2_Code/model_ensemble_meta_learning/data/s3/bad-models'
exps_data = core.load_exps_data([data_path], False)


def plot_from_exps(exp_data,
                   filters={},
                   split_figures_by=None,
                   split_plots_by=None,
                   x_key='n_timesteps',
                   y_key=None,
                   plot_name='./bad-models',
                   subfigure_titles=None,
                   plot_labels=None,
                   x_label=None,
                   y_label=None,
                   fontsize=11):

    exp_data = filter(exp_data, filters=filters)
    exps_per_plot = group_by(exp_data, group_by_key=split_figures_by)
    fig, axarr = plt.subplots(1, len(exps_per_plot.keys()), figsize=(15, 4))
    fig.tight_layout(pad=3.0, w_pad=4.0, h_pad=3.0, rect=[0, 0, 0.95, 1])

    # x axis formatter
    xfmt = matplotlib.ticker.ScalarFormatter()
    xfmt.set_powerlimits((3, 3))

    # iterate over subfigures
    for i, (default_plot_title, plot_exps) in enumerate(sorted(exps_per_plot.items())):
        plots_in_figure_exps = group_by(plot_exps, split_plots_by)
        subfigure_title = subfigure_titles[i] if subfigure_titles else default_plot_title
        axarr[i].set_title(subfigure_title, fontsize=fontsize)
        axarr[i].xaxis.set_major_formatter(xfmt)
        axarr[i].xaxis.set_major_locator(plt.MaxNLocator(5))

        # iterate over plots in figure
        for j, (default_label, exps) in enumerate(sorted(plots_in_figure_exps.items())):
            x, y_mean, y_std = prepare_data_for_plot(exps, x_key=x_key, y_key=y_key)

            label = plot_labels[j] if plot_labels else default_label
            label = label if i == 0 else "__nolabel__"
            axarr[i].plot(x, y_mean, label=label)
            axarr[i].fill_between(x, y_mean + y_std, y_mean - y_std, alpha=0.2)

            # axis labels
            axarr[i].set_xlabel(x_label if x_label else x_key, fontsize=fontsize)
            axarr[i].set_ylabel(y_label if y_label else y_key, fontsize=fontsize)

    fig.legend(loc='center right', ncol=1, bbox_transform = plt.gcf().transFigure)
    fig.savefig(plot_name + '.png')
    fig.savefig(plot_name + '.pdf')


filter_dict = {'output_bias_range': [0, 0.1]}

exps_data_filtered = filter(exps_data, filter_dict)


plot_from_exps(exps_data,
               split_figures_by='output_bias_range',
               split_plots_by='exp_prefix',
               y_key='EnvTrajs-AverageReturn',
               subfigure_titles=['output_bias_range [0.0, 0.1]',
                                'output_bias_range [0.0, 0.5]',
                                'output_bias_range [0.0, 1.0]'],
               plot_labels=['ME-MPG', 'ME-TRPO'],
               x_label='time steps',
               y_label='average return',
               )