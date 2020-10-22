"""Configure seaborn for plotting."""
import matplotlib

PAD_INCHES = 0.00

rc_params = {
             # FONT
             'font.size': 9,
             'font.sans-serif': 'Myriad Pro',
             'font.stretch': 'condensed',
             'font.weight': 'normal',
             'mathtext.fontset': 'custom',
             'mathtext.fallback_to_cm': True,
             'mathtext.rm': 'Minion Pro',
             'mathtext.it': 'Minion Pro:italic',
             'mathtext.bf': 'Minion Pro:bold:italic',
             'mathtext.cal': 'Minion Pro:italic',  # TODO find calligraphy font
             'mathtext.tt': 'monospace',
             # AXES
             'axes.linewidth': 0.5,
             'axes.spines.top': False,
             'axes.spines.right': False,
             'axes.labelsize': 9,
             # TICKS
             'xtick.major.size': 3.5,
             'xtick.major.width': 0.8,
             'ytick.major.size': 3.5,
             'xtick.labelsize': 8,
             'ytick.labelsize': 8,
             'ytick.major.width': 0.8,
             # LEGEND
             'legend.fontsize': 9,
             # SAVING FIGURES
             'savefig.bbox': 'tight',
             'savefig.pad_inches': PAD_INCHES,
             'pdf.fonttype': 42,
             'savefig.dpi': 300}


def mm2inch(*tupl, pad):
    inch = 25.4
    if isinstance(tupl[0], tuple):
        return tuple(i/inch - pad for i in tupl[0])
    else:
        return tuple(i/inch - pad for i in tupl)


def set_figure_size(*size):
    params = {'figure.figsize': mm2inch(size, pad=PAD_INCHES)}
    matplotlib.rcParams.update(params)
