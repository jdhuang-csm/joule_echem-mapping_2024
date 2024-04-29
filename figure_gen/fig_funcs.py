# Utility funcions for plotting

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import numpy as np
import os
import colorsys
from scipy.interpolate import interp1d

plotdir = './figures'

def set_plotdir(path):
    global plotdir
    plotdir = path


def set_plot_formatting():
	plt.rcParams['font.size'] = 8
	# plt.rcParams['axes.labelsize'] = 8
	plt.rcParams['legend.fontsize'] = 8
	# plt.rcParams['legend.title_fontsize'] = 8
	plt.rcParams['xtick.labelsize'] = 7
	plt.rcParams['ytick.labelsize'] = 7
	
	half_width = 112 / 25.4 
	full_width = 172 / 25.4 

	default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
	# colors = [(74,111,227), (211,63,106), (239,151,8), (17,198,56)]
	# colors = [np.array(c) / 255 * 0.9 for c in colors]
	
	dark_colors = [adjust_lightness('#0000a7', 1.25), adjust_lightness('#c1272d', 0.8), '#008176']
	base_colors = [adjust_lightness(c, 1.25) for c in dark_colors]
	light_colors = [adjust_lightness(c, 1.25) for c in base_colors]

	# linestyles = ['-', '--', '-.', ':']
	# markers = ['o', '^', 's', 'v']
	# line_formats = [{'ls': ls, 'c': c} for ls, c in zip(linestyles, colors)]

	# data_kw = dict(facecolors='none', edgecolors='k', alpha=0.3, s=10)
	
	return full_width, half_width, light_colors, base_colors, dark_colors


def add_letters(axes, loc=(-0.25, 1.0), size=9, start_index=0):
	axes = np.atleast_1d(axes)
	if type(loc) == tuple:
		loc = [loc] * len(axes.ravel())
	for i, ax in enumerate(axes.ravel()):
		ax.text(*loc[i], '{}'.format(chr(97 + start_index + i)).upper(), transform=ax.transAxes, 
				ha='left', va='top', fontsize=size, fontweight='bold')


def savefig(fig, name, ext='pdf', dpi=300, **kw):
	fig.savefig(os.path.join(plotdir, f'{name}.{ext}'), dpi=dpi, **kw)
	
	
def expand_axlim(ax, axis, frac):
	for axis_i in axis:
		lim = getattr(ax, f'get_{axis_i}lim')()
		range = lim[1] - lim[0]
		extend = frac * range / 2
		new_lim = (lim[0] - extend, lim[1] + extend)
		getattr(ax, f'set_{axis_i}lim')(new_lim)
		

def zero_axlim(ax, axis):
	lims = getattr(ax, f'get_{axis}lim')()
	getattr(ax, f'set_{axis}lim')(np.array(lims) - lims[0])
	
	
def get_gs_bounds(nrow, ncol, left=0.03, right=1.04, bottom=0.03, top=1.05, hspace=0.1, wspace=0.06):
	# get bounds for individual gridspecs
	hrange = top - bottom
	top_bounds = top - hspace - (1 / nrow) * hrange * np.arange(nrow)
	bot_bounds = top + hspace - (1 / nrow) * hrange * np.arange(1, nrow + 1)

	wrange = right - left
	left_bounds = left + wspace + (1 / ncol) * wrange * np.arange(ncol)
	right_bounds = left -wspace + (1 / ncol) * wrange * np.arange(1, ncol + 1)

	return {'left': left_bounds, 'right': right_bounds, 'bottom': bot_bounds, 'top': top_bounds}
	

def make_gridspec(fig, gs_bounds, nrow, ncol, fig_row, fig_col, **kw):
	hbounds = {k: gs_bounds[k][fig_row] for k in ['top', 'bottom']}
	wbounds = {k: gs_bounds[k][fig_col] for k in ['left', 'right']}
	bounds = hbounds | wbounds
	gs_kw = dict(bounds, **kw)
	
	return fig.add_gridspec(nrow, ncol, **gs_kw)
	
	
def adjust_lightness(color, amount):
	try:
		c = mpl.colors.cnames[color]
	except:
		c = color
	c = colorsys.rgb_to_hls(*mpl.colors.to_rgb(c))
	# print(c)
	return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
	
	
def set_sl(color, s, l):
	try:
		c = mpl.colors.cnames[color]
	except:
		c = color
	c = colorsys.rgb_to_hls(*mpl.colors.to_rgb(c))
	# Keep hue, adjust saturation and lightness
	return colorsys.hls_to_rgb(c[0], l, s)
	

def make_color_seq(base_color, length, start_lightness=1.5, end_lightness=0.5):
	lightness_vals = np.linspace(start_lightness, end_lightness, length)
	colors = [adjust_lightness(base_color, amt) for amt in lightness_vals]
	return colors
	

def colorline(ax, x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
	"""
	http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
	http://matplotlib.org/examples/pylab_examples/multicolored_line.html
	Plot a colored line with coordinates x and y
	Optionally specify colors in the array z
	Optionally specify a colormap, a norm function and a line width
	"""
	# path = mpath.Path(np.column_stack([x, y]))
	# verts = path.interpolated(steps=3).vertices
	# x, y = verts[:, 0], verts[:, 1]

	# Default colors equally spaced on [0,1]:
	if z is None:
		z = np.linspace(0.0, 1.0, len(x))

	# Special case if a single number:
	if not hasattr(z, "__iter__"):	# to check for numerical input -- this is a hack
		z = np.array([z])

	z = np.asarray(z)

	segments = make_segments(x, y)
	lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
							  linewidth=linewidth, alpha=alpha)

	ax.add_collection(lc)

	return lc


def make_segments(x, y):
	"""
	Create list of line segments from x and y coordinates, in the correct format
	for LineCollection: an array of the form numlines x (points per line) x 2 (x
	and y) array
	"""

	points = np.array([x, y]).T.reshape(-1, 1, 2)
	segments = np.concatenate([points[:-1], points[1:]], axis=1)
	return segments
	
	
def y_to_alpha(y, y1, y2, alpha1, alpha2):
    d1 = np.abs(y - y1)
    d2 = np.abs(y - y2)
    deno = np.maximum(d1 + d2, 1e-8)
    return (d1 * alpha2 + d2 * alpha1) / deno
	

def gradient_fill_between(x, y1, y2, alpha1=0, alpha2=1, fill_color=None, ax=None, nx=50, ny=50, **kwargs):
    """
    Plot a line with a linear alpha gradient filled beneath it.
    Adapted from Joe Kington's answer at https://stackoverflow.com/questions/29321835/is-it-possible-to-get-color-gradients-under-curve-in-matplotlib

    Parameters
    ----------
    x, y : array-like
        The data values of the line.
    fill_color : a matplotlib color specifier (string, tuple) or None
        The color for the fill. If None, the color of the line will be used.
    ax : a matplotlib Axes instance
        The axes to plot on. If None, the current pyplot axes will be used.
    Additional arguments are passed on to matplotlib's ``plot`` function.

    Returns
    -------
    line : a Line2D instance
        The line plotted.
    im : an AxesImage instance
        The transparent gradient clipped to just the area beneath the curve.
    """
    if ax is None:
        ax = plt.gca()

    line1, = ax.plot(x, y1, alpha=alpha1, **kwargs)
    line2, = ax.plot(x, y2, alpha=alpha2, **kwargs)
    if fill_color is None:
        fill_color = line1.get_color()

    zorder = line1.get_zorder()
#     alpha = line1.get_alpha()
#     alpha = 1.0 if alpha is None else alpha

    # Get image data limits
    xmin, xmax, ymin, ymax = x.min(), x.max(), min(y1.min(), y2.min()), max(y1.max(), y2.max())
    
    # Set up image color/alpha values
    x_grid = np.linspace(xmin, xmax, nx)
    y_grid = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(x_grid, y_grid)
    
    y1_interp = interp1d(x, y1)(xx)
    y2_interp = interp1d(x, y2)(xx)
    
    z = np.empty((ny, nx, 4), dtype=float)

    rgb = mpl.colors.colorConverter.to_rgb(fill_color)
    z[:,:,:3] = rgb
    z[:,:,-1] = y_to_alpha(yy, y1_interp, y2_interp, alpha1, alpha2)

    
    im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                   origin='lower', zorder=zorder)

    xy1 = np.column_stack([x, y1])
    xy2 = np.column_stack([x, y2])
    xy = np.vstack([xy1, xy2[::-1], xy1[0]])
    clip_path = mpl.patches.Polygon(xy, facecolor='none', edgecolor='none', closed=True)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)

    ax.autoscale(True)
    return line1, im
	
	
def format_table(data, col_names=None, fmt=None):
	if col_names is not None:
		header = ' & '.join(col_names) + ' \\\n\hline\n'
	else:
		header = ''
	
	txt = []
	for i in range(len(data)):
		row = list(data[i])
		if format is None:
			row = [str(r) for r in row]
		else:
			row = ['{:{fmt}}'.format(r, fmt=fmt) for r in row]
		txt.append(' & '.join(row))
		
	txt = '\\\\\n'.join(txt)
	txt = header + txt
	print(txt)
	
	
def sci_not(x, prec):
	pwr = np.floor(np.log10(x))
	factor = x / (10 ** pwr)
	return '${:.{prec}f} \\times 10^{{{:.0f}}}$'.format(factor, pwr, prec=prec)
	
	
	
