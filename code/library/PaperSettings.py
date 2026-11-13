from matplotlib import pyplot
from os import path
import numpy
import matplotlib.lines as mlines
import easygui

# Some parameters
wingspan = 0.13
bw = 0.06 #for plotting the KDEs
throw_away_steps = 50 #do not count distances and collisions in the first n steps

# Paths for saving and reading data
code_image_folder = 'output'
paper_image_folder = 'output' # This is a secondary folder for saving images
results_folder = 'results'

result_folder_exists = path.exists(results_folder)
paper_image_folder_exists = path.exists(paper_image_folder)
if not paper_image_folder_exists: paper_image_folder = code_image_folder

if not result_folder_exists:
    msg = '''
    You need to provide an existing folder to which to save the results.
    Set the [results_folder] variable in PaperSettings.py
    '''
    easygui.msgbox(msg)


# Set the fonts for plots
font = {'family': 'serif', 'serif': ['DejaVu Serif']}
pyplot.rc('font', **font)
# Tol's bright colors
pulse_type_colors = numpy.array(["#EE6677", "#228833", "#4477AA", "#CCBB44", "#66CCEE", "#AA3377"])
# Colors from the Paul Tol's "muted" set
red ='#882255'
blue = '#332288'
green = '#117733'
# Face colors for plots
facecolor1 = '#FAFAFA'
facecolor2 = '#F5F5F5'
corridor_color = '#F7F6F2'

# These are some support variables used for plotting the data
pulse_labels = ['Bodies', 'Walls', 'Calls', 'Bodies$_2$', 'Walls$_2$', 'Perched']
legend_handles = []
for i in range(5):
    handle = mlines.Line2D([], [], color=pulse_type_colors[i], marker='.', linestyle='None', markersize=10, label=pulse_labels[i])
    legend_handles.append(handle)

# This is the range over which we want to plot the empirical bat distance data
# This avoids outliers
distance_distribution_range = (0, 3)

# support functions for saving images
def save_figure(figure_type, arena, condition):
    base_name = figure_type + '_' + arena + '_' + condition + '.pdf'
    save_figure_with_base_name(base_name)

def save_figure_with_base_name(base_name):
    full_paper_name = path.join(paper_image_folder, base_name)
    full_code_name = path.join(code_image_folder, base_name)
    pyplot.savefig(full_paper_name)
    pyplot.savefig(full_code_name)

