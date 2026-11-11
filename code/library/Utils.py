import os
import shutil
import sys
from os.path import join

import easygui
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import natsort
import numpy
import pandas
from matplotlib import pyplot
from scipy.stats import truncnorm

from library import PaperSettings
from pyBat import FileOperations
from pyBat import SaveLoad
from pyBat import Visualize
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde

def count_hits(df, separation=5):
    def count_instances(group):
        # Ensure 'step' is numeric
        group['step'] = pd.to_numeric(group['step'], errors='coerce')
        # Sort by step
        group = group.sort_values('step')
        # Calculate step differences
        group['step_diff'] = group['step'].diff().fillna(float(separation))
        # Mark rows with sufficient step difference
        group['instance'] = (group['step_diff'] >= separation).cumsum()
        # Count unique instances
        return group['instance'].nunique()
    # Group by replication and bat, count valid instances, then sum across all replications
    result = df.groupby(['replication']).apply(count_instances)
    if result.shape[0] == 0: return pd.DataFrame(columns=['replication', 'hits'])
    result = result.reset_index()
    result.columns = ['replication', 'hits']
    return result
    #return df.groupby(['replication', 'bat']).apply(count_instances).sum()

def get_last_part_of_path(path):
    return os.path.basename(os.path.normpath(path))


def create_folder(folder_path, ask=False):
    choices = ['Clear it', 'Abort', 'Keep it']
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        if not ask: return
        choice = easygui.choicebox( f"Folder '{folder_path}' already exists.", choices=choices)
        if choice == 'Clear it':
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)
        if choice == 'Abort':
            sys.exit(0)


def get_hist_data(axes, index, x_value=False, plot=False):
    lines = axes.get_lines()
    line = lines[index]
    x_data = line.get_xdata()
    y_data = line.get_ydata()
    max_index = np.argmax(y_data)
    x_provided = True
    if not x_value:
        x_provided = False
        x_value = np.array(x_data[max_index])
    y_interpolated = np.interp(x_value, x_data, y_data)
    if plot:
        pyplot.figure()
        pyplot.plot(x_data, y_data)
        pyplot.scatter(x_value, [y_interpolated], 'ro')
        pyplot.show()
    if x_provided: return y_interpolated
    return x_value, y_interpolated


def map_colors(data, max_value):
    norm = mcolors.Normalize(vmin=0, vmax=max_value)
    cmap = cm.viridis  # You can choose any colormap you prefer
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = mappable.to_rgba(data)
    return colors


def overall_max(array1, array2):
    maximum = numpy.maximum(array1, array2)
    max = numpy.max(maximum)
    return max


def get_time_axis(array, fs):
    n = len(array)
    max_time = n / fs
    axis = numpy.linspace(0, max_time, n)
    return axis


def max_point(time_axis, series):
    max_index = numpy.argmax(series)
    x = time_axis[max_index]
    y = series[max_index]
    return x, y


def series_range(series):
    min_v = numpy.min(series)
    max_v = numpy.max(series)
    rng = max_v - min_v
    return rng


def truncated_normal(mean, sd, lower, upper):
    a = (lower - mean) / sd
    b = (upper - mean) / sd
    return truncnorm.rvs(a, b, loc=mean, scale=sd)


def symmetric_uniform(array, rng):
    result = numpy.random.uniform(-rng, rng, array.shape)
    return result


def real_bat_data():
    data = pandas.read_csv('ScienceDirect_files_28Nov2023/closest.csv')
    data = data.closest.values
    return data

def plot_swarm(result_file, step_size=10, plot_time=True, plot_types=False):

    results_folder = PaperSettings.results_folder
    selected = join(results_folder, result_file)
    selected_results = SaveLoad.pickle_load(selected)
    selected_swarm = selected_results['swarm']


    outside_positions = numpy.array(selected_swarm.outside_positions)
    shape_outside_positions = outside_positions.shape
    nr_of_outside_positions = shape_outside_positions[0]

    selected_arena = selected_results['arena']
    sorted_edge_points = selected_arena['sorted_edge_points']
    end_region = selected_arena['end_region']
    all_bats = selected_swarm.all_bats

    colors = PaperSettings.pulse_type_colors

    max_steps = 0
    for current_bat in all_bats:
        pulse_types = current_bat.get_history('pulse_type_code')
        steps = len(pulse_types)
        max_steps = max(max_steps, steps)

    # pyplot.figure()
    nr_of_plotted_bats = 0
    for current_bat in all_bats:
        x_path, y_path, _ = current_bat.get_path()
        pulse_types = current_bat.get_history('pulse_type_code')

        steps = len(pulse_types)
        selected_steps = numpy.arange(0, steps, step_size)

        selected_x = x_path[selected_steps]
        selected_y = y_path[selected_steps]

        if plot_time:
            pyplot.plot(x_path, y_path, color='k', alpha=0.5)
            time_colors = map_colors(selected_steps, max_value=max_steps)
            pyplot.scatter(selected_x, selected_y, c=time_colors, s=2, zorder=100)

        if plot_types:
            if nr_of_plotted_bats < 30:
                selected_colors = colors[pulse_types]
                selected_colors = selected_colors[selected_steps]
                pyplot.scatter(selected_x, selected_y, c=selected_colors, s=10, zorder=100)
            if nr_of_plotted_bats < 1:
                pyplot.plot(x_path, y_path, color='k', alpha=0.5)

        if not plot_time and not plot_types:
            pyplot.plot(x_path, y_path, color='k', alpha=0.5)

        nr_of_plotted_bats = nr_of_plotted_bats + 1

    edges_x = sorted_edge_points[:-1, 0]
    edges_y = sorted_edge_points[:-1, 1]

    base_name = selected_arena['base_name']
    if base_name not in ['open']:
        pyplot.plot(edges_x, edges_y, color='k', alpha=0.5)
    else:
        pyplot.scatter(edges_x, edges_y, color='gray', s=1)

    end_region_x = end_region[:, 0]
    end_region_y = end_region[:, 1]
    pyplot.scatter(end_region_x, end_region_y, s=5)

    pyplot.xlabel('X coordinate (m)')
    pyplot.ylabel('Y coordinate (m)')

    unique_positions = numpy.unique(outside_positions, axis=0)
    nr_unique_positions = unique_positions.shape[0]

    #if nr_of_outside_positions > 0: pyplot.scatter(outside_positions[:, 2], outside_positions[:, 3], color='r', s=20)

    Visualize.square_axis()
    return selected_results


def aggregate_data(condition_pattern, n=None, select_last=False):
    results_folder = PaperSettings.results_folder
    files = FileOperations.get_files(results_folder, pattern=condition_pattern)
    files = natsort.natsorted(files)

    if n is not None:
        files = files[-n:] if select_last else files[:n]

    all_pulse_data = []
    all_bat_data = []
    all_outside_positions = []
    replication = 0
    for file in files:
        selected = join(results_folder, file)
        selected_results = SaveLoad.pickle_load(selected)
        selected_swarm = selected_results['swarm']

        outside_positions = selected_swarm.outside_positions
        outside_positions = pandas.DataFrame(outside_positions, columns=['bat', 'step', 'x', 'y'])
        outside_positions['replication'] = replication
        size = outside_positions.shape[0]
        if size > 0: all_outside_positions.append(outside_positions)
        all_bats = selected_swarm.all_bats

        bat_data = selected_swarm.export_bat_data()
        bat_data['file'] = file
        bat_data['replication'] = replication
        all_bat_data.append(bat_data)

        for current_bat in all_bats:
            pulse_types = current_bat.get_history('pulse_type_code')
            maxed = current_bat.get_history('maxed')
            simulation_step = current_bat.get_history('simulation_step')
            pulse_types = pandas.DataFrame({'bat': current_bat.id, 'replication': replication, 'pulse_type': pulse_types, 'maxed': maxed, 'simulation_step': simulation_step})
            all_pulse_data.append(pulse_types)

        replication = replication + 1

    all_bat_data = pandas.concat(all_bat_data)
    all_pulse_data = pandas.concat(all_pulse_data)
    #types_count = count_types(all_pulse_types, cutoff=-110)
    if len(all_outside_positions) > 0:
        all_outside_positions = pandas.concat(all_outside_positions)
    else:
        all_outside_positions = pandas.DataFrame()

    result = {}
    result['all_pulse_data'] = all_pulse_data
    result['all_bat_data'] = all_bat_data
    result['all_outside_positions'] = all_outside_positions
    return result





def summarize_per_replication(df, column, stat="mode", cutoff=None, bandwidth=0.2):
    """
    Computes summary statistics for a given column, grouped by 'file'.

    Parameters:
    df (pd.DataFrame): The input dataframe with a 'file' column.
    column (str): The column to analyze.
    stat (str): The statistic to compute. Options: 'mode', 'mode_kde', 'mean', 'median', 'std', 'min', 'max', 'below_cutoff'.
    cutoff (float, optional): Required if stat='below_cutoff', defines the threshold.
    bandwidth (float, optional): Bandwidth for KDE when using 'mode_kde'.

    Returns:
    pd.DataFrame: Summary statistics per file.
    """

    def kde_mode(x, bw):
        """Estimate the mode using Kernel Density Estimation (KDE)."""
        if len(x) < 2:
            return np.nan  # KDE requires at least two points
        kde = gaussian_kde(x, bw_method=bw)
        x_grid = np.linspace(min(x), max(x), 1000)
        return x_grid[np.argmax(kde(x_grid))]

    if stat == "mode":
        result = df.groupby("replication")[column].agg(lambda x: x.mode().iat[0] if not x.mode().empty else None)
    elif stat == "mode_kde":
        result = df.groupby("replication")[column].apply(lambda x: kde_mode(x.dropna(), bandwidth))
    elif stat == "mean":
        result = df.groupby("replication")[column].mean()
    elif stat == "median":
        result = df.groupby("replication")[column].median()
    elif stat == "std":
        result = df.groupby("replication")[column].std()
    elif stat == "min":
        result = df.groupby("file")[column].min()
    elif stat == "max":
        result = df.groupby("replication")[column].max()
    elif stat == "below_cutoff":
        if cutoff is None:
            raise ValueError("cutoff must be provided for 'below_cutoff' statistic.")
        result = df.groupby("replication")[column].apply(lambda x: (x < cutoff).mean() * 100)
    else:
        raise ValueError(
            f"Unknown statistic '{stat}'. Choose from 'mode', 'mode_kde', 'mean', 'median', 'std', 'min', 'max', 'below_cutoff'.")

    return result.reset_index().rename(columns={column: stat})

