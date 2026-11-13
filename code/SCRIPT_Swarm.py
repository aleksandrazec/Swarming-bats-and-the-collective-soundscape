import matplotlib as mpl
mpl.use("TkAgg")

import copy
import time

from os import path

from matplotlib import pyplot

from library import PaperSettings
from library import ReadArena
from library import Swarm
from library import Utils
#from pyBat import PushOver
from pyBat import SaveLoad


arena_names = ['branched'] #default 'branched'
conditions = ['default'] #default 'default'

replications = 1
number_of_bats = 5 #default 5
max_steps = 55 # default 55
Utils.create_folder(PaperSettings.results_folder)

#%%
for arena_name in arena_names:
    for condition in conditions:
        pyplot.close('all')
        start_time = time.time()
        for replication in range(replications):
            pyplot.close('all')
            replication_string = '%0.3i' % replication
            result_output_file = 'run_' + condition + '_' + arena_name + '_' + replication_string + '.pck'
            result_output_file = path.join(PaperSettings.results_folder, result_output_file)
            arena = ReadArena.read_arena(arena_name, scale=0.1, plot=False)
            test_swarm = Swarm.Swarm(number_of_bats, arena=arena, condition=condition, replication=replication)

            # Run
            positions = test_swarm.bat_positions
            distances = []
            number_of_active_bats = []
            for step in range(max_steps):
                end_time = time.time()
                #print(step, 'Duration', end_time - start_time)
                number_of_active_bats.append(len(test_swarm.active_bats))
                if len(test_swarm.active_bats) < number_of_bats / 2: break
                test_swarm.set_perched_bats(proportion=0.25)
                if condition == 'perched2': test_swarm.set_perched_bats(proportion=0.5)
                test_swarm.simulation_step()
                if arena_name != 'open': test_swarm.remove_bats_in_end_region()
                closest_distances = test_swarm.get_closest_distances()
                closest_distances = list(closest_distances)
                distances = distances + closest_distances

            test_swarm.plot_traces()
            bat_data = test_swarm.export_bat_data()

            result_output = {}
            result_output['number_of_bats'] = number_of_bats
            result_output['steps'] = step
            result_output['swarm'] = copy.copy(test_swarm)
            result_output['arena_name'] = arena_name
            result_output['arena'] = arena
            result_output['replication'] = replication
            result_output['number_of_active_bats'] = number_of_active_bats
            SaveLoad.pickle_save(result_output_file, result_output)

            stamp = time.asctime()
            bullet_message = f"{stamp}, Replication {replication + 1} of {replications}, arena {arena_name}, condition {condition}"
            #PushOver.send(bullet_message)

        pyplot.figure()
        pyplot.hist(bat_data.closest, alpha=0.5, bins=250, density=True, cumulative=True)
        pyplot.vlines(0.15, 0, 1)
        pyplot.show()

        pyplot.figure()
        pyplot.hist(bat_data.maxed)
        pyplot.show()

        pyplot.figure()
        pyplot.hist(bat_data.speed)
        pyplot.show()
        #%%
        pyplot.figure()
        Utils.plot_swarm(path.basename(result_output_file))
        pyplot.show()
