import numpy as np
from pyBat import Signal
from pyBat import Wiegrebe
from pyBat import Acoustics
import time

emission_duration = 9 / 1000
f1 = 50000
f2 = 30000
center = int((f1 + f2) / 2)

emission_db = 130
speed_of_sound = 343
fs = 250000

_, call = Signal.sweep(f1, f2, emission_duration, sample_rate=fs)
length_call = len(call)
model = Wiegrebe.ModelWiegrebe(fs, center, bands=5)

def random_delays(nr, max_delay):
    random_values = np.random.uniform(low=0.0005, high=max_delay, size=nr)
    return random_values

def get_decibels(pulses, side):
    if side == 'left':
        decibels = pulses.left.values + emission_db
    else:
        decibels = pulses.right.values + emission_db
    decibels[decibels < 0] = 0
    return decibels

# def get_delays(pulses, max_delay):
#     distances = pulses.distance.values
#     delays = (2 * distances) / speed_of_sound # Default delays
#     values_to_search = ['calls', 'echoes2', 'walls2'] # These are the pulse types for which random delays are generated
#     indices = pulses.index[pulses['pulse_type'].isin(values_to_search)]
#     num_indices = len(indices)
#     rd = random_delays(num_indices, max_delay)
#     delays[indices] = rd
#     return delays


def get_delays(pulses, max_delay):
    distances = pulses.distance.values
    delays = (2 * distances) / speed_of_sound  # Default delays
    values_to_search = ['calls', 'echoes2', 'walls2']  # Pulse types for which random delays are generated
    # Get the positional indices corresponding to the matching pulse types
    indices = pulses.index[pulses['pulse_type'].isin(values_to_search)].to_numpy()
    positional_indices = pulses.index.get_indexer(indices)  # Map to 0-based positions
    # Generate random delays
    num_indices = len(positional_indices)
    rd = random_delays(num_indices, max_delay)
    # Update the delays array
    delays[positional_indices] = rd
    return delays

def pulses2cochlear(pulses, side, condition, delays, max_delay):
    selected = pulses.query('maxed > -110000')
    max_number_of_samples = int(max_delay * fs)
    if condition == 'no_others': selected = selected.query('pulse_type != "echoes"')
    if condition == 'no_others': selected = selected.query('pulse_type != "calls"')
    selected = selected.reset_index()

    start = time.time()
    decibels = get_decibels(selected, side=side)

    impulse_responses = Acoustics.make_impulse_response(delays, decibels, emission_duration, fs)
    impulses = impulse_responses['ir_result']

    convolved_signal = np.convolve(call, impulses, mode='full')
    convolved_signal = Signal.dechirp(call, convolved_signal)
    model.run_model(convolved_signal)
    model_result = model.result
    model_result[max_number_of_samples:] = 0
    end = time.time()
    duration = end - start
    internals = {}
    internals['model'] = model
    internals['duration'] = duration
    internals['impulses'] = impulses
    internals['convolved_signal'] = convolved_signal
    internals['model_result'] = model_result
    return model_result, internals

