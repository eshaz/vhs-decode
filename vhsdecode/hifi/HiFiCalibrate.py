
import argparse

from dataclasses import dataclass

import string
from random import SystemRandom
import csv

from numba import njit

from multiprocessing import (
    cpu_count,
    Pipe,
    SimpleQueue,
    Process,
    freeze_support,
    current_process,
    Event,
    resource_tracker,
    set_start_method
)
from multiprocessing.shared_memory import SharedMemory
from concurrent.futures import ProcessPoolExecutor, as_completed, wait

import numpy as np
import soundfile as sf

from vhsdecode.hifi.HiFiDecode import (
    Deemphasis,
    Expander,
    DEFAULT_EXPANDER_GAIN,
    DEFAULT_EXPANDER_RATIO,
    DEFAULT_EXPANDER_ATTACK_TAU,
    DEFAULT_EXPANDER_RELEASE_TAU,
    DEFAULT_VHS_EXPANDER_WEIGHTING_TAU_1,
    DEFAULT_VHS_EXPANDER_WEIGHTING_TAU_2,
    DEFAULT_VHS_EXPANDER_WEIGHTING_DB_PER_OCTAVE,
    DEFAULT_VHS_EXPANDER_WEIGHTING_BANDWIDTH,
    DEFAULT_VHS_DEEMPHASIS_TAU_1,
    DEFAULT_VHS_DEEMPHASIS_TAU_2,
    DEFAULT_VHS_DEEMPHASIS_DB_PER_OCTAVE,
    DEFAULT_VHS_DEEMPHASIS_BANDWIDTH,
)

default_threads = cpu_count()
parser = argparse.ArgumentParser(
    description="Derives De-Emphasis and Expander settings for hifi-decode"
)
parser.add_argument(
    "in_decoded",
    metavar="in_decoded",
    type=str,
    help="source decoded file",
    nargs="?",
    default="",
)
parser.add_argument(
    "in_reference",
    metavar="in_reference",
    type=str,
    help="source reference file",
    nargs="?",
    default="",
)
parser.add_argument(
    "--threads",
    "-t",
    metavar="threads",
    type=int,
    default=default_threads,
    help="number of CPU threads to use",
)


@dataclass
class CalibrateDecodedFile():
    name: str
    channels: int
    length: int
    sample_rate: int
    dtype: np.dtype

    def __init__(self, buffer_name, length, channels, sample_rate, dtype):
        self.name = buffer_name
        self.channels = channels
        self.length = length
        self.sample_rate = sample_rate
        self.dtype = dtype

@dataclass
class CalibrateResult():
    expander_attack_tau: float
    expander_release_tau: float
    expander_gain: float
    expander_ratio: float
    expander_weighting_tau_1: float
    expander_weighting_tau_2: float
    expander_weighting_db_per_octave: float
    expander_weighting_bandwidth: float
    deemphasis_tau_1: float
    deemphasis_tau_2: float
    deemphasis_db_per_octave: float
    deemphasis_bandwidth: float

    correlation_results = ()
    correlation_lags = ()

    keys = [
        'expander_attack_tau',
        'expander_release_tau',
        'expander_gain',
        'expander_ratio',
        'expander_weighting_tau_1',
        'expander_weighting_tau_2',
        'expander_weighting_db_per_octave',
        'expander_weighting_bandwidth',
        'deemphasis_tau_1',
        'deemphasis_tau_2',
        'deemphasis_db_per_octave',
        'deemphasis_bandwidth'
    ]

    def __init__(self,
        expander_attack_tau,
        expander_release_tau,
        expander_gain,
        expander_ratio,
        expander_weighting_tau_1,
        expander_weighting_tau_2,
        expander_weighting_db_per_octave,
        expander_weighting_bandwidth,
        deemphasis_tau_1,
        deemphasis_tau_2,
        deemphasis_db_per_octave,
        deemphasis_bandwidth
    ):
        self.expander_attack_tau = expander_attack_tau
        self.expander_release_tau = expander_release_tau
        self.expander_gain = expander_gain
        self.expander_ratio = expander_ratio

        self.expander_weighting_tau_1 = expander_weighting_tau_1
        self.expander_weighting_tau_2 = expander_weighting_tau_2
        self.expander_weighting_db_per_octave = expander_weighting_db_per_octave
        self.expander_weighting_bandwidth = expander_weighting_bandwidth

        self.deemphasis_tau_1 = deemphasis_tau_1
        self.deemphasis_tau_2 = deemphasis_tau_2
        self.deemphasis_db_per_octave = deemphasis_db_per_octave
        self.deemphasis_bandwidth = deemphasis_bandwidth
        

class CalibrateSharedMemory():
    def __init__(self, decoded_file: CalibrateDecodedFile):
        self.shared_memory = SharedMemory(name=decoded_file.name)
        resource_tracker.unregister(
            self.shared_memory._name,
            "shared_memory"
        )

        self.size = self.shared_memory.size
        self.buf = self.shared_memory.buf
        self.name = self.shared_memory.name
        self.close = self.shared_memory.close
        self.unlink = self.shared_memory.unlink

        self.channels = decoded_file.channels
        self.length = decoded_file.length
        self.dtype = decoded_file.dtype

        self.audio = np.ndarray(
            (self.channels, self.length),
            dtype=self.dtype,
            buffer=self.buf
        )

    @staticmethod
    def create_shared_memory(name, channels, length, dtype):
        byte_size = channels * length * np.dtype(dtype).itemsize

        # allow more than one instance to run at a time
        system_random = SystemRandom()
        name += "_" + "".join(
            system_random.choice(string.ascii_lowercase + string.digits)
            for _ in range(8)
        )

        # this instance must be saved in a variable that persists on both processes
        # Windows will remove the shared memory if it garbage collects the handle in any of the processes it is open in
        # https://stackoverflow.com/a/63717188
        shm = SharedMemory(size=byte_size, name=name, create=True)
        resource_tracker.unregister(
            shm._name,
            "shared_memory"
        )

        return shm, name


def decode_worker(in_file, conn):
    decoded_dtype = np.float32

    data, sample_rate = sf.read(in_file, dtype=decoded_dtype, always_2d=True)
    length = data.shape[0]
    channels = data.shape[1]

    # create and manage the shared memory from the parent process
    conn.send((channels, length, decoded_dtype))
    shared_memory_name = conn.recv()

    calibrate_decoded_file = CalibrateDecodedFile(shared_memory_name, length, channels, sample_rate, decoded_dtype)
    calibrate_shared_memory = CalibrateSharedMemory(calibrate_decoded_file)

    for i in range(channels):
        np.copyto(calibrate_shared_memory.audio[i], data[:, i])

    calibrate_shared_memory.close()

    conn.send(calibrate_decoded_file)

def decode_input_files(in_raw, in_reference):
    in_raw_parent, in_raw_child = Pipe()
    in_raw_process = Process(None, decode_worker, args=(in_raw, in_raw_child))
    in_raw_process.start()

    in_reference_parent, in_reference_child = Pipe()
    in_reference_process = Process(None, decode_worker, args=(in_reference, in_reference_child))
    in_reference_process.start()

    in_raw_channels, in_raw_length, in_raw_dtype = in_raw_parent.recv()
    in_raw_shm, in_raw_shm_name = CalibrateSharedMemory.create_shared_memory("hifi-raw", in_raw_channels, in_raw_length, in_raw_dtype)
    in_raw_parent.send(in_raw_shm_name)

    in_reference_channels, in_reference_length, in_reference_dtype = in_reference_parent.recv()
    in_reference_shm, in_reference_shm_name = CalibrateSharedMemory.create_shared_memory("hifi-ref", in_reference_channels, in_reference_length, in_reference_dtype)
    in_reference_parent.send(in_reference_shm_name)

    decoded_raw = in_raw_parent.recv()
    decoded_reference = in_reference_parent.recv()

    in_raw_process.join()
    in_reference_process.join()

    return in_raw_shm, in_reference_shm, decoded_raw, decoded_reference

def limited_xcorr_best(x, y, max_lag):
    x = np.asarray(x) - np.mean(x)
    y = np.asarray(y) - np.mean(y)

    corr = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            corr.append(np.dot(x[:lag], y[-lag:]))
        elif lag > 0:
            corr.append(np.dot(x[lag:], y[:-lag]))
        else:
            corr.append(np.dot(x, y))

    corr = np.array(corr)
    corr /= (np.std(x) * np.std(y) * len(x))

    idx = np.argmax(corr)
    return corr[idx], idx - max_lag


@njit(nogil=True,fastmath=True,cache=True)
def correlate(decoded_processed_channel, decoded_reference_channel):
    return np.corrcoef(decoded_processed_channel, decoded_reference_channel)[0, 1]

def test_decode_params(params: CalibrateResult, decoded_raw: CalibrateDecodedFile, decoded_reference: CalibrateDecodedFile, lag, get_lag = False):
    results = []
    lags = []
    decoded_raw_shm = CalibrateSharedMemory(decoded_raw)
    decoded_reference_shm = CalibrateSharedMemory(decoded_reference)

    for channel in range(decoded_raw.channels):
        decoded_raw_channel = decoded_raw_shm.audio[channel]
        decoded_reference_channel = decoded_reference_shm.audio[channel]

        if lag != 0:
            x = decoded_raw_channel
            y = decoded_reference_channel
            x_start = max(lag, 0)
            y_start = max(-lag, 0)

            # length of overlap
            n = min(len(x) - x_start, len(y) - y_start)
            decoded_raw_channel = x[x_start : x_start + n]
            decoded_reference_channel = y[y_start : y_start + n]

        decoded_processed_channel = decoded_raw_channel.copy()

        deemphasis = Deemphasis(
            decoded_raw.sample_rate,
            params.deemphasis_tau_1,
            params.deemphasis_tau_2,
            params.deemphasis_db_per_octave,
            params.deemphasis_bandwidth
        )

        expander = Expander(
            decoded_raw.sample_rate,
            params.expander_gain,
            params.expander_ratio,
            params.expander_attack_tau,
            params.expander_release_tau,
            params.expander_weighting_tau_1,
            params.expander_weighting_tau_2,
            params.expander_weighting_db_per_octave,
            params.expander_weighting_bandwidth
        )

        deemphasis.process(decoded_processed_channel)
        # prime expander
        expander.process(
            decoded_raw_channel[:decoded_raw.sample_rate],
            np.copy(decoded_processed_channel[:decoded_raw.sample_rate])
        )
        expander.process(
            decoded_raw_channel,
            decoded_processed_channel
        )

        if get_lag:
            result, lag = limited_xcorr_best(decoded_processed_channel, decoded_reference_channel, 1000)
            results.append(result)
            lags.append(lag)
        else:
            # compare the two values
            results.append(correlate(decoded_processed_channel, decoded_reference_channel))
            # results.append(np.dot(decoded_processed_channel, decoded_reference_channel) / len(decoded_processed_channel))

    decoded_raw_shm.close()
    decoded_reference_shm.close()

    params.correlation_results = tuple(results)
    params.correlation_lags = tuple(lags)
    return params

def float_range(min_val, max_val, step):
    """Return a list of floats from min_val to max_val inclusive."""
    num_steps = int(round((max_val - min_val) / step)) + 1
    return [round(min_val + i * step, 12) for i in range(num_steps)]

def partial_filter(combo):
    """
    Early pruning filter for partial combinations.
    combo: list of floats for the first N parameters
    Returns True if this partial combination could still be valid
    """
    # combo[0]: ea_tau, combo[1]: er_tau
    if len(combo) >= 2 and combo[0] >= combo[1]:
        return False
    
    # combo[4]: ew_tau1, combo[5]: ew_tau2
    if len(combo) >= 6 and combo[4] <= combo[5]:
        return False
    
    # combo[8]: d_tau1, combo[9]: d_tau2
    if len(combo) >= 10 and combo[8] <= combo[9]:
        return False
    
    # combo[7]: ew_bw, combo[11]: d_bw
    if len(combo) >= 8 and combo[7] <= 0:
        return False
    if len(combo) >= 12 and combo[11] <= 0:
        return False
    return True

def recursive_generate(ranges, combo=None, depth=0):
    """
    Recursively generate all combinations of values in `ranges`,
    applying the filter function.
    
    ranges: list of generators/lists for each parameter
    combo: current partial combination
    depth: current recursion depth
    """
    if combo is None:
        combo = []

    if not partial_filter(combo):
        return

    if depth == len(ranges):
        # Full combination ready, check filter
        yield CalibrateResult(*combo)
        return

    # Recurse over current parameter
    for value in ranges[depth]:
        yield from recursive_generate(ranges, combo + [value], depth + 1)

# Main generator
def get_ranges(param_dict):
    """
    param_dict: {
        'expander_attack_tau': {'min':0.1,'max':0.5,'step':0.1},
        ...
    }
    """
    # Generate all ranges
    ranges = []
    for key in CalibrateResult.keys:
        r = float_range(param_dict[key]['min'], param_dict[key]['max'], param_dict[key]['step'])
        ranges.append(r)

    return ranges


def main() -> int:
    """
    hifi-calibrate
    
    * Generate sweeps, whitenoise, etc. for testing
    * Otherwise, normal audio should work
    * Import hifi recorded, digital or lineout copy
    * Run the whole set of deemphasis and expander settings
      Gather the coorelation between processed and digital copy
    * Plot the last 6 settings on a graph
    
    * Run hifi-decode
    * Decode input comparison file
    * Maybe synchronize them somehow, or force the user to do that...
    
    * Input Decoded file, Input Reference file
    * Open 2 shared memory instances (or one split in two)
      Save the hifi-decode result, and the input comparison result
    * Split up the deemphasis / expander settings into n chunks
    * Spawn n processes to correlate the audio using the settings
    * Wait until all processes complete their chunks
    * Sort and plot the highest 6 results
    * Possibly export to cli parameters for hifi-decode
    
    """

    args = parser.parse_args()

    in_raw_shm, in_reference_shm, decoded_raw, decoded_reference = decode_input_files(args.in_decoded, args.in_reference)

    # get the result with the existing defaults, use this to determine any delay in the filtering
    params = CalibrateResult(
        DEFAULT_EXPANDER_ATTACK_TAU,
        DEFAULT_EXPANDER_RELEASE_TAU,
        DEFAULT_EXPANDER_GAIN,
        DEFAULT_EXPANDER_RATIO,
        DEFAULT_VHS_EXPANDER_WEIGHTING_TAU_1,
        DEFAULT_VHS_EXPANDER_WEIGHTING_TAU_2,
        DEFAULT_VHS_EXPANDER_WEIGHTING_DB_PER_OCTAVE,
        DEFAULT_VHS_EXPANDER_WEIGHTING_BANDWIDTH,
        DEFAULT_VHS_DEEMPHASIS_TAU_1,
        DEFAULT_VHS_DEEMPHASIS_TAU_2,
        DEFAULT_VHS_DEEMPHASIS_DB_PER_OCTAVE,
        DEFAULT_VHS_DEEMPHASIS_BANDWIDTH,
    )
    results = test_decode_params(params, decoded_raw, decoded_reference, 0, True)
    lag = results.correlation_lags[0]

    # Example usage
    param_dict = {
        'expander_attack_tau': {'min':DEFAULT_EXPANDER_ATTACK_TAU,'max':DEFAULT_EXPANDER_ATTACK_TAU,'step':1},
        'expander_release_tau': {'min':DEFAULT_EXPANDER_RELEASE_TAU,'max':DEFAULT_EXPANDER_RELEASE_TAU,'step':1},
        'expander_gain': {'min':DEFAULT_EXPANDER_GAIN,'max':DEFAULT_EXPANDER_GAIN,'step':1},
        'expander_ratio': {'min':DEFAULT_EXPANDER_RATIO,'max':DEFAULT_EXPANDER_RATIO,'step':1},
        
        'expander_weighting_tau_1': {'min':0.000150,'max':0.000300,'step':0.000001},
        'expander_weighting_tau_2': {'min':0.000010,'max':0.000100,'step':0.000001},
        'expander_weighting_db_per_octave': {'min':DEFAULT_VHS_EXPANDER_WEIGHTING_DB_PER_OCTAVE,'max':DEFAULT_VHS_EXPANDER_WEIGHTING_DB_PER_OCTAVE,'step':1},
        'expander_weighting_bandwidth': {'min':1.5,'max':3.5,'step':0.02},
        
        'deemphasis_tau_1': {'min':0.000150,'max':0.000300,'step':0.000001},
        'deemphasis_tau_2': {'min':0.000010,'max':0.000100,'step':0.000001},
        'deemphasis_db_per_octave': {'min':DEFAULT_VHS_DEEMPHASIS_DB_PER_OCTAVE,'max':DEFAULT_VHS_DEEMPHASIS_DB_PER_OCTAVE,'step':1},
        'deemphasis_bandwidth': {'min':1.5,'max':3.5,'step':0.02},
    }
    
    # Generate results lazily
    ranges = get_ranges(param_dict)
    generator = recursive_generate(ranges)
    #generator = list(islice(generator, 30))
    max_workers = args.threads

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = set()  # keep track of running futures
        with open(args.in_reference + "_results.csv", mode='w', newline='') as csv_file:
            try:
                writer = csv.writer(csv_file)
                writer.writerow([*CalibrateResult.keys, "correlation_0", "correlation_1"])

                for params in generator:
                    # Submit new task
                    future = executor.submit(test_decode_params, params, decoded_raw, decoded_reference, lag)
                    futures.add(future)

                    # Keep the number of running futures under max_workers
                    if len(futures) >= max_workers:
                        # Wait for at least one to complete
                        done, futures = wait(futures, return_when="FIRST_COMPLETED")
                        for f in done:
                            result = f.result()
                            row = [
                                result.expander_attack_tau, result.expander_release_tau,
                                result.expander_gain, result.expander_ratio,
                                result.expander_weighting_tau_1, result.expander_weighting_tau_2,
                                result.expander_weighting_db_per_octave, result.expander_weighting_bandwidth,
                                result.deemphasis_tau_1, result.deemphasis_tau_2,
                                result.deemphasis_db_per_octave, result.deemphasis_bandwidth,
                                result.correlation_results[0],
                                result.correlation_results[1]
                            ]
                            writer.writerow(row)
                            print(row)

                # Wait for any remaining futures to complete
                for f in as_completed(futures):
                    result = f.result()
                    row = [
                        result.expander_attack_tau, result.expander_release_tau,
                        result.expander_gain, result.expander_ratio,
                        result.expander_weighting_tau_1, result.expander_weighting_tau_2,
                        result.expander_weighting_db_per_octave, result.expander_weighting_bandwidth,
                        result.deemphasis_tau_1, result.deemphasis_tau_2,
                        result.deemphasis_db_per_octave, result.deemphasis_bandwidth,
                        result.correlation_results[0],
                        result.correlation_results[1]
                    ]
                    writer.writerow(row)
                    print(row)
            finally:
                csv_file.close()

    in_raw_shm.close()
    in_raw_shm.unlink()
    in_reference_shm.close()
    in_reference_shm.unlink()


if __name__ == "__main__":
    set_start_method('fork')
    main()