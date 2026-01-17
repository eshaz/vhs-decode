import argparse

from dataclasses import dataclass

import string
from random import SystemRandom
import csv

from numba import njit
import numba

from multiprocessing import (
    cpu_count,
    Pipe,
    Process,
    resource_tracker,
    set_start_method
)

from multiprocessing.shared_memory import SharedMemory
from concurrent.futures import ProcessPoolExecutor, as_completed, wait

import numpy as np
import soundfile as sf
import scipy

from vhsdecode.hifi.HiFiDecode import (
    Deemphasis,
    Expander,
    DEFAULT_EXPANDER_GAIN,
    DEFAULT_EXPANDER_RATIO,
    DEFAULT_EXPANDER_ATTACK_TAU,
    DEFAULT_EXPANDER_RELEASE_TAU,
    DEFAULT_VHS_EXPANDER_WEIGHTING_TAU_1,
    DEFAULT_VHS_EXPANDER_WEIGHTING_TAU_2,
    DEFAULT_VHS_EXPANDER_WEIGHTING_BANDWIDTH,
    DEFAULT_VHS_EXPANDER_WEIGHTING_LOW_PASS,
    DEFAULT_VHS_EXPANDER_WEIGHTING_LOW_PASS_TRANSITION,
    DEFAULT_VHS_DEEMPHASIS_TAU_1,
    DEFAULT_VHS_DEEMPHASIS_TAU_2,
    DEFAULT_VHS_DEEMPHASIS_BANDWIDTH,
    DEFAULT_VHS_NR_DEEMPHASIS_TAU_1,
    DEFAULT_VHS_NR_DEEMPHASIS_TAU_2,
    DEFAULT_VHS_NR_DEEMPHASIS_BANDWIDTH
)

default_threads = cpu_count()
parser = argparse.ArgumentParser(
    description="Derives De-Emphasis and Expander settings for hifi-decode"
)
parser.add_argument(
    "-i",
    type=str,
    help="source decoded file",
    default="",
)
parser.add_argument(
    "-r",
    type=str,
    help="source reference file",
    default="",
)
parser.add_argument(
    "-o",
    type=str,
    help="output csv",
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
class CalibrateAudioData():
    name: str
    length: int
    sample_rate: int
    dtype: np.dtype

    def __init__(self, buffer_name, length, sample_rate, dtype):
        self.name = buffer_name
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
    expander_weighting_bandwidth: float
    expander_weighting_low_pass: float
    expander_weighting_low_pass_transition: float
    deemphasis_tau_1: float
    deemphasis_tau_2: float
    deemphasis_bandwidth: float
    nr_deemphasis_tau_1: float
    nr_deemphasis_tau_2: float
    nr_deemphasis_bandwidth: float
    similarity: float
    max_gain_error: float
    rms_error: float

    keys = [
        'expander_attack_tau',
        'expander_release_tau',
        'expander_gain',
        'expander_ratio',
        'expander_weighting_tau_1',
        'expander_weighting_tau_2',
        'expander_weighting_bandwidth',
        'expander_weighting_low_pass',
        'expander_weighting_low_pass_transition',
        'deemphasis_tau_1',
        'deemphasis_tau_2',
        'deemphasis_bandwidth',
        'nr_deemphasis_tau_1',
        'nr_deemphasis_tau_2',
        'nr_deemphasis_bandwidth',
    ]

    def __init__(self,
        expander_attack_tau,
        expander_release_tau,
        expander_gain,
        expander_ratio,
        expander_weighting_tau_1,
        expander_weighting_tau_2,
        expander_weighting_bandwidth,
        expander_weighting_low_pass,
        expander_weighting_low_pass_transition,
        deemphasis_tau_1,
        deemphasis_tau_2,
        deemphasis_bandwidth,
        nr_deemphasis_tau_1,
        nr_deemphasis_tau_2,
        nr_deemphasis_bandwidth,
    ):
        self.expander_attack_tau = expander_attack_tau
        self.expander_release_tau = expander_release_tau
        self.expander_gain = expander_gain
        self.expander_ratio = expander_ratio

        self.expander_weighting_tau_1 = expander_weighting_tau_1
        self.expander_weighting_tau_2 = expander_weighting_tau_2
        self.expander_weighting_bandwidth = expander_weighting_bandwidth
        self.expander_weighting_low_pass = expander_weighting_low_pass
        self.expander_weighting_low_pass_transition = expander_weighting_low_pass_transition

        self.deemphasis_tau_1 = deemphasis_tau_1
        self.deemphasis_tau_2 = deemphasis_tau_2
        self.deemphasis_bandwidth = deemphasis_bandwidth

        self.nr_deemphasis_tau_1 = nr_deemphasis_tau_1
        self.nr_deemphasis_tau_2 = nr_deemphasis_tau_2
        self.nr_deemphasis_bandwidth = nr_deemphasis_bandwidth
        

class CalibrateSharedMemory():
    def __init__(self, decoded_file: CalibrateAudioData):
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

        self.length = decoded_file.length
        self.dtype = decoded_file.dtype

        self.audio = np.ndarray(
            self.length,
            dtype=self.dtype,
            buffer=self.buf
        )

    @staticmethod
    def create_shared_memory(name, length, item_size):
        byte_size = length * item_size

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

    # create and manage the shared memory from the parent process
    conn.send((length, decoded_dtype))
    shared_memory_name = conn.recv()

    calibrate_decoded_file = CalibrateAudioData(shared_memory_name, length, sample_rate, decoded_dtype)
    calibrate_shared_memory = CalibrateSharedMemory(calibrate_decoded_file)

    # only use the left channel
    np.copyto(calibrate_shared_memory.audio, data[:, 0])

    calibrate_shared_memory.close()

    conn.send(calibrate_decoded_file)

def decode_input_files(in_raw, in_reference):
    in_raw_parent, in_raw_child = Pipe()
    in_raw_process = Process(None, decode_worker, args=(in_raw, in_raw_child))
    in_raw_process.start()

    in_reference_parent, in_reference_child = Pipe()
    in_reference_process = Process(None, decode_worker, args=(in_reference, in_reference_child))
    in_reference_process.start()

    in_raw_length, in_raw_dtype = in_raw_parent.recv()
    in_raw_shm, in_raw_shm_name = CalibrateSharedMemory.create_shared_memory("hifi-raw", in_raw_length, np.dtype(in_raw_dtype).itemsize)
    in_raw_parent.send(in_raw_shm_name)

    in_reference_length, in_reference_dtype = in_reference_parent.recv()
    in_reference_shm, in_reference_shm_name = CalibrateSharedMemory.create_shared_memory("hifi-ref", in_reference_length, np.dtype(in_reference_dtype).itemsize)
    in_reference_parent.send(in_reference_shm_name)

    decoded_raw = in_raw_parent.recv()
    decoded_reference = in_reference_parent.recv()

    in_raw_process.join()
    in_reference_process.join()

    # get the fft for the reference data
    decoded_raw_shm = CalibrateSharedMemory(decoded_raw)
    decoded_reference_shm = CalibrateSharedMemory(decoded_reference)

    _, H_ref = estimate_transfer_function(
        decoded_raw_shm.audio,
        decoded_reference_shm.audio,
        fs=decoded_raw.sample_rate,
    )

    decoded_raw_shm.close()
    decoded_reference_shm.close()

    reference_fft_shm, reference_fft_shm_name = CalibrateSharedMemory.create_shared_memory("hifi-ref-fft", len(H_ref), np.dtype(H_ref.dtype).itemsize)
    reference_fft = CalibrateAudioData(reference_fft_shm_name, len(H_ref), decoded_reference.sample_rate, np.dtype(H_ref.dtype))

    # copy to shared memory
    reference_fft_instance = CalibrateSharedMemory(reference_fft)
    np.copyto(reference_fft_instance.audio, H_ref)
    reference_fft_instance.close()

    return (
        in_raw_shm, in_reference_shm, reference_fft_shm,
        decoded_raw, decoded_reference, reference_fft
    )

@njit(
    [
        (
            numba.types.Array(numba.types.float32, 1, "A"),
            numba.types.Array(numba.types.float32, 1, "C")
        )
    ],
    cache=True,
    fastmath=False,
    nogil=True,
)
def correlate(decoded_processed_channel, decoded_reference_channel):
    return np.corrcoef(decoded_processed_channel, decoded_reference_channel)[0, 1]

def normalized_fft(audio):
    # window = np.hanning(len(audio)).astype(np.float32, copy=False)
    fft = scipy.fft.rfft(audio).real
    #fft_abs = np.abs(fft)
    #fft_norm = fft / np.max(fft_abs)

    return fft

def estimate_transfer_function(x, y, fs, nperseg=8192):
    eps = 1e-12

    f, Pxy = scipy.signal.csd(y, x, fs=fs, nperseg=nperseg)
    _, Pxx = scipy.signal.csd(x, x, fs=fs, nperseg=nperseg)

    H = Pxy / (Pxx + eps)
    return f, H


def test_decode_params(params: CalibrateResult, decoded_raw: CalibrateAudioData, reference_fft: CalibrateAudioData):
    decoded_raw_shm = CalibrateSharedMemory(decoded_raw)
    reference_fft_shm = CalibrateSharedMemory(reference_fft)

    decoded_raw_channel = decoded_raw_shm.audio

    decoded_processed_channel = decoded_raw_channel.copy()

    deemphasis = Deemphasis(
        decoded_raw.sample_rate,
        params.deemphasis_tau_1,
        params.deemphasis_tau_2,
        params.deemphasis_bandwidth,
        params.nr_deemphasis_tau_1,
        params.nr_deemphasis_tau_2,
        params.nr_deemphasis_bandwidth
    )

    expander = Expander(
        decoded_raw.sample_rate,
        params.expander_gain,
        params.expander_ratio,
        params.expander_attack_tau,
        params.expander_release_tau,
        params.deemphasis_tau_1,
        params.deemphasis_tau_2,
        params.deemphasis_bandwidth,
        params.expander_weighting_tau_1,
        params.expander_weighting_tau_2,
        params.expander_weighting_bandwidth,
        params.expander_weighting_low_pass,
        params.expander_weighting_low_pass_transition,
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

    H_ref = reference_fft_shm.audio
    f, H_test = estimate_transfer_function(
        decoded_raw_channel,
        decoded_processed_channel,
        fs=decoded_raw.sample_rate,
    )

    eps = 1e-12
    H_ref_db  = 20 * np.log10(np.abs(H_ref)  + eps)
    H_test_db = 20 * np.log10(np.abs(H_test) + eps)

    band = (f >= 100) & (f <= 16_000)
    H_ref_db  = H_ref_db[band]
    H_test_db = H_test_db[band]

    # frequency response error
    delta_db = H_test_db - H_ref_db
    # delta_db -= np.mean(delta_db)

    rms_err_db = np.sqrt(np.mean(delta_db**2))
    max_err_db = np.max(np.abs(delta_db))

    # remove constant gain offset
    a = H_ref_db - np.mean(H_ref_db)
    b = H_test_db - np.mean(H_test_db)

    # normalize energy (L2 norm)
    a /= np.linalg.norm(a) + 1e-12
    b /= np.linalg.norm(b) + 1e-12

    cos_sim = np.dot(a, b)

    # store metrics
    params.similarity = cos_sim
    params.max_gain_error = max_err_db
    params.rms_error = rms_err_db

    decoded_raw_shm.close()
    reference_fft_shm.close()

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
    # attack < release
    if len(combo) >= 2 and combo[0] >= combo[1]:
        return False
    
    # expander_weighting_tau_1 > expander_weighting_tau_2
    if len(combo) >= 6 and combo[4] <= combo[5]:
        return False

    # deemphasis_tau_1 > deemphasis_tau_2
    if len(combo) >= 11 and combo[9] <= combo[10]:
        return False

    # nr_deemphasis_tau_1 > nr_deemphasis_tau_2
    if len(combo) >= 14 and combo[12] <= combo[13]:
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

    param_dict = {
        'expander_attack_tau': {'min':DEFAULT_EXPANDER_ATTACK_TAU,'max':DEFAULT_EXPANDER_ATTACK_TAU,'step':1e-4},
        'expander_release_tau': {'min':DEFAULT_EXPANDER_RELEASE_TAU,'max':DEFAULT_EXPANDER_RELEASE_TAU,'step':1e-4},
        'expander_gain': {'min':DEFAULT_EXPANDER_GAIN,'max':DEFAULT_EXPANDER_GAIN,'step':1},
        'expander_ratio': {'min':DEFAULT_EXPANDER_RATIO,'max':DEFAULT_EXPANDER_RATIO,'step':1},
        
        'expander_weighting_tau_1': {'min':DEFAULT_VHS_EXPANDER_WEIGHTING_TAU_1,'max':DEFAULT_VHS_EXPANDER_WEIGHTING_TAU_1,'step':1e-7},
        'expander_weighting_tau_2': {'min':DEFAULT_VHS_EXPANDER_WEIGHTING_TAU_2,'max':DEFAULT_VHS_EXPANDER_WEIGHTING_TAU_2,'step':1e-6},
        'expander_weighting_bandwidth': {'min':DEFAULT_VHS_EXPANDER_WEIGHTING_BANDWIDTH,'max':DEFAULT_VHS_EXPANDER_WEIGHTING_BANDWIDTH,'step':0.01},
        'expander_weighting_low_pass': {'min':1000,'max':500000,'step':500},
        'expander_weighting_low_pass_transition': {'min':1000,'max':25000,'step':200},
        
        'deemphasis_tau_1': {'min':DEFAULT_VHS_NR_DEEMPHASIS_TAU_1,'max':DEFAULT_VHS_NR_DEEMPHASIS_TAU_1,'step':1},
        'deemphasis_tau_2': {'min':DEFAULT_VHS_NR_DEEMPHASIS_TAU_2,'max':DEFAULT_VHS_NR_DEEMPHASIS_TAU_2,'step':1},
        'deemphasis_bandwidth': {'min':DEFAULT_VHS_NR_DEEMPHASIS_BANDWIDTH,'max':DEFAULT_VHS_NR_DEEMPHASIS_BANDWIDTH,'step':0.001},
        
        'nr_deemphasis_tau_1': {'min':DEFAULT_VHS_DEEMPHASIS_TAU_1,'max':DEFAULT_VHS_DEEMPHASIS_TAU_1,'step':1},
        'nr_deemphasis_tau_2': {'min':DEFAULT_VHS_DEEMPHASIS_TAU_2,'max':DEFAULT_VHS_DEEMPHASIS_TAU_2,'step':1},
        'nr_deemphasis_bandwidth': {'min':DEFAULT_VHS_DEEMPHASIS_BANDWIDTH,'max':DEFAULT_VHS_DEEMPHASIS_BANDWIDTH,'step':0.001},
    }

    ranges = get_ranges(param_dict)
    generator = recursive_generate(ranges)
    max_workers = args.threads
    best_similarity = -1
    best_rms_error = np.inf

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = set()  # keep track of running futures
        with open(args.o, mode='w', newline='') as csv_file:
            try:
                in_raw_shm, in_reference_shm, reference_fft_shm, decoded_raw, decoded_reference, reference_fft = decode_input_files(args.i, args.r)

                writer = csv.writer(csv_file)
                writer.writerow([*CalibrateResult.keys, "similarity", "max_gain_error", "rms_error"])

                for params in generator:
                    # Submit new task
                    future = executor.submit(test_decode_params, params, decoded_raw, reference_fft)
                    futures.add(future)

                    # Keep the number of running futures under max_workers
                    if len(futures) >= max_workers:
                        # Wait for at least one to complete
                        done, futures = wait(futures, return_when="FIRST_COMPLETED")
                        for f in done:
                            result = f.result()
                            row = [
                                result.expander_attack_tau,
                                result.expander_release_tau,
                                result.expander_gain,
                                result.expander_ratio,
                                result.expander_weighting_tau_1,
                                result.expander_weighting_tau_2,
                                result.expander_weighting_bandwidth,
                                result.expander_weighting_low_pass,
                                result.expander_weighting_low_pass_transition,
                                result.deemphasis_tau_1,
                                result.deemphasis_tau_2,
                                result.deemphasis_bandwidth,
                                result.nr_deemphasis_tau_1,
                                result.nr_deemphasis_tau_2,
                                result.nr_deemphasis_bandwidth,
                                result.similarity,
                                result.max_gain_error,
                                result.rms_error
                            ]

                            if result.similarity >= best_similarity or result.rms_error <= best_rms_error:
                                print("new best result", row)

                                writer.writerow(row)
                                if result.similarity >= best_similarity:
                                    best_similarity = result.similarity
                                if result.rms_error <= best_rms_error:
                                    best_rms_error = result.rms_error
                            else:
                                print(row, end="\r")

                # Wait for any remaining futures to complete
                for f in as_completed(futures):
                    result = f.result()
                    row = [
                        result.expander_attack_tau,
                        result.expander_release_tau,
                        result.expander_gain,
                        result.expander_ratio,
                        result.expander_weighting_tau_1,
                        result.expander_weighting_tau_2,
                        result.expander_weighting_bandwidth,
                        result.expander_weighting_low_pass,
                        result.expander_weighting_low_pass_transition,
                        result.deemphasis_tau_1,
                        result.deemphasis_tau_2,
                        result.deemphasis_bandwidth,
                        result.nr_deemphasis_tau_1,
                        result.nr_deemphasis_tau_2,
                        result.nr_deemphasis_bandwidth,
                        result.similarity,
                        result.max_gain_error,
                        result.rms_error
                    ]

                    if result.similarity >= best_similarity or result.rms_error <= best_rms_error:
                        print("new best result", row)

                        writer.writerow(row)
                        if result.similarity >= best_similarity:
                            best_similarity = result.similarity
                        if result.rms_error <= best_rms_error:
                            best_rms_error = result.rms_error
                    else:
                        print(row, end="\r")
            finally:
                csv_file.close()

                in_raw_shm.close()
                in_raw_shm.unlink()
                in_reference_shm.close()
                in_reference_shm.unlink()
                reference_fft_shm.close()
                reference_fft_shm.unlink()


if __name__ == "__main__":
    set_start_method('fork')
    main()