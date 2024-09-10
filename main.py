import struct
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import scipy

parser = ArgumentParser(description="Convert audio to MakeCode Arcade hex buffers!")
parser.add_argument("-i", "--input", metavar="PATH", type=Path, required=True,
                    help="The input MONO WAV file.")
parser.add_argument("-o", "--output", metavar="PATH", type=Path,
                    required=False,
                    help="The output TypeScript file which contains MakeCode Arcade "
                         "code.")
parser.add_argument("-p", "--period", metavar="MILLISECONDS", type=int, default=25,
                    help="The period in milliseconds between each DFT for the spectrogram.")
parser.add_argument("--debug", action="store_true",
                    help="Print human readable strings instead of hex buffers for debugging0")
args = parser.parse_args()

debug_output = args.debug
can_log = args.output is not None or debug_output
spectrogram_period = args.period

if can_log:
    print(f"Arguments received: {args}")

input_path = args.input.expanduser().resolve()
if can_log:
    print(f"Opening audio {input_path}")

sample_rate, data = scipy.io.wavfile.read(input_path)
channel_count = data.shape[1] if len(data.shape) > 1 else 1
if channel_count > 1:
    print(f"Audio has {channel_count} channels, but only the first will be used.")
sample_count = data.shape[0]
track_length = sample_count / sample_rate

if can_log:
    print(f"Audio has {sample_count} samples at {sample_rate} Hz, "
          f"which is {track_length:.2f} seconds long.")


def constrain(value, min_value, max_value):
    return min(max(value, min_value), max_value)


def create_sound_instruction(start_freq: int, end_freq: int, start_vol: int,
                             end_vol: int, duration: int) -> str:
    """
    Generate a MakeCode Arcade sound instruction.

    :param start_freq: Start frequency of the sound.
    :param end_freq: Ending frequency of the sound.
    :param start_vol: Start volume of the sound.
    :param end_vol: Ending volume of the sound.
    :param duration: Duration of the sound.
    :return: MakeCode Arcade sound instruction hex string buffer literal.
    """
    return struct.pack("<BBHHHHH",
                       3,  # sine waveform (8 bits)
                       0,  # unused (8 bits)
                       max(start_freq, 1),  # start frequency in hz (16 bits)
                       duration,  # duration in ms (16 bits)
                       constrain(start_freq, 0, 1024),  # start volume (16 bits)
                       constrain(end_freq, 0, 1024),  # end volume (16 bits)
                       max(end_freq, 1)  # end frequency in hz (16 bits)
                       ).hex()


def audio_to_makecode_arcade(data, sample_rate, period) -> str:
    """
    Convert audio to MakeCode Arcade hex buffers.

    :param data: Single channel audio data.
    :param sample_rate: Sample rate of the audio.
    :param period: Period for the spectrogram.
    :return: MakeCode Arcade code.
    """
    spectrogram_frequency = period / 1000
    if can_log:
        print(
            f"Generating spectrogram with a period of {period} ms. (nperseg = {round(spectrogram_frequency * sample_rate)})")
    f, t, Sxx = scipy.signal.spectrogram(data, sample_rate, nperseg=round(
        spectrogram_frequency * sample_rate))

    frequency_buckets = [50, 159, 200, 252, 317, 400, 504, 635, 800, 1008, 1270, 1600,
                         2016, 2504, 3200, 4032, 5080, 7000, 9000, 10240]

    max_freqs = 30
    if can_log:
        print(f"Gathering {max_freqs} loudest frequencies and amplitudes")

    loudest_indices = np.argsort(Sxx, axis=0)[-max_freqs:]
    loudest_frequencies = f[loudest_indices].transpose()
    loudest_amplitudes = Sxx[loudest_indices, np.arange(Sxx.shape[1])].transpose()
    max_amp = np.max(Sxx)

    if can_log:
        print(f"Generating sound instructions")

    def find_loudest_freq_index_in_bucket(slice_index: int, bucket_index: int) -> int:
        freqs = loudest_frequencies[slice_index]
        low = frequency_buckets[bucket_index - 1] if bucket_index > 0 else 0
        high = frequency_buckets[bucket_index]
        # Start at the end of the frequency array because they are sorted in ascending order
        for i in range(len(freqs) - 1, -1, -1):
            if low <= freqs[i] <= high:
                return i
        return -1

    if debug_output:
        print(f"{"":>6}", end="")
        for bucket in frequency_buckets:
            print(f"{bucket:>16}", end="")
        print()

    sound_instruction_buffers = ["hex`"] * len(frequency_buckets)
    for slice_index in range(len(loudest_frequencies)):
        if debug_output:
            print(f"{slice_index:<6}", end="")
        for bucket_index in range(len(frequency_buckets)):
            freq_index = find_loudest_freq_index_in_bucket(slice_index, bucket_index)
            if freq_index != -1:
                freq = round(loudest_frequencies[slice_index, freq_index])
                amp = round(
                    loudest_amplitudes[slice_index, freq_index] / max_amp * 1024)
                sound_instruction_buffers[bucket_index] += create_sound_instruction(
                    freq,
                    freq,
                    amp,
                    amp,
                    period)
                if debug_output:
                    print(f"{f"{freq} Hz {amp} amp":>16}", end="")
            else:
                sound_instruction_buffers[bucket_index] += create_sound_instruction(0,
                                                                                    0,
                                                                                    0,
                                                                                    0,
                                                                                    period)
                if debug_output:
                    print(f"{"0 Hz 0 amp":>16}", end="")
        if debug_output:
            print()
    sound_instruction_buffers = [buffer + "`" for buffer in sound_instruction_buffers]

    return f"""namespace music {{
    //% shim=music::queuePlayInstructions
    export function queuePlayInstructions(timeDelta: number, buf: Buffer) {{}}
}}

const soundInstructions = [
    {",\n    ".join(sound_instruction_buffers)}
];

for (const soundInstruction of soundInstructions) {{
    music.queuePlayInstructions(100, soundInstruction);
}}"""


code = audio_to_makecode_arcade(data, sample_rate, spectrogram_period)
if args.output is not None:
    output_path = args.output.expanduser().resolve()
    if can_log:
        print(f"Writing to {output_path}")
    output_path.write_text(code)
else:
    print(code)
