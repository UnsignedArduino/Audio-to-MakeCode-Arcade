import struct
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import scipy
from tqdm import trange

parser = ArgumentParser(description="Convert audio to MakeCode Arcade hex buffers!")
parser.add_argument("-i", "--input", metavar="PATH", type=Path, required=True,
                    help="The input MONO WAV file.")
parser.add_argument("-o", "--output", metavar="PATH", type=Path,
                    required=False,
                    help="The output TypeScript file which contains MakeCode Arcade "
                         "code.")
parser.add_argument("-p", "--period", metavar="MILLISECONDS", type=int, default=25,
                    help="The period in milliseconds between each DFT for the spectrogram.")
args = parser.parse_args()

can_log = args.output is not None or args.preview
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

    max_freqs = 20
    print(f"Gathering {max_freqs} loudest frequencies and amplitudes")

    loudest_indices = np.argsort(Sxx, axis=0)[-max_freqs:]
    loudest_frequencies = f[loudest_indices]
    loudest_amplitudes = Sxx[loudest_indices, np.arange(Sxx.shape[1])]

    code = f"const instructionLines = [\n"
    for i in trange(max_freqs):
        code += "    hex`"
        for j in range(len(t)):
            freq = round(loudest_frequencies[i, j])
            amp = round(loudest_amplitudes[i, j] / 2 ** 15 * 1024)
            prevAmp = round(
                loudest_amplitudes[i, j - 1] / 2 ** 15 * 1024) if j > 0 else amp
            code += create_sound_instruction(freq, freq, prevAmp, amp, period)
        code += "`,\n"
    code += "];"

    code += """

for (const instructions of instructionLines) {
    music.playInstructions(100, instructions);
}
"""

    return code


code = audio_to_makecode_arcade(data, sample_rate, spectrogram_period)
if args.output is not None:
    output_path = args.output.expanduser().resolve()
    if can_log:
        print(f"Writing to {output_path}")
    output_path.write_text(code)
else:
    print(code)
