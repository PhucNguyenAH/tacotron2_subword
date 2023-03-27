from pydub import AudioSegment
from glob import glob
from tqdm import tqdm
import sys
import os

def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms
    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0  # ms
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold:
        trim_ms += chunk_size
        if trim_ms > len(sound):
            return None

    return trim_ms

Test_infer = glob("Outdir/demo/audio/*.wav")
if not os.path.exists("benchmark"):
    os.makedirs("benchmark")
for infer in tqdm(Test_infer):
    benchmark = infer.replace("Outdir/demo/audio","benchmark")
    sound = AudioSegment.from_file(infer, format="wav")

    start_trim = detect_leading_silence(sound)
    if start_trim is None:
        continue
    end_trim = detect_leading_silence(sound.reverse())

    duration = len(sound)
    trimmed_sound = sound[start_trim:duration-end_trim]
    trimmed_sound.export(benchmark, format="wav")