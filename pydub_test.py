
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.silence import detect_nonsilent

sound = AudioSegment.from_wav("chordprog.wav")

print(detect_nonsilent(sound, silence_thresh=-40))
chunks = split_on_silence(sound, silence_thresh=-40)

print(sound)

print(chunks)