import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt

def plotGraph(location):
    audio = tfio.audio.AudioIOTensor(location)

    audio_slice = audio[100:]

    audio_tensor = tf.squeeze(audio_slice, axis=[-1])

    tensor = tf.cast(audio_tensor, tf.float32) / 32768.0

    position = tfio.audio.trim(tensor, axis=0, epsilon=0.1)
    print(position)

    start = position[0]
    stop = position[1]
    print(start, stop)

    processed = tensor[start:stop]

    plt.figure()
    plt.title(location)
    plt.plot(processed.numpy())
    plt.show()

plotGraph('chords/a-chord.wav')
plotGraph('chords/e-chord.wav')
    


