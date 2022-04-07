from gettext import find
import os
import wave
import tensorflow as tf
import pathlib
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import numpy as np
import librosa

dataDir = pathlib.Path('chords')
chords = np.array(tf.io.gfile.listdir(str(dataDir)))


def decode(audio_bin):
    audio, _ = tf.audio.decode_wav(contents=audio_bin)
    return tf.squeeze(audio, axis=1)

def get_label(file_path):
    parts = tf.strings.split(
        input=file_path,
        sep=os.path.sep
    )

    return parts[-2]

def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode(audio_binary)
    return waveform, label

def findData(path='chords'):
    dataDir = pathlib.Path(path)
    chords = np.array(tf.io.gfile.listdir(str(dataDir)))
    print('chords:', chords)
    filenames = tf.io.gfile.glob(str(dataDir) + '/*/*')
    filenames = tf.random.shuffle(filenames)
    num_samples = len(filenames)
    print("number of examples: ", num_samples)
    
    files_ds = tf.data.Dataset.from_tensor_slices(filenames)
    batch_size = 4
    files_processed = process_data(files_ds)
    files_processed.cache().prefetch(tf.data.AUTOTUNE)

    
    validation_ds = files_processed.batch(batch_size)
    
    waveform_ds = files_ds.map(
        map_func=get_waveform_and_label,
        num_parallel_calls=tf.data.AUTOTUNE
        )
    
    spect_ds = waveform_ds.map(
         map_func=get_spect_and_id,
         num_parallel_calls=tf.data.AUTOTUNE
    )

    train_ds = spect_ds

    train_ds = train_ds.batch(batch_size)

    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)

    for spectrogram, _ in spect_ds.take(1):
        input_shape = spectrogram.shape
    print('Input shape:', input_shape)
    num_labels = len(chords)

    norm_layer = tf.keras.layers.Normalization()

    norm_layer.adapt(data=spect_ds.map(map_func=lambda spec, label: spec))

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Resizing(32, 32),
        norm_layer,
        tf.keras.layers.Conv2D(32, 5, activation='relu'),
        tf.keras.layers.Conv2D(64, 5, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_labels),
    ])

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )


    EPOCHS = 30 
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=10),
    )
    

    testdataDir = pathlib.Path("test")
    testfilenames = tf.io.gfile.glob(str(testdataDir) + '/*/**')
    testfilenames = tf.random.shuffle(testfilenames)

    test_ds = tf.data.Dataset.from_tensor_slices(testfilenames)
    test_ds = process_data(test_ds)

    for spectrogram, label in test_ds.batch(1):
        prediction = model(spectrogram)
        print(prediction)
        plt.bar(chords, tf.nn.softmax(prediction[0]))
        print(label)
        plt.title('Predictions for file {}'.format(chords[label[0]]))
        plt.show()
        
    model.save('./')

    # test_audio = []
    # test_labels = []

    # for audio, label in test_ds:
    #     test_audio.append(audio.numpy())
    #     test_labels.append(label.numpy())

    # test_audio = np.array(test_audio)
    # test_labels = np.array(test_labels)
    # y_pred = np.argmax(model.predict(test_audio), axis=1)
    # y_true = test_labels

    # test_acc = sum(y_pred == y_true) / len(y_true)
    # print(f'Test set accuracy: {test_acc:.0%}')


def wave_to_spect(wave):
    input_len = 115000
    wave = wave[:input_len]
    zero_pad = tf.zeros( 
        [115000] - tf.shape(wave),
        dtype=tf.float32
    )

    wave = tf.cast(wave, dtype=tf.float32)
    length = tf.concat([wave, zero_pad], 0)

    spectrogram = tf.signal.stft(
        length, frame_length=255, frame_step=128
    )

    spectrogram = tf.abs(spectrogram)

    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def get_spect_and_id(audio, label):
    spectrogram = wave_to_spect(audio)
    spectid = tf.argmax(label == chords)
    return spectrogram, spectid

def plot_spect(spect, ax):
    if len(spect.shape) > 2:
        assert len(spect.shape) == 3 
        spect = np.squeeze(spect, axis=-1)

    log = np.log(spect.T + np.finfo(float).eps)
    height = log.shape[0]
    width = log.shape[1]
    x = np.linspace(0, np.size(spect), num=width, dtype=int)
    y = range(height)
    ax.pcolormesh(x, y, log)


def convertSong(path):
    audio = tfio.audio.AudioIOTensor(path)

    audio_slice = audio[100:]

    audio_tensor = tf.squeeze(audio_slice, axis=[-1])

    tensor = tf.cast(audio_tensor, tf.float32) / 32768.0

    spectrogram = tfio.audio.spectrogram(tensor, nfft=512, window=512, stride=256)
    
    plt.figure()
    plt.plot(tf.math.log(spectrogram).numpy())
    plt.show()

def songToWav(file_path):
    audio_binary = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    
    samples = 150
    n = int(tf.size(audio) / samples)
    wav = audio[:(n * samples)]
    x = tf.reshape(tensor=wav, shape=(n, samples))
    plt.figure()
    plt.plot(x.numpy())
    plt.show()
    

def process_data(files):
    output_ds = files.map(
        map_func=get_waveform_and_label,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    output_ds = output_ds.map(
        map_func=get_spect_and_id,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return output_ds
    
findData()
#best website https://www.tensorflow.org/tutorials/audio/simple_audio

#compare notes from 2 tracks to