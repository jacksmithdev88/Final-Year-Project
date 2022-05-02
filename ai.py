import os
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import numpy as np

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
    directory = pathlib.Path(path)
    chords = np.array(tf.io.gfile.listdir(str(directory)))
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

    rows = 4
    cols = 4
    n = rows*cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    for i, (spectrogram, label_id) in enumerate(spect_ds.take(n)):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        plot_spect(spectrogram.numpy(), ax)
        ax.set_title(chords[label_id.numpy()])
        ax.axis('off')

    plt.show()

    train_ds = spect_ds

    train_ds = train_ds.batch(batch_size)

    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)

    for spectrogram, _ in spect_ds.take(1):
        input_shape = spectrogram.shape
        print("SHAPE")
        print(spectrogram.shape)
    print('Input shape:', input_shape)
    num_labels = len(chords)

    layer = tf.keras.layers.Normalization()

    layer.adapt(data=spect_ds.map(map_func=lambda spec, label: spec))

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Resizing(32, 32),
        layer,
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


    model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=30,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=10),
    )
    

    testdataDir = pathlib.Path("test")
    testfilenames = tf.io.gfile.glob(str(testdataDir) + '/*/**')
    testfilenames = tf.random.shuffle(testfilenames)

    test_ds = tf.data.Dataset.from_tensor_slices(testfilenames)
    test_ds = process_data(test_ds)
    for spectrogram, label in test_ds.batch(1):
        prediction = model(spectrogram)
        plt.bar(chords, tf.nn.softmax(prediction[0]))
        plt.title('Predictions for file {}'.format(chords[label[0]]))
        plt.show()
        
    model.save('model/')

    test_audio = []
    test_labels = []

    for audio, label in test_ds:
        test_audio.append(audio.numpy())
        test_labels.append(label.numpy())

    test_audio = np.array(test_audio)
    test_labels = np.array(test_labels)
    y_pred = np.argmax(model.predict(test_audio), axis=1)
    y_true = test_labels

    test_acc = sum(y_pred == y_true) / len(y_true)
    print(f'Test set accuracy: {test_acc:.0%}')

    splitSong('chordprog.wav')

    song_filenames = tf.io.gfile.glob('temporary/*')
    song_ds = tf.data.Dataset.from_tensor_slices(song_filenames)
    song_ds = process_data(song_ds)

    for spectrogram, label in song_ds.batch(1):
        prediction = model(spectrogram)
        pos = (np.argmax(prediction[0], axis=-1))
        print("ESTIMATE")
        print(chords[pos])
        

    # all_spects = []

    # for i, chunk in enumerate(chunks):
    #     path='temporary/chunk{0}.wav'.format(i)
    #     chunk.export(path, format="wav")
    #     audio_binary = tf.io.read_file(path)
    #     waveform = decode(audio_binary)
    #     spectrogram = wave_to_spect(waveform)
    #     all_spects.append(spectrogram)

    # song_ds = process_data(all_spects)
    # for spectrogram in song_ds.batch(1):
    #     prediction = model(spectrogram)
    #     plt.bar(chords, tf.nn.softmax(prediction[0]))
    #     plt.show()
    
        
    

def wave_to_spect(wave):
    input_len = 85000
    wave = wave[:input_len]
    zero_pad = tf.zeros( 
        [85000] - tf.shape(wave),
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


def get_spect_and_id(waveform, label):
    spectrogram = wave_to_spect(waveform)
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


def process_data(files):
    tune = tf.data.AUTOTUNE
    processed = files.map(
        map_func=get_waveform_and_label,
        num_parallel_calls=tune
    )

    processed = processed.map(
        map_func=get_spect_and_id,
        num_parallel_calls=tune
    )
    return processed
    


# song conversion must be different as it could be stereo / mono. Also must be seperated 
def splitSong(path):
    sound = AudioSegment.from_wav(path)
    times = detect_nonsilent(sound, min_silence_len=400, silence_thresh=-45)
    print(times)
    snippet = split_on_silence(sound, min_silence_len=400, silence_thresh=-45)

    for i, chunk in enumerate(snippet):
        path='temporary/chunk{0}.wav'.format(i)
        chunk.export(path, format="wav")
        audio_binary = tf.io.read_file(path)
        waveform = decode(audio_binary)
        spectrogram = wave_to_spect(waveform)
        print(spectrogram.shape)
    return snippet
  

def convertSong(chunks):
    createTemp()
    
    all_spects = []


    

    rows = 1
    cols = 4
    n = rows*cols
    fig, axes = plt.subplots(cols, figsize=(20, 20))
    for i, (spectrogram) in enumerate(all_spects):
        c = i % cols
        ax = axes[c]
        ax.set_title("Chunk {}".format(i))
        plot_spect(spectrogram.numpy(), ax)
        ax.axis('off')

    plt.show()


    return all_spects
        
    

def createTemp():
    if os.path.exists("temporary"):
        files = os.listdir("temporary")
        for file in files: 
            os.remove("temporary/" +file)
    else: 
        os.mkdir("temporary")



#best website https://www.tensorflow.org/tutorials/audio/simple_audio
#splitSong("chordprog.wav")
findData()
# compare notes from 2 tracks to