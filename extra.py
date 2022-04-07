#wavToSpect(convertChordToWaveform('chords/a-chord.wav'))
#wavToSpect(convertChordToWaveform('chords/e-chord.wav'))
#wavToSpect(convertChordToWaveform('chords/b-chord.wav'))


# def graphs(): 
#     dataDir = pathlib.Path(path)
#     chords = np.array(tf.io.gfile.listdir(str(dataDir)))
#     print('chords:', chords)
#     filenames = tf.io.gfile.glob(str(dataDir) + '/*/*')
#     filenames = tf.random.shuffle(filenames)
    
#     autotune = tf.data.AUTOTUNE
#     files_ds = tf.data.Dataset.from_tensor_slices(filenames)
#     waveform_ds = files_ds.map(
#     map_func=get_waveform_and_label,
#     num_parallel_calls=autotune
#     )
    
#     spect_ds = waveform_ds.map(
#         map_func=wave_to_spect,
#         num_parallel_calls=autotune
#     )


#     for waveform, label in waveform_ds.take(1):
#         label = label.numpy().decode('utf-8')
#         spectrogram = wave_to_spect(waveform, label)

#     print('Label:', label)
#     print('Waveform shape:', waveform.shape)
#     print('Spectrogram shape:', spectrogram[0].shape)


#     fig, axes = plt.subplots(2, figsize=(12, 8))
#     timescale = np.arange(waveform.shape[0])
#     axes[0].plot(timescale, waveform.numpy())
#     axes[0].set_title(label +' Waveform')
#     axes[0].set_xlim([0, 115000])

#     plot_spect(spectrogram[0].numpy(), axes[1])
#     axes[1].set_title('Spectrogram')
#     plt.show()

#     rows = 3 
#     cols = 3 
#     n = rows * cols
#     fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

#     for i, (spect, label) in enumerate(spect_ds.take(n)):
#         r = i // cols
#         c = i % cols
#         ax = axes[r][c]
#         plot_spect(spect.numpy(), ax)
#         ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
#         ax.set_title(label.numpy().decode('utf-8') + " chord")

#     plt.show()