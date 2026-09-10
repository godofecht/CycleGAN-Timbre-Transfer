# CycleGAN-Timbre-Transfer

One-to-one timbre transfer between two instrument datasets, using a cycle
consistent adversarial network over constant-Q spectrograms.

Audio never reaches the network directly. `spectrogram.py` turns each clip into a
256 by 256 CQT magnitude spectrogram, padded from the native 256 by 126, and the
CycleGAN learns the mapping between the two spectrogram domains. The generated
spectrogram is inverted back to audio at the end.

Because the mapping is trained in both directions with a cycle consistency loss,
the two datasets do not need to be paired. There is no requirement that a string
recording and a flute recording play the same note.

## Results

Trained pairs are checked in under `Audio + Spectrograms/`, each with the input
clip, the prediction and the spectrogram for both.

| Directory | Mapping |
|---|---|
| `BEST STRING2FLUTE/` | strings to flute |
| `BEST FLUTE2STRING/` | flute to strings |
| `BEST STRING2GUITAR/` | strings to guitar |

## Files

| | |
|---|---|
| `spectrogram.py` | CQT analysis, padding, display, and resynthesis back to audio. |
| `build_data.py` | Walks the input directories and writes TFRecords for each domain. |
| `cyclegan.py` | The two generators, the two discriminators and the training loop. Generator blocks come from the pix2pix reference implementation in `tensorflow_examples`. |
| `eval.py` | Runs a trained model over a directory of clips. |

## Run

```bash
python spectrogram.py          # audio to spectrograms
python build_data.py --X_input_dir output/TrainA --Y_input_dir output/TrainB
python cyclegan.py
```

TensorFlow 2 with `tensorflow_datasets`, `tensorflow_examples`, `librosa` and
`scipy`. Sample rate is 16 kHz throughout.
