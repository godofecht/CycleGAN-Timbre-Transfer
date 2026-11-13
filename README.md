# CycleGAN-Timbre-Transfer

One-to-one timbre transfer between source and target datasets, using a modified cycle-consistent adversarial network.

![Spectrogram of a string recording used as input. Bright, evenly spaced horizontal bands fill the lower half, which are the note's harmonics, and they run for most of the clip before stopping.](Audio%20%2B%20Spectrograms/BEST%20STRING2FLUTE/Test0.png)

![Spectrogram of the network's flute-style output for the same clip. There are fewer, thinner bands, mostly near the bottom, and the picture is darker overall.](Audio%20%2B%20Spectrograms/BEST%20STRING2FLUTE/Prediction0.png)

## What this is

Timbre is the quality that makes a flute and a violin sound different when they play the same note at the same loudness. Timbre transfer means taking a recording of one instrument and making it sound like another while keeping the notes.

This project tries to turn string recordings into flute or guitar sounds, and flute into strings. It does not work on sound waves directly. It first turns each clip into a spectrogram, a picture of sound with time going across, pitch going up, and brightness showing loudness. A CycleGAN (explained below) then learns to change pictures of one instrument into pictures of the other, and the new picture is turned back into sound.

The two pictures above show one example: a string clip going in, and the flute-style result coming out. The repo has ten examples for each of three instrument pairs, as pictures and as audio you can play.

## What you can learn from it

- Music Technology and Physics: what timbre is, and how harmonics show up in a spectrogram.
- Computer Science: generative adversarial networks (GANs), where two networks learn by competing.
- Computer Science: how audio can be treated as an image so image-processing networks can work on it.
- Computer Science: cycle consistency, a way to train without matched pairs of examples.
- Maths: Fourier analysis and the mel scale, which spaces pitches the way human hearing does.

## Run it

### Listen to the results

You do not need to install anything to hear the results. Open the [Audio + Spectrograms](Audio%20%2B%20Spectrograms) folder on GitHub, choose a folder, and open any `.wav` file. Use the download button to save it, then play it on your computer.

| Folder | Mapping |
| --- | --- |
| `BEST STRING2FLUTE/` | strings to flute |
| `BEST FLUTE2STRING/` | flute to strings |
| `BEST STRING2GUITAR/` | strings to guitar |

In each folder, `audio_test3.wav` is input clip number 3 and `audio_prediction3.wav` is the network's version of it. `Test3.png` and `Prediction3.png` are their spectrograms. The input clips were also turned into spectrograms and back, so both files have the same slightly rough sound from that process.

### On your own computer

Training needs your own audio, because the training clips are not in the repo. The code was last run in 2020 and uses library versions that are now old. These steps come from reading the code and have not been tested.

**You will need:**

- [Python](https://www.python.org/downloads/) 3.7 or 3.8 and [Git](https://git-scm.com/downloads).
- These Python packages: TensorFlow 2, `tensorflow-datasets`, `tensorflow-examples`, `librosa` older than 0.8 (newer versions removed `librosa.output.write_wav`), `numpy`, `scipy`, `matplotlib` and `ipython`. `eval.py` also needs `fastdtw`.
- A graphics card is not required, but training on a processor alone will be slow.
- Short clips of two instruments, ideally 4 seconds long. The code expects exactly 139 training clips of the first instrument, 160 of the second, and 10 test clips.

1. Download the code:

   ```bash
   git clone https://github.com/godofecht/CycleGAN-Timbre-Transfer.git
   cd CycleGAN-Timbre-Transfer
   ```

2. Install the packages:

   ```bash
   pip install tensorflow tensorflow-datasets "librosa<0.8" numpy scipy matplotlib ipython fastdtw
   pip install git+https://github.com/tensorflow/examples.git
   ```

3. Put your clips in these folders: `audio/TrainA` (139 clips of the first instrument), `audio/TrainB` (160 clips of the second), `audio/TestA` and `audio/TestB` (10 clips each).

4. Make the output folders, which the scripts do not create:

   ```bash
   mkdir -p output/TrainA output/TrainB output/TestA output/TestB SpectrogramImages
   ```

5. Turn the audio into spectrograms, saved as `.npy` files:

   ```bash
   python spectrogram.py
   ```

6. Pack the spectrograms into TFRecord files (a TensorFlow data format), then copy them to where `cyclegan.py` looks for them:

   ```bash
   python build_data.py --X_input_dir output/TrainA --Y_input_dir output/TrainB
   cp output/tfrecords/trainA.tfrecord output/tfrecords/trainB.tfrecord output/tfrecords/testA.tfrecord .
   ```

7. Train:

   ```bash
   python cyclegan.py
   ```

   It trains for 20 epochs (passes over the data) and prints the losses as it goes. It saves a checkpoint every 10 epochs in `checkpoints/train`. When it finishes, it converts the 10 test clips and writes `Test0.png` to `Test9.png` and `Prediction0.png` to `Prediction9.png` into `SpectrogramImages/`, and `audio_test0.wav` to `audio_prediction9.wav` into the main folder.

## How it works

**Spectrograms.** `spectrogram.py` loads each clip and makes a mel spectrogram with 256 frequency bands up to 8 kHz. The mel scale spaces the bands the way human hearing does, closer together at low pitches. A 4-second clip at 16 kHz gives 126 time steps, so each spectrogram is 256 by 126. It is padded with zeros to 256 by 256, because the network expects a square picture.

**A GAN.** A generative adversarial network has two parts. The generator makes fake examples. The discriminator looks at real and fake examples and tries to tell them apart. Each one learns from the other's mistakes: the discriminator gets better at spotting fakes, and the generator gets better at making fakes it cannot spot.

**A CycleGAN.** This project has no matched pairs, so there is no "this flute clip is the right answer for this string clip". A CycleGAN gets round that with two generators and two discriminators:

- Generator G turns strings into flute, and generator F turns flute into strings.
- Discriminator Y judges whether a flute spectrogram is real, and discriminator X does the same for strings.
- Cycle consistency: if you turn a string clip into flute with G and back with F, you should get the original string clip. The difference is added to the loss (the number the networks try to make small). This stops G from making any flute sound at all, because it has to keep enough of the input for F to rebuild it.

**Back to sound.** After training, G converts each test spectrogram. The program reverses the scaling and uses the Griffin-Lim method in librosa to estimate a sound wave that matches the picture. A spectrogram keeps loudness but loses the timing of each wave (the phase), so this step has to guess it, which adds roughness.

| File or folder | What is in it |
| --- | --- |
| `spectrogram.py` | audio to mel spectrograms, padding, and a function to display one |
| `build_data.py` | packs the spectrogram files into TFRecord files for each domain |
| `cyclegan.py` | the two generators, the two discriminators, the losses, the training loop and the conversion back to audio |
| `eval.py` | compares a generated clip with the two training sets using MFCCs (a compact summary of timbre) |
| `Audio + Spectrograms/` | results: input and output audio and spectrograms for three instrument pairs |

## Ideas to try

1. Listen to `audio_test0.wav` and `audio_prediction0.wav` in `Audio + Spectrograms/BEST STRING2FLUTE/`, then look at `Test0.png` and `Prediction0.png`. Which harmonics did the network keep and which did it remove? How does that match what a real flute sounds like?
2. In `cyclegan.py`, `LAMBDA` sets how much the cycle consistency loss counts. It is 1 here; the original CycleGAN paper used 10. Try 10 and compare the results.
3. In `cyclegan.py`, change `EPOCHS` from 20 and listen to how the output changes with more or less training.
4. In `cyclegan.py`, `random_jitter` uses `numpy.resize`, which repeats the data to fill the bigger array instead of stretching the picture. Replace it with `tf.image.resize` and see whether training improves.
5. In `cyclegan.py`, replace `librosa.output.write_wav` with `soundfile.write` from the `soundfile` package, so the code runs with current versions of librosa.
6. `spectrogram.py` has commented-out lines for a constant-Q transform (CQT), which spaces bands by musical notes. Switch to it, and change the conversion back to audio in `cyclegan.py` to match.

## Technical details

- Audio is converted to spectrograms with `librosa.feature.melspectrogram(n_mels=256, fmax=8000)` at the file's own sample rate, then padded from 256 by 126 to 256 by 256 and saved as `.npy`. Commented-out lines in `spectrogram.py` use a constant-Q transform (CQT) instead.
- Because the mapping is trained in both directions with a cycle consistency loss, the two datasets do not need to be paired. There is no requirement that a string recording and a flute recording play the same note.
- Before training, each spectrogram is compressed with log(x + 1), divided by its maximum, scaled to -1 to 1, enlarged to 284 by 284 with `numpy.resize` and randomly cropped back to 256 by 256.
- The generator is a U-Net (an encoder-decoder with skip connections between matching layers): 8 downsampling blocks from 64 to 512 filters and 7 upsampling blocks, with dropout in the first three, and a tanh output. The discriminator is a PatchGAN, which scores 30 by 30 patches of the input. Both are adapted from the pix2pix reference implementation (arXiv 1611.07004). `pix2pix` from `tensorflow_examples` is imported but not called.
- Losses: binary cross-entropy for the adversarial terms, L1 cycle loss weighted by `LAMBDA = 1`, and L1 identity loss weighted by 0.5 x `LAMBDA`. Four Adam optimisers with learning rate 2e-4 and beta1 0.5.
- Training runs 20 epochs over the first 100 clips of each domain, one pair per step. Checkpoints go to `./checkpoints/train` (two kept). TensorBoard logs go to `logs/gradient_tape/`.
- Output audio is made with `librosa.feature.inverse.mel_to_audio` (Griffin-Lim), normalised and written at 16 kHz. Sample rate is 16 kHz throughout.
- `eval.py` loads the first 139 clips of `audio/TrainA`, the first 160 of `audio/TrainB` and one clip from `audio/GeneratedB`, computes 13 MFCCs for each, and prints the average cosine distance from the generated clip to each training set, using the first MFCC row. It also defines a dynamic time warping (DTW) function that is not called.

## Project status

This is a 2020 research experiment and is not being developed. Known problems:

- The training audio is not in the repo, and there is no note of where it came from.
- File counts are fixed in the code (139, 160 and 10). A different number of clips makes the reshape in `cyclegan.py` fail.
- `cyclegan.py` reads `trainA.tfrecord` from the main folder, while `build_data.py` writes to `output/tfrecords/`.
- `librosa.output.write_wav` was removed in librosa 0.8, so the last step fails with current librosa.
- `BEST STRING2GUITAR/` is missing `audio_prediction1.wav` and `audio_prediction6.wav`, and its `FINALSTRINGGUITAR/` subfolder repeats the same pictures.
- The spectrogram pictures are labelled with a note axis meant for CQT (`y_axis='cqt_note'`), although they are mel spectrograms. The saved pictures hide the axes, so this does not show.

## Licence

You can use, study, share and change this code. If you share a changed version, you must share its source code under the same licence. See [LICENSE](LICENSE) (GNU GPL v3).

## Credits

- Written by Abhishek Shivakumar.
- CycleGAN: Jun-Yan Zhu, Taesung Park, Phillip Isola and Alexei A. Efros (2017), "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks".
- pix2pix: Phillip Isola, Jun-Yan Zhu, Tinghui Zhou and Alexei A. Efros (2017), "Image-to-Image Translation with Conditional Adversarial Networks", [arXiv:1611.07004](https://arxiv.org/abs/1611.07004).
- Built with [TensorFlow](https://www.tensorflow.org/) and [librosa](https://librosa.org/).
