# Audio-Captioning
Automatic audio captions are intended to automatically generate a natural language
description of an audio clip. This is an intermodal translation task (rather than a
voice-to-text conversion) where the system takes an audio signal as input and returns a
textual description of the audio signal. Automatic audio captioning methods can model
concepts, physical properties of objects and environments, and knowledge at a high level.
This modeling can be used in a variety of applications, from automated content
description to intelligent, content-aware machine-to-machine interactions. Most closed
captioning models follow the encoder-decoder architecture. In this architecture, the
decoder predicts a word sequence based on the audio features extracted by the encoder.
Convolutional neural networks (CNNs) and recurrent neural networks (RNNs) are
commonly used as audio encoders. The input to the encoder is a log mel spectrogram
calculated from an audio signal while the model is predicting a sequence of words. We
replaced the encoder part of the basic architecture and added an Inception V3 model to
extract audio functionality from the log mel spectrogram of the automatic audio subtitle
task.


# Introduction
Automated Audio Captioning (AAC) is a new and challenging task that includes a
variety of modalities. This can be described as generating a textual description(i.e
caption) when given an audio signal. The textual description obtained should be as close
as possible to the one assigned by humans. Unlike automatic speech recognition, which
simply converts speech to text, Automated Audio Captioning converts environmental
sound to text. More information is needed to generate accurate captions, such as sound
events, acoustic scenes, spatiotemporal relationships between sound sources, foreground
and background discrimination, concepts, and physical properties of objects and
surroundings. Lack of training data is one of the most challenging part of Automated
Audio Captioning task.
The Automatic audio captioning task combines audio with natural language processing
(NLP) to create meaningful natural language sentences. The purpose of audio captioning
is different from audio processing tasks such as audio event / scene detection and audio
labeling. These tasks are not aimed at creating descriptive text in natural language, but
audio captioning is aimed at capturing relationships between events, scenes, and objects
to create meaningful text.The resulting captions contain information on different aspects
of the audio signal’s content, ranging from detecting audio events to understanding
spatiotemporal interactions, foreground and background disambiguation, environment,
textures, and other high-level information.All the published works focusing on
Automated Audio Captioning completely employ deep learning methods. The majority
of them use an encoder-decoder system and approach the challenge as a
sequence-to-sequence (seq2seq) learning problem.
