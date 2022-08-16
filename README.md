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


## Introduction
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

## Proposed Method
The proposed model consists of a CNN encoder and a Transformer decoder.The encoder
produces an audio feature vector based on the log mel-spectrogram of an audio sample.
Based on the feature vector generated by the encoder and previously generated words,
the decoder estimates the posterior probability of the words.

<img src=https://github.com/Monishraj50/Audio-Captioning/blob/main/utils/model_arch.png height =300>

## Model Architecture
### Encoder

Convolutional neural networks (CNNs) have been widely used in audio processing research,
and they have demonstrated a strong ability to extract audio information. To extract
audio features and avoid over-fitting, a pre-trained Inception V3 model was utilised as the
encoder.
Positional encoding is used to feed those audio features into the embedding layer, which
uses sine and cosine functions of various frequencies. Create a vector using the cos
function for every odd index on the input vector, and a vector using the sin function for
every even index. After that, add those vectors to their corresponding input embeddings,
which successfully provides the network information on each vector’s position followed by
a transformer encoder

### Decoder
The decoder consists of three parts,
• a word embedding layer,
• a standard Transformer decoder and
• a linear layer.
The word embedding layer converts each input word into a vector with a fixed dimension.
The word embedding layer can be thought of as a V*d embedding look-up matrix, where
V is the vocabulary size and d is the word vector dimension. This layer is randomly
initialised and frozen during the training phase.Transformer is built to handle sequential
data and displays cutting-edge performance in natural language production tasks. The
Transformer decoder here is used as the multi-modal decoder. The transformer decoder
receives the word embeddings from the word embedding layer, as well as audio features
from the encoder, and incorporates them using a multi-head attention method. The
decoder employed only has two transformer decoder blocks with four heads because the
captions in the datasets are typically short. The dimension of the hidden layer is 128.
Finally, a linear layer is utilised to generate a probability distribution for the vocabulary.

## Transfer Learning
Transfer learning tries to transfer the information from one domain to another domain in
order to solve the problem of insufficient training data and increase the model’s
generalisation capabilities. Transfer learning is most commonly employed in tasks that
need only one modality. Two transfer learning strategies are offered for this cross-modal
(audio to text) translation job, the first transferring from an upstream task and the
other from an in-domain dataset.
The encoder extracts the audio features from an audio clip by using the Inception-v3
pretrained model.Inception-v3 is a convolutional neural network architecture that has a
total of 42 layers and a lower error rate than its predecessors. Label Smoothing, Factorized
7 x 7 convolutions, and the inclusion of an auxiliary classifer to transfer label information
lower down the network are all enhancements made by the Inception family (along with
the use of batch normalisation for layers in the sidehead).
