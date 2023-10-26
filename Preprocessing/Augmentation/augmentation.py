'''
UASpeech: For our experiments, we use the Dysarthric Speech Database for Universal Access Research (UASpeech) [6].
The database contains utterances of 449 unique isolated words consisting of both uncommon and common words such as digits,
computer commands, and radio alphabet words. Of the 449 words, 288 are spoken once each, 6 are spoken twice each and 155
are spoken 3 times each. 7 microphones are used to record each utterance. Recordings are provided from both normal speakers
and speakers with Cerebral Palsy who self-report a diagnosis of dysarthria. Speech intelligibility ratings are given for
each speaker, which are calculated empirically from human annotators’ ability to correctly transcribe each person’s speech.
Transcription accuracy of 0-25% produces an intelligibility rating of "Very Low," 25-50% is rated "Low," 50-75% is rated
"Mid," and 75-100% is rated "High." For our experiments, we use 9 control speakers and 4 speakers with dysarthria.
Speakers F05 and M14 are female and male speakers with dysarthria, respectively, with intelligibility ratings of "High."
Speakers F04 and M05 have intelligibility ratings of "Mid." We would expect to use
(288+6×2+155×3)×(7 microphones)×(13 speakers) = 69615 total utterances, but due to our preprocessing some files are
excluded (69254 total utterances).

Preprocessing and Feature Extraction: There is some stationary noise in the recordings, which we remove using Noisereduce
[14]. Next, we trim silence from the beginning and end of each recording. Finally, we extract the mel log spectrogram
using Librosa [15]. We use 80 mel frequency bins and a 10ms frame shift. We allow a dynamic range of 120db by clipping
anything below -120db, then normalize amplitude in the range zero to one.

Attention-Based Voice Conversion: Since UASpeech is a parallel corpus, we time-align matching normal and dysarthric
utterances using dynamic time warping (DTW). DTW is performed using 12 MCEP features per frame, with the same frame shift
as the mel log spectrogram features (10ms). Normal and dysarthric utterances are aligned using DTW on MCEP, then the DTW
path is used to time-align the mel log spectrogram features.

Our voice conversion model consists of 6 layers of multihead attention [16]. We implement this model using the
TransformerEncoderLayer from PyTorch where each multihead attention layer has 8 heads and a model dimension of 80
(number of frequency bins). To train the voice conversion model, we pass the time-aligned normal utterance through the
network and apply the mean-squared error (MSE) loss function between the network output and the matching time-aligned
dysarthric utterance. We use the Adam optimizer, a batch-size of one, and train for 150,000 iterations.

Attention: (Proposed system, Figure 1) Normal and dysarthric sentences from the seen partition are paired to train the
voice conversion model. For both the Attention and DCGAN voice conversion approaches, we train a separate model for each
of the 9 × 4 = 36 speaker pairs, each pair including one speaker with and one without dysarthria. The trained voice
conversion model is then used to convert data from the normal unseen train and val datasets into artificial dysarthric
utterances. We use the converted utterances from all 36 models to train and validate an ASR model and evaluate the ASR
model performance on dysarthric unseen test data.
'''

from Augmentation_Preprocessing import extraction_conversion, tts_model

if __name__ == '__main__':

    # TODO execute extraction iterating on a list of folder paths to analyse all the dataset's folders

    aligner = extraction_conversion.Extraction("../SR_Nor/",
                                               "../SR_Dys/")

    aligned_normal_mel_list, aligned_dysarthric_mel_list = aligner.execute()

    # TODO slice arrays into training and validation
    model = tts_model.train(aligned_normal_mel_list, aligned_dysarthric_mel_list)





