import librosa.feature
import numpy as np
import pyworld
import pysptk
from fastdtw import fastdtw

if __name__ == '__main__':

    normal_audio_file = "C:\\Users\\Filippo\\Projects\\DysarthriaEmoEnhancer\\Preprocessing\\Non-Silenced-Audio-1.wav"
    dysarthric_audio_file = "C:\\Users\\Filippo\\Projects\\DysarthriaEmoEnhancer\\Preprocessing\\Non-Silenced-Audio.wav"

    normal_audio, normal_sample_rate = librosa.load(normal_audio_file)
    dysarthric_audio, dysarthric_sample_rate = librosa.load(dysarthric_audio_file)

    # Spectrogram extraction
    
    normal_mel_spectrogram = librosa.feature.melspectrogram(y=normal_audio, sr = normal_sample_rate, n_fft = 2048, hop_length = int(0.01 * normal_sample_rate), n_mels = 80)
    dysarthric_mel_spectrogram = librosa.feature.melspectrogram(y=dysarthric_audio, sr = dysarthric_sample_rate, n_fft = 2048, hop_length = int(0.01 * dysarthric_sample_rate), n_mels = 80)

    normal_log_mel_spectrogram = librosa.power_to_db(normal_mel_spectrogram, ref=np.max)
    dysarthric_log_mel_spectrogram = librosa.power_to_db(dysarthric_mel_spectrogram, ref=np.max)

    top_db = 120

    # Clip values below -120dB
    normal_log_mel_spectrogram = np.maximum(normal_log_mel_spectrogram, normal_log_mel_spectrogram.max() - top_db)
    dysarthric_log_mel_spectrogram = np.maximum(dysarthric_log_mel_spectrogram, dysarthric_log_mel_spectrogram.max() - top_db)

    # Normalize amplitude in the range [0, 1]
    normal_log_mel_spectrogram = (normal_log_mel_spectrogram - normal_log_mel_spectrogram.min()) / (normal_log_mel_spectrogram.max() - normal_log_mel_spectrogram.min())
    dysarthric_log_mel_spectrogram = (dysarthric_log_mel_spectrogram - dysarthric_log_mel_spectrogram.min()) / (dysarthric_log_mel_spectrogram.max() - dysarthric_log_mel_spectrogram.min())

    def wav2mcep_numpy(current_audio, sample_rate):

        # MCEP extraction

        # Use WORLD vocoder to spectral envelope
        _, sp, _ = pyworld.wav2world(current_audio.astype(np.double), fs=sample_rate,
                                     frame_period=10, fft_size=512)

        # Extract MCEP features
        mgc = pysptk.sptk.mcep(sp, order=12, alpha=0.65, maxiter=0,
                               etype=1, eps=1.0E-8, min_det=0.0, itype=3)

        return mgc


    normal_mcep = wav2mcep_numpy(normal_audio, normal_sample_rate)
    dysarthric_mcep = wav2mcep_numpy(dysarthric_audio, dysarthric_sample_rate)

    # Calculate DTW alignment
    distance, path = fastdtw(normal_mcep, dysarthric_mcep)

    # The 'path' variable now contains the optimal alignment between the two sequences

    num_frequency_bins = len(normal_log_mel_spectrogram)

    aligned_normal_mel = [[] for _ in range(num_frequency_bins)]
    aligned_dysarthric_mel = [[] for _ in range(num_frequency_bins)]

    
    for i, j in path:
        for bin_idx in range(num_frequency_bins):
            aligned_normal_mel[bin_idx].append(normal_log_mel_spectrogram[bin_idx][i])
            aligned_dysarthric_mel[bin_idx].append(dysarthric_log_mel_spectrogram[bin_idx][j])



