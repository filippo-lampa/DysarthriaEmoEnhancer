import librosa.feature
import numpy as np
import pyworld
import pysptk
from fastdtw import fastdtw
import os


class Extraction:

    normal_audio_folder_path = ""
    dysarthric_audio_folder_path = ""

    def __init__(self, normal_audio_folder_path, dysarthric_audio_folder_path):
        self.normal_audio_folder_path = normal_audio_folder_path
        self.dysarthric_audio_folder_path = dysarthric_audio_folder_path
        return

    def extract_log_spectrograms(self, audio, sample_rate):
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=2048,
                                                         hop_length=int(0.01 * sample_rate), n_mels=80)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # top_db = 120
        # Clip values below -120dB
        # log_mel_spectrogram = np.maximum(log_mel_spectrogram, log_mel_spectrogram.max() - top_db)

        # Normalize amplitude in the range [0, 1]
        """log_mel_spectrogram = (log_mel_spectrogram - log_mel_spectrogram.min()) / (log_mel_spectrogram.max() -
                                                                                   log_mel_spectrogram.min())"""
        return log_mel_spectrogram


    def wav2mcep_numpy(self, current_audio, sample_rate):
        # Use WORLD vocoder to spectral envelope
        _, sp, _ = pyworld.wav2world(current_audio.astype(np.double), fs=sample_rate, frame_period=10, fft_size=512)
        # Extract MCEP features
        mgc = pysptk.sptk.mcep(sp, order=12, alpha=0.65, maxiter=0, etype=1, eps=1.0E-8, min_det=0.0, itype=3)
        return mgc

    def align_mcep_and_mel(self, normal_mcep, dysarthric_mcep, normal_log_mel_spectrogram,
                           dysarthric_log_mel_spectrogram):
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
        return aligned_normal_mel, aligned_dysarthric_mel

    def execute(self, number_of_elements_to_analyse=float("inf")):
        aligned_normal_mel_list = []
        aligned_dysarthric_mel_list = []
        for index, filename in enumerate(os.listdir(self.dysarthric_audio_folder_path)):

            if index == number_of_elements_to_analyse:
                break

            dysarthric_audio, dysarthric_sample_rate = librosa.load(self.dysarthric_audio_folder_path + filename)

            if not os.path.exists(self.normal_audio_folder_path + "C" + filename):
                break

            normal_audio, normal_sample_rate = librosa.load(self.normal_audio_folder_path + "C" + filename)

            normal_log_mel_spectrogram = self.extract_log_spectrograms(normal_audio, normal_sample_rate)
            dysarthric_log_mel_spectrogram = self.extract_log_spectrograms(dysarthric_audio, dysarthric_sample_rate)

            normal_mcep = self.wav2mcep_numpy(normal_audio, normal_sample_rate)
            dysarthric_mcep = self.wav2mcep_numpy(dysarthric_audio, dysarthric_sample_rate)

            aligned_normal_mel, aligned_dysarthric_mel = self.align_mcep_and_mel(normal_mcep, dysarthric_mcep,
                                                                            normal_log_mel_spectrogram,
                                                                            dysarthric_log_mel_spectrogram)
            aligned_normal_mel_list.append(aligned_normal_mel)
            aligned_dysarthric_mel_list.append(aligned_dysarthric_mel)

            print("Preprocessed " + str(index) + " elements")

        return aligned_normal_mel_list, aligned_dysarthric_mel_list

    def execute_for_generation(self, audio_path):

        audio, sample_rate = librosa.load(audio_path)

        log_mel_spectrogram = self.extract_log_spectrograms(audio, sample_rate)

        print("File preprocessed")

        return log_mel_spectrogram, sample_rate
