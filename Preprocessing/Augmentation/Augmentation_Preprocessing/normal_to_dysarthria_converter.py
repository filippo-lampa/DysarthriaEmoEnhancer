import tts_model
import extraction_conversion
import librosa.feature
import soundfile as sf
import torch
import numpy as np

if __name__ == '__main__':
    extr = extraction_conversion.Extraction("", "")
    normal_utterance, sample_rate = extr.execute_for_generation(
        "C:\\Users\\Filippo\\Projects\\DysarthriaEmoEnhancer\\Preprocessing\\SR_Nor\\CF02_B1_C1_M3_trimmed.wav")

    converted_utterance = tts_model.convert(normal_utterance).permute(2, 0, 1)
    converted_utterance = torch.squeeze(converted_utterance)

    def invert_log_mel_spectrogram(log_mel_spectrogram, sample_rate):
        mel_spectrogram = librosa.db_to_power(log_mel_spectrogram.detach().numpy())

        audio = librosa.feature.inverse.mel_to_audio(mel_spectrogram, sr=sample_rate, n_fft=2048,
                                                     hop_length=int(0.01 * sample_rate))

        # Normalize the audio
        audio = audio / np.max(np.abs(audio))

        return audio

    wav_file = invert_log_mel_spectrogram(converted_utterance, sample_rate)

    sf.write("test1.wav", wav_file, sample_rate)
