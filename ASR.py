from transformers import pipeline
from Dataset_Manager import Dataset_Manager

if __name__ == '__main__':
    asr = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-100h")
    dataset = Dataset_Manager().get_dataset()
    result = asr(dataset['train'][6]['audio'].copy())
    print(result)