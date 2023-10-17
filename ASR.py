from transformers import pipeline
from Dataset_Manager import Dataset_Manager
import torch

if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)
    dataset = Dataset_Manager().get_dataset()
    result = asr(dataset['train'][1]['audio'].copy(), generate_kwargs={"task": "transcribe"})
    print(result)