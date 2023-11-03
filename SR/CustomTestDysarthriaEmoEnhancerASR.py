from transformers import pipeline
from Dataset_Manager import Dataset_Manager
import torch


def get_test_accuracy():
    correct_samples = 0
    count = 0
    for i in range(0, len(dataset['valid'])):
        count += 1
        result = asr(dataset['valid'][i]['audio'].copy(), generate_kwargs={"language": "english", "task": "transcribe"})['text']
        correct = dataset['valid'][i]['text']
        if result == correct:
            correct_samples += 1
        print("Predicted: ", correct, " - Actual: ", result, " - Current Accuracy: ",
              correct_samples / count)
    return correct_samples / len(dataset['valid'])


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    asr = pipeline("automatic-speech-recognition", model="FilippoLampa/whisper-small-dv", device=device)
    dataset = Dataset_Manager().get_dataset()
    accuracy = get_test_accuracy()
    print(accuracy)