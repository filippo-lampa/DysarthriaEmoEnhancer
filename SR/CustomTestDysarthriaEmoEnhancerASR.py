from transformers import pipeline
from Dataset_Manager import Dataset_Manager
import torch
import re
import numpy as np


def calculate_wer(reference, hypothesis):
    # Split the reference and hypothesis sentences into words
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    # Initialize a matrix with size |ref_words|+1 x |hyp_words|+1
    # The extra row and column are for the case when one of the strings is empty
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    # The number of operations for an empty hypothesis to become the reference
    # is just the number of words in the reference (i.e., deleting all words)
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    # The number of operations for an empty reference to become the hypothesis
    # is just the number of words in the hypothesis (i.e., inserting all words)
    for j in range(len(hyp_words) + 1):
        d[0, j] = j
    # Iterate over the words in the reference and hypothesis
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            # If the current words are the same, no operation is needed
            # So we just take the previous minimum number of operations
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                # If the words are different, we consider three operations:
                # substitution, insertion, and deletion
                # And we take the minimum of these three possibilities
                substitution = d[i - 1, j - 1] + 1
                insertion = d[i, j - 1] + 1
                deletion = d[i - 1, j] + 1
                d[i, j] = min(substitution, insertion, deletion)
    # The minimum number of operations to transform the hypothesis into the reference
    # is in the bottom-right cell of the matrix
    # We divide this by the number of words in the reference to get the WER
    wer = d[len(ref_words), len(hyp_words)] / len(ref_words)
    return wer


def get_test_accuracy():
    correct_samples_1 = 0
    correct_samples_2 = 0
    correct_samples_3 = 0
    count = 0
    total_wer_1 = 0
    total_wer_2 = 0
    total_wer_3 = 0

    regex = re.compile('[,\.!?\"\']')

    for i in range(0, len(dataset['test'])):

        result1 = regex.sub('', asr1(dataset['test'][i]['audio'].copy(),
                                     generate_kwargs={"language": "english", "task": "transcribe"})['text'].strip().lower())
        result2 = regex.sub('', asr2(dataset['test'][i]['audio'].copy(),
                                     generate_kwargs={"language": "english", "task": "transcribe"})['text'].strip().lower())
        result3 = regex.sub('', asr3(dataset['test'][i]['audio'].copy(),
                                     generate_kwargs={"language": "english", "task": "transcribe"})['text'].strip().lower())

        correct = regex.sub('', dataset['test'][i]['text'].strip().lower())

        # Ignore the wrong samples in the dataset
        if correct != "xxx" and "\\" not in correct:

            count += 1

            if result1 == correct:
                correct_samples_1 += 1
            if result2 == correct:
                correct_samples_2 += 1
            if result3 == correct:
                correct_samples_3 += 1

            total_wer_1 += calculate_wer(correct, result1)
            total_wer_2 += calculate_wer(correct, result2)
            total_wer_3 += calculate_wer(correct, result3)

            print("Predicted1: ", result1, " - Actual: ", correct, " - Current Accuracy: ",
                  str("%.2f" % (correct_samples_1 / count)) + "%", " - Current WER: ",
                  str("%.2f" % ((total_wer_1 / count) * 100)) + "%")
            print("Predicted2: ", result2, " - Actual: ", correct, " - Current Accuracy: ",
                  str("%.2f" % (correct_samples_2 / count)) + "%", " - Current WER: ",
                  str("%.2f" % ((total_wer_2 / count) * 100)) + "%")
            print("Predicted3: ", result3, " - Actual: ", correct, " - Current Accuracy: ",
                  str("%.2f" % (correct_samples_3 / count)) + "%", " - Current WER: ",
                  str("%.2f" % ((total_wer_3 / count) * 100)) + "%")
            print("--------------------------------------------------")

    return (correct_samples_1 / len(dataset['test']), correct_samples_2 / len(dataset['test']),
            correct_samples_3 / len(dataset['test']), total_wer_1 / len(dataset['test']),
            total_wer_2 / len(dataset['test']), total_wer_3 / len(dataset['test']))


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    asr1 = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)
    asr2 = pipeline("automatic-speech-recognition", model="FilippoLampa/whisper-small-dv", device=device)
    asr3 = pipeline("automatic-speech-recognition", model="FilippoLampa/dysarthria-emo-enhancer", device=device)

    dataset = Dataset_Manager().get_dataset()
    accuracy1, accuracy2, accuracy3, wer1, wer2, wer3 = get_test_accuracy()
    print("Base whisper acc: " + str("%.2f" % accuracy1) + "%" + " - Custom whisper def dataset acc: " +
          str("%.2f" % accuracy2) + "%" +
          " - Custom whisper enh dataset acc: " + str("%.2f" % accuracy3) + "%")
    print("Base whisper WER: " + str("%.2f" % (wer1 * 100)) + "% " + "- Custom whisper def dataset WER: " +
          str("%.2f" % (wer2 * 100)) + "% " +
          " - Custom whisper enh dataset WER: " + str("%.2f" % (wer3 * 100)) + "% ")