import torch
import transformers
from transformers import pipeline
from datasets import Dataset, Audio
import argparse
import analyser

def main(filepath):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    transformers.logging.set_verbosity_error()
    print("Loading the audio file...")
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)
    dataset = Dataset.from_dict({"audio": [filepath]}).cast_column("audio", Audio())
    print("Understanding the sentence...")
    result = asr(dataset[0]['audio'].copy(), generate_kwargs={"task": "transcribe"})
    print("Analysing the sentiment...")
    anlysr = analyser.SentimentAnalyser()
    print("Sentence recognized: " + result['text'])
    print("--------------------")
    print("The sentence is likely to be interpreted as positive" if anlysr.get_sentiment(result['text']) == 1 else
          "The sentence is likely to be interpreted as negative")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--filepath", required=True, help="path to the audio file")
    args = vars(ap.parse_args())
    main(args['filepath'])