import torch
import transformers
from transformers import pipeline
from datasets import Dataset, Audio
import argparse
import analyser
import gradio as gr
import os


def main(audio):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    transformers.logging.set_verbosity_error()
    print("Loading the audio file...")
    asr = pipeline("automatic-speech-recognition", model="FilippoLampa/dysarthria-emo-enhancer", device=device,
                   generate_kwargs={"language": "english", "task": "transcribe"})
    dataset = Dataset.from_dict({"audio": [audio]}).cast_column("audio", Audio())
    print("Understanding the sentence...")
    result = asr(dataset[0]['audio'].copy(), generate_kwargs={"task": "transcribe"})

    print("Analysing the sentiment...")
    anlysr = analyser.SentimentAnalyser()
    print("Sentence recognized: " + result['text'])
    print("--------------------")

    return "The sentence is likely to be interpreted as positive" if anlysr.get_sentiment(result['text']) == 1 else \
        "The sentence is likely to be interpreted as negative"


if __name__ == '__main__':

    custom_css = ".gradio-container{position: absolute; -webkit-transform: translate(-0%, 10%);\
                    -moz-transform: translate(-0%, 10%);\
                    -ms-transform: translate(-0%, 10%);\
                    -o-transform: translate(-0%, 10%);\
                    transform: translate(-0%, 10%);}"

    analysis_interface = gr.Interface(
        main,
        gr.Audio(sources=["upload","microphone"], type="filepath"),
        gr.Label(num_top_classes=1),
        examples=[
            [os.path.join(os.path.dirname(__file__),"../Customized_Dataset/kaggle/input/dysarthria-and-nondysarthria-"
                                                    "speech-dataset/Dysarthria and Non Dysarthria/Dataset/Male_Dysarthria"
                                                    "/M02/Session2/Wav/0191.wav")],
            [os.path.join(os.path.dirname(__file__),"../Customized_Dataset/kaggle/input/dysarthria-and-nondysarthria-"
                                                    "speech-dataset/Dysarthria and Non Dysarthria/Dataset/"
                                                    "Female_dysarthria/F03/Session3/Wav/0130.wav")],
            [os.path.join(os.path.dirname(__file__),"../Customized_Dataset/kaggle/input/dysarthria-and-nondysarthria-"
                                                    "speech-dataset/Dysarthria and Non Dysarthria/Dataset/"
                                                    "Male_Dysarthria/M03/Session2/Wav/0379.wav")],
            [os.path.join(os.path.dirname(__file__),"../Customized_Dataset/kaggle/input/dysarthria-and-nondysarthria-"
                                                    "speech-dataset/Dysarthria and Non Dysarthria/Dataset/"
                                                    "Female_dysarthria/F03/Session3/Wav/0127.wav")],
            [os.path.join(os.path.dirname(__file__),"../Customized_Dataset/kaggle/input/dysarthria-and-nondysarthria-"
                                                    "speech-dataset/Dysarthria and Non Dysarthria/Dataset/"
                                                    "Female_dysarthria/F03/Session3/Wav/0107.wav")],
            [os.path.join(os.path.dirname(__file__),"../Customized_Dataset/kaggle/input/dysarthria-and-nondysarthria-"
                                                    "speech-dataset/Dysarthria and Non Dysarthria/Dataset/"
                                                    "Male_Dysarthria/M03/Session2/Wav/0373.wav")],
        ],
        css=custom_css,
        allow_flagging="never",
        title="Dysarthria Emo Enhancer",
        description="<div style='text-align: center;'>Upload an audio file from a dysarthric speaker from your system "
                    "or record it using the microphone to know weather the sentence is likely to be interpreted as "
                    "positive or negative.</div>"
    )

    analysis_interface.launch()