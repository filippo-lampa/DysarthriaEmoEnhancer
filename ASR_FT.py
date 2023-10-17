from transformers import WhisperProcessor
from Dataset_Manager import Dataset_Manager
from DataCollatorSpeechSeq2SeqWithPadding import DataCollatorSpeechSeq2SeqWithPadding
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import WhisperForConditionalGeneration
from functools import partial
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

import torch
import evaluate


def prepare_dataset(ds):
    audio = ds["audio"]

    ds = processor(
        audio=audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=ds["text"],
    )

    # compute input length of audio sample in seconds
    ds["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    return ds


def is_audio_in_length_range(length):
    return length < max_input_length


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # compute orthographic wer
    wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)

    # compute normalised WER
    pred_str_norm = [normalizer(pred) for pred in pred_str]
    label_str_norm = [normalizer(label) for label in label_str]
    # filtering step to only evaluate the samples that correspond to non-zero references:
    pred_str_norm = [
        pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0
    ]
    label_str_norm = [
        label_str_norm[i]
        for i in range(len(label_str_norm))
        if len(label_str_norm[i]) > 0
    ]

    wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)

    return {"wer_ortho": wer_ortho, "wer": wer}


if __name__ == '__main__':

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    processor = WhisperProcessor.from_pretrained("openai/whisper-small")

    dataset = Dataset_Manager().get_dataset()

    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=1)

    max_input_length = 30.0

    dataset["train"] = dataset["train"].filter(
        is_audio_in_length_range,
        input_columns=["input_length"],
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    metric = evaluate.load("wer")

    normalizer = BasicTextNormalizer()

    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

    # disable cache during training since it's incompatible with gradient checkpointing
    model.config.use_cache = False

    # set language and task for generation and re-enable cache
    model.generate = partial(
        model.generate, use_cache=True
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-small-dv",  # name on the HF Hub
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=50,
        max_steps=500,  # increase to 4000 if you have your own GPU or a Colab paid plan
        gradient_checkpointing=True,
        fp16=True,
        fp16_full_eval=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=500,
        eval_steps=500,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=True,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )

    trainer.train()

    trainer.evaluate()