import numpy as np
from datasets import load_dataset, load_metric

import os

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from transformers import Trainer
from transformers import TrainingArguments
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2ForSequenceClassification
from ser_pitch_model import Wav2Vec2AMSoftmax
from sklearn.metrics import balanced_accuracy_score
import torch
import torchaudio

checkpoint_dir = 'no_joint_visec'
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

train_dataset = load_dataset("csv", data_files='train.csv', split="train")
test_dataset = load_dataset("csv", data_files='valid.csv', split="train")

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, padding_side='right', do_normalize=True, return_attention_mask=False)
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["accent"] = batch["emotion_id"]
    return batch

train_dataset = train_dataset.map(speech_file_to_array_fn)
test_dataset = test_dataset.map(speech_file_to_array_fn)

def prepare_dataset(batch):
    # check that all files have the correct sampling rate
    assert (
        len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {feature_extractor.sampling_rate}."

    batch["input_values"] = feature_extractor(batch["speech"],\
        sampling_rate=batch["sampling_rate"][0]).input_values

    batch["label"] = batch["accent"]
    return batch

train_dataset = train_dataset.map(prepare_dataset, batch_size=8, num_proc=4, batched=True)
test_dataset = test_dataset.map(prepare_dataset, batch_size=8, num_proc=4, batched=True)


@dataclass
class DataCollatorCTCWithPadding:

    processor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor([feature["label"] for feature in features])

        return batch

data_collator = DataCollatorCTCWithPadding(processor=feature_extractor, padding=True)

metric = load_metric("f1")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {'weighted_acc': balanced_accuracy_score(labels, predictions)}


model = Wav2Vec2ForSequenceClassification.from_pretrained(
    'nguyenvulebinh/wav2vec2-base-vi', 
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.1,
    final_dropout=0.1,
    mask_time_prob=0.05,
    layerdrop=0.1,
    num_labels=4,
    classifier_proj_size=256
)
model.freeze_feature_extractor()
# model = torch.compile(model)

training_args = TrainingArguments(
  output_dir=checkpoint_dir,
  group_by_length=False,
  per_device_train_batch_size=2,
  gradient_accumulation_steps=1, 
  evaluation_strategy="epoch",
  save_strategy="epoch",
  logging_strategy="epoch",
  num_train_epochs=30,
  dataloader_num_workers=6,
  learning_rate=1.5e-5,
  warmup_steps=0,
  save_total_limit=30,
  eval_accumulation_steps=1,
  report_to='tensorboard'
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()