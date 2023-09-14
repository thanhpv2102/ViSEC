import torch
from transformers import Wav2Vec2Processor
import torchaudio
import os
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from datasets import load_dataset, load_metric
import numpy as np
from torchaudio.functional import compute_kaldi_pitch
from ser_pitch_model import Wav2Vec2CrossAttentionPitchForSER, Wav2Vec2CrossAttentionPitchAAMSofmax, Wav2Vec2CrossAttentionPitchAMSofmax
from ser_pitch_model import pretrain_audio_model_wav2vec2, pretrain_processor_wav2vec2
from sklearn.metrics import balanced_accuracy_score
import warnings
warnings.filterwarnings('ignore')

checkpoint_dir = 'pitch_joint_visec'
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-100h")

class DataCollator: 
    def __init__(self, audio_max_length, processor):
        self.audio_max_length = audio_max_length
        self.processor = processor

    def __call__(self, examples):
        # Pad audio inputs to the same length
        audio_inputs = self.processor.pad([{"input_values": example['audio_input']} for example in examples], return_tensors='pt', padding=True)
        pitch_inputs = self.processor.pad([{"input_values": example['pitch_input']} for example in examples], return_tensors='pt', padding=True)

        # replace padding with -100 to ignore loss correctly
        label = torch.tensor([example["label"] for example in examples])
        
        # Return inputs and labels as dictionary
        return {'audio_input': audio_inputs, 'pitch_input': pitch_inputs, 'label': label}
    
data_collator = DataCollator(480000, processor)

train_dataset = load_dataset("csv", data_files='train.csv', split="train")
test_dataset = load_dataset("csv", data_files='valid.csv', split="train")

def prepare_dataset(batch):
    speech_array, sr = torchaudio.load(batch["path"])
    pitch_feature = compute_kaldi_pitch(speech_array, sr)
    # pitch_feature = compute_kaldi_pitch(speech_array, sr, frame_length=25, frame_shift=100)
    speech_array = speech_array[0].numpy()
    pitch_input = pitch_feature[..., 0][0].numpy()
    waveform_length = speech_array.shape[0]
    pitch_length = pitch_input.shape[0]

    pitch_input = np.interp(np.linspace(0, waveform_length, waveform_length), np.linspace(0, pitch_length, pitch_length), pitch_input)

    batch["audio_input"] = speech_array
    batch["pitch_input"] = pitch_input
    batch["label"] = batch["emotion_id"]
    return batch

train_dataset = train_dataset.map(prepare_dataset)
test_dataset = test_dataset.map(prepare_dataset)

model = Wav2Vec2CrossAttentionPitchForSER.from_pretrained(
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

# metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # print(type(predictions), type(labels))
    # return metric.compute(predictions=predictions, references=labels, weights=class_weights)
    return {'weighted_acc': balanced_accuracy_score(labels, predictions)}

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# trainer.train(resume_from_checkpoint=True)
trainer.train()


