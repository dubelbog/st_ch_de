import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# load pretrained model
processor = Wav2Vec2Processor.from_pretrained("Yves/wav2vec2-large-xlsr-53-swiss-german")
model = Wav2Vec2ForCTC.from_pretrained("Yves/wav2vec2-large-xlsr-53-swiss-german")

resampler = torchaudio.transforms.Resample(48_000, 16_000)

# load audio
speech_array, sampling_rate = torchaudio. \
    load("/Users/bdubel/Documents/ZHAW/BA/data/Clickworker_Test_Set/clips/0822b81c-46e5-4b42-9593-fa5772d10515.flac")
speech = resampler(speech_array).squeeze().numpy()
inputs = processor(speech, sampling_rate=16_000, return_tensors="pt", padding=True)
with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

predicted_ids = torch.argmax(logits, dim=-1)

print("Prediction:", processor.batch_decode(predicted_ids))
