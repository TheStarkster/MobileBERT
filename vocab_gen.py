"""
Export google/mobilebert-uncased MLM head to TFLite with full-sequence output.

Requirements:
    pip install transformers tensorflow

The key fix vs the naive from_keras_model() approach:
  - Use @tf.function with a concrete input_signature so TFLite traces the graph
    with fixed shapes [1, SEQ_LEN] rather than shape=[1, 1] (the Keras default),
    which previously produced output shape [1, 1, 30522] instead of [1, 128, 30522]
    and caused the model to only ever read position 0 (CLS), always predicting ".".

Output files:
  mobilebert_mlm.tflite  — copy to app/src/main/assets/ime/nlp/
  vocab.txt              — copy to app/src/main/assets/ime/nlp/
"""
from transformers import TFMobileBertForMaskedLM, AutoTokenizer
import tensorflow as tf

SEQ_LEN = 128
BATCH = 1

print("Loading model and tokenizer from google/mobilebert-uncased ...")
tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
model = TFMobileBertForMaskedLM.from_pretrained("google/mobilebert-uncased")

# Save vocab.txt
tokenizer.save_vocabulary(".")
print("vocab.txt saved.")

# Define a concrete function pinned to [BATCH, SEQ_LEN] so TFLite infers the
# full output shape [BATCH, SEQ_LEN, vocab_size] = [1, 128, 30522].
@tf.function(input_signature=[
    tf.TensorSpec(shape=[BATCH, SEQ_LEN], dtype=tf.int32, name="input_ids"),
    tf.TensorSpec(shape=[BATCH, SEQ_LEN], dtype=tf.int32, name="attention_mask"),
    tf.TensorSpec(shape=[BATCH, SEQ_LEN], dtype=tf.int32, name="token_type_ids"),
])
def serving_fn(input_ids, attention_mask, token_type_ids):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        training=False,
    )
    return outputs.logits  # [1, 128, 30522]

print("Tracing model with concrete shapes ...")
concrete_fn = serving_fn.get_concrete_function()

print("Converting to TFLite (dynamic-range quantization) ...")
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn], model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("mobilebert_mlm.tflite", "wb") as f:
    f.write(tflite_model)

size_mb = len(tflite_model) / 1e6
print(f"mobilebert_mlm.tflite saved: {size_mb:.1f} MB")
print()
print("Next steps:")
print("  cp mobilebert_mlm.tflite app/src/main/assets/ime/nlp/")
print("  cp vocab.txt             app/src/main/assets/ime/nlp/")
print()
print("Expected TFLite tensor specs:")
print("  Input  input_ids        [1, 128]  int32")
print("  Input  attention_mask   [1, 128]  int32")
print("  Input  token_type_ids   [1, 128]  int32")
print("  Output logits           [1, 128, 30522]  float32")
