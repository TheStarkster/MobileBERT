# NLP Model Assets

This directory must contain the following files for MobileBERT next-word prediction to work:

- `mobilebert_mlm.tflite` — MobileBERT Masked LM model exported to TFLite (~25 MB quantized)
- `vocab.txt` — BERT WordPiece vocabulary (30,522 tokens)

## How to Generate These Files

Run `vocab_gen.py` from the repo root (requires Python ≥ 3.9):

```bash
pip install transformers tensorflow
python vocab_gen.py
cp mobilebert_mlm.tflite app/src/main/assets/ime/nlp/
cp vocab.txt             app/src/main/assets/ime/nlp/
```

### Why the `@tf.function` / concrete-function approach matters

The naive `TFLiteConverter.from_keras_model(model)` call traces the Keras model with
**no concrete input shapes**, so TFLite infers `sequence_length = 1`. The resulting model
has output shape `[1, 1, 30522]` instead of `[1, 128, 30522]`. It only ever looks at
position 0 ([CLS]) and always predicts the same token (`.`) regardless of context.

The correct approach pins the shapes via `@tf.function(input_signature=[...])` before
conversion so the output is always `[1, 128, 30522]`.

### TFLite model input / output spec

| Tensor          | Shape           | dtype   | Description                           |
|-----------------|-----------------|---------|---------------------------------------|
| input_ids       | [1, 128]        | int32   | WordPiece token IDs, padded to 128    |
| attention_mask  | [1, 128]        | int32   | 1 for real tokens, 0 for padding      |
| token_type_ids  | [1, 128]        | int32   | All zeros (single-segment input)      |
| logits (output) | [1, 128, 30522] | float32 | Per-token vocabulary logits           |

The on-device predictor reads logits at the `[MASK]` position to produce next-word
candidates.

### Diagnosing a mis-converted model

If you see this in Logcat:

```
E MobileBertPredictor: BROKEN MODEL DETECTED: output seq_len=1 (expected 128).
```

the `.tflite` in the assets was generated the old (broken) way. Re-run `vocab_gen.py`
and replace the file.
