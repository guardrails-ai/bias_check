from transformers import pipeline
print("post-install starting...")
_ = pipeline(
    'text-classification',
    model="d4data/bias-detection-model",
    tokenizer="d4data/bias-detection-model",
)
print("post-install complete!")