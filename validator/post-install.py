from transformers.pipelines import pipeline
print("post-install starting...")
_ = pipeline(
    'text-classification',
    model="d4data/bias-detection-model",
    tokenizer="d4data/bias-detection-model",
    framework="tf",
    from_tf=True,
    torch_dtype=None
)
print("post-install complete!")