from transformers import pipeline
print("post-install starting...")
_ = pipeline("text-classification", "d4data/bias-detection-model")
print("post-install complete!")