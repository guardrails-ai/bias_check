print("post-install starting...")
from transformers import pipeline
_ = pipeline("text-classification", "d4data/bias-detection-model")
print("post-install complete!")