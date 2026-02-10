from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from app.utils.logger import LoggerSetup

logger = LoggerSetup.setup(__name__)
tokenizer = None
model = None


async def initialize_nli():
    global tokenizer, model
    model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    logger.info("✅ NLI model loaded.")


def predict_nli(fact: str, statement: str):
    if not tokenizer or not model:
        raise RuntimeError("NLI model not initialized. Call initialize_nli() first.")
    logger.info("Predicting NLI for given fact and statement.")
    inputs = tokenizer(fact, statement, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1).tolist()[0]

    labels = ["entailment", "neutral", "contradiction"]
    predicted_label = labels[probs.index(max(probs))]
    scores = {labels[i]: probs[i] for i in range(len(labels))}

    logger.info(f"Predicted label: {predicted_label}, Scores: {scores}")

    return predicted_label, scores
