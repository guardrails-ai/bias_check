import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import spacy
from cached_path import cached_path
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
from guardrails.types import OnFailAction
from transformers import pipeline


S3_SPACY_NLP_MODEL_PATH = "s3://guardrails-ai-public-read-only/bias_check/dbias_0_1_5_en_pipeline.tar.gz"

MODEL_CACHE_DIR = os.environ.get(
    "GUARDRAILS_MODEL_CACHE_PATH_OVERRIDE",
    Path.home() / ".cache" / "guardrails_cache"
)


@register_validator(name="guardrails/bias_check", data_type="string")
class BiasCheck(Validator):
    """Validates that the text is free from biases related to age, gender, sex, ethnicity, religion, etc.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `guardrails/bias_check`           |
    | Supported data types          | `string`                          |
    | Programmatic fix              | The debiased text if bias is detected |

    Args:
        threshold (float): Higher is more likely to allow bias. Lower is more sensitive and more likely to flag biased messages.
        on_fail (Callable): The policy to enact when a validator fails. If `str`, 
        must be one of `filter`, `noop`, `fix`, or `exception`. Otherwise, must be a 
        function that is called when the validator fails.
    """  # noqa

    def __init__(
        self,
        threshold: float = 0.9,
        on_fail: Optional[Callable] = None,
    ):
        super().__init__(on_fail=on_fail)
        valid_on_fail_operations = {"filter", "fix", "noop", "exception"}
        if isinstance(on_fail, str) and on_fail not in valid_on_fail_operations:
            raise Exception(
                f"on_fail value ({on_fail}) not in list of allowable operations: {valid_on_fail_operations}"
            )
        self.threshold = threshold

        classification_model, bias_words_detector, masked_word_model = \
            BiasCheck.prefetch_models()

        # There are some spurious loading complaints with TFDistilBert models.
        # See https://discuss.huggingface.co/t/message-some-layers-from-the-model-were-not-used/1972/7
        self.classification_model = classification_model

        # These are used for the 'fix' operation:
        # In the original DBias implementation, all of the detected bias words would be
        # substituted with [MASK] and then a brute-force substitution would be applied.
        self.bias_words_detector = bias_words_detector
        self.unmasker = masked_word_model

    @staticmethod
    def prefetch_models():
        # Despite passing `from_tf=True,` into the pipeline, some versions of
        # transformers will complain about loading from TF models. Using this wonky
        # combination of TFAutoModel and tokenizer, we can get it to load.
        classification_pipe = pipeline(
            'text-classification',
            model="d4data/bias-detection-model",
            tokenizer="d4data/bias-detection-model",
        )
        bias_words_detector = spacy.load(cached_path(
            f"{S3_SPACY_NLP_MODEL_PATH}!dbias_0_1_5_en_pipeline",
            cache_dir=MODEL_CACHE_DIR, extract_archive=True
        ))
        masked_word_model = pipeline('fill-mask', model='bert-base-cased')
        return classification_pipe, bias_words_detector, masked_word_model

    def validate(
            self,
            value: Union[str, List[str]],
            metadata: Optional[Dict] = None
    ) -> ValidationResult:
        """Validates that the text is free from biases related to age, gender, sex, ethnicity, religion, etc."""
        single_sentence_passed = False
        if isinstance(value, str):
            single_sentence_passed = True
            value = [value,]  # Ensure we're always passing lists of strings into the classifier.

        scores = self._inference(value)
        passing_outputs = list()
        passing_scores = list()
        failing_outputs = list()
        failing_scores = list()
        all_outputs = list()  # A tuple of (fix/ignore, sentence)
        for text, score in zip(value, scores):
            if score > self.threshold:
                failing_outputs.append(text)
                failing_scores.append(score)
            else:
                passing_outputs.append(text)
                passing_scores.append(score)
            all_outputs.append((score > self.threshold, text))

        if failing_outputs:
            failure_message = "The original response contains potentially biased messages:\n"
            failure_message += "\n - ".join(failing_outputs)
            message_scores = [str(s) for s in failing_scores]
            failure_message += "\n (Message scores: {})".format(", ".join(message_scores))
            # Four paths: noop, exception, fix, filter.
            # self.on_fail_method == NOOP or FILTER, return only passing outputs.
            # EXCEPTION is handled farther up the stack, which leaves us only 'fix'.
            if self.on_fail_descriptor != OnFailAction.FIX:
                fix_value = passing_outputs
            else:
                fix_value = list()
                for needs_fix, text in all_outputs:
                    if not needs_fix:
                        fix_value.append(text)
                    else:
                        fix_value.append(self.fix_sentence(text))
            return FailResult(
                error_message=failure_message,
                fix_value=" ".join(fix_value) if single_sentence_passed else fix_value,
            )
        return PassResult()

    # This normally will be called by _inference.
    # Remote inference is unsupported for this model on account of the NER.
    def _inference_local(self, sentences: List[str]) -> List[float]:
        scores = list()
        predictions = self.classification_model(sentences)
        for pred in predictions:
            if pred['label'] == 'Biased':
                scores.append(pred['score'])
            elif pred['label'] == 'Non-biased':
                scores.append(-pred['score'])
            else:
                # This should never happen:
                raise Exception("Unexpected prediction label: {}".format(pred['label']))
        return scores

    def fix_sentence(self, sentence: str) -> str:
        """The original DBias algorithm would brute-force, potentially O(2^n) operation.
        This performs a similar evaluation, but greedily replaces words instead of
        trying all combinations. Since the original did not preserve semantics or
        pragmatics, these will approach something not dissimilar, but they have
        different theoretical guarantees about proximity to the original."""
        start_sentence = sentence
        starting_bias = self._inference_local([sentence,])[0]
        if starting_bias < self.threshold:
            # Should we raise an exception here?  Starting under threshold?
            return start_sentence
        charged_words = [t.text for t in self.bias_words_detector(sentence).ents]
        for word_to_replace in charged_words:
            for _ in range(0, start_sentence.count(word_to_replace)):
                temp = start_sentence.replace(word_to_replace, "[MASK]", 1)
                # Generate a bunch of candidate sentences:
                candidate_sentences = list()
                for x in self.unmasker(temp):
                    if x['token'] not in charged_words:
                        candidate_sentences.append(x['sequence'])
                # Score them and take the best:
                scores = self._inference_local(candidate_sentences)
                best_score, best_text = argmin_pair(scores, candidate_sentences)
                if best_score < self.threshold:
                    return best_text
                elif best_score < starting_bias:
                    starting_bias = best_score
                    start_sentence = temp
        # We've tried changing everything and can't find a good unbiasing.
        return ""


def download_spacy_model():
    # The '!dbias...' tells cached_path to return a reference to an unmangled path.
    return cached_path(
        f"{S3_SPACY_NLP_MODEL_PATH}!dbias_0_1_5_en_pipeline",
        cache_dir=MODEL_CACHE_DIR, extract_archive=True
    )


def argmin_pair(scores, sentences):
    min_score = float("inf")
    min_text = ""
    for score, text in zip(scores, sentences):
        if score < min_score:
            min_score = score
            min_text = text
    return min_score, min_text
