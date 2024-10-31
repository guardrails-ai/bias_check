from typing import Any, Callable, Dict, Optional

from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)

from transformers import pipeline

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
        on_fail (Callable): The policy to enact when a validator fails. If `str`, must be one of `filter`, `noop`, or `exception`. Otherwise, must be a function that is called when the validator fails.
    """  # noqa

    def __init__(
        self,
        threshold: float = 0.9,
        on_fail: Optional[Callable] = None,
    ):
        super().__init__(on_fail=on_fail)
        valid_on_fail_operations = {"filter", "noop", "exception"}
        if isinstance(on_fail, str) and on_fail not in valid_on_fail_operations:
            raise Exception(
                f"on_fail value ({on_fail}) not in list of allowable operations: {valid_on_fail_operations}"
            )
        self.threshold = threshold
        self.model = pipeline(
            'text-classification',
            model="d4data/bias-detection-model",
        )
        # There are some spurious loading complaints with TFDistilBert models.
        # See https://discuss.huggingface.co/t/message-some-layers-from-the-model-were-not-used/1972/7

    def validate(self, value: Any, metadata: Dict = None) -> ValidationResult:
        """Validates that the text is free from biases related to age, gender, sex, ethnicity, religion, etc."""
        if isinstance(value, str):
            value = [value,]  # Ensure we're always passing lists of strings into the classifier.

        classified_examples = self.model(value)
        passing_outputs = list()
        passing_scores = list()
        failing_outputs = list()
        failing_scores = list()
        for text, prediction in zip(value, classified_examples):
            if prediction['label'] == 'Biased':
                score = prediction['score']
            elif prediction['label'] == 'Non-biased':
                score = -prediction['score']  # Note the negation!
            else:
                raise Exception(f"Got unexpected prediction label: {prediction['label']}")
            if score > self.threshold:
                failing_outputs.append(text)
                failing_scores.append(score)
            else:
                passing_outputs.append(text)
                passing_scores.append(score)

        if failing_outputs:
            failure_message = "The original response contains potentially biased messages:\n"
            failure_message += "\n - ".join(failing_outputs)
            message_scores = [str(s) for s in failing_scores]
            failure_message += "\n (Message scores: {})".format(", ".join(message_scores))
            # Do we need to call the on_fail_method here?
            return FailResult(
                error_message=failure_message,
                fix_value=" ".join(passing_outputs),
            )
        return PassResult()
