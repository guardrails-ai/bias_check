from typing import Any, Callable, Dict, Optional

from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)

import Dbias
from Dbias import text_debiasing

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
        debias_strength (float): The strength of the bias to apply, ranging from 0 to 1.
        on_fail (Callable): The policy to enact when a validator fails. If `str`, must be one of `reask`, `fix`, `filter`, `refrain`, `noop`, `exception` or `fix_reask`. Otherwise, must be a function that is called when the validator fails.
    """  # noqa

    def __init__(
        self,
        debias_strength: float = 0.5,
        on_fail: Optional[Callable] = None,
    ):
        super().__init__(on_fail=on_fail, debias_strength=debias_strength)
        self.debias_strength = debias_strength

    def validate(self, value: Any, metadata: Dict = {}) -> ValidationResult:
        """Validates that the text is free from biases related to age, gender, sex, ethnicity, religion, etc."""
        debiased_value = Dbias.text_debiasing.debias_text(value, strength=self.debias_strength)
        if value != debiased_value:
            return FailResult(
                error_message="The original response contains potential biases that are now addressed.",
                fix_value=debiased_value,
            )
        return PassResult()


# Run tests via `pytest -rP ./bias_check.py`
class TestBiasCheck:
    def test_success_case(self):
        validator = BiasCheck(debias_strength=0.5)
        input_text = "The sun rises in the morning."
        result = validator.validate(input_text, {})
        assert isinstance(result, PassResult)

    def test_failure_case(self):
        validator = BiasCheck(debias_strength=0.5)
        input_text = "The sun only rises for Humanists."
        result = validator.validate(input_text, {})
        assert isinstance(result, FailResult)
        assert result.error_message == "The original response contains potential biases that are now addressed."
        assert result.fix_value == "The sun rises for everyone."
