# to run these, run 
# make tests

from guardrails import Guard
import pytest
from validator import BiasCheck
from guardrails.validator_base import FailResult, PassResult

# We use 'exception' as the validator's fail action,
#  so we expect failures to always raise an Exception
# Learn more about corrective actions here:
#  https://www.guardrailsai.com/docs/concepts/output/#%EF%B8%8F-specifying-corrective-actions
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