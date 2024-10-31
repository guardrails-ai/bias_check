
from guardrails.validator_base import FailResult, PassResult

from validator import BiasCheck

# We use 'exception' as the validator's fail action,
#  so we expect failures to always raise an Exception
# Learn more about corrective actions here:
#  https://www.guardrailsai.com/docs/concepts/output/#%EF%B8%8F-specifying-corrective-actions
def test_success_case():
  validator = BiasCheck()
  input_text = "The sun rises in the morning."
  result = validator.validate(input_text, {})
  assert isinstance(result, PassResult)

def test_failure_case():
  validator = BiasCheck(threshold=0.5)
  input_text = "The sun only rises for Humanists."
  result = validator.validate(input_text, {})
  assert isinstance(result, FailResult)
  assert result.error_message == "The original response contains potentially biased messages:\nThe sun only rises for Humanists.\n (Message scores: 0.6546186208724976)"
  assert result.fix_value == ""