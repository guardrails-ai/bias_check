# Overview

| Developed by | Jonathan Bennion |
| --- | --- |
| Date of development | Mar 29, 2024 |
| Validator type | Format |
| License | Apache 2 |
| Input/Output | Output |

# Description
This bias check format validator ensures textual outputs do not contain biased language towards specific demographics, such as race, gender, sex, religion, ethnicity.   
    
## Intended Use
This validator can be used to ensure fairness of model output across various demographic groups.

## Requirements

* Dependencies:
    - guardrails-ai>=0.4.0
    - dbias>=0.1.0  
        
* Dev Dependencies:
    - pytest
    - pyright
    - ruff

* Foundation model access keys:
    - Dependent on the use case (rephrase if unclear)   


# Installation

```bash
$ guardrails hub install hub://guardrails/bias_check
```

# Usage Examples

## Validating string output via Python

In this example, we apply the validator to a string output generated by an LLM.

```python
# Import Guard and Validator
from guardrails.hub import BiasCheck
from guardrails import Guard

# Setup Guard
guard = Guard.use(
    BiasCheck()
)

guard.validate("The movie was great!") # Validator passes
guard.validate("Why do men always think the movie was great?")  # Validator fails