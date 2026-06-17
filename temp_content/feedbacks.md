# purpose of the file
This file contains the feedback from the user. 
The feedback is intended as interactive feedback for all proposals or explicit requests from the codex.
This file must be only edited by the user, and is read-only to the codex.
For documentation purpose or cache purpose, the file may contain multiple feedbacks for different sections. 
The user takes the responsiibility to ensure that the feedback for different sections are properly labeled.
If you cannot locate the exact feedback for the current prompt, explicitly raise the request to clarify.

# feedback

### milestone 0 phase B

rem choice 1: keep your recommendation.
rem choice 2: keep your recommendation.
rem choice 3: keep your recommendation, though I didn't see why this is important. As I have made my decision you don't need to further argue if this item is clearly resolved.
rem choice 4: you are correct that large scale experiment does not fit my scope. keep simple verification experiments as planned. if the paper has documented very quick tests that can be implemented optionally. otherwise, do not implement paper replication at this stage.
rem choice 5: report both result summary with useful findings and saved raw results. you should put them inside the same place for the milestone. defer plots as you recommend.

### milestone 0 phase D

clarifications.
using diffsim smooth-clamped semantics is good for the first hand experiments. Later on it can be changed if any results are against it.

### milestone 0 phase F

questions.
in the results.json, it seems that only braking recovery scenario is reported. have you carried out all candidate scenarios for replication correctness?
briefly list what you have done for the testing and explain why they pass. you should dump this to ./temp_content/temp_impl/milestone_0/phase_e.md.