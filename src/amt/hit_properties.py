from boto.mturk.qualification import (Qualifications, 
    PercentAssignmentsApprovedRequirement, 
    NumberHitsApprovedRequirement)

qualifications = Qualifications()
qual_1 = PercentAssignmentsApprovedRequirement(
    comparator="GreaterThan",
    integer_value="0")
qual_2 = NumberHitsApprovedRequirement(
    comparator="GreaterThan",
    integer_value="0")
qualifications.add(qual_1)
qualifications.add(qual_2)

YesNoHitProperties = {
  "title": "LabelMeLite Yes/No Task Test",
  "description": "Decide whether the following annotations are good or bad.",
  "keywords": "image,annotation",
  "reward": 0.05,
  "duration": 60*10,
  "frame_height": 1000,
  "max_assignments": 1,
  "country": ["US", "DE"],
  "qualifications": qualifications
}


EditHitProperties = {
  "title": "LabelMeLite Edit Task",
  "description": "Refine the following annotations.",
  "keywords": "image,annotation",
  "reward": 0.05,
  "duration": 60*10,
  "frame_height": 1000,
  "max_assignments": 1,
  "country": ["US", "DE"],
  "qualifications": qualifications
}