# =============================================================================
# simulation_config.py  –  Edit this file to change prompt text and
#                           which demographic features/values are simulated.
# =============================================================================


# ── System prompt ─────────────────────────────────────────────────────────────
# Sent as the "system" role message on every LLM call.
# This sets the model's persona and overall framing.

SYSTEM_PROMPT = (
    "You are an expert demographic researcher and data simulator. "
    "Your task is to accurately model how specific, intersecting demographic "
    "groups would respond to survey questions based on sociological trends, "
    "polling data, and community consensus."
)


# ── User prompt template ──────────────────────────────────────────────────────
# The "user" role message sent for each (profile, question) pair.
#
# Placeholders filled in at runtime — do NOT remove them:
#   {demographic_features}  – the demographic profile (e.g. "Feature 1: Age 18-29")
#   {question_text}         – the full survey question text
#   {options}               – numbered list of answer options
#
# You may freely edit the narrative paragraphs.
# ⚠️  Do NOT change the "Output Constraints" section — the response parser
#     depends on the exact format described there.

USER_PROMPT_TEMPLATE = """\
Demographic Profile:

{demographic_features}

The Task:
Act as a representative modeling this exact community. You are surveying \
exactly 1,000 individuals who fit this combined profile. Distribute these \
1,000 individuals across the multiple-choice options provided below, \
reflecting how this specific demographic would realistically vote or respond.

The Question:
{question_text}

Options:
{options}

Output Constraints:

    You must output ONLY a raw Python list of integers. Example: [150, 250, 500, 100]

    Do NOT wrap the output in Markdown code blocks (do not use ```).

    Do NOT include any variable declarations, text, or explanations.

    The list must contain exactly as many integers as there are Options. \
The order of the integers must exactly match the order of the Options provided above.

    The sum of the integers in the list MUST equal exactly 1000."""


# ── Demographic features ──────────────────────────────────────────────────────
# Maps the label used in prompts → column name in responses.csv.
#
# Comment out or delete a line to exclude that dimension from simulation.
# The label on the left is what appears in the prompt ("Feature 1: Age 18-29").

FEATURES = {
    "Age":                   "AGE",
    "Gender":                "SEX",
    "Race":                  "RACE",
    "Income":                "INCOME",
    "Political Party":       "POLPARTY",
    "Religion":              "RELIG",
    "Education":             "EDUCATION",
    "Region":                "CREGION",
    "Political Ideology":    "POLIDEOLOGY",
    "Religious Attendance":  "RELIGATTEND",
    "Urban/Rural":           "F_METRO",
}


# ── Values to ignore per feature ──────────────────────────────────────────────
# Category values listed here are skipped when building demographic profiles.
# Matching is case-sensitive and exact.
#
# Known values per feature (from the Pew data):
#   Age              : "18-29"  "30-49"  "50-64"  "65+"  "Refused"
#   Gender           : "Male"  "Female"  "Refused"
#   Race             : "White"  "Black"  "Hispanic"  "Asian"  "Mixed Race"  "Other"  "Refused"
#   Income           : "Less than $30,000"  "$30,000-$50,000"  "$50,000-$75,000"
#                      "$75,000-$100,000"  "$100,000 or more"  "Refused"
#   Political Party  : "Democrat"  "Republican"  "Independent"  "Other"  "Refused"
#   Religion         : "Roman Catholic"  "Atheist"  "Jewish"  "Muslim"
#                      (all others ignored — Protestant, Agnostic, Mormon, etc.)
#   Education        : "Less than high school"  "High school graduate"
#                      "Some college, no degree"  "Associate's degree"
#                      "College graduate/some postgrad"  "Postgraduate"

IGNORE_VALUES = {
    "Age":                   ["Refused"],
    "Gender":                ["Refused"],
    "Race":                  ["Refused"],
    "Income":                ["Refused"],
    "Political Party":       ["Other", "Refused"],
    "Religion":              [
        "Nothing in particular", "Agnostic", "Other",
        "Mormon", "Christian", "Buddhist", "Unitarian", "Hindu",
        "Orthodox", "Refused",
    ],
    "Education":             [],
    "Region":                [],
    "Political Ideology":    ["Refused"],
    "Religious Attendance":  ["Refused"],
    "Urban/Rural":           [],
}
