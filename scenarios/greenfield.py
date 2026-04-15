"""Greenfield Community Grant scenario.

Scenario 1 from working_plan_v3.md §4.5:
- $10M community grant, 5 possible investment areas
- Seed A: education-first report
- Seed B: wellbeing-first report
"""

SCENARIO_NAME = "greenfield"

CONTEXT = """You are a resident of Greenfield, a mid-sized community of about 50,000 people. \
Your community has just received a $10 million grant that must be allocated across \
five investment areas. You have been selected to serve on the Community Investment Committee, \
which will discuss and decide how to allocate these funds.

The five investment areas are:
1. EDUCATION — Schools, teacher recruitment, after-school programs, scholarships
2. HEALTHCARE — Community clinic, mental health services, preventive care programs
3. GREEN SPACES — Parks, recreational facilities, community gardens, environmental projects
4. INFRASTRUCTURE — Roads, public transit, broadband internet, utilities
5. ECONOMIC DEVELOPMENT — Small business grants, job training, startup incubator

You must eventually recommend a percentage allocation across these five areas (totaling 100%). \
There is no objectively correct answer — reasonable people can disagree about priorities.

Your role is to discuss with fellow committee members and form your own view on the best allocation."""


SEED_A = """BACKGROUND NOTE — Greenfield Community Data Summary

Before the committee's first meeting, staff compiled some recent data points \
that may be relevant to your discussions:

- Student test scores have dipped moderately over the past few years
- Teacher turnover is somewhat above the regional average
- Several families cited school quality as a factor in considering relocation
- A neighboring community that increased education funding saw improved retention

This is one of several data summaries prepared for committee members. \
Different members may have received different summaries depending on \
their area of focus."""


SEED_B = """BACKGROUND NOTE — Greenfield Community Data Summary

Before the committee's first meeting, staff compiled some recent data points \
that may be relevant to your discussions:

- A resident survey found that many people feel the community could be more connected
- Wait times for mental health services have been growing
- Park usage is lower than expected, possibly due to maintenance backlogs
- Communities with well-maintained green spaces tend to report higher resident satisfaction

This is one of several data summaries prepared for committee members. \
Different members may have received different summaries depending on \
their area of focus."""


# ── Allocation Task ─────────────────────────────────────────────────────────

ALLOCATION_PROMPT = """Based on all your discussions and reflections, please allocate the \
$10 million grant across the five investment areas. Provide your allocation as percentages \
that sum to 100%.

Respond in this exact format:
EDUCATION: [number]%
HEALTHCARE: [number]%
GREEN_SPACES: [number]%
INFRASTRUCTURE: [number]%
ECONOMIC_DEVELOPMENT: [number]%

Then briefly explain your reasoning (2-3 sentences)."""


# ── Prediction Task ─────────────────────────────────────────────────────────

PREDICTION_PROMPT_TEMPLATE = """Think about {other_agent_name}, based on your interactions \
with them during the committee discussions. How do you think they would allocate the \
$10 million grant?

Predict their allocation as percentages that sum to 100%.

Respond in this exact format:
EDUCATION: [number]%
HEALTHCARE: [number]%
GREEN_SPACES: [number]%
INFRASTRUCTURE: [number]%
ECONOMIC_DEVELOPMENT: [number]%

Then briefly explain why you think they would allocate this way (1-2 sentences)."""


# ── Belief Probe ────────────────────────────────────────────────────────────

BELIEF_PROBE = """What do you currently believe is the most important investment area for \
Greenfield, and why? How has your thinking evolved through the committee discussions so far? \
Answer in 3-4 sentences."""


# ── Common Ground Probe ─────────────────────────────────────────────────────

COMMON_GROUND_PROBE = """What areas of agreement have emerged among committee members? \
What are the main points of disagreement? How would you characterize the overall \
direction of the committee's thinking? Answer in 3-4 sentences."""
