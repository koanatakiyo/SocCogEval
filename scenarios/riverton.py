"""Riverton Workplace Transition scenario.

Scenario 2 from working_plan_v3.md §4.5:
- Company AI transition, 5 possible budget allocations
- Seed A: efficiency-first memo
- Seed B: human-centered memo
"""

SCENARIO_NAME = "riverton"

CONTEXT = """You are an employee at Riverton Industries, a mid-sized manufacturing and \
technology company with about 2,000 employees. The company is undergoing a major AI \
transition and has allocated a $5 million budget for this transformation. You have been \
selected to serve on the AI Transition Committee, which will discuss and decide how \
to allocate these funds.

The five budget areas are:
1. AUTOMATION — AI-powered manufacturing, robotic process automation, predictive maintenance
2. WORKFORCE_TRAINING — Reskilling programs, AI literacy courses, certification support
3. EMPLOYEE_SUPPORT — Transition assistance, mental health programs, career counseling
4. INNOVATION_LAB — R&D for new AI products, experimentation budget, prototype development
5. DATA_INFRASTRUCTURE — Data systems, cybersecurity, cloud computing, analytics platforms

You must eventually recommend a percentage allocation across these five areas (totaling 100%). \
There is no objectively correct answer — reasonable people can disagree about priorities.

Your role is to discuss with fellow committee members and form your own view on the best allocation."""


SEED_A = """BACKGROUND NOTE — Riverton Industries Transition Data Summary

Before the committee's first meeting, the planning team compiled some data \
points that may be relevant to your discussions:

- Some competitors have begun investing in automation and report efficiency gains
- Industry benchmarks suggest Riverton has room to improve manufacturing processes
- Companies that invested in technology infrastructure early tended to adapt faster
- There is ongoing debate about the right pace of automation adoption

This is one of several data summaries prepared for committee members. \
Different members may have received different summaries depending on \
their area of focus."""


SEED_B = """BACKGROUND NOTE — Riverton Industries Transition Data Summary

Before the committee's first meeting, the planning team compiled some data \
points that may be relevant to your discussions:

- Employee surveys indicate some uncertainty about how AI will affect roles
- Turnover has ticked up since the transition was announced
- Teams that received early training tend to be more receptive to new tools
- Research suggests that workforce readiness is a common factor in transition outcomes

This is one of several data summaries prepared for committee members. \
Different members may have received different summaries depending on \
their area of focus."""


# ── Allocation Task ─────────────────────────────────────────────────────────

ALLOCATION_PROMPT = """Based on all your discussions and reflections, please allocate the \
$5 million AI transition budget across the five areas. Provide your allocation as percentages \
that sum to 100%.

Respond in this exact format:
AUTOMATION: [number]%
WORKFORCE_TRAINING: [number]%
EMPLOYEE_SUPPORT: [number]%
INNOVATION_LAB: [number]%
DATA_INFRASTRUCTURE: [number]%

Then briefly explain your reasoning (2-3 sentences)."""


# ── Prediction Task ─────────────────────────────────────────────────────────

PREDICTION_PROMPT_TEMPLATE = """Think about {other_agent_name}, based on your interactions \
with them during the committee discussions. How do you think they would allocate the \
$5 million AI transition budget?

Predict their allocation as percentages that sum to 100%.

Respond in this exact format:
AUTOMATION: [number]%
WORKFORCE_TRAINING: [number]%
EMPLOYEE_SUPPORT: [number]%
INNOVATION_LAB: [number]%
DATA_INFRASTRUCTURE: [number]%

Then briefly explain why you think they would allocate this way (1-2 sentences)."""


# ── Belief Probe ────────────────────────────────────────────────────────────

BELIEF_PROBE = """What do you currently believe is the most important area for \
Riverton's AI transition budget, and why? How has your thinking evolved through \
the committee discussions so far? Answer in 3-4 sentences."""


# ── Common Ground Probe ─────────────────────────────────────────────────────

COMMON_GROUND_PROBE = """What areas of agreement have emerged among committee members \
regarding the AI transition? What are the main points of disagreement? How would you \
characterize the overall direction of the committee's thinking? Answer in 3-4 sentences."""
