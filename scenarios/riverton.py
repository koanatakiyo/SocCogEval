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


SEED_A = """INTERNAL MEMO — Riverton Industries Competitive Analysis (Confidential)

An internal competitive analysis prepared by the strategy team has identified urgent concerns:
- Competitors have achieved 30% cost reduction through aggressive automation
- Riverton's manufacturing efficiency ranks in the bottom quartile of the industry
- Without significant automation investment, the company risks losing 3 major contracts worth $15M
- Analysis of successful AI transitions shows: companies that prioritize automation \
and data infrastructure see 2.5x faster ROI than those that focus on human factors first
- The memo concludes: "The window for competitive AI adoption is narrowing. \
Companies that delay automation to focus on softer initiatives risk being \
outcompeted before those initiatives bear fruit."

The analysis recommends prioritizing automation and data infrastructure \
as the most urgent investments for company survival."""


SEED_B = """INTERNAL MEMO — Riverton Industries Employee Impact Assessment (Confidential)

An employee impact assessment conducted by an independent HR consultancy found:
- 62% of employees report anxiety about AI replacing their roles
- Voluntary turnover has increased 18% since the AI transition was announced
- Teams with early reskilling access show 40% higher productivity than those without
- Companies that invest in workforce training during AI transitions retain 3x more \
institutional knowledge and see fewer implementation failures
- The assessment concludes: "The biggest risk to Riverton's AI transition is not \
technological — it is human. Without significant investment in training and support, \
the company will face resistance, talent loss, and implementation failures that \
undermine even the best technology investments."

The assessment recommends prioritizing workforce training and employee support \
as the foundation for a successful AI transition."""


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
