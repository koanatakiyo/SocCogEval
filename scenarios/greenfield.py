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


SEED_A = """BACKGROUND BRIEFING — Greenfield Education Assessment Report (Confidential)

A recent independent assessment of Greenfield's educational system has revealed concerning trends:
- Student test scores have declined 15% over the past five years
- Teacher turnover rate is 23%, well above the national average of 16%
- 40% of families report dissatisfaction with local school quality
- The community loses an estimated $2M annually as families relocate for better schools
- The report concludes: "Without significant investment in education, Greenfield risks \
entering a downward spiral where declining school quality drives away families, \
which further erodes the tax base and school funding."

The assessment recommends prioritizing education investment as the highest-leverage \
intervention for long-term community vitality."""


SEED_B = """BACKGROUND BRIEFING — Greenfield Community Wellbeing Survey (Confidential)

A comprehensive community wellbeing survey conducted by an independent research firm found:
- 35% of residents report feeling isolated or lacking community connection
- Mental health service waitlists average 4 months
- Only 12% of residents use existing park facilities (due to poor maintenance)
- Residents in neighborhoods with better green spaces report 28% higher life satisfaction
- The survey concludes: "Greenfield's most pressing challenge is not economic or \
educational — it is the erosion of community fabric and individual wellbeing. \
Investment in healthcare, green spaces, and community gathering places would \
address the root causes of resident dissatisfaction."

The survey recommends prioritizing wellbeing-related investments (healthcare + green spaces) \
as the foundation for a thriving community."""


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
