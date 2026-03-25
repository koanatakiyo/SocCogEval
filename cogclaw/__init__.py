"""CogClaw: Computational Social Cognition Experiment Pipeline.

Modules:
    config      - Configuration, API params, model strategy
    llm         - Multi-provider LLM client (Qwen/Llama/GPT/DeepSeek/Gemini/Grok)
    agent       - Agent class with HypergraphMemory
    memory      - HypergraphMemory (graph + vector + RRF retrieval) + extraction
    reasoning   - Fork-based ReasoningGraph for S2 deliberate reasoning
    experience  - Advantage-weighted ExperienceLibrary for agent learning
    probes      - SocCogBench measurement battery (BFI-10, conformity, allocation, prediction)
    network     - Network topology (all-to-all, segregated, none)
    orchestrator- Conversation orchestrator with integrated pre/post measurement
    conditions  - 5-condition factorial design factory
"""
