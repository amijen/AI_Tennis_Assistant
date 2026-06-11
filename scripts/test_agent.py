"""
Manual testing script for the Tennis Rules Agent.

Run with:
    python -m scripts.test_agent

Set VERBOSE=True to see the agent's reasoning step-by-step.
"""
from app.agents.agent import build_agent, ask_agent


# ============================================================
# CONFIG
# ============================================================
VERBOSE = False  # Set False for cleaner output


# ============================================================
# TEST QUESTIONS
# ============================================================
TEST_QUESTIONS = [
    "How many sets are played in a men's Grand Slam match?",
    "What is the tie-break rule?",
    
    "Compare tie-break rules between ITF and Grand Slam",
    "What's the difference between ITF and Grand Slam coaching rules?",
    "How do medical timeouts differ between ITF and Grand Slam?",
    
    "Who won Wimbledon in 2023?",
    
    "What does the rulebook say about the warm-up?",
]


# ============================================================
# DISPLAY HELPERS
# ============================================================

def pretty_print_result(question: str, result: dict):
    """Print a question's answer in a readable format."""
    print(f"\n{'='*70}")
    print(f"❓ QUESTION: {question}")
    print(f"{'='*70}")
    
    # Show tools that were called
    if result.get("steps"):
        print("\n🔧 TOOLS USED:")
        for i, step in enumerate(result["steps"], start=1):
            tool_name = step.get("tool", "unknown")
            tool_input = step.get("input", {})
            print(f"   {i}. {tool_name}({tool_input})")
    
    # Show the final answer
    print(f"\n💡 ANSWER:")
    print(result["answer"])
    print()


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "="*70)
    print("🎾 TENNIS RULES AGENT — TEST")
    print("="*70)
    
    print("\n🔨 Building agent... (this takes ~5-10s)")
    agent = build_agent(verbose=VERBOSE)
    print("✅ Agent ready!\n")
    
    # Run all test questions
    for question in TEST_QUESTIONS:
        try:
            result = ask_agent(question)
            pretty_print_result(question, result)
        except Exception as e:
            print(f"\n❌ ERROR on '{question}':")
            print(f"   {type(e).__name__}: {e}\n")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()