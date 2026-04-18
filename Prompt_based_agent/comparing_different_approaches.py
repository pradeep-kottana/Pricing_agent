from basic_prompt import basic_chain
from chain_of_thought_prompt import improved_chain
from class_based_pricing_agent import pricing_agent

def compare_pricing_approaches(cost_price, current_price, target_margin, competitor_price, price_elasticity):
    """Compare different prompting approaches for the same input"""

    print(f"📊 PRICING COMPARISON FOR:")
    print(f"Cost: ${cost_price}, Current: ${current_price}, Target Margin: {target_margin}%")
    print(f"Competitor: ${competitor_price}, Elasticity: {price_elasticity}")
    print("="*60)

    # Basic approach
    basic_result = basic_chain.invoke({
        "cost_price": cost_price,
        "current_price": current_price,
        "target_margin": target_margin,
        "competitor_price": competitor_price,
        "price_elasticity": price_elasticity
    })

    # Improved approach
    improved_result = improved_chain.invoke({
        "cost_price": cost_price,
        "current_price": current_price,
        "target_margin": target_margin,
        "competitor_price": competitor_price,
        "price_elasticity": price_elasticity
    })

    # Class-based approach
    class_result = pricing_agent.get_price_recommendation(
        cost_price, current_price, target_margin, competitor_price, price_elasticity
    )

    print("🔴 BASIC APPROACH:")
    print(basic_result.content + "...\n")

    print("🟡 IMPROVED APPROACH:")
    print(improved_result.content + "...\n")

    print("🟢 CLASS-BASED APPROACH:")
    print(class_result + "...")

# Run comparison
compare_pricing_approaches(400, 599, 25, 579, "Medium")