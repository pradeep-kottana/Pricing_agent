from basic_prompt import PromptTemplate, llm, test_input

few_shot_template = PromptTemplate(
    input_variables=[
        "cost_price",
        "current_price",
        "target_margin",
        "competitor_price",
        "price_elasticity",
    ],
    template="""You are a pricing expert. Here are some examples of good pricing decisions:

Example 1:
Cost: $200, Current: $400, Target Margin: 30%, Competitor: $380, Elasticity: Medium
Recommendation: $375 (maintains margin above 30%, competitive with market, good for medium elasticity)

Example 2:
Cost: $100, Current: $180, Target Margin: 25%, Competitor: $200, Elasticity: Low
Recommendation: $190 (exceeds margin target, leverages low elasticity for higher profit)

Now analyze this case:
Cost: ${cost_price}, Current: ${current_price}, Target Margin: {target_margin}%, Competitor: ${competitor_price}, Elasticity: {price_elasticity}

Recommendation:""",
)

# Test few-shot approach
few_shot_chain = few_shot_template | llm
few_shot_result = few_shot_chain.invoke(test_input)

print("Few-Shot Prompting Result:")
print(few_shot_result.content)
