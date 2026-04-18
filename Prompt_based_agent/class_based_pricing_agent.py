from dotenv import load_dotenv

load_dotenv()
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
# Importing necessary libraries from langchain
from langchain_groq import ChatGroq


class PricingAgent:
    def __init__(self):

        # self.llm
        self.llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

        # self.system_message
        self.system_message = SystemMessage(
            content="You are a retail pricing agent. Recommend an optimal selling price balancing margin and competation"
        )

        # self.pricing_template
        self.pricing_template = PromptTemplate(
            input_variables=[
                "cost_price",
                "current_price",
                "target_margin",
                "competitor_price",
                "price_elasticity",
            ],
            template="""
        You are a retail pricing agent. Recommend an optimal selling price balancing margin and competation by step by step analysis.
        Use below formula for recomended price:
        1. target_price = cost_price*(1+(target_margin/100))


        Inputs:{cost_price},{current_price},{target_margin},{competitor_price},{price_elasticity}
        """,
        )

    def get_price_recommendation(
        self,
        cost_price,
        current_price,
        target_margin,
        competitor_price,
        price_elasticity,
    ):
        """Get price recommendation with detailed analysis"""

        messages = [
            self.system_message,
            HumanMessage(
                content=self.pricing_template.format(
                    cost_price=cost_price,
                    current_price=current_price,
                    target_margin=target_margin,
                    competitor_price=competitor_price,
                    price_elasticity=price_elasticity,
                )
            ),
        ]

        response = self.llm.invoke(messages)
        return response.content

    def quick_price_check(self, cost_price, target_margin, competitor_price):
        """Quick price check with minimal inputs"""
        quick_prompt = f"""Quick pricing check:
        Cost: ${cost_price}, Target Margin: {target_margin}%, Competitor: ${competitor_price}

        Provide a quick price recommendation with brief reasoning."""

        response = self.llm.invoke([HumanMessage(content=quick_prompt)])
        return response.content


print("Initializing Pricing Agent...")
pricing_agent = PricingAgent()
print("Pricing Agent Ready!")

print("Testing Assignment Example:")
print("=" * 50)

result = pricing_agent.get_price_recommendation(
    cost_price=400,
    current_price=599,
    target_margin=25,
    competitor_price=579,
    price_elasticity="Medium",
)

print(result)

print("\nTest Case 1: High Elasticity Scenario")
print("=" * 50)
result1 = pricing_agent.get_price_recommendation(
    cost_price=100,
    current_price=200,
    target_margin=30,
    competitor_price=180,
    price_elasticity="High",
)
print(result1)

# Test Case 2: Low elasticity scenario
print("\nTest Case 2: Low Elasticity Scenario")
print("=" * 50)
result2 = pricing_agent.get_price_recommendation(
    cost_price=50,
    current_price=100,
    target_margin=40,
    competitor_price=120,
    price_elasticity="Low",
)
print(result2)

print("Quick Price Check Test:")
print("=" * 30)

quick_result = pricing_agent.quick_price_check(
    cost_price=250, target_margin=20, competitor_price=350
)
print(quick_result)
