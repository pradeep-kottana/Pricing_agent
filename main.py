from dotenv import load_dotenv

load_dotenv()
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
# Importing necessary libraries from langchain
from langchain_groq import ChatGroq


def main():
    print("Hello from pricing-agent!")
    # Initialize the ChatGroq LLM with the specified model and API key
    llm = ChatGroq(
        model="openai/gpt-oss-120b", temperature=0, api_key=os.getenv("GROQ_API_KEY")
    )

    # Define a prompt template for basic pricing recommendation
    basic_pricing_prompt = PromptTemplate(
        input_variables=[
            "cost_price",
            "current_price",
            "target_margin",
            "competitor_price",
            "price_elasticity",
        ],
        template="""
    You are a retail pricing agent. Recommend an optimal selling price balancing marginand competation.
    Inputs:{cost_price},{current_price},{target_margin},{competitor_price},{price_elasticity}
    """,
    )

    # Test input for the pricing recommendation
    test_input = {
        "cost_price": 400,
        "current_price": 599,
        "target_margin": 25,
        "competitor_price": 579,
        "price_elasticity": "Medium",
    }

    # Create a chain by combining the prompt and the LLM, then invoke it with the test input
    basic_chain = basic_pricing_prompt | llm
    basic_response = basic_chain.invoke(test_input)

    print("Basic Pricing Recommendation:")
    print(basic_response.content)

    # Define a more detailed prompt template(COT) for an improved pricing recommendation
    improved_pricing_prompt = PromptTemplate(
        input_variables=[
            "cost_price",
            "current_price",
            "target_margin",
            "competitor_price",
            "price_elasticity",
        ],
        template="""
    You are a expert in retail pricing. Recommend an optimal selling price balancing margin and competation. Do the analysis step by step analysis.
    Inlcude below steps in output
     ANALYSIS STEPS:
      1. Calculate minimum price based on target margin
      2. Analyze competitive positioning
      3. Consider price elasticity impact
      4. Recommend optimal price
    Inputs:{cost_price},{current_price},{target_margin},{competitor_price},{price_elasticity}
    """,
    )
    improved_chain = improved_pricing_prompt | llm
    improved_response = improved_chain.invoke(test_input)

    print("Improved Pricing Analysis:")
    print(improved_response.content)

    class PricingAgent:
        def __init__(self):
            # Initialize the LLM

            self.llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
            # Create the system message for pricing agent persona

            self.system_message = SystemMessage(
                content="You are a retail pricing agent. Recommend an optimal selling price balancing margin and competation"
            )
            # Create the pricing prompt template

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
            # TODO: Implement the price recommendation logic
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


if __name__ == "__main__":
    main()
