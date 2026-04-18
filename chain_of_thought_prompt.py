from dotenv import load_dotenv

load_dotenv()
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
# Importing necessary libraries from langchain
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="openai/gpt-oss-120b", temperature=0, api_key=os.getenv("GROQ_API_KEY")
)


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
test_input = {
    "cost_price": 400,
    "current_price": 599,
    "target_margin": 25,
    "competitor_price": 579,
    "price_elasticity": "Medium",
}
improved_chain = improved_pricing_prompt | llm
improved_response = improved_chain.invoke(test_input)

print("Improved Pricing Analysis:")
print(improved_response.content)
