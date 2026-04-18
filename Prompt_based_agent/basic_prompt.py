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
