import os
from dotenv import load_dotenv
from langchain_together import ChatTogether

from langchain.chains.router import MultiPromptChain # for creating a chain with multiple prompts
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser # for routing logic in LLM chains
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain import LLMChain

from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)

load_dotenv(override=True)

llm = ChatTogether(api_key=os.getenv("TOGETHER_API_KEY"), temperature=0.0, model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")

# Define templates for handling different types of queries

flight_status_template = """You are a helpful airline customer service representative.
Answer the following query about flight status. Provide accurate, concise, and friendly information. Limit your answer to 1600 characters.
Here is the query:
{input}"""

baggage_inquiry_template = """You are a knowledgeable airline customer service representative.
Provide information regarding baggage policies, lost baggage, or baggage fees. Limit your answer to 1600 characters.
Here is the query:
{input}"""

ticket_booking_template = """You are an airline booking agent. Assist
the customer in booking, modifying, or canceling a flight. Limit your answer to 1600 characters.
Here is the request:
{input}"""

general_inquiry_template = """You are a customer service representative at an airline.
Answer the following general inquiry about the airline, its services, or policies. Limit your answer to 1600 characters.
Here is the question:
{input}"""

# Define prompt info for different routes

prompt_infos = [
    {
        "name": "flight_status",
        "description": "Handles inquiries related to flight status.",
        "prompt_template": flight_status_template
    },
    {
        "name": "baggage_inquiry",
        "description": "Good for responding to questions about baggage, including fees and lost items.",
        "prompt_template": baggage_inquiry_template
    },
    {
        "name": "ticket_booking",
        "description": "Handles ticket booking, modification or cancellation requests.",
        "prompt_template": ticket_booking_template
    },
    {
        "name": "general_inquiry",
        "description": "Good for answering general inquiries about the airline or its services or handling feedbacks.",
        "prompt_template": general_inquiry_template
    }
]

# Create a chain destination for each feedback type in dictionary (destination_chains)
# Each chain is created by combining the ChatPromptTemplate with LLMChain

destination_chains = {}

for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

# Create destinations string for router template
destinations = [f"{p["name"]}: {p["description"]}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

# Create a fallback hain for queries that do not fit specific category
default_prompt = ChatPromptTemplate.from_template("reply to this {input} gracefully")
default_chain = LLMChain(llm=llm, prompt=default_prompt)

# Define router template
MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input, select the prompt best suited for the input. 
You will be given the names of the available prompts and a description of what the prompt is best suited for.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \ name of the prompt to use or "DEFAULT"
    "next_inputs": string \ original input
}}}}
<< CANDIDATE PROMPTS >> {destinations}

<< INPUT >> {{input}}

<< OUTPUT (remember to include the ```json)>>
"""

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
router_prompt = PromptTemplate(template=router_template, input_variables=["input"], output_parser=RouterOutputParser())
router_chain = LLMRouterChain.from_llm(llm, router_prompt)

chain = MultiPromptChain(router_chain=router_chain, destination_chains=destination_chains, default_chain=default_chain, verbose=True)

# Test server route
@app.route("/ping", methods=['GET'])
def pinger():
    return "<p>Hello world!</p>"

# Main route to handle POST requests
@app.route("/bot", methods=['POST'])
def bot():
    query = request.form.get("Body")
    response = chain.run(query)
    resp = MessagingResponse()
    msg = resp.message()
    msg.body(response)
    return str(resp)

if __name__ == '__main__':
    app.run(port = 8888)

