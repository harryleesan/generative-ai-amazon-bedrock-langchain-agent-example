import os
import json
import boto3
from langchain.agents.tools import Tool
from urllib.parse import urlparse

bedrock = boto3.client('bedrock-runtime', region_name="eu-west-3")
# bedrock = boto3.client('bedrock-runtime', region_name=os.environ['AWS_REGION'])
HW_INFHOST=os.getenv("HW_INFHOST")
HW_INFERENCE_URL=os.getenv("HW_INFERENCE_URL")
RETRIEVE_K_DOCS = 2
INFERENCE_REQUEST_KWARGS = {
    "headers": {
        "Content-Type": "application/json",
        "Infhost": HW_INFHOST
    },
    "timeout": 30,
}

class Tools:

    def __init__(self) -> None:
        print("Initializing Tools")
        self.tools = [
            Tool(
                name="AnyCompany",
                func=self.kendra_search,
                description="Use this tool to answer questions about AnyCompany.",
            )
        ]

    def get_inference_payload(self, query: str, k: int) -> dict:
        """Construct inference payload dict that will be sent with POST request."""
        payload = {
            "inputs": [
                {
                    "name": "input-0",
                    "shape": [1],
                    "datatype": "BYTES",
                    "parameters": None,
                    "data": [
                        {
                            "query": query,
                            "k": k
                        }
                    ]
                }
            ]
        }
        return payload
        
    def get_context_from_vectorstore(self, query: str):
        """
        Send an inference request to the model and get
        the context from knowledge base.
        """
        try:
            payload = self.get_inference_payload(
                query=query,
                k=RETRIEVE_K_DOCS
            )
            response = requests.post(
                HW_INFERENCE_URL, data=json.dumps(payload), **INFERENCE_REQUEST_KWARGS
            )
    
            if response.status_code == 200:
                docs = response.json()["outputs"][0]["data"]
                context = "\n\n".join(doc["page_content"] for doc in docs)
            else:
                context = ""
                print(
                    f"Request error. Code: {response.status_code}, Message: {response.text}"
                )
    
        except Exception as ex:
            context = ""
            print(f"Inference Error: {ex}")
    
        return context

    # def parse_kendra_response(self, kendra_response):
    #     """
    #     Extracts the source URI from document attributes in Kendra response.
    #     """
    #     modified_response = kendra_response.copy()

    #     result_items = modified_response.get('ResultItems', [])

    #     for item in result_items:
    #         source_uri = None
    #         if item.get('DocumentAttributes'):
    #             for attribute in item['DocumentAttributes']:
    #                 if attribute.get('Key') == '_source_uri':
    #                     source_uri = attribute.get('Value', {}).get('StringValue', '')

    #         if source_uri:
    #             print(f"Amazon Kendra Source URI: {source_uri}")
    #             item['_source_uri'] = source_uri

    #     return modified_response

    def kendra_search(self, question):
        """
        Performs a Kendra search using the Query API.
        """
        # kendra = boto3.client('kendra')

        # kendra_response = kendra.query(
        #     IndexId=os.getenv('KENDRA_INDEX_ID'),
        #     QueryText=question,
        #     PageNumber=1,
        #     PageSize=5  # Limit to 5 results
        # )

        # parsed_results = self.parse_kendra_response(kendra_response)

        context = self.get_context_from_vectorstore(question)

        print(f"HW Chroma Query Item: {context}")

        # passing in the original question, and various Kendra responses as context into the LLM
        return self.invokeLLM(question, context)

    def invokeLLM(self, question, context):
        """
        Generates an answer for the user based on the Kendra response.
        """
        prompt_data = f"""
        Human:
        Imagine you are AnyCompany's Mortgage AI assistant. You respond quickly and friendly to questions from a user, providing both an answer and the sources used to find that answer.

        Format your response for enhanced human readability.

        At the end of your response, include the relevant sources if information from specific sources was used in your response. Use the following format for each of the sources used: [Source #: Source Title - Source Link].

        Using the following context, answer the following question to the best of your ability. Do not include information that is not relevant to the question, and only provide information based on the context provided without making assumptions. 

        Question: {question}

        Context: {context}

        \n\nAssistant:
        """

        # Formatting the prompt as a JSON string
        json_prompt = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "temperature": 0.5,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_data
                        }
                    ]
                }
            ]
        })

        # Invoking Claude3, passing in our prompt
        response = bedrock.invoke_model(
            body=json_prompt,
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            accept="application/json",
            contentType="application/json"
        )

        # Getting the response from Claude3 and parsing it to return to the end user
        response_body = json.loads(response['body'].read())
        answer = response_body['content'][0]['text']

        return answer

# Pass the initialized retriever and llm to the Tools class constructor
tools = Tools().tools
