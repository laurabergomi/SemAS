import requests
import argparse
import os
from dotenv import load_dotenv
from call_llm import query_gpt

load_dotenv("global_variables.env")
llm_api_key = os.getenv("GPT_API_KEY")


def read_content_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None


# Extract relations using LLM with zero-shot approach and constrained relations
def extract_relations():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Send the input text to the API endpoint for relation extraction.')
    parser.add_argument('input_text_file', type=str, help='Path to the file containing the input text.')
    parser.add_argument('instruction_file', type=str, help='Path to the file containing the instructions.')

    # Parse the arguments
    args = parser.parse_args()
    input_text = read_content_from_file(args.input_text_file)
    instructions = read_content_from_file(args.instruction_file)

    if input_text and instructions:
        # If you want to consider only "Data" element
        # input_text = input_text.split("Content returned:")[1].split("**Data:**")[1].split("**Warrant:**")[0]

        # Concatenate input_text and instructions to form the complete message
        user_message = f"{input_text}\n\n{instructions}"

        response = query_gpt(prompt=user_message, llm_api_key=llm_api_key, temperature=1)

        print(f"%%% Message sent:\n{user_message}\n\n")
        print(f"Content returned:\n{response}\n")

    else:
        print("Failed to read the input_text or instructions from the files.")


if __name__ == "__main__":
    extract_relations()
