from relation_extraction import *
import argparse
import os
import numpy as np
from dotenv import load_dotenv
load_dotenv("global_variables.env")
llm_api_key = os.getenv("GPT_API_KEY")


def llm_evaluation():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Send the encounter and explanation to the API endpoint for alignment evaluation by the LLM')
    parser.add_argument('encounter_file', type=str, help='Path to the file containing the encounter')
    parser.add_argument('explanation_file', type=str, help='Path to the file containing the explanation')
    parser.add_argument('instruction_file', type=str, help='Path to the file containing the instructions.')

    # Parse the arguments
    args = parser.parse_args()
    encounter = read_content_from_file(args.encounter_file)
    explanation = read_content_from_file(args.explanation_file)
    instructions = read_content_from_file(args.instruction_file)

    if encounter and explanation and instructions:
        user_message = f"{instructions.format(patient_case=encounter, explanation=explanation)}"
        response = query_gpt(prompt=user_message, llm_api_key=llm_api_key, model_name="o3", temperature=1)
        print(f"%%% Message sent:\n{user_message}\n\nContent returned:\n{response}\n")
    else:
        print("Failed to read the inputs.")


if __name__ == "__main__":
    llm_evaluation()

