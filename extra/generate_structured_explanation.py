import requests
import argparse
import os
from dotenv import load_dotenv
from call_llm import query_gpt

load_dotenv()
llm_api_key = os.getenv("GPT_API_KEY")


def read_content_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None


def generate_explanation():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Send a concatenated patient case, question and instructions message to the API endpoint.')
    parser.add_argument('case_file', type=str, help='Path to the file containing the case information')
    parser.add_argument('question_file', type=str, help='Path to the file containing the question')
    parser.add_argument('instruction_file', type=str, help='Path to the file containing the instructions')

    # Parse the arguments
    args = parser.parse_args()

    # Read the case and question content from the provided file paths
    case_content = read_content_from_file(args.case_file)
    question_content = read_content_from_file(args.question_file)
    instructions = read_content_from_file(args.instruction_file)

    if case_content and question_content and instructions:
        user_message = f"{case_content}\n\n{question_content}\n\n{instructions}"
        response = query_gpt(prompt=user_message, llm_api_key=llm_api_key, temperature=1)

        print(f"%%% Message sent:\n{user_message}\n\n")
        print(f"Content returned:\n{response}\n")
    else:
        print("Failed to read the case or question or instructions from the files.")


if __name__ == "__main__":
    generate_explanation()
