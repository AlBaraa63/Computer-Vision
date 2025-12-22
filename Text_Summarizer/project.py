# project.py
# A script to summarize text or PDF files using the Hugging Face BART large CNN model.
# - Supports both .txt and .pdf input files.
# - Requires the HF_TOKEN environment variable for API access.
# - Outputs the summary to output.txt.
#
# Usage:
#   1. Place your text or PDF file in the same directory.
#   2. Run the script and enter the file name when prompted (or press Enter for input.txt).
#   3. The summary will be printed and saved to output.txt.
#
# Author: [AlBaraa Mohammed]
# Date: [24/5/2025]

import pdfplumber
import sys
import os
import requests


# Main function to coordinate the summarization process
def main(summary_style="briefly"):
    # Get the input file from the user (or use default)
    input_file = get_input_file()
    output_file = "output.txt"

    if input_file.lower().endswith(".pdf"):
        # If the input file is a PDF, read it using pdfplumber
        print(f"Reading PDF file: {input_file}")
        text = read_pdf(input_file)
    else:
        # Otherwise, treat it as a text file
        print(f"Reading text file: {input_file}")
        # Read the input file
        try:
            with open(input_file, encoding="utf-8") as f:
                text = f.read()
        except FileNotFoundError:
            sys.exit(f"Error: The file {input_file} was not found.")

    if not text.strip():
        sys.exit("Error: The input file is empty.")

    print(f"Summary coming next ({summary_style})…")

    # Generate the summary using the Hugging Face API
    summary = summarize_text(text, summary_style)

    # If the summary is generated successfully, print and save it
    if summary:
        print("\n--- Summary ---\n")
        print(summary)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"\nSummary saved to '{output_file}'")
    else:
        print("Failed to generate summary.")


# Get the input file name from the user
# Returns the file name as a string
def get_input_file():
    input_file = input("Enter the input file name (default: input.txt): ").strip()
    if not input_file:
        input_file = "input.txt"
    if not os.path.exists(input_file):
        sys.exit(f"Error: The file {input_file} was not found.")
    return input_file


# Read and extract text from a PDF file using pdfplumber
# Returns the extracted text as a string
def read_pdf(filepath):
    try:
        with pdfplumber.open(filepath) as pdf:
            text = ""
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
            return text.strip()
    except Exception as e:
        sys.exit(f"Error reading PDF file: {e}")


# Summarize text using the Hugging Face API
# Returns the summary as a string
# Exits on error
def summarize_text(text, summary_style):
    api_token = os.getenv("HF_TOKEN")
    if not api_token:
        sys.exit("Hugging Face API token not found. Set HF_TOKEN environment variable.")
    headers = {"Authorization": f"Bearer {api_token}"}
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    prompt = f"Summarize the following text {summary_style}:\n{text}"
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
        response.raise_for_status()
        result = response.json()
        return result[0]["summary_text"]
    except Exception as e:
        sys.exit(f"Hugging Face API Error: {e}")


if __name__ == "__main__":
    summary_style = "briefly"
    if len(sys.argv) > 1:
        summary_style = sys.argv[1]
    main(summary_style)
