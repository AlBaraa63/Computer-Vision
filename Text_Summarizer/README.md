# AI Text Summarizer

## Overview

**AI Text Summarizer** is a Python command-line application that summarizes the content of any text or PDF file using advanced artificial intelligence models from Hugging Face. The user can specify the desired summary style, such as "briefly," "in detail," or "as bullet points," either as a command-line argument or interactively. The program prints the summary and saves it to `output.txt`.

---

## Features

* Summarizes any `.txt` **or** `.pdf` file using Hugging Face's BART model
* Lets the user pick a summary style (e.g., briefly, in detail, bullet points)
* Handles errors gracefully (e.g., missing files, missing API token, empty file)
* Easy command-line interface and clear prompts
* Fully tested with pytest, including mocks for API calls and user input

---

## Files

* **project.py** — Main code file with all functions and logic, including:

  * `main`: The main function to run the app
  * `get_input_file`: Prompts user for file name and validates existence
  * `read_pdf`: Extracts text from PDF files
  * `summarize_text`: Interacts with Hugging Face API to summarize text
* **test\_project.py** — Automated tests for main functions using pytest (mocks API calls and file input)
* **requirements.txt** — Lists required Python packages (`requests`, `pdfplumber`, `pytest`)
* **README.md** — This documentation file
* **input.txt/output.txt** — Example input and output files

---

## How to Use

1. **Install dependencies:**

   ```
   pip install -r requirements.txt
   ```

2. **Get a free API key from Hugging Face:**
   [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

3. **Set your API token as an environment variable:**

   * On Windows (PowerShell):

     ```powershell
     $env:HF_TOKEN="your_huggingface_token"
     ```
   * On Linux/Mac:

     ```bash
     export HF_TOKEN="your_huggingface_token"
     ```

4. **Prepare your input text file or PDF (default: `input.txt` or any `.pdf`).**

5. **Run the program:**

   ```bash
   python project.py                    # Uses "briefly" by default
   python project.py "in detail"        # For a detailed summary
   python project.py "as bullet points" # For bullet points summary
   ```

6. **Follow the prompt to enter the filename, or just hit Enter to use `input.txt`.**

7. **The summary is printed to the console and saved as `output.txt`.**

---

## Running Tests

Run all automated tests with:

```bash
pytest test_project.py
```

---

## Design Choices & Notes

* All logic is kept in one file (`project.py`) for easy review, as required by CS50P.
* The program checks for required files and environment variables, exiting gracefully with a helpful message if anything is missing.
* The summary style is passed as a prompt modifier, making it easy to experiment with different summary "moods."
* Tests use `pytest` and Python's `monkeypatch` to simulate user input and API calls.
* The code is structured to make future expansions (like PDF support, chunking large texts, or more APIs) straightforward.

**Note:**
The free Hugging Face summarization API used in this project is not always as “friendly” or flexible as commercial APIs like OpenAI—sometimes it may ignore detailed summary style instructions or take longer to respond. However, it reliably generates good summaries for most standard texts, and offers a free and accessible way to experiment with AI summarization.
This project is designed to be easily upgradable: you can swap in a different API, support other file types, or add features like chunking and a graphical interface in the future.

---

## Example

**Input (input.txt):**

```
Artificial Intelligence (AI) has become a pervasive force in our daily lives...
```

**Command:**

```
python project.py "as bullet points"
```

**Output:**

```
- AI is transforming daily life
- Powers tools like voice assistants and recommendations
- Raises ethical concerns about bias, automation, and transparency
```

---

## Author & Submission Details

* **Name:** AlBaraa Mohammad
* **GitHub Username:** AlBaraa-1
* **edX Username:** albaraa\_63
* **City/Country:** UAE/AD
* **Video Demo:** [https://youtu.be/7m9xP-cEmDo](https://youtu.be/7m9xP-cEmDo)