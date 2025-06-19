import os
import pandas as pd
from google import genai
from google.generai import types
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Retrieve your API key from environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in a .env file.")

# Configure the genai library with your API key
genai.configure(api_key=GEMINI_API_KEY)

# Define the structure of your CSV data
CSV_COLUMNS = [
    "docgen", "doc_no", "f1", "datedoc", "doc_title",
    "org_out", "org_in", "f2", "f3"
]
NUM_COLUMNS = len(CSV_COLUMNS)

# Define the detailed SYSTEM INSTRUCTION for the AI model
# This is now the primary set of instructions for the model's behavior.
SYSTEM_INSTRUCTION = f"""
You are a data extraction specialist. Your task is to analyze the provided document and extract information into a CSV format where '|' separates columns. Use this exact column order:
{'|'.join(CSV_COLUMNS)}.

Follow these rules strictly:
1.  The column 'f1' must always be 0.
2.  The output must always have exactly {NUM_COLUMNS} columns per row.
3.  'doc_no' must be the second column and 'datedoc' must be the fourth. Do not swap them.
4.  Any field that is blank or cannot be found must contain a single '0'.
5.  If you encounter a row in the document that you cannot understand or parse, skip that row entirely in your output.
6.  Do not output a header row. Only output the data rows.

Example of a valid output row:
1|01038/2559|0|1/7/2559|รายงานการสำรองข้อมูลประจำเดือน มิย.59|งานสารบรรณ (สบ.)|กองบริหารการคลัง (กค.)|เดินเรื่องเอง เดินเรื่องเอง|0
"""

def process_pdf_file(file_path, model):
    """
    Uploads a PDF file to Gemini, processes it using the pre-configured model,
    and returns the data as a pandas DataFrame.
    """
    try:
        print(f"Uploading file: {file_path}")
        # Use the simpler, high-level upload_file function
        uploaded_file = genai.upload_file(path=file_path, mime_type='application/pdf')
        print(f"Uploaded file '{uploaded_file.display_name}' as: {uploaded_file.uri}")

        # The model already knows its instructions from the system_instruction.
        # We only need to provide the file as the content.
        response = model.generate_content(
            contents=[uploaded_file],
            generation_config=types.GenerateContentConfig(
                response_mime_type="text/plain",
                temperature=0.0 # Keep temperature low for consistency
            )
        )

        csv_text = response.text.strip()
        lines = csv_text.split('\n')

        data = [line.split('|') for line in lines if line] # Filter out empty lines

        # Validate the extracted data
        valid_data = [row for row in data if len(row) == NUM_COLUMNS]

        if valid_data:
            df = pd.DataFrame(valid_data, columns=CSV_COLUMNS)
            print(f"Successfully processed and validated data from {file_path}")
            return df
        else:
            print(f"Invalid or no data format extracted for file {file_path}. Skipping...")
            return None

    except Exception as e:
        print(f"An error occurred while processing file {file_path}: {e}")
        return None

def main():
    """
    Main function to orchestrate the PDF processing and data compilation.
    """
    # --- Model Initialization with System Instruction ---
    # Create the GenerativeModel instance once with our system instruction.
    model = genai.GenerativeModel(
        model_name="gemini-2.5-pro-preview-05-06", # or your preferred model
        system_instruction=SYSTEM_INSTRUCTION
    )

    # Get the directory containing PDF files from the user
    user_input = input("Enter the path to the directory containing your PDF files: ")
    if not os.path.isdir(user_input):
        print("The provided path is not a valid directory.")
        return

    files = [f for f in os.listdir(user_input) if f.lower().endswith('.pdf')]
    if not files:
        print("No PDF files found in the specified directory.")
        return

    all_dataframes = []
    output_csv_file = "gemini_extracted_data.csv"

    for file_name in files:
        file_path = os.path.join(user_input, file_name)

        # Pass the pre-configured model to the processing function
        df = process_pdf_file(file_path, model)
        if df is not None:
            all_dataframes.append(df)
            print(f"Queued data from: {file_name}")

    # Combine all dataframes into one and save to a CSV file
    if all_dataframes:
        combined_data = pd.concat(all_dataframes, ignore_index=True)
        combined_data.to_csv(output_csv_file, sep='|', index=False, header=True)
        print(f"\nAll files processed. Data successfully saved to '{output_csv_file}'.")
    else:
        print("\nNo valid data was extracted from any of the files.")

if __name__ == "__main__":
    main()
