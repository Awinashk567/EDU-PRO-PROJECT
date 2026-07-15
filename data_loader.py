import pandas as pd

# 1. Define the file path (Make sure the Excel file is in the same folder as this script)
file_name = "Copy of EduPro Online Platform.xlsx"

try:
    print("Loading the Excel file... This might take a few seconds.")
    
    # 2. Load the entire Excel file to check what sheets it contains
    excel_data = pd.ExcelFile(file_name)
    
    # Print the names of the sheets to understand our data structure
    print(f"Success! Found the following sheets in the dataset: {excel_data.sheet_names}\n")
    
    # 3. Load each sheet into its own Pandas DataFrame
    # We use a dictionary to store them automatically based on their sheet names
    dataframes = {}
    for sheet in excel_data.sheet_names:
        dataframes[sheet] = pd.read_excel(file_name, sheet_name=sheet)
        print(f"Loaded '{sheet}' sheet with {dataframes[sheet].shape[0]} rows and {dataframes[sheet].shape[1]} columns.")
        
    print("\nAll data loaded successfully! We are ready for analysis.")

except FileNotFoundError:
    print(f"Error: Could not find the file '{file_name}'. Please ensure it is in the same folder as this script.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")