import pandas as pd
import os
import sys
import glob
from pathlib import Path

def convert_xlsx_to_csv(xlsx_file, output_file=None, sheet_name=0):
    try:
        # Read Excel file
        df = pd.read_excel(xlsx_file, sheet_name=sheet_name)
        
        # Determine output filename if not specified
        if output_file is None:
            output_file = os.path.splitext(xlsx_file)[0] + '.csv'
        
        # Write to CSV
        df.to_csv(output_file, index=False)
        print(f"Successfully converted {xlsx_file} to {output_file}")
        
    except Exception as e:
        print(f"Error converting {xlsx_file}: {str(e)}")

def main():

    input_path = "/home/basvercru/Documents/Visual Code Workspaces/BelHisFirm/index_parser/CRF/BelHisFirm-GT/1913_GT_annotated.xlsx"
    output_file = "/home/basvercru/Documents/Visual Code Workspaces/BelHisFirm/index_parser/CRF/BelHisFirm-GT/1913_GT_annotated.csv"
    
    convert_xlsx_to_csv(input_path, output_file)

if __name__ == "__main__":
    main()