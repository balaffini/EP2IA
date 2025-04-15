
import pandas as pd
import sys

def transform_csv(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Replace dots with commas in all numeric columns
    df = df.applymap(lambda x: str(x).replace('.', ','))
    
    # Save the transformed dataframe to a new CSV file
    output_file_path = file_path.replace('.csv', '_transformed.csv')
    df.to_csv(output_file_path, index=False, header=False)
    
    print(f'Transformed file saved as: {output_file_path}')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python transform_csv.py <path_to_csv_file>")
    else:
        file_path = sys.argv[1]
        transform_csv(file_path)
