import csv
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('csv', type=str, help='csv file to convert')
parser.add_argument('json', type=str, help='json file to write')
parser.add_argument('--columns', type=int, default = 0, help="number of columns to trim")
args = parser.parse_args()

# open a csv file
def open_csv_file(csv_file):
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

# write a dict to json
def write_json_file(json_file, data):
    with open(json_file, 'w') as f:
        json.dump(data, f)
    return True

def main():
    data = open_csv_file(args.csv)

    # Add to dictionary by filename
    data_dict = {}
    for i in range(1,len(data)):
        filename = data[i][0]
        if filename not in data_dict:
            data_dict[filename] = []
        data_dict[filename].append(data[i][args.columns:])
    
    if write_json_file(args.json, data_dict):
        print("Successfully wrote to json file")


if __name__ == "__main__":
    main()
