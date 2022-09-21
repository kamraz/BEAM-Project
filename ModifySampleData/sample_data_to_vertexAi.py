import csv
import sys

# open csv file
def open_csv_file(csv_file):
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

# write csv file
def write_csv_file(csv_file, data):
    with open(csv_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(data)

csv_file = sys.argv[1]
cloud_prefix = 'gs://eagle-test-bucket/'

# ['prev_filename', 'filename', 'label_type', 'image_width', 'image_height', 'xmin', 'xmax', 'ymin', 'ymax']
data = open_csv_file(csv_file)

final_csv = []
for i in range(1,len(data)):
    row = []
    row.append(cloud_prefix+data[i][0])
    row.append(data[i][2])

    # corrdinates as a proportion of the image size
    xmin = float(data[i][5])/float(data[i][3])
    xmax = float(data[i][6])/float(data[i][3])
    ymin = float(data[i][7])/float(data[i][4])
    ymax = float(data[i][8])/float(data[i][4])
    row = row + [xmin, ymin, '', '', xmax, ymax, '', '']
    final_csv.append(row)

print(final_csv[0])
write_csv_file("final.csv", final_csv)