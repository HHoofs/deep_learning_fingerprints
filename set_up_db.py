import csv
import glob
import os
import re

from tqdm import tqdm

VARS = ['id', 'full_path', 'subdir', 'gender', 'pattern']

def read_files_sd04(vars=VARS):
    list_ids = glob.glob('sd04/png_txt/*/*.txt')

    with open('test.csv', 'w', newline='') as csv_db:
        writer = csv.writer(csv_db, quoting=csv.QUOTE_ALL)
        writer.writerow(vars)

        for id in tqdm(list_ids):
            variables = extract_file_path_info(id)
            with open(id) as y_txt:
                extract_gender_and_pattern(variables, y_txt)
            writer.writerow(variables)


def extract_gender_and_pattern(variables, y_txt):
    for row, line in enumerate(y_txt.readlines()):
        if row == 0:
            variables.append(re.search('Gender: (.)', line).group(1))
        if row == 1:
            variables.append(re.search('Class: (.)', line).group(1))


def extract_file_path_info(id):
    variables = []
    variables.append(os.path.splitext(os.path.basename(id))[0])
    variables.append(os.path.splitext(id)[0])
    variables.append(os.path.dirname(id).split(os.sep)[-1])
    return variables


if __name__ == '__main__':
    read_files_sd04()