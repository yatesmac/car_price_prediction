'''sample.py Samples a single random row from test CSV file and writes it to a new CSV file.'''

import csv
import random
import os


def generate_name():
    '''Generate an incremental filename.'''
    root = '../../data/test/random_rows'
    base_name = f'{root}/sample'
    ext = 'csv'
    counter = 1

    while True:
        output = f"{base_name}_{counter}.{ext}"
        if not os.path.exists(output):
            break
        counter += 1

    return output


def sample_row_csv(input):
    '''Samples a single random row'''
    try:
        with open(input, 'r') as f_in:
            reader = csv.reader(f_in)
            rows = list(reader)  # Read all rows into a list

        if not rows:
            raise ValueError('Input CSV file is empty.')
        
        random_row = random.choice(rows)
        output = generate_name()
        with open(output, 'w', newline='') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(random_row)

    except FileNotFoundError:
        print(f"Error: Input file '{input}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return output
    

