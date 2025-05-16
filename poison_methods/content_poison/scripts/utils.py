import csv
import numpy as np
from prettytable import PrettyTable
def format_print(table):
    table = table.copy()
    for row in table:
        for i in range(len(row)):
            # print(type(row[i]))
            if type(row[i]) == float or type(row[i]) == np.float64: 
                if row[i] <= 1:
                    row[i] = str(round(row[i]*100, 2))+"%"
                else:
                    row[i] = round(row[i], 2)
    t = PrettyTable()
    t._validate_field_names = lambda *a, **k: None
    t.field_names = table[0]
    t.add_rows(table[1:])
    print(t)


def write_csv(table, file_name):
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(table)
    print("write to", file_name)