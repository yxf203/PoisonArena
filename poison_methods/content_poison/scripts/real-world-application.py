import json
import os
import sys
sys.path.append(".")

from scripts.utils import format_print, write_csv

doc_applications = ["ChatChat", "Quivr"]
revew_applications = ["Comment_Analyzer", "Amz-Review-Analyzer"]
file_format = ["PDF", "Markdown", "HTML"]
data_type = ["Products", "Books"]
if __name__ == "__main__":
    table6 = [["Applications", "PDF", "Markdown", "HTML", "Total"], ["ChatChat"], ["Quivr"]]
    print("Evaluating ASR of Real-World Applications.")
    print("-" * 40)
    print("Processing word-level attack on ChatChat and Quivr.")
    for app in doc_applications:
        res_file = os.path.join("results", "applications",app+".json")
        tot_succ = 0
        total = []
        index = doc_applications.index(app)+1
        print("\t Processing result of", app)
        with open(res_file, "r") as f:
            res = json.load(f)
        for format in file_format:
            succ = res[format]["succ_index"]
            tot = res[format]["total"]
            print("\t\t success index of ", format, "is ", succ)
            table6[index].append(len(succ))
            tot_succ += len(succ)
            total.append(tot)
        table6[index].append(tot_succ)
        total.append(sum(total))

    for line in range(1,3,1):
        for i in range(1,5,1):
            table6[line][i] = str(round(table6[line][i] / total[i-1] *100, 2)) + "% " + "(" + str(table6[line][i]) + "/" +str(total[i-1]) + ")"

    table7 = [["Applications", "Products", "Books", "Total"], ["Comment Analyzer"], ["Amz Review Analyzer"]]

    print("-" * 40)
    print("Processing whole-content attack on Comment Analyzer and Amz Review Analyzer.")
    for app in revew_applications:
        res_file = os.path.join("results", "applications",app+".json")
        tot_succ = 0
        total = []
        index = revew_applications.index(app)+1
        print("\t Processing result of", app)
        with open(res_file, "r") as f:
            res = json.load(f)
        for format in data_type:
            succ = res[format]["succ_index"]
            tot = res[format]["total"]
            print("\t\t success index of ", format, "is ", succ)
            table7[index].append(len(succ))
            tot_succ += len(succ)
            total.append(tot)
        table7[index].append(tot_succ)
        total.append(sum(total))

    for line in range(1,3,1):
        for i in range(1,4,1):
            table7[line][i] = str(round(table7[line][i] / total[i-1] *100, 2)) + "% " + "(" + str(table7[line][i]) + "/" +str(total[i-1]) + ")"
    
    print("\n")
    print("Table 6: The evaluation on document Q&A applications.")
    format_print(table6)
    write_csv(table6, "all_tables/table6.csv")

    print("\n")
    print("Table 7: The evaluation on summarization applications.")
    format_print(table7)
    write_csv(table7, "all_tables/table7.csv")


