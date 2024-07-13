import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np



def calculate_accuracy(file_path):
    # Read the csv file
    df = pd.read_csv(file_path)
    #print(df.head())
    # add a new column to the dataframe to store the result of the comparison
    # trim the columns correct QID and output QID
    df['correct QID'] = df['correct QID'].str.strip()
    df['output QID'] = df['output QID'].str.strip()
    df['result'] = np.where(df['correct QID'] == df['output QID'], 1, 0)
    #save the results in a new csv file
    df.to_csv(file_path, index=False)

    # # filter the rows where the result is 0
    # d_errors = df[df['result'] == 0]
    # # filter the columns 'prompt', 'output', 'correct QID', 'output QID'
    # d_errors = d_errors[[ 'correct QID', 'output QID']]
    # count = d_errors.shape[0]
    # print(d_errors)
    # # save the errors in a new xlsx file
    # d_errors.to_excel(file_path.replace(".csv", "_errors.xlsx"), index=False)

    # Compare the two columns and calculate the accuracy
    accuracy = accuracy_score(df['correct QID'], df['output QID'])

    # remove the rows where correct QID is 0
    df = df[df['correct QID'] != 0]

    # calculate F1 score
    f1 = f1_score(df['correct QID'], df['output QID'], average='weighted')

    # confusion matrix
    #confusion_matrix = pd.crosstab(df['correct QID'], df['output QID'], rownames=['Actual'], colnames=['Predicted'])
    #print(confusion_matrix)

    return accuracy, f1

def find_errors(path1, path2, path3):
    # Read the csv file
    df1 = pd.read_csv(path1)
    if path2 != "":
        df2 = pd.read_csv(path2)
    if path3 != "":
        df3 = pd.read_csv(path3)

    print(df1.head())
    if path2 != "":
        print(df2.head())
    if path3 != "":
        print(df3.head())
    # add a new column to the dataframe to store the result of the comparison
    # trim the columns correct QID and output QID
    df1['correct QID'] = df1['correct QID'].str.strip()
    df1['output QID'] = df1['output QID'].str.strip()
    #df1['result'] = np.where(df1['correct QID'] == df1['output QID'], 1, 0)
   
    if path2 != "":
        df2['correct QID'] = df2['correct QID'].str.strip()
        df2['output QID'] = df2['output QID'].str.strip()
    #df2['result'] = np.where(df2['correct QID'] == df2['output QID'], 1, 0)
    
    if path3 != "":
        df3['correct QID'] = df3['correct QID'].str.strip()
        df3['output QID'] = df3['output QID'].str.strip()
    #df3['result'] = np.where(df3['correct QID'] == df3['output QID'], 1, 0)
   
    # filter the rows where the result is 0
    #d_errors1 = df1[df1['result'] == 0]
    #d_errors2 = df2[df2['result'] == 0]
    #d_errors3 = df3[df3['result'] == 0]
    # filter the columns 'prompt', 'output', 'correct QID', 'output QID'
    df1 = df1[['group', 'correct QID', 'output QID']]
    if path2 != "":
        df2 = df2[[ 'group', 'correct QID', 'output QID']]
    if path3 != "":
        df3 = df3[[ 'group', 'correct QID', 'output QID']]
    
    # merge the dataframes horizontally where correct QID is the same and output QID is different
    # Merge the first two DataFrames on 'correct QID'
    if path2 != "":
        combined_errors = pd.merge(df1, df2, on=['group', 'correct QID'], how='outer', suffixes=('_1', '_2'))
    else:
        combined_errors = df1

    # Merge the result with the third DataFrame on 'correct QID'
    if path3 != "":
        combined_errors = pd.merge(combined_errors, df3, on=['group', 'correct QID'], how='outer', suffixes=('', '_3'))
    
    # remove the duplicate rows
    combined_errors = combined_errors.drop_duplicates()

    if path2 == "" and path3 =="":
        combined_errors['result'] = np.where((combined_errors['correct QID'] == combined_errors['output QID']), 1, 0)
    elif path3 !="":
    # if correct QID == output QID_1 and correct QID == output QID_2 and correct QID == output QID_3 then result = 1 else 0
        combined_errors['result'] = np.where((combined_errors['correct QID'] == combined_errors['output QID_1']) & (combined_errors['correct QID'] == combined_errors['output QID_2'])& (combined_errors['correct QID'] == combined_errors['output QID']), 1, 0)
    else:
        combined_errors['result'] = np.where((combined_errors['correct QID'] == combined_errors['output QID_1']) & (combined_errors['correct QID'] == combined_errors['output QID_2']), 1, 0)

    # filter the rows where the result is 0
    combined_errors = combined_errors[combined_errors['result'] == 0]


    # Save to an Excel file with the dataframes next to each other (horizontally)
    combined_errors.to_excel(path1.replace(".csv", "_merged.xlsx"), index=False)

    # get the list of groups
    groups = combined_errors['group'].unique()
    return groups



path1 = "../Dataset/2T_Round4/2T_Round4_results_0.csv"
path2 = "../Dataset/2T_Round4/2T_Round4_results_1.csv"
path3 = "../Dataset/2T_Round4/2T_Round4_results_2.csv"

accuracy1, f11 = calculate_accuracy(path1)
accuracy2, f12 = calculate_accuracy(path2)
accuracy3, f13 = calculate_accuracy(path3)

# find errors
groups1 = find_errors(path1, path2, path3)


# calculate the average accuracy and F1 score
average_accuracy = (accuracy1 + accuracy2 + accuracy3) / 3
average_f1 = (f11 + f12 + f13) / 3

path1_table = "../Dataset/2T_Round4/2T_Round4_table_results_0.csv"
path2_table = "../Dataset/2T_Round4/2T_Round4_table_results_1.csv"

accuracy1_table, f11_table = calculate_accuracy(path1_table)
accuracy2_table, f12_table = calculate_accuracy(path2_table)

average_accuracy_table = (accuracy1_table + accuracy2_table) / 2
average_f1_table = (f11_table + f12_table) / 2

# find errors
groups_table = find_errors(path1_table, path2_table, "")


path1_RAG1 = "../Dataset/2T_Round4/2T_Round4_RAG_results_0_v1.csv"
path2_RAG1 = "../Dataset/2T_Round4/2T_Round4_RAG_results_1_v1.csv"
path3_RAG1 = "../Dataset/2T_Round4/2T_Round4_RAG_results_2_v1.csv"

accuracy1_RAG1, f11_RAG1 = calculate_accuracy(path1_RAG1)
accuracy2_RAG1, f12_RAG1 = calculate_accuracy(path2_RAG1)
accuracy3_RAG1, f13_RAG1 = calculate_accuracy(path3_RAG1)

average_accuracy_RAG1 = (accuracy1_RAG1 + accuracy2_RAG1 + accuracy3_RAG1) / 3
average_f1_RAG1 = (f11_RAG1 + f12_RAG1 + f13_RAG1) / 3

# find errors
groups_RAG1 = find_errors(path1_RAG1, path2_RAG1, path3_RAG1)

path_RAG_ts = "../Dataset/2T_Round4/2T_Round4_RAG_results_0_ts.csv"
path_RAG_ts2 = "../Dataset/2T_Round4/2T_Round4_RAG_results_1_ts.csv"
path_RAG_ts3 = "../Dataset/2T_Round4/2T_Round4_RAG_results_2_ts.csv"

accuracy_RAG_ts, f1_RAG_ts = calculate_accuracy(path_RAG_ts)
#accuracy_RAG_ts2, f1_RAG_ts2 = calculate_accuracy(path_RAG_ts2)
#accuracy_RAG_ts3, f1_RAG_ts3 = calculate_accuracy(path_RAG_ts3)

average_accuracy_RAG_ts = (accuracy_RAG_ts + 0 + 0) / 1
average_f1_RAG_ts = (f1_RAG_ts + 0 + 0) / 1#

# find errors
groups_RAG_ts = find_errors(path_RAG_ts, "", "")



#path1_RAG = "../Dataset/2T_Round4/2T_Round4_RAG_results_0_v2.csv"
path2_RAG = "../Dataset/2T_Round4/2T_Round4_RAG_results_1_v2.csv"
path3_RAG = "../Dataset/2T_Round4/2T_Round4_RAG_results_2_v2.csv"

#accuracy1_RAG, f11_RAG = calculate_accuracy(path1_RAG)
accuracy2_RAG, f12_RAG = calculate_accuracy(path2_RAG)
accuracy3_RAG, f13_RAG = calculate_accuracy(path3_RAG)

average_accuracy_RAG = (accuracy2_RAG + accuracy3_RAG) / 2
average_f1_RAG = (f12_RAG + f13_RAG) / 2

# find errors
groupsRAG2 = find_errors(path2_RAG, path3_RAG, "")

path1_RAG_v3 = "../Dataset/2T_Round4/2T_Round4_RAG_results_0_v3.csv"
path2_RAG_v3 = "../Dataset/2T_Round4/2T_Round4_RAG_results_1_v3.csv"
path3_RAG_v3 = "../Dataset/2T_Round4/2T_Round4_RAG_results_2_v3.csv"

accuracy1_RAG_v3, f11_RAG_v3 = calculate_accuracy(path1_RAG_v3)
accuracy2_RAG_v3, f12_RAG_v3 = calculate_accuracy(path2_RAG_v3)
accuracy3_RAG_v3, f13_RAG_v3 = calculate_accuracy(path3_RAG_v3)

average_accuracy_RAG_v3 = (accuracy1_RAG_v3 + accuracy2_RAG_v3 + accuracy3_RAG_v3) / 3
average_f1_RAG_v3 = (f11_RAG_v3 + f12_RAG_v3 + f13_RAG_v3) / 3

# find errors
groupsRAG3 = find_errors(path1_RAG_v3, path2_RAG_v3, path3_RAG_v3)

all_groups = np.concatenate((groups1, groups_table, groups_RAG1, groups_RAG_ts, groupsRAG2, groupsRAG3))
all_groups = np.unique(all_groups)
print(all_groups)
# length of the array
print(len(all_groups))

# save all_groups to a file
np.savetxt("../Dataset/2T_Round4/train_2T_Round4.txt", all_groups, fmt='%s')

# find errors
#find_errors(path1_RAG_v3, path2_RAG_v3, path3_RAG_v3)

# print the results in a table round two decimal places
print("Results for 2T Round 4")
print("table                    |", round(accuracy1_table,2), "|", round(accuracy2_table,2), "|","0.00" ,"|", round(average_accuracy_table, 2), "|", round(average_f1_table, 2))
print("title and summary        |", round(accuracy1,2), "|", round(accuracy2,2) ,"|", round(accuracy3,2) ,"|", round(average_accuracy, 2), "|", round(average_f1, 2))
print("title and summary v2     |", round(accuracy1_RAG_v3,2), "|", round(accuracy2_RAG_v3,2) ,"|", round(accuracy3_RAG_v3,2) ,"|", round(average_accuracy_RAG_v3, 2), "|", round(average_f1_RAG_v3, 2))
print("RAG table wikidata       |", round(accuracy1_RAG1,2), "|", round(accuracy2_RAG1,2) ,"|", round(accuracy3_RAG1,2) ,"|", round(average_accuracy_RAG1, 2), "|", round(average_f1_RAG1, 2))
# print("RAG title summary wikidata |", round(accuracy_RAG_ts,2), "|", "0.00" ,"|", "0.00" ,"|", round(average_accuracy_RAG_ts, 2), "|", round(average_f1_RAG_ts, 2))
# print("RAG both                 |", 0.00, "|", round(accuracy2_RAG,2) ,"|", round(accuracy3_RAG,2) ,"|", round(average_accuracy_RAG, 2), "|", round(average_f1_RAG, 2))


