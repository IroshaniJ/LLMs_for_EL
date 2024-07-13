from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun
from langchain import hub
from langchain.agents import AgentExecutor, create_json_chat_agent
import os
import re
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
import pandas as pd
from ollama import Client
import argparse

  
def main():

    # Initialize the parser
    parser = argparse.ArgumentParser(description="Process some inputs.")

    # Adding arguments
    parser.add_argument("dataset", type=str, help="Dataset name", choices=["Round1_T2D", "Round3_2019", "2T_Round4", "Round4_2020", "HardTablesR2", "HardTablesR3"])
    parser.add_argument("dataset_desc", type=str, help="Dataset_Desc file name (.csv)", choices=["Round1_T2D_f3.csv", "Round3_f3.csv", "2T-2020_f3.csv", "Round4_f3.csv", "HardTableR2-2021_f3.csv", "HardTableR3-2021_f3.csv"])
    parser.add_argument("title_summary", type=str, help="Table title summary file name (.csv)", choices=["Round1_T2D.csv", "Round3.csv", "2T_Round4.csv", "Round4.csv", "HardTableR2.csv", "HardTableR3.csv"])

    # Parsing arguments
    args = parser.parse_args()

    dataset = args.dataset
    dataset_desc = args.dataset_desc
    title_summary = args.title_summary

    # Using arguments
    print(f"Dataset: {dataset}")
    print(f"Dataset_Desc: {dataset_desc}")
    print(f"Title_Summary: {title_summary}")
    
    path = "../../../../projects/nn10072k/EMD/data/datasets_to_index_3.json"  # Update this to the correct path
    csvs ="../../../../projects/nn10072k/EMD/data/Datasets_Desc/"
    table_title_summary_path = "../../../../projects/nn10072k/EMD/data/tables_with_title_summary/"
    
    # read the file
    with open(path, 'r') as f:
        data = json.load(f)

    # read the list of tables for each group
    #Round1_T2D_f3 = data["Round1_T2D"]
    #Round3_f3 = data["Round3_2019"]
    #Round2T_2020_f3 = data["2T_Round4"]
    #Round4_f3 = data["Round4_2020"]
    #HardTableR2_2021_f3 = data["HardTablesR2"]
    HardTableR3_2021_f3 = data[dataset]

    # read .csv files from csvs
    #round1_t2d = pd.read_csv(csvs + "Round1_T2D_f3.csv")
    #round3 = pd.read_csv(csvs + "Round3_f3.csv")
    #round2t_2020 = pd.read_csv(csvs + "Round2T_2020_f3.csv")
    #round4_2020 = pd.read_csv(csvs + "Round4_f3.csv")
    #hardtable_r2_2021 = pd.read_csv(csvs + "HardTableR2_2021_f3.csv")
    hardtable_r3_2021 = pd.read_csv(csvs + dataset_desc)

    table_title_summary = pd.read_csv(table_title_summary_path + title_summary)
    # remove the spaces and strip the string
    table_title_summary["tableName"] = table_title_summary["tableName"].str.strip()

    # create a list of pandas dataframes
    #dfs = [round1_t2d, round3, round2t_2020, round4_2020, hardtable_r2_2021, hardtable_r3_2021]
    # datasets list 
    #datasets = ["Round1_T2D", "Round3_2019", "2T_Round4", "Round4_2020", "HardTableR2", "HardTableR3"]
    # run mistral using ollama

    df = hardtable_r3_2021
    # select only the the keys available in data
    df = df[df["key"].isin(HardTableR3_2021_f3)]
    new_df = pd.DataFrame()
    grouped = df.groupby("group")
    # save results in a new dataframe with columns tableName,key, group, correct QID, output QID
    results = pd.DataFrame(columns=["tableName", "key", "group","prompt", "output", "correct QID", "output QID",])

    llm = ChatOllama(model="mistral:latest")
    wikidata = WikidataQueryRun(api_wrapper=WikidataAPIWrapper(),handle_tool_error=True, handle_validation_error=False )
    tools = [wikidata]
    #prompt = hub.pull("hwchase17/react-chat-json")

    client = Client(host='http://localhost:11434')
    # filter rows where group = 7798
    #grouped = grouped.filter(lambda x: x["group"].iloc[0] == 7798)
    #print(grouped)
    #grouped = grouped.groupby("group")
    # Iterate over each group
    for iter in range(3):
        count = 0
        # save results in a new dataframe with columns tableName,key, group, correct QID, output QID
        results = pd.DataFrame(columns=["tableName", "key", "group","prompt", "output", "correct QID", "output QID",])

        for group_name, group_df in grouped:
            #print("start time: ", time.time())    
            # Read the table file
            # Get the key
            #print("group_name", group_name)
            #print("group_df", group_df)
            key_parts = group_df["key"].iloc[0].split()
            table_name = key_parts[0].strip().replace('s/+',"")

            table_file = f"../../../../projects/nn10072k/EMD/data/Dataset/{dataset}/tables/{table_name}.csv"
            table_df = pd.read_csv(table_file)

            
            row_number = int(key_parts[1])-1
            col_number = int(key_parts[2])
                
            # Get the value and column name
            value = table_df.iloc[row_number, col_number]
            col_name = table_df.columns[col_number]
                
            # Save the value as X and the column name as COL
            X = value
            COL = col_name
                
            # Format the content of the table with the format “[TAB] col: | Media | MIX | [SEP] row 1: | Dainik Jagran | 27.5 | [SEP] row 2: | Dainik Bhaskar | 14 | [SEP] row 3: | Aajtak TV | 7 | [SEP] row 4: | CNN Editions (International) | 6 | [SEP] row 5: | Dinakaran | 5 | [SEP] row 6: | Malayala Manorama | 4 | [SEP] row 7: | Divya Bhaskar | 3.8 | [SEP] row 8: | Dinamalar | 3.5 | [SEP] row 9: | Huffington Post | 3 | [SEP] row 10: | foxnews | 3 | [SEP] row 11: | bbc Hindi | 2.6 | [SEP] row 12: | indosiar | 2.5 | [SEP] row 13: | Softpedia | 2.41 | [SEP] row 14: | Dina Thanthi | 2.1 | [SEP] row 15: | CNN | 2 | [SEP] row 16: | People's Daily (Renmin Ri Bao) | 2 | [SEP] row 17: | USA Today | 2 | [SEP] row 18: | Navbharat Times | 2 | [SEP] row 19: | Sahara Samay English | 1.8 | [SEP] row 20: | Punjab Kesari | 1.75 | 
            #where "Media" and "MIX" are the column names and the rest are the values in the table.
            content = "col: | "
            for col in table_df.columns:
                content += f"{col} | "
            content += " "
            for index, row in table_df.iterrows():
                if index == row_number:
                    content += f" row {index + 1}: | "
                    for col in table_df.columns:
                        content += f"{row[col]} | "
                    content += " "
            
            #print("TABLE_ROW", content)
                
            # Save the content as TABLE
            TABLE_ROW = content

            # Get the table title and summary
            table_title = table_title_summary[table_title_summary["tableName"] == table_name]
            #print("table_title_summary", table_title)
            try:
                TABLE_TITLE = table_title["title"].iloc[0]
                TABLE_SUMMARY = table_title["summary"].iloc[0]
            except:
                TABLE_TITLE = "None"
                TABLE_SUMMARY = "None"
                count += 1
            print(f"missing {table_name} title and summary count: {count}")
            #print("TABLE_TITLE", TABLE_TITLE)
            #print("TABLE_SUMMARY", TABLE_SUMMARY)
                    
            # Get the "id", "name", "types" columns in the selected group
            selected_group = group_df[["id", "name", "description", "types"]]
                
            # Format the content
            entity_candidates = ""
            for index, row in selected_group.iterrows():
                # if row["types"] has " and replace it with '
                row["types"] = row["types"].replace('"', "'")
                # if row['description'] has " and replace it with ' if it is a string
                if isinstance(row['description'], str):
                    row['description'] = row['description'].replace('"', "'")
                # if row['name'] has " and replace it with ' if it is a string
                if isinstance(row['name'], str):
                    row['name'] = row['name'].replace('"', "'")
                if row["types"] == "[]":
                    entity_candidates += f"<[ID] {row['id']} {row['name']} [DESCRIPTION] {row['description']} [TYPE] None>,"
                # iterate over each type
                else:
                    entity_candidates += f"<[ID] {row['id']} {row['name']} [DESCRIPTION] {row['description']} [TYPE] {row['types']}>,"
            # Save the content as REFERENT ENTITY CANDIDATES
            REFERENT_ENTITY_CANDIDATES = entity_candidates
            WIKI_OUTPUT = wikidata.run(X)
            try:
                WIKI_OUTPUT = wikidata.run(X)
                # format the WIKI_OUTPUT to JSON without indentation
                WIKI_OUTPUT = json.dumps(WIKI_OUTPUT, indent=0)
                # check if WIKI_OUTPUT contains Result Q6831776:\nLabel:
                pattern = r"Result Q\d+:\nLabel:"
                if not re.search(pattern, WIKI_OUTPUT):
                    WIKI_OUTPUT = REFERENT_ENTITY_CANDIDATES
            except:
                WIKI_OUTPUT = REFERENT_ENTITY_CANDIDATES
            

            #print(WIKI_OUTPUT)
            QID = group_df[group_df["target"] == 1]["id"]
            # extract the value starting from Q to space
            if QID.empty:
                QID = "0"
            else:
                QID = re.findall(r'Q\d+', QID.iloc[0])
    
            
            #prompt0 = f"Given the table title \"{TABLE_TITLE}\", summary \"{TABLE_SUMMARY}\", row \"{TABLE_ROW}\", along with referent entity candidates {WIKI_OUTPUT} from wikidata, what is ID of the correct entity for {X}. Provide Only the ID of the correct entity {X}."
            prompt0 = f"Link the entity mention \"{X}\" from the table cell to its Wikidata entity. Given the table \"{TABLE_ROW}\", along with referent entity candidates {WIKI_OUTPUT} from wikidata, identify the correct referent entity candidate for {X}."
            #print("PROMPT", prompt0)
            response = client.chat(model='mistral', messages=[
            {
            'role': 'user',
            'content': prompt0,
            },
            ])
            content_0 = response["message"]["content"]
            
            #print("OUTPUT", content_0)
            #extract Q9626>  from the content2
            try:
                qid_0 =re.findall(r'Q\d+>', content_0)
                # if the list is empty, then look for Q\d+ in the content
                if not qid_0:
                    qid_0 = re.findall(r'Q\d+', content_0)[0]
                # else remove > from the string
                else:
                    qid_0 = qid_0[0][:-1]
            except:
                qid_0 = "0"
            print("Correct:",QID, "Output", qid_0)


            # Save the results in the dataframe
            df = pd.DataFrame({"tableName": group_df["tableName"].iloc[0], "key": group_df["key"].iloc[0], "group": group_name, "prompt":prompt0, "output":content_0, "correct QID": QID, "output QID":qid_0},index=[0])
            results = [results, df]
            # convert the list to a dataframe
            results = pd.concat(results)
            print("end time: ", time.time())
            

            # Save the results to a new csv file
            results.to_csv(f"../../../../projects/nn10072k/EMD/data/Dataset/{dataset}/{dataset}_RAG_results_{iter}_v1.csv", index=False)

    
if __name__ == "__main__":
    main()
 

