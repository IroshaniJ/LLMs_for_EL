import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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
import argparse
from peft import PeftModel

  
def main():

    # Initialize the parser
    parser = argparse.ArgumentParser(description="Process some inputs.")

    

    tables = "../../../../data/iroshanij/entitylinking/Dataset/Turl/tables/"
    list_of_tables = "../../../../data/iroshanij/entitylinking/Dataset/Turl/gt/cea_gt.csv"
    dataset_desc = "../../../../data/iroshanij/entitylinking/Datasets_Desc/Turl.csv"
    table_title_summary = pd.read_csv("../../../../data/iroshanij/entitylinking/tables_with_title_summary/Turl.csv")


    # Using arguments
    print(f"Dataset: {tables}")
    print(f"Dataset_Desc: {dataset_desc}")
 
    # Read the dataset
    data_desc = pd.read_csv(dataset_desc)

    # filter the data_desc to get only the rows with tableName in list_of_tables
    # read the csv file to get the list of tables and get the unique values in 1st column
    list_of_tables = pd.read_csv(list_of_tables)
    list_of_tables = list_of_tables.iloc[:, 0].unique()
    # filter the data_desc to get only the rows with tableName in list_of_tables
    data_desc = data_desc[data_desc["tableName"].isin(list_of_tables)]


    df = data_desc
    new_df = pd.DataFrame()
    grouped = df.groupby("group")
  
    wikidata = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())

    base_model_id = "mistralai/Mistral-7B-v0.1"
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Mistral, same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
    )

    eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)
    ft_model = PeftModel.from_pretrained(base_model, "mistral-journal-finetune/checkpoint-50")
   
    # Iterate over each group
    for iter in range(3):
        # save results in a new dataframe with columns tableName,key, group, correct QID, output QID
        results = pd.DataFrame(columns=["tableName", "key", "group","prompt", "output", "correct QID", "output QID",])
        count = 0
        for group_name, group_df in grouped:
            
            key_parts = group_df['key'].apply(lambda x: pd.Series(x.split('-')[-2:]))
            table_name = key_parts[0].strip().replace('s/+',"")

            table_file = f"{tables}/{table_name}.csv"
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
            #WIKI_OUTPUT = wikidata.run(X)
            try:
                WIKI_OUTPUT = wikidata.run(X)
                # format the WIKI_OUTPUT to JSON without indentation
                WIKI_OUTPUT = json.dumps(WIKI_OUTPUT, indent=0)
                # check if WIKI_OUTPUT contains Result Q6831776:\nLabel:
                #pattern = r"Result Q\d+:\nLabel:"
                #if not re.search(pattern, WIKI_OUTPUT):
                #    WIKI_OUTPUT = ""
            except:
                WIKI_OUTPUT = ""
            

            print(WIKI_OUTPUT)
            QID = group_df[group_df["target"] == 1]["id"]
            # extract the value starting from Q to space
            if QID.empty:
                QID = "0"
            else:
                QID = re.findall(r'Q\d+', QID.iloc[0])
    
            
            prompt0 = f"### Question: Given the table title \"{TABLE_TITLE}\", summary \"{TABLE_SUMMARY}\", row \"{TABLE_ROW}\", along with referent entity candidates {REFERENT_ENTITY_CANDIDATES} and wikidata {WIKI_OUTPUT}, identify the correct referent entity candidate for {X}. ### Answer:"
        
            #prompt0 = f"Link the entity mention \"{X}\" from the table cell to its Wikidata entity. Given the table \"{TABLE_ROW}\", along with referent entity candidates {REFERENT_ENTITY_CANDIDATES} and wikidata {WIKI_OUTPUT}, identify the correct referent entity candidate for {X}."
            #print("PROMPT", prompt0)
            model_input = eval_tokenizer(prompt0, return_tensors="pt").to("cuda")
            ft_model.eval()
            with torch.no_grad():
                response = eval_tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=100, repetition_penalty=1.15)[0], skip_special_tokens=True)
                print(response)
                # Extract the content from  ### Answer: in response
                # Define the marker from where you want to start the extraction
                marker = "### Answer:"

                # Find the position of the marker in the string
                marker_position = response.find(marker)

                # Check if the marker is found in the string
                if marker_position != -1:
                    # Extract everything after the marker
                    content_0 = response[marker_position + len(marker):].strip()
                else:
                    content_0 = "Marker not found in the string."
                            
            print("OUTPUT", content_0)
            
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
            results.to_csv(f"../../../../data/iroshanij/entitylinking/Dataset/Turl/turl_RAG_results_{iter}_ts_ft.csv", index=False)

    
if __name__ == "__main__":
    main()
 

