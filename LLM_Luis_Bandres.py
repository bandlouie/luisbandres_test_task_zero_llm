#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import io
import json
import markdown
import warnings
import pandas as pd
import gradio as gr
import IPython.display
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationTokenBufferMemory
from langchain.chains import SequentialChain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain


# In[2]:


warnings.filterwarnings('ignore')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file


# In[3]:


llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.0)


# ## LLM Mapping Chain
# 
# It doesn't need history for completion. Therefore, is token-efficiente and it doesn't require the memory of GPT.

# In[4]:


load_template_file_prompt = ChatPromptTemplate.from_template("""

You will be provided with a table in a markdown format as << INPUT >>.

If there is not a markdown table in the input return an empty JSON object. Otherwise, return a JSON object formatted to look like:

<< FORMATTING >>
{{{{
    "template_metadata": [
        {{{{
            "header": string, \ name of the column. If the input does not contain a header suggest a name for the column based on its data.
            "type": string, \ type of the data column of the markdown file in the input. 
            "sample": \ put a not null samble of the column. This sample should have the most common value which is not null.
            "categorical": bool \ check if the column is categorical (true) or not categorical (false).
            "categories_list": [] \ list of the unique values if the column is categorical.
            "date_format": null except if type is date suggest SQL DATE FORMAT for converting the column values to date.
            "description": string \ descrption of this column based only on its data.
        }}}},
        ...
    ]
}}}}


<< INPUT >>
{input_template}

<< OUTPUT >>
""")


# In[5]:


load_file_prompt = ChatPromptTemplate.from_template("""

You will be provided with a table in a markdown format as << INPUT >>.
Also you will provided with the name of the table as << TABLE >>

If there is not a markdown table in the input return an empty JSON object. Otherwise, return a JSON object formatted to look like:

<< FORMATTING >>
{{{{ 
    "table_name": string, \ name of the table. you can find it at the begining of the input.
    "file_metadata": [
        {{{{
            "header": string, \ name of the column. If the input does not contain a header suggest a name for the column based on its data.
            "type": string, \ type of the data column of the markdown file in the input. 
            "sample": \ put a not null samble of the column. This sample should have the most common value which is not null.
            "date_format": string, \ null except if type is date suggest SQL DATE FORMAT for converting the column values to date.
            "description": string \ description of this column based only on its data.
        }}}},
        ...
    ],
    "table" : string \ the complete markdown table in the input
}}}}

<< TABLE >>
{table_name}

<< INPUT >>
{input_file}

<< OUTPUT >>
Only return a JSON Object, no more!! The only valid output is a JSON Object.

""")


# In[6]:


formating_header_prompt = ChatPromptTemplate.from_template("""
You will receive two JSON Objects called table_info and template_description as inputs << INPUT >.

Follow the following instruction:

    Step 1: Go over the list table_info['file_metadata'] and find what is the N header in the list template_description['template_metadata'] most similar to the "new_file" table header.
    Step 2: return the same "table_info" JSON object adding the following information:

"header_match": [
        {{{{
            "table_header": string, \ header of "new_file" table most similar to the N header of "template" table
            "template_header": string, \ header of "template" table
        }}}},
        ...
    ],

"template_header" is the N header of "template" table most similar to the correspoding header of "new_file" table. Determine this similarity based only on the metadata such as:
    * Data types
    * Samples of both tables
    * Description

<< INPUT >>
{table_info}
{template_description}

<< OUTPUT >>
Only return a JSON Object, no more!! The only valid output is a JSON Object.

""")


# In[7]:


table_proposal_prompt = ChatPromptTemplate.from_template("""
You will receive two JSON Objects called table_header_match and template_description as inputs << INPUT >.

Follow the next instructions for generating a markdown table:
    
    Step 1: Go over the list table_header_match["header_match"].
    Step 2: For each one of the table_header_match["header_match"]["table_header"], replace the header of the markdown table in table_header_match["table"] by table_header_match["header_match"]["template_header"]
    Step 3: The new markdown table must have only the columns in listed in table_header_match["header_match"]["template_header"]. Remove all the remaining columns different to table_header_match["header_match"]["template_header"].

Return the new markdown table in a JSON Object formatted to look like:

<< FORMATTING >>
{{{{ 
    "table_name": string, \ name of the table. you can find it in table_header_match["table_name"].
    "file_metadata": [
        {{{{
            "header": string, \ name of the column.
            "type": string, \ type of the data column of the markdown file in the input. 
            "sample": \ put a not null samble of the column. This sample should have the most common value which is not null.
            "categorical": bool \ check if the column is categorical (true) or not categorical (false).
            "categories_list": [] \ list of the unique values if the column is categorical.
            "date_format": null except if type is date suggest SQL DATE FORMAT for converting the column values to date.
            "description": string \ descrption of this column based only on its data.
        }}}},
        ...
    ],
    "modified_table" : string, \ the new markdown table.
    "template_metadata": template_description["template_metadata"]  \ the template metadata.
}}}}

<< INPUT >>
{table_header_match}
{template_description}

<< OUTPUT >>
Only return a JSON Object, no more!! The only valid output is a JSON Object.

""")


# In[8]:


formating_categories_prompt = ChatPromptTemplate.from_template("""
You will receive a JSON Object called simple_table as input << INPUT >.

Based on the given input, the task is to find the column in the "modified_table" that is most similar to the N header in the "template_metadata" list. Then, select only the categorical columns and return the "simple_table" JSON object with the added information.

Follow the next instructions for generating a markdown table:
Step 1: compare the headers in the "modified_table" with the headers in the "template_metadata" list. We will  iterate over the columns in the "modified_table" and find the column that is most similar to the N header in the "template_metadata" list.
Step 2: After finding the most similar column, check if it is a categorical column by checking the "categorical" key in the "file_metadata" list.
Step 3: Add the following information to the "simple_table" JSON object.
Step 4: Return the updated  "simple_table" JSON object.

"categories_match": [
    {{{{
        "categories_list": string, \ list of categories in simple_table['template_metadata']
        "table_header": string, \ header of markdown table in simple_table["modified_table"] most similar to the N header of "template_metadata"
    }}}},
    ...
],

<< INPUT >>
{simple_table}

<< OUTPUT >>
Return the updated  "simple_table" JSON object. Only return a JSON Object, no more!! The only valid output is a JSON Object.
""")


# In[9]:


categories_result_prompt = ChatPromptTemplate.from_template("""
You will receive a JSON Object called table_categories_match as input << INPUT >.

Based on the given input, the task is to generate a markdown table by replacing each value in the categorical columns of the "modified_table" with the most similar item from the "categories_list" in the "categories_match" section. The updated table should be returned as a JSON object.

Follow the next instructions for generating a markdown table:
Step 1: Iterate over each categorical column in the "modified_table".
Step 2: Replace each value in the column with the most similar item from the "categories_list" in the "categories_match" section.
Setp 3: the new markdown table will be called correct_cats_markdown_table

Return the new markdown table (correct_cats_markdown_table) in a JSON Object formatted to look like:

<< FORMATTING >>
{{{{ 
    "table_name": string, \ name of the table. you can find it in table_categories_match["table_name"].
    "file_metadata": [
    {{{{
        "header": string, \ name of the column.
        "type": string, \ type of the data column of the markdown file in the input. 
        "sample": \ put a not null samble of the column. This sample should have the most common value which is not null.
        "categorical": bool \ check if the column is categorical (true) or not categorical (false).
        "categories_list": [] \ list of the unique values if the column is categorical.
        "date_format": null except if type is date suggest SQL DATE FORMAT for converting the column values to date.
        "description": string \ descrption of this column based only on its data.
    }}}},
    ...
    ],
    "table" : string, \ the new markdown table (correct_cats_markdown_table).
    "template_metadata": table_categories_match["template_metadata"]  \ the template metadata.
}}}}

<< INPUT >>
{table_categories_match}

<< OUTPUT >>
Return only the JSON Object. Only return a JSON Object, no more!! The only valid output is a JSON Object.

""")


# In[10]:


formating_dates_prompt = ChatPromptTemplate.from_template("""
You will receive a JSON Object called table_categories_result as input << INPUT >.

Change the format of each one the rows of date columns in the markdown in table_categories_result["table"] according to the date format in the list table_categories_result["template_metadata"]

The new markdown table will be called correct_dates_markdown_table

Return the new markdown table (correct_dates_markdown_table) in a JSON Object formatted to look like:

<< FORMATTING >>
{{{{ 
    "table_name": string, \ name of the table. you can find it in table_categories_result["table_name"].
    "table" : string, \ the new markdown table (correct_dates_markdown_table).
    "template_metadata": table_categories_result["template_metadata"]  \ the template metadata.
}}}}

<< INPUT >>
{table_categories_result}

<< OUTPUT >>
Return only the JSON Object. Only return a JSON Object, no more!! The only valid output is a JSON Object.

""")


# In[11]:


formating_strings_prompt = ChatPromptTemplate.from_template("""
You will receive a JSON Object called table_dates_result as input << INPUT >.

Based on the given input, the task is to find the column in the markdown "table" that is most similar to the N header in the "template_metadata" list. Then, select only the string columns and return the "table_dates_result" JSON object with the added information.

Follow the next instructions for generating a markdown table:
Step 1: compare the headers in the "table" with the headers in the "template_metadata" list. We will  iterate over the columns in the "table" and find the column that is most similar to the N header in the "template_metadata" list.
Step 2: After finding the most similar column, check if it is a string column by checking the "type" key in the "file_metadata" list.
Step 3: Ignore if it is a categorical, numerical or date column by checking the "categorical" key in the "file_metadata" list.
Step 4: Add the following information to the "table_dates_result" JSON object.
Step 5: Transform all the rows of string columns of markdown table so they look like than their columns in "template_metadata".
Step 5: Return the updated  "table_dates_result" JSON object.

"strings_match": [
    {{{{
        "selected_sample": string, \ sample of data in table_dates_result["template_metadata"]
        "table_header": string, \ header of markdown table in table_dates_result["table"] most similar to the N header of "template_metadata"
    }}}},
    ...
],

<< INPUT >>
{table_dates_result}

<< OUTPUT >>
Return the updated "table_dates_result" JSON object. Only return a JSON Object, no more!! The only valid output is a JSON Object.
""")


# In[12]:


chain_template_load = LLMChain(llm=llm, prompt=load_template_file_prompt, 
                     output_key="template_description"
                    )
chain_load = LLMChain(llm=llm, prompt=load_file_prompt, 
                     output_key="table_info"
                    )
chain_header_formatting = LLMChain(llm=llm, prompt=formating_header_prompt, 
                     output_key="table_header_match"
                    )
chain_proposal = LLMChain(llm=llm, prompt=table_proposal_prompt, 
                     output_key="simple_table"
                    )
chain_cats_formatting = LLMChain(llm=llm, prompt=formating_categories_prompt, 
                     output_key="table_categories_match"
                    )
chain_cats_result = LLMChain(llm=llm, prompt=categories_result_prompt, 
                     output_key="table_categories_result"
                    )
chain_dates_result = LLMChain(llm=llm, prompt=formating_dates_prompt, 
                     output_key="table_dates_result"
                    )
chain_strings_formatting = LLMChain(llm=llm, prompt=formating_strings_prompt, 
                     output_key="table_strings_match"
                    )


# In[13]:


mapping_chain = SequentialChain(
    chains=[chain_template_load, chain_load, chain_header_formatting, chain_proposal, chain_cats_formatting, chain_cats_result, chain_dates_result, chain_strings_formatting],
    input_variables=["input_template","table_name","input_file"],
    output_variables=["table_header_match","table_categories_match","table_strings_match"],
    verbose=True
)


# ## LLM Coding Chain
# 
# It doesn't need history for completion. Therefore, is token-efficiente and it doesn't require the memory of GPT.

# In[14]:


def extract_data_attributes(analysis_chain):
    return dict(
        # Inputs
        template_table = analysis_chain['input_template'],
        initial_table = analysis_chain['input_file'],
        # Metadata
        file_metadata = json.loads(analysis_chain['table_header_match'])['file_metadata'],
        template_metadata = json.loads(analysis_chain['table_categories_match'])['template_metadata'],
        # Feature Mappings
        header_match = json.loads(analysis_chain['table_header_match'])['header_match'],
        categories_match = json.loads(analysis_chain['table_categories_match'])['categories_match'],
        strings_match = json.loads(analysis_chain['table_strings_match'])['strings_match'],
        # Final Table
        final_table = json.loads(analysis_chain['table_strings_match'])['table']
    )

# def get_python_code_prompt():
#     global execution_chain
#     # try:
#     map_process = extract_data_attributes(execution_chain)
#     return f"""
#     Also, you will be provided with a initial_table in a markdown format as << INITIAL_TABLE >>. You need to transform the initial_table to the same format than  template table.
#     Finally, you will be provided with a template_table in a markdown format as << TEMPLATE_TABLE >>. The template_table is the expected result you need to achieve.

#     The objective is to create a python code for transforming the initial_table into template_table. You need to use the mapping from renaming columns of initial_table to making their headers the same than template_table.


#     Please follow the next 12 (twelve) steps for creating a python code:

#     STEP 1: Load initial_table using pandas 1.3.1 as a dataframe.
    
#     STEP 2: Run this step if and only if any "table_header" in "categories_list" << CATEGORIES REQUIRED >>> is in initial_table headers. Omit this step if "categories_list" is an empty list.
    
#     Replace each value in the strings column with the most similar item from the "categories_list" << CATEGORIES REQUIRED >>>. The python code needs to replace each value in the categorical columns of the dataframe with the most similar item from the "categories_list" in the << CATEGORIES REQUIRED >>>.
#     << CATEGORIES REQUIRED >>>
#     ```json
#     {json.dumps(map_process['categories_match'])}
#     ```
#     When calculate similarity, not use index [0] if difflib.get_close_matches() returns an empty list. In that case use the original category value.
#     You need to implement in this step a code for calculating similarities between strings. All resulting categories columns must be string columns. Considers the previous steps.
    
#     STEP 3: Rename columns using HEADERS MAPPING. New headers names are in "template_header" section.
#     << HEADERS MAPPING >>
#     ```json
#     {json.dumps(map_process['header_match'])}
#     ```
    
#     STEP 3.5: Run the same processes in STEP if and only if any "table_header" in "categories_list" << CATEGORIES REQUIRED >>> is in updated initial_table headers. Omit this step if "categories_list" is an empty list.
    
#     STEP 4: In the following steps only consider the new headers of the columns assigned in STEP 3. Also discard other columns.

#     STEP 5: Transform the date columns of inital_table to the same date format than template_table. Considers the previous steps.
    
#     STEP 6: Transform all the rows of string columns (except categories columns in STEP 3) of dataframe so they look like than their columns in template_table. Considers the previous steps.
    
#     STEP 7: Transform all the rows of columns (for serials) of dataframe so they look like than their columns (for serials) in template_table. Considers the previous steps.
    
#     STEP 8: Transform values in numeric columns of inital_table so they look like than their numeric columns in template_table. Considers the previous steps.

#     STEP 9: Read all the code and be sure that all required libraries in the code are correctly imported. Considers the previous steps.
    
#     STEP 10: Fix all syntaxis errors for python 3.9.
    
#     STEP 11: Remove any duplicated column.

#     STEP 12: Save the dataframe as csv file called "transformed_table".

#     Remember the objective is to reproduce the template_table using python 3.9 or above.

#     CONSTRAINTS
#     a) Code will receive only initial_table as a csv file.
#     b) Avoid inplae parameters in pandas transformations
#     c) The code need to save the transformed table as csv.
#     d) Categorical columns must be filled (not completely empty).
#     d) you need to test that produced result table is exactly the same than template_table provided.
#     e) You must return only a complete python script.
#     f) template_table is only a markdown that only exists in this prompt for your guidance. final table is not a file nor is it a dataframe.
#     g) template_table is ignored by the resulting python code.
#     h) fix all syntaxis error for python 3.9

#     << INITIAL_TABLE >>
#     {json.dumps(map_process['initial_table'])}

#     << TEMPLATE_TABLE >>
#     {json.dumps(map_process['final_table'])}

#     << OUTPUT >>
#     You must return only a complete python script. Please avoid make extra comments, I need only the python script.

#     """
#     # except Exception as e:
#     #     print(f"{e}")
#     #     return None
# In[92]:


def get_python_code_prompt():
    global execution_chain
    # try:
    map_process = extract_data_attributes(execution_chain)
    cat_list = sum([d['categories_list'] for d in map_process['categories_match']],[])
    cat_headers = ', '.join([d['table_header'] for d in map_process['categories_match'] if len(d['categories_list'])>0])
    return f"""
    You will be provided with a initial_table in a markdown format as << INITIAL_TABLE >>.
    You will be provided with a template_table in a markdown format as << TEMPLATE_TABLE >>.
    You will be provided with a JSON object with headers mapping as << HEADERS MAPPING >>>
    You will be provided with a list of allowed categories as << CATEGORIES ALLOWED >>>

    Create a python code for transforming the initial_table into template_table so initial_table will be indetical to template_table. Python Code must handle exceptions at each step: Python Code must end without errors.
    
    initial_table must be loaded from csv file as a dataframe of only strings using pandas 1.3.1. and python 3.9. change name to dataframe.
    
    template_table is only a markdown (is not a csv file) that only exists in this prompt as a guide.
    
    Headers must be renamed according to << HEADERS MAPPING >> 
    
    All the rows of columns of renamed dataframe must look like than their columns in template_table: must have the same punctuation and letter cases
    
    Transform all the rows of columns (for serials) of renamed dataframe so they look like than their columns (for serials) in template_table.
    
    Transform the string columns with dates in dataframe to have the same date format than template_table. Consider the previous steps.
    
    Replace each value in the categories columns ({cat_headers}) with the most similar (difflib.get_close_matches()) item from the list in << CATEGORIES ALLOWED >>>. When calculate similarity, not use index [0] if difflib.get_close_matches() returns an empty list. In that case use the original category value. All resulting categories columns must be string columns. The python code needs to replace each value in the categorical columns of the renamed dataframe with the most similar item from list in the << CATEGORIES ALLOWED >>>. All categories must be kept as strings always.
    
    Only keep the same columns than template_table.

    Save the dataframe as csv file called "transformed_table".

    << INITIAL_TABLE >>
    ```markdown
    {json.dumps(map_process['initial_table'])}
    ```

    << TEMPLATE_TABLE >>
    ```markdown
    {json.dumps(map_process['final_table'])}
    ```
        
    << HEADERS MAPPING >>
    ```json
    {json.dumps(map_process['header_match'])}
    ```
    
    << CATEGORIES ALLOWED >>>
    ```json
    {json.dumps(cat_list)}
    ```

    << OUTPUT >>
    You must return only a complete python script. Please avoid make extra comments, I need only the python script.

    """
    # except Exception as e:
    #     print(f"{e}")
    #     return None


# ## User Interface Functions

# In[17]:


def markdown_to_html(md_table_string):
    return markdown.markdown(md_table_string, extensions=['markdown.extensions.tables'])


# In[18]:


def process_csv(file, file_label):
    global _file_buffer
    
    # Read the uploaded CSV file with pandas
    df = pd.read_csv(io.StringIO(file.decode('utf-8')))
    
    # Convert the DataFrame to an HTML table with added styles
    html_table = df.to_html(classes='table table-striped')
    
    # Add CSS for scrollable table
    styled_table = f"""
    <div style="max-width: 100%; overflow-x: auto;">
        {html_table}
    </div>
    """
    _file_buffer[file_label] = df.to_markdown()
    
    return styled_table

def process_template(file):
    return process_csv(file, 'template')

def process_new_file(file):
    return process_csv(file, 'new_file')


# In[19]:


_file_buffer = {
    'template':'',
    'new_file':'',
}
execution_chain = None
def tables_analysis():
    global _file_buffer
    global execution_chain
    execution_chain = mapping_chain({
        "input_template":_file_buffer['template'],
        "table_name":"new_file",
        "input_file":_file_buffer['new_file']
    })
    result_chain = json.loads(execution_chain['table_strings_match'])
    show_table_html = markdown_to_html(result_chain['table'])
    system_context = f"""
This is the transformed data according to the template.

Transformed Data:

{result_chain['table']}

Template:

{execution_chain['input_template']}

Input File:

{execution_chain['input_file']}
    """
    return show_table_html, system_context
    


# In[20]:


anaylisis_check = False
def feedback_analysis(res):
    global anaylisis_check
    anaylisis_check = (res=='Yes')


# In[21]:


python_text = ''
is_new_chat = True
def generate_python_code():
    global python_text
    global is_new_chat
    if anaylisis_check:
        try:
            del python_code_conv
        except:
            pass
        try:
            # No uses memory
            python_code_conv = ConversationChain(
                llm=llm, 
                verbose=False
            )
            python_text = python_code_conv.predict(input=get_python_code_prompt())
            python_text = python_text.split('```python')[1].split('```')[0]
            is_new_chat = True
        except:
            python_text = ""
    else:
        python_text = "Please confirm the file was mapped correctly."
    return python_text


# In[22]:


python_code_check = False
def feedback_python_code(res):
    global anaylisis_check
    global python_code_check
    python_code_check = (res=='Yes')
    if anaylisis_check:
        if python_code_check:
            return "Python Code Valid!"
        else:
            return "Python Code Invalid!"
    else:
        return "Please confirm Analysis at Step 2."


# In[23]:


def download_python_code():
    global anaylisis_check
    global python_code_check
    global python_text
    if (anaylisis_check & python_code_check):
        # Save the string to a file
        filename = "output_python_code.py"
        with open(filename, 'w') as f:
            f.write(python_text)

        # Return the file path so Gradio can allow the user to download it
        return filename


# ### Chatbot functions
# There is a chatbot, here GPT memory handle the token usage.

# In[24]:


memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=4000)
bot_conversation = ConversationChain(
    llm=llm, 
    verbose=False,
    memory=memory
)


# In[25]:


def respond(message, chat_history, instruction, temperature=0.0):  
    global is_new_chat
    if is_new_chat:
        memory.save_context({"output": python_text})
        is_new_chat = False
    else:
        prompt = message
    bot_message = bot_conversation.predict(input=message)
    chat_history.append((message, bot_message))
    return "", chat_history


# In[26]:


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.HTML('<h1 align="center">Test Task Submission</h1>')
            gr.HTML('<h2 align="center">Luis Bandres</h2>')
            gr.HTML('<p align="center">Add description here</p>')
    with gr.Row():
        with gr.Column():
            # Load Data
            gr.HTML('<h2 align="center">Step 1: Load Data</h2>')
            
            upload_template = gr.inputs.File(type="bytes", label="Upload Template")
            data_template = gr.outputs.HTML(label="Template")
            upload_template.upload(process_template, inputs=upload_template, outputs=data_template)
            
            upload_file = gr.inputs.File(type="bytes", label="Upload New File")
            data_file = gr.outputs.HTML(label="New File")
            upload_file.upload(process_new_file, inputs=upload_file, outputs=data_file)

        with gr.Column():
            # Analyse Data
            gr.HTML('<h2 align="center">Step 3: Transform using LLM </h2>')
            gr.HTML('<h3 align="left">Dont leave this page while processing.</h3>')
            gr.HTML('<p align="left">This process could take 5 minutes approximately...</p>')
            btn_analyse = gr.Button("Transform Table")
            data_proposal = gr.outputs.HTML(label="Data Mapping Result")
            chk_analysis = gr.Radio(["Yes", "No"], label="Data was mapped correctly?")

            # Generating Code
            gr.HTML('<h2 align="center">Step 4: Generate Python Code </h2>')
            btn_python_code = gr.Button("Generate")
            text_python_code = gr.Textbox(value="Please Generate Python Code",label="Python Code")
            chk_python_code = gr.Radio(["Yes", "No"], label="Python code is correct?")
            
            # Edit Code
            gr.HTML('<h2 align="center">Step 5: Download Python Code </h2>')
            gr.HTML('<h3 align="left">Requisites:</h3>')
            gr.HTML('<p align="left">   * Step 2 must be confirmed.</p>')
            gr.HTML('<p align="left">   * Step 3 must be confirmed.</p>')
            python_code_result = gr.outputs.HTML(label="Result Python Code")
            btn_download_python = gr.Button("Save Code")
            download_python = gr.outputs.File(label="Generated Python Code")
    
    with gr.Row():    
        with gr.Column():
            # Chatbot
            gr.HTML('<h2 align="center">Step 6: Advanced Options </h2>')
            gr.HTML('<b align="left">This is a chatbot powered by OpenAI GPT 3.5 so it can help you to editing the code with AI Assistance. The Table Analysis and the Generated Python Code have been loaded to this assistant.</p>')
            chatbot = gr.Chatbot(height=446, label='Chatbot') #just to fit the notebook
            msg = gr.Textbox(label="Prompt")
            with gr.Accordion(label="Settings",open=False):
                system_context = gr.Textbox(label="System Context", lines=2, value="A conversation between a user and an LLM-based AI python coding assistant. The assistant gives helpful, honest, and precise answers. The assistant must act as a programmer.")
            btn = gr.Button("Submit")
            clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")
            
    # Actions
    btn_analyse.click(tables_analysis, inputs=None, outputs=[data_proposal,system_context])
    chk_analysis.change(feedback_analysis,inputs=chk_analysis, outputs=None)

    btn_python_code.click(generate_python_code,inputs=None,outputs=text_python_code)
    chk_python_code.change(feedback_python_code,inputs=chk_python_code, outputs=python_code_result)

    btn_download_python.click(download_python_code,inputs=None,outputs=download_python)

    btn.click(respond, inputs=[msg, chatbot, system_context], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot, system_context], outputs=[msg, chatbot]) #Press enter to submit

gr.close_all()
demo.queue().launch(share=False, server_port=int(os.environ['GRADIO_SERVER_PORT']))


# ## Generating Examples for Further Training GPT Model

# In[51]:


import pickle


# In[93]:


number_task = '12'

with open(f'./additional_task/training_data/sample_{number_task}.pickle', 'wb') as handle:
    pickle.dump({
        'prompt':get_python_code_prompt(),
        'completion':python_text
    }, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[94]:


with open(f'./additional_task/training_data/sample_{number_task}.pickle', 'rb') as handle:
    bb = pickle.load(handle)


# In[95]:


print(bb['prompt'])


# In[96]:


print(bb['completion'])


# ## END

# In[ ]:





# In[ ]:




