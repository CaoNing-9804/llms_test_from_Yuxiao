import os
# import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    # BitsAndBytesConfig,
    # HfArgumentParser,
    pipeline,
    # logging,
)

import json

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,3"
# os.environ['CUDA_VISIBLE_DEVICES'] = "5"
os.environ['CUDA_VISIBLE_DEVICES'] = "1,3,4"

# DEFAULT_SYSTEM_PROMPT = """\
# You are a helpful, respectful and honest assistant with a deep knowledge of code and software design. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
# """

# DEFAULT_SYSTEM_PROMPT = """\
# You are a helpful, respectful and honest assistant with a deep knowledge of code and software design. Always answer as helpfully as possible.\
# """

DEFAULT_SYSTEM_PROMPT = ""




def read_json(file_path):
  with open(file_path,'r', encoding="utf-8") as f:
    label = json.load(f)
    return label

def write_json(new_data, file_name, folder_name = "result"):
  if not os.path.exists(folder_name):
    os.makedirs(folder_name)
  json_path = os.path.join(folder_name, file_name)

  with open(json_path,'w+', encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False)    
  print('Total write ', len(new_data))

def write_txt(data, file_name, folder_name = "result"):
  if not os.path.exists(folder_name):
    os.makedirs(folder_name)
  text_path = os.path.join(folder_name, file_name)
  with open(text_path, "w", encoding='utf-8-sig') as file:
    file.write(data)

def get_prompt(chat_history:str, usr_input: str, index_num: int) -> str:

  if index_num==1:
    texts = [f'<s>[INST] <<SYS>>\n{DEFAULT_SYSTEM_PROMPT}\n<</SYS>>\n\n']
    # message = message.strip() if do_strip else message
    texts.append(f'{usr_input} [/INST]')
    return ''.join(texts)
  else:
    texts = [f'{chat_history.strip()} </s><s>[INST] ']
    texts.append(f'{usr_input} [/INST]')
    return ''.join(texts)






def main() -> int:

  ##### Test Model Name
  model_name = "codellama/CodeLlama-7b-Instruct-hf"
  # model_name = "TheBloke/CodeLlama-7B-Instruct-GPTQ"

  # Load base model
  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      device_map="auto",
      # max_memory={0: "29GiB", 3: "29GiB"}
      # max_memory={5: "29GiB"}
      max_memory={1: "24GiB", 3: "25GiB", 4: "18GiB",}
  )

  ##### Device map check
  print("\n\n\ndevice_map:\n", model.hf_device_map)

  #test
  model.config.use_cache = False
  model.config.pretraining_tp = 1

  # Load LLaMA tokenizer
  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "right"

  # pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=800)
  pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=4096)

  ##### Memory Test
  # usr1 = "remember the variable: index=8"
  # usr2 = "what is the value of the variable 'index'?"
  # usr_inputs=[usr1,usr2]
  # index_num = 0
  # chat_history = ""
  # for usr_input in usr_inputs:
  #   index_num+=1
  #   prompt_temp = get_prompt(chat_history, usr_input, index_num)
  #   response_result = pipe(prompt_temp)
  #   print("\nindex_num:\n", index_num, "\n")
  #   chat_history = response_result[0]['generated_text']
  #   print(chat_history)




  # import pdb
  # pdb.set_trace()




  ##### Test Dataset
  test_data_json_path = "test_data_groundtruth.json"
  test_data_dict = read_json(test_data_json_path)
  ##### Example Dataset
  example_data_json_path = "example_data_groundtruth.json"
  example_data_dict = read_json(example_data_json_path)

  num=0
  test_text=""
  prompt_question_response_dict = {}
  # response_list = []
  for (k,ele) in test_data_dict.items():
    num+=1
    codesample = ele["code"]

    ##### Prompt format and list
    prompt = "The following code contaning a code issue of 'Fetch the whole entity only to check existence':\n" + example_data_dict["1"]["code"]
    prompt += "\nAccording to the above example, answer the following question with only 'Yes' or 'No' - is there any code issue of 'Fetch the whole entity only to check existence' in the following codes?\n" + codesample
    usr_inputs=[prompt]

    chat_history = ""
    index_num = 0
    prompt_history=[]
    for usr_input in usr_inputs:
      index_num+=1
      prompt_temp = get_prompt(chat_history, usr_input, index_num)
      prompt_history.append(prompt_temp)
      response_result = pipe(prompt_temp)
      chat_history = response_result[0]['generated_text']
      str_num = "\n NUMBER: " +str(k) + "\n"
    output_dict = {
      "question_prompt": prompt_history[-1],
      "response": chat_history
    }
    prompt_question_response_dict.update({k:output_dict})
    test_text+=str_num
    test_text+=chat_history
    print("\n NUMBER: ", num, "\n")
    # print(chat_history)
  write_txt(test_text, "cl7_test_result.txt")
  write_json(prompt_question_response_dict, "cl7_test_result.json")

  

if __name__ == '__main__':
  main()