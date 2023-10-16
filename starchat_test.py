import os
# import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    # BitsAndBytesConfig,
    # HfArgumentParser,
    # TrainingArguments,
    pipeline,
    # logging,
)
import argparse
from tqdm import tqdm
import json


os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,3"
# os.environ['CUDA_VISIBLE_DEVICES'] = "5"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,3"

# DEFAULT_SYSTEM_PROMPT = "Below is a dialogue between a human user and an AI assistant. The assistant is happy to help with almost anything, and will do its best to understand exactly what is needed."

# DEFAULT_SYSTEM_PROMPT = """\
# You are a helpful, respectful and honest assistant with a deep knowledge of code and software design. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
# """

# DEFAULT_SYSTEM_PROMPT = """\
# You are a helpful, respectful and honest assistant with a deep knowledge of code and software design. Always answer as helpfully as possible.\
# """

DEFAULT_SYSTEM_PROMPT = ""


def parse_args():
    parse = argparse.ArgumentParser(description='llms test')
    parse.add_argument('-m', '--model', default="HuggingFaceH4/starchat-beta", type=str, help='Model Name')
    parse.add_argument('-t', '--test_data', default="test_data_groundtruth.json", type=str, help='Test data file path')
    parse.add_argument('-e', '--example_data', default="example_data_groundtruth.json", type=str, help='Example data file path')
    parse.add_argument('-n', '--num_ex', default="1", type=int, help='Number of code examples provided')
    parse.add_argument('-o', '--output', default="starchat_test_result", type=str, help='Output file name')

    args = parse.parse_args()
    return args

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

##prompt
def get_starchat_prompt(chat_history:str, usr_input: str, index_num: int) -> str:
  system_token: str = "<|system|>"
  user_token: str = "<|user|>"
  assistant_token: str = "<|assistant|>"
  end_token: str = "<|end|>"
  if index_num==1:
    prompt = system_token + "\n" + DEFAULT_SYSTEM_PROMPT + end_token + "\n"
    prompt += user_token + "\n" + usr_input + end_token + "\n"
    prompt += assistant_token + "\n"
    return prompt
  else:
    prompt = chat_history + end_token + "\n"
    prompt += user_token + "\n" + usr_input + end_token + "\n"
    prompt += assistant_token + "\n"
    return prompt








def main(model_name, test_data_json_path, example_data_json_path, output_name, ex_num) -> int:

  # Load base model
  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      device_map="auto",
      max_memory={0: "27GiB", 1: "24GiB", 3: "24GiB"}
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

  pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=800)

  # ##### Memory Test
  # usr1 = "remember the variable: index=8"
  # usr2 = "what is the value of the variable 'index'?"
  # usr_inputs=[usr1,usr2]
  # chat_history = ""
  # index_num=0
  # for usr_input in usr_inputs:
  #   index_num+=1
  #   prompt_temp = get_starchat_prompt(chat_history, usr_input, index_num)
  #   # response_result = pipe(prompt_temp)
  #   # response_result = pipe(prompt_temp, max_new_tokens=256, do_sample=True, temperature=0.2, top_k=50, top_p=0.95, eos_token_id=49155)
  #   response_result = pipe(prompt_temp, do_sample=True, temperature=0.2, top_k=50, top_p=0.95, eos_token_id=49155)
  #   print("\nindex_num:\n", index_num, "\n")
  #   chat_history = response_result[0]['generated_text']
  #   print(chat_history)



  # import pdb
  # pdb.set_trace()


  ##### Test Dataset Example Dataset

  test_data_dict = read_json(test_data_json_path)
  example_data_dict = read_json(example_data_json_path)

  num=0
  test_text=""
  prompt_question_response_dict = {}
  # response_list = []
  for (k,ele) in tqdm(test_data_dict.items()):
    num+=1
    codesample = ele["code"]

    ##### Prompt format and list
    prompt = f"The following {str(ex_num)} code examples contan a code issue of 'Fetch the whole entity only to check existence':\n"
    for i in range(ex_num):
      prompt += example_data_dict[str(i+1)]["code"] + "\n"
    prompt += "\nAccording to the above examples, answer the following question with only 'Yes' or 'No' - is there any code issue of 'Fetch the whole entity only to check existence' in the following codes?\n" + codesample
    usr_inputs=[prompt]

    chat_history = ""
    index_num = 0
    prompt_history=[]
    for usr_input in usr_inputs:
      index_num+=1
      prompt_temp = get_starchat_prompt(chat_history, usr_input, index_num)
      prompt_history.append(prompt_temp)
      # response_result = pipe(prompt_temp, max_new_tokens=256, do_sample=True, temperature=0.2, top_k=50, top_p=0.95, eos_token_id=49155)
      response_result = pipe(prompt_temp, do_sample=True, temperature=0.2, top_k=50, top_p=0.95, eos_token_id=49155)
      chat_history = response_result[0]['generated_text']
      str_num = "\n NUMBER: " +str(k) + "\n"
      # print(chat_history)
    output_dict = {
      "question_prompt": prompt_history[-1],
      "response": chat_history
    }
    prompt_question_response_dict.update({k:output_dict})
    test_text+=str_num
    test_text+=chat_history
    print("\n k NUMBER: ", str_num, "\n")
    # print(chat_history)
  write_txt(test_text, output_name+".txt")
  write_json(prompt_question_response_dict, output_name+".json")


if __name__ == '__main__':
  args = parse_args()
  main(args.model, args.test_data, args.example_data, args.output, args.num_ex)