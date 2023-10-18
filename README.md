# llms_test
code_review



# Requirements

`pip install git+https://github.com/huggingface/transformers.git@main accelerate`

`pip install optimum`

`pip install auto-gptq`


## How To Use

Run codellama_test.py for the test of CodeLlama

Run starchat_test.py for the test of Starchat

Bothe files generate output results in "result" folder 

Run llms_test_matrix.py to generate Confusion Matrix and Accuracy

### Example

`python codellama_test.py -s 1 -n 1 -o cl7_test_result_sys1_1p1ex`

`python llms_test_matrix.py -p cl7_test_result_sys1_1p1ex.json -g test_data_groundtruth.json`

