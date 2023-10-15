import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def read_json(file_name, folder_name="result"):
    file_path = os.path.join(folder_name, file_name)
    with open(file_path,'r', encoding="utf-8") as f:
        label = json.load(f)
        return label




def main() -> int:
    # output_data_json_path = "cl7_example_description_oneprompt_According to the above example, answer the following question with only 'Yes' or 'No' - is there any code issue of 'Fetch the whole entity only to check existence' in the following codes.json"
    output_data_json_path = "cl7_test_result.json"
    groundtruth_data_json_path = "test_data_groundtruth.json"
    output_data_dict = read_json(output_data_json_path)
    groundtruth_data_dict = read_json(groundtruth_data_json_path, folder_name=".")

    pre = []
    gt = []
    length = len(output_data_dict)
    black_space_for_codellama = "  "
    if not output_data_json_path.startswith('cl'):
        black_space_for_codellama=""
    for i in range(length):
        # print(output_data_dict[str(i+1)])
        # print(groundtruth_data_dict[str(i+1)])

        if output_data_dict[str(i+1)]["question_prompt"]+black_space_for_codellama in output_data_dict[str(i+1)]["response"]:
            answer = output_data_dict[str(i+1)]["response"].replace(output_data_dict[str(i+1)]["question_prompt"]+black_space_for_codellama,'')
            # print(answer)
            answer_lower = answer.lower()
            if answer_lower.startswith("yes"):
                pre.append("T")
            elif answer_lower.startswith("no"):
                pre.append("F")
            else:
                raise Exception("Not answer withn yes or no!!!!!")
            gt.append(groundtruth_data_dict[str(i+1)]["index"])
        else:
            raise Exception("Question prompt not in generation!!!!!")

    print(pre)
    print(gt)

    c_matrix = confusion_matrix(gt, pre)
    tn, fp, fn, tp = confusion_matrix(gt, pre).ravel()
    acc=(tp+tn)/(tn+tp+fn+fp)
    print(f"c_matrix:\n{c_matrix}")
    print("tp:",tp)
    print("tn:",tn)
    print("fp:",fp)
    print("fn:",fn)
    print("acc:",acc)
    # ConfusionMatrixDisplay.from_predictions(gt, pre)
    # plt.show()
    


if __name__ == '__main__':
  main()


