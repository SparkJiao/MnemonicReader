import json
from sys import argv


def main(input_file1, input_file2, output_file):
    answers = list()
    with open(input_file1, 'r') as f:
        data = json.load(f)
        for key in data:
            answers.append(data[key])
    with open(input_file2, 'r') as f:
        data = json.load(f)
        for i, item in enumerate(data):
            item['answer'] = answers[i]
            data[i] = item
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == '__main__':
    input_file1 = argv[1]
    input_file2 = argv[2]
    output_file = argv[3]
    main(input_file1, input_file2, output_file)
