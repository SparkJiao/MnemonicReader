import json
from sys import argv


def main(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
        output = list()
        for key in data:
            value = data[key]
            keys = key.split('-')
            if len(keys) == 2:
                id = keys[0]
                turn_id = int(keys[1])
                answer = value
                output.append({'id': id, 'turn_id': turn_id, 'answer': answer})
            elif len(keys) == 3:
                assert keys[2] == 'yesno'
                assert keys[0] == output[-1]['id']
                assert int(keys[1]) == output[-1]['turn_id']
                # if value == 'y':
                #     output[-1]['answer'] = 'yes'
                # elif value == 'n':
                #     output[-1]['answer'] = 'no'
                output[-1]['yesno'] = value
            else:
                raise RuntimeError("Bad key: %s", key)
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=4)


if __name__ == '__main__':
    input_file = argv[1]
    output_file = argv[2]
    main(input_file, output_file)
