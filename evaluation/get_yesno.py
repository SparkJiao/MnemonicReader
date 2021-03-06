import json
from sys import argv


def main(input_file1, input_file2, output_file):
    with open(input_file1, 'r') as f:
        yesno_pred = list()
        data1 = json.load(f)
    with open(input_file2, 'r') as f:
        data2 = json.load(f)['data']
        yesno_sou = list()
    i = 0
    for article in data2:
        id = article['id']
        turn_id = article['questions'][0]['turn_id']
        input_text = article['answers'][0]['input_text'].strip().replace('\n', '').lower()
        if input_text == 'yes':
            yesno_sou.append('y')
        elif input_text == 'no':
            yesno_sou.append('n')
        else:
            yesno_sou.append('x')
        for j in range(i, len(data1), 1):
            if data1[j]['id'] == id and data1[j]['turn_id'] == turn_id:
                yesno_pred.append(data1[j]['yesno'])
                i = j + 1
                break

    y_tp = 0
    y_fp = 0
    y_tn = 0
    y_fn = 0
    n_tp = n_fp = n_tn = n_fn = 0
    x_tp = x_fp = x_tn = x_fn = 0
    for a, b in zip(yesno_pred, yesno_sou):
        if a == 'y' and b == 'y':
            y_tp += 1
        elif a == 'y' and (b == 'n' or b == 'x'):
            y_fp += 1
        elif (a == 'n' or a == 'x') and (b == 'n' or b == 'x'):
            y_tn += 1
        elif (a == 'n' or a == 'x') and b == 'y':
            y_fn += 1
        if a == 'n' and b == 'n':
            n_tp += 1
        elif a == 'n' and (b == 'y' or b == 'x'):
            n_fp += 1
        elif (a == 'y' or a == 'x') and (b == 'y' or b == 'x'):
            n_tn += 1
        elif (a == 'y' or a == 'x') and b == 'n':
            n_fn += 1
        if a == 'x' and b == 'x':
            x_tp += 1
        elif a == 'x' and (b == 'y' or b == 'n'):
            x_fp += 1
        elif (a == 'y' or a == 'n') and (b == 'y' or b == 'n'):
            x_tn += 1
        elif (a == 'y' or a == 'n') and b == 'x':
            x_fn += 1
    y_recall = y_tp * 1.0 / (y_tp + y_fn)
    y_precision = y_tp * 1.0 / (y_tp + y_fp)
    y_f1 = 2 * y_recall * y_precision / (y_recall + y_precision)
    n_recall = n_tp * 1.0 / (n_tp + n_fn)
    n_precision = n_tp * 1.0 / (n_tp + n_fp)
    n_f1 = 2 * n_recall * n_precision / (n_recall + n_precision)
    x_recall = x_tp * 1.0 / (x_tp + x_fn)
    x_precision = x_tp * 1.0 / (x_tp + x_fp)
    x_f1 = 2 * x_recall * x_precision / (x_recall + x_precision)
    f1 = {"yes": {"recall": y_recall, "precision": y_precision, "f1": y_f1},
          "no": {"recall": n_recall, "precision": n_precision, "f1": n_f1},
          "x": {"recall": x_recall, "precision": x_precision, "f1": x_f1}}
    with open(output_file, 'w') as f:
        json.dump(f1, f, indent=4)


if __name__ == '__main__':
    input_file1 = argv[1]
    input_file2 = argv[2]
    output_file = argv[3]
    main(input_file1, input_file2, output_file)
