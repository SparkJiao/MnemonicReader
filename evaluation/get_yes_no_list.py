import json
from sys import argv


def main(input_file1, input_file2, output_file):
    with open(input_file1, 'r') as f:
        yesno_pred = list()
        data = json.load(f)
        for key in data:
            yesno_pred.append(key['yesno'])
    with open(input_file2, 'r') as f:
        yesno_sou = list()
        data = json.load(f)['data']
        for article in data:
            answers = article['answers']
            for answer in answers:
                input_text = answer['input_text'].strip().replace('\n', '').lower()
                if input_text == 'yes':
                    yesno_sou.append('y')
                elif input_text == 'no':
                    yesno_sou.append('n')
                else:
                    yesno_sou.append('x')
    y_tp = 0
    y_fp = 0
    y_tn = 0
    y_fn = 0
    n_tp = n_fp = n_tn = n_fn = 0
    x_tp = x_fp = x_tn = x_fn = 0
    for a, b in zip(yesno_pred, yesno_sou):
        # yes
        if a == 'y' and b == 'y':
            y_tp += 1
        elif a == 'y' and (b == 'n' or b == 'x'):
            y_fp += 1
        elif (a == 'n' and b == 'n') or (a == 'x' and b == 'x'):
            y_tn += 1
        elif (a == 'n' or a == 'x') and b == 'y':
            y_fn += 1
        # no
        if a == 'n' and b == 'n':
            n_tp += 1
        elif a == 'n' and (b == 'y' or b == 'x'):
            n_fp += 1
        elif (a == 'y' and b == 'y') or (a == 'x' and b == 'x'):
            n_tn += 1
        elif (a == 'y' or a == 'x') and b == 'n':
            n_fn += 1
        # not
        if a == 'x' and b == 'x':
            x_tp += 1
        elif a == 'x' and (b == 'y' or b == 'n'):
            x_fp += 1
        elif (a == 'y' and b == 'y') or (a == 'n' and b == 'n'):
            x_tn += 1
        elif (a == 'y' or a == 'n') and b == 'x':
            x_fn += 1
    y_recall = y_tp * 1.0 / (y_tp + y_fn)
    y_precision = y_tp * 1.0 / (y_tp + y_fp)
    y_f1 = 2 * y_recall * y_precision / (y_recall + y_precision)
    y_acc = (y_tp + y_tn) * 1.0 / (y_tp + y_tn + y_fp + y_fn)
    n_recall = n_tp * 1.0 / (n_tp + n_fn)
    n_precision = n_tp * 1.0 / (n_tp + n_fp)
    n_f1 = 2 * n_recall * n_precision / (n_recall + n_precision)
    n_acc = (n_tp + n_tn) * 1.0 / (n_tp + n_tn + n_fp + n_fn)
    x_recall = x_tp * 1.0 / (x_tp + x_fn)
    x_precision = x_tp * 1.0 / (x_tp + x_fp)

    x_f1 = 2 * x_recall * x_precision / (x_recall + x_precision)
    f1 = {"yes": {"recall": y_recall, "precision": y_precision, "f1": y_f1, "acc": y_acc},
          "no": {"recall": n_recall, "precision": n_precision, "f1": n_f1, "acc": n_acc},
          "x": {"recall": x_recall, "precision": x_precision, "f1": x_f1, "acc": ""}}
    with open(output_file, 'w') as f:
        json.dump(f1, f, indent=4)


def two_label_f1(input_file1, input_file2, output_file):
    with open(input_file1, 'r') as f:
        yesno_pred = list()
        data = json.load(f)
        for key in data:
            yesno_pred.append(key['yesno'])
    with open(input_file2, 'r') as f:
        yesno_sou = list()
        data = json.load(f)['data']
        for article in data:
            answers = article['answers']
            for answer in answers:
                input_text = answer['input_text'].strip().replace('\n', '').lower()
                if input_text == 'yes':
                    yesno_sou.append('y')
                elif input_text == 'no':
                    yesno_sou.append('n')
                else:
                    yesno_sou.append('x')
    y_tp = 0
    y_fp = 0
    y_tn = 0
    y_fn = 0
    n_tp = n_fp = n_tn = n_fn = 0
    for a, b in zip(yesno_pred, yesno_sou):
        if b == 'x':
            continue
        # yes
        if a == 'y' and b == 'y':
            y_tp += 1
        elif a == 'y' and b == 'n':
            y_fp += 1
        elif a == 'n' and b == 'n':
            y_tn += 1
        elif a == 'n' and b == 'y':
            y_fn += 1
        # no
        if a == 'n' and b == 'n':
            n_tp += 1
        elif a == 'n' and b == 'y':
            n_fp += 1
        elif a == 'y' and b == 'y':
            n_tn += 1
        elif a == 'y' and b == 'n':
            n_fn += 1
    y_recall = y_tp * 1.0 / (y_tp + y_fn)
    y_precision = y_tp * 1.0 / (y_tp + y_fp)
    y_f1 = 2 * y_recall * y_precision / (y_recall + y_precision)
    y_acc = (y_tp + y_tn) * 1.0 / (y_tp + y_tn + y_fp + y_fn)
    n_recall = n_tp * 1.0 / (n_tp + n_fn)
    n_precision = n_tp * 1.0 / (n_tp + n_fp)
    n_f1 = 2 * n_recall * n_precision / (n_recall + n_precision)
    n_acc = (n_tp + n_tn) * 1.0 / (n_tp + n_tn + n_fp + n_fn)
    f1 = {"yes": {"recall": y_recall, "precision": y_precision, "f1": y_f1, "acc": y_acc},
          "no": {"recall": n_recall, "precision": n_precision, "f1": n_f1, "acc": n_acc}}
    with open(output_file, 'w') as f:
        json.dump(f1, f, indent=4)


if __name__ == '__main__':
    input_file1 = argv[1]
    input_file2 = argv[2]
    output_file = argv[3]
    # main(input_file1, input_file2, output_file)
    two_label_f1(input_file1, input_file2, output_file)
