import torch
from torch.nn.functional import gumbel_softmax


def get_f1_matrix(span_start: torch.Tensor, span_end: torch.Tensor, passage_length: int):
    """
    return the f1 value matrix from span_start and span_end with passage_length
    :param span_start: batch_size
    :param span_end: batch_size
    :param passgae_length: batch_size
    :return: f1_matrix: batch_size, passage_Length, passage_length
    """
    batch_size = span_start.size(0)
    span_start_m = torch.tensor(span_start, dtype=torch.float, device=span_start.device).view(batch_size, 1, 1).expand(
        batch_size, passage_length, passage_length)
    span_end_m = torch.tensor(span_end, dtype=torch.float, device=span_end.device).view(batch_size, 1, 1).expand(
        batch_size, passage_length, passage_length)
    tmp = list()
    for i in range(passage_length):
        tmp.append(i)
    p_start = torch.tensor(tmp, dtype=torch.float, device=span_start.device)
    p_end = torch.tensor(tmp, dtype=torch.float, device=span_end.device)
    # p_start_m[b][i][j] = j
    p_start_m = p_start.view(1, 1, passage_length).expand(batch_size, passage_length, passage_length)
    # p_end_m[b][i][j] = i
    p_end_m = p_end.view(1, passage_length, 1).expand(batch_size, passage_length, passage_length)
    # start[b][i][j] = max(span_start[b], j)
    start = torch.max(span_start_m, p_start_m)
    # end[b][i][j] = min(span_end[b], i)
    end = torch.min(span_end_m, p_end_m)
    # exist matrix
    zero = torch.zeros(1, dtype=torch.float, device=span_start.device)
    one = torch.ones(1, dtype=torch.float, device=span_start.device)
    # exist_matrix[i][j] = max(i - j, 0) = i - j + 1 if i >= j else 0
    exist_matrix = torch.max(p_end_m - p_start_m + 1, zero)
    # exist_matrix[i][j] = 1 if i >= j else 0
    exist_matrix = torch.min(exist_matrix, one)
    # num_same[b][i][j] = f1(j, i, s[b], e[b])
    num_same = end - start + 1
    # choose num_same > 0
    num_same = torch.max(num_same, zero)
    # num_same -> 1
    num_same = torch.min(num_same, one)
    # num_same -> correct value -> 1
    num_same = num_same * exist_matrix
    # span[b][i][j] = span_end[b] - span_start[b] + 1
    span = span_end_m - span_start_m + 1
    # p_span[b][i][j] = i - j + 1
    p_span = (p_end_m - p_start_m + 1) * exist_matrix
    f1 = 2 * num_same / (p_span + span + 2)
    return f1.transpose(2, 1)


def test_get_f1_matrix():
    span_start = torch.tensor([2, 3])
    span_end = torch.tensor([5, 9])
    f1_matrix = get_f1_matrix(span_start, span_end, 12)
    print(f1_matrix)


def evidence_f1_loss(p_start: torch.Tensor, p_end: torch.Tensor, span_start: torch.Tensor, span_end: torch.Tensor,
                     f1_matrix: torch.Tensor):
    """
    sum(p_i * p_j * F1(i, j)), 0 <= i <= j < passage_length
    :param p_start: batch_size * passage_length(masked 0)
    :param p_end: batch_size * passage_length(masked 0)
    :param span_start: batch_size
    :param span_end: batch_size
    :param f1_matrix: batch_size * p * p
    :return: loss
    """
    assert p_start.dim() == 2
    assert p_end.dim() == 2
    assert span_start.dim() == 1
    assert span_end.dim() == 1

    batch_size = p_start.size(0)
    span_prob = torch.bmm(p_start.unsqueeze(2), p_end.unsqueeze(2).transpose(2, 1))
    f1 = (span_prob * f1_matrix).reshape(batch_size, -1)
    loss = torch.sum(f1, 1).squeeze(-1)
    return loss


def cal_f1(p_start: torch.Tensor, p_end: torch.Tensor, span_start: torch.Tensor, span_end: torch.Tensor):
    """
    :param p_start: 1
    :param p_end: 1
    :param span_start: batch_size
    :param span_end: batch_size
    :return: f1
    """
    batch_size = span_end.size(0)
    # batch_size
    span_start = torch.tensor(span_start, dtype=torch.float, device=span_start.device)
    span_end = torch.tensor(span_end, dtype=torch.float, device=span_end.device)
    start = torch.max(p_start, span_start)
    end = torch.min(p_end, span_end)
    num_same = torch.max(end - start + 1, torch.zeros(1, dtype=torch.float, device=span_start.device))
    num_same = torch.min(num_same, torch.ones(1, dtype=torch.float, device=span_start.device))
    # # precision = num_same / (p_end - p_start + 1)
    # # recall = num_same / (span_end - span_start + 1)
    # # f1 = (2 * precision * recall) / (precision + recall)
    f1 = 2 * num_same / (span_end - span_start + p_end - p_start + 2)
    # for i in range(batch_size):
    #     if num_same[i] <= 0:
    #         f1[i] = 0
    assert len(f1) == batch_size
    # f1 = torch.randn(batch_size)
    return f1


def test_cal_f1():
    p_start = torch.tensor([3.])
    p_end = torch.tensor([5.])
    span_start = torch.tensor([2, 3])
    span_end = torch.tensor([5, 7])
    f1 = cal_f1(p_start, p_end, span_start, span_end)
    print(f1)


def answer_f1_loss(p_start: torch.Tensor, p_end: torch.Tensor, span_start: torch.Tensor,
                   span_end: torch.Tensor, f1_matrix: torch.Tensor):

    """
    :param p_start: batch_size * passage_length
    :param p_end: batch_size * passage_length
    :param span_start:  batch_size
    :param span_end:  batch_size
    :param f1_matrix: batch_size * passage_length * passage_length
    :return:
    """
    assert p_start.dim() == 2
    assert p_end.dim() == 2
    assert span_start.dim() == 1
    assert span_end.dim() == 1

    batch_size = p_start.size(0)
    # batch_size * passage_length * 1
    start_prob = p_start.unsqueeze(2)
    end_prob = p_end.unsqueeze(2)
    # batch_size * (passage_length * passage_length)
    span_prob = torch.bmm(start_prob, end_prob.transpose(2, 1)).reshape(batch_size, -1)
    # batch_size * (passage_length * passage_length) * 1
    span_prob_gumbel = gumbel_softmax(span_prob, hard=True).unsqueeze(-1)
    # batch_size * 1 * (p * p)
    f1_matrix = f1_matrix.reshape(batch_size, -1).unsqueeze(1)
    return torch.bmm(f1_matrix, span_prob_gumbel).squeeze(-1).squeeze(-1)


def f1_loss(p_start, p_end, span_start, span_end):

    assert p_start.dim() == 2
    assert p_end.dim() == 2
    assert span_start.dim() == 1
    assert span_end.dim() == 1

    passage_length = p_start.size(1)
    batch_size = span_start.size(0)

    f1_matrix = get_f1_matrix(span_start, span_end, passage_length)

    # print("f1_matrix:")
    # print(f1_matrix)

    answer_loss = answer_f1_loss(p_start, p_end, span_start, span_end, f1_matrix)

    # print("answer_loss:")
    # print(answer_loss)

    evidence_loss = evidence_f1_loss(p_start, p_end, span_start, span_end, f1_matrix)

    # print("evidence_loss:")
    # print(evidence_loss)

    return answer_loss + evidence_loss
