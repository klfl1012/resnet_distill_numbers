import torch
import torch.nn as nn
from teacher import get_teacher_model
from student import StudentCNN


def kd_loss(student_logits, teacher_logits, labels, alpha=0.5, temperature=4.0):

    student_probs = nn.functional.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = nn.functional.softmax(teacher_logits / temperature, dim=1)
    kl_loss = nn.functional.kl_div(student_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
    ce_loss = nn.CrossEntropyLoss()(student_logits, labels)
    total_loss = alpha * kl_loss + (1 - alpha) * ce_loss

    return total_loss









