import torch
import torch.nn as nn


def kl_loss(student_logits, teacher_logits, labels, temperature=4.0):
    student_probs = nn.functional.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = nn.functional.softmax(teacher_logits / temperature, dim=1)
    kl_loss = nn.functional.kl_div(student_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

    return kl_loss

def policy_distillation_loss(student_logits, teacher_logits, temperature=4.0):  
    student_probs = nn.functional.softmax(student_logits / temperature, dim=1)
    teacher_probs = nn.functional.softmax(teacher_logits / temperature, dim=1)
    policy_loss = nn.functional.mse_loss(student_probs, teacher_probs)

    return policy_loss

def attention_loss(student_feats, teacher_feats):
    student_feats_map = torch.nn.functional.normalize(student_feats, p=2, dim=1)
    teacher_feats_map = torch.nn.functional.normalize(teacher_feats, p=2, dim=1)

    return nn.functional.mse_loss(student_feats_map, teacher_feats_map) 

def feature_distillation_loss(student_feats, teacher_feats):
    return nn.functional.mse_loss(student_feats, teacher_feats)


if __name__ == "__main__":
    pass




