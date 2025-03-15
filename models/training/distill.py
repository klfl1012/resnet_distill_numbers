import torch
import torch.nn as nn


def kd_loss(student_logits, teacher_logits, labels, alpha=0.5, temperature=4.0):

    student_probs = nn.functional.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = nn.functional.softmax(teacher_logits / temperature, dim=1)
    kl_loss = nn.functional.kl_div(student_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
    ce_loss = nn.CrossEntropyLoss()(student_logits, labels)
    total_loss = alpha * kl_loss + (1 - alpha) * ce_loss

    return total_loss


def policy_distillation_loss(student_logits, teacher_logits, temperature=4.0):  

    student_probs = nn.functional.softmax(student_logits / temperature, dim=1)
    teacher_probs = nn.functional.softmax(teacher_logits / temperature, dim=1)
    loss = nn.functional.mse_loss(student_probs, teacher_probs)

    return loss


class Discriminator(nn.Module):

    def __init__(self, feature_dim, hidden_units=256):
        super(Discriminator, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)
    

def adversarial_distillation(student, teacher, discriminator, optimizer_S, optimizer_D, criterion, trainloader, device):
    teacher.eval()
    student.train()

    for imgs, _ in trainloader:
        imgs = imgs.to(device)

        with torch.no_grad():
            teacher_outs = teacher(imgs)

        student_outs = student(imgs)

        student_features = student_outs.extract_features(imgs)

        optimizer_D.zero_grad(set_to_none=True) 
        real_preds = discriminator(teacher_outs)
        fake_preds = discriminator(student_features.detach())

        d_loss = criterion(real_preds, torch.ones_like(real_preds)) + criterion(fake_preds, torch.zeros_like(fake_preds))
        d_loss.backward()
        optimizer_D.step()

        optimizer_S.zero_grad(set_to_none=True)
        fake_preds = discriminator(student_features)
        s_loss = criterion(fake_preds, torch.ones_like(fake_preds))
        s_loss.backward()
        optimizer_S.step()

if __name__ == "__main__":
    pass




