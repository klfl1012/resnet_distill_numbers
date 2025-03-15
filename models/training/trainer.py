import torch, torch.nn as nn, torch.optim as optim
from models.student import StudentCNN
from models.teacher import Teacher
from models.training.distill import policy_distillation_loss, adversarial_distillation, kd_loss, Discriminator
from utils import get_dataloaders


class Trainer:

    def __init__(self, teacher_path, student_params, trainloader, testloader, device, method="kd", alpha=0.5, temperature=4.0, lr=0.001):
        self.device = device
        self.teacher = Teacher().load_state_dict(torch.load(teacher_path)).to(device)
        self.student = StudentCNN(**student_params).to(device)  
        self.trainloader = trainloader
        self.testloader = testloader
        self.method = method
        self.alpha = alpha
        self.temperature = temperature
        self.lr = lr

        self.optimizer_S = optim.Adam(self.student.parameters(), lr=self.lr)

        if method == "adv":
            feature_dim = student_params["hidden_units"]
            self.discriminator = Discriminator(feature_dim=feature_dim).to(device)
            self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr)
            self.ciriterion_adv = nn.BCELoss()  


    def train(self, epochs):
        self.teacher.eval()
        self.student.train()

        for epoch in range(epochs):

            total_loss = 0.0

            if self.method == "adv":
                adversarial_distillation(
                    self.student, self.teacher, self.discriminator, self.optimizer_S, self.optimizer_D, self.criterion_adv, self.trainloader, self.device
                )
                continue

            for imgs, labels in self.trainloader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                self.optimizer_S.zero_grad(set_to_none=True)

                teacher_outs = self.teacher(imgs).detach()
                student_outs = self.student(imgs)

                if self.method == "kd":
                    loss = kd_loss(student_outs, teacher_outs, labels, self.alpha, self.temperature)
                elif self.method == "policy":
                    loss = policy_distillation_loss(student_outs, teacher_outs, self.temperature)

                else:
                    raise ValueError("Invalid distillation method")

                loss.backward()
                self.optimizer_S.step()

                total_loss += loss.item()
            
            print(f"Epoch {epoch + 1}, loss: {total_loss / len(self.trainloader):.4f}")

    def evaluate(self):
        self.student.eval()
        correct, total = 0, 0.0

        with torch.no_grad():
            for imgs, labels in self.testloader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outs = self.student(imgs)
                _, predicted = torch.max(outs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total    
        print(f"Test accuracy: {accuracy:.2f}%")



if __name__ == "__main__":
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    teacher_path = "teacher.pth"
    student_params = {
        "num_filters1": 10,
        "num_filters2": 5,
        "kernel_size1": 1,
        "kernel_size2": 1,
        "padding1": 1,
        "padding2": 1,
        "padding3": 1,
        "hidden_units": 32
    }

    trainloader, testloader = get_dataloaders(batch_size=32, resize=(112, 112)) 
    trainer = Trainer(

    )




