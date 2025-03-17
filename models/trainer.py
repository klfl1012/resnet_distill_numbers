import torch, torch.nn as nn, torch.optim as optim
from student import StudentCNN
from teacher import Teacher
from distill import policy_distillation_loss, kl_loss, attention_loss, feature_distillation_loss, adversarial_distillation, Discriminator
from utils import get_dataloaders


class Trainer:

    def __init__(self, teacher_path, student_params, trainloader, testloader, device, method="kd", alpha=0.5, temperature=4.0, lr=0.001):
        self.device = device
        self.teacher = Teacher().to(device)
        self.teacher.load_state_dict(torch.load(teacher_path, map_location=device))
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
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_S, mode="min", factor=0.1, patience=3)
        relative_improvement_threshold = 0.01
        best_loss = float("inf")
        patience = 0
        epochs_no_improvement = 0

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

                criterion = nn.CrossEntropyLoss()
                hard_loss = criterion(student_outs, labels)

                if self.method == "kd":
                    soft_loss = kl_loss(student_outs, teacher_outs, labels, self.alpha, self.temperature)

                elif self.method == "policy":
                    soft_loss = policy_distillation_loss(student_outs, teacher_outs, self.temperature)

                elif self.method == "attention":
                    student_feats = self.student.extract_features(imgs, layers=["final"])
                    teacher_feats = self.teacher.extract_features(imgs, layers=["final"])
                    soft_loss = attention_loss(student_feats, teacher_feats)

                elif self.method == "feature":
                    student_feats = self.student.extract_features(imgs, layers=["final"])
                    teacher_feats = self.teacher.extract_features(imgs, layers=["final"])
                    soft_loss = feature_distillation_loss(student_feats, teacher_feats)

                else:
                    raise ValueError("Invalid distillation method")

                loss = alpha * hard_loss + (1 - alpha) * soft_loss
                loss.backward()
                self.optimizer_S.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.trainloader)
            scheduler.step(loss)
            print(f"Epoch {epoch + 1}, loss: {avg_loss:.4f}")

            if avg_loss < best_loss - relative_improvement_threshold:
                best_loss = avg_loss
                patience = 0
            else:
                patience += 1
            if epochs_no_improvement > patience:
                print("Early stopping")
                break

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

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    img_size = (28, 28)
    lr = 0.001
    temperature = 4.0   
    batch_size = 32
    teacher_path = "./trained_models/trained_teacher_state_dict_1.pth"
    student_params = {
        "num_filters1": 8,
        "num_filters2": 5,
        "kernel_size1": 1,
        "kernel_size2": 1,
        "padding1": 1,
        "padding2": 1,
        "padding3": 1,
        "hidden_units": 32,
        "img_size": img_size
    }
    alphas = [0, 0.5, 1]
    methods = ["attention", "feature"]
    # methods = ["kd", "policy", "attention", "feature"]

    epochs = 15
    trainloader, testloader = get_dataloaders(batch_size=batch_size, resize=img_size) 

    for method in methods:
        for alpha in alphas:
            print(f"Training student with method: {method}, alpha: {alpha}")   
            trainer = Trainer(
                teacher_path=teacher_path,
                student_params=student_params,
                trainloader=trainloader,    
                testloader=testloader,
                device=device,
                method="kd",
                alpha=alpha,
                temperature=temperature,
                lr=lr
            )
            trainer.train(epochs=epochs)
            trainer.evaluate()
            torch.save(trainer.student.state_dict(), f"./trained_models/d_s_{method}_{alpha}_1.pth")

    print("Fininshed training")



# relative impr threshold mit acc sodass net overfitted