import torch, torch.nn as nn, torch.optim as optim, numpy as np, pandas as pd, psutil, os, time, json, logging
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import List

from student import StudentCNN
from teacher import TeacherCNN, Teacher
from distill import policy_distillation_loss, kl_loss, attention_loss, feature_distillation_loss
from utils import get_dataloaders


class TrainManager:

    def __init__(
            self, 
            teacher: nn.Module, 
            student: nn.Module, 
            trainloader: torch.utils.data.DataLoader, 
            testloader: torch.utils.data.DataLoader, 
            device: str,
            method: List=["kd", "policy", "attention", "feature", "self", "adv"], 
            alpha: float=0.5, 
            temperature: float=4.0, 
            lr: float=0.001
        ):
        self.device = device
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.trainloader = trainloader
        self.testloader = testloader
        self.method = method
        self.alpha = alpha
        self.temperature = temperature
        self.lr = lr
        self.optimizer = optim.Adam(self.student.parameters(), lr=self.lr)


    def train(self, epochs):
        self.teacher.eval()
        self.student.train()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.1, patience=3)
        relative_improvement_threshold = 0.01
        best_loss = float("inf")
        patience = 0
        epochs_no_improvement = 0

        for epoch in range(epochs):

            total_loss = 0.0

            for imgs, labels in self.trainloader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)

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
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.trainloader)
            scheduler.step(loss)
            logging.info(f"Epoch {epoch + 1}, loss: {avg_loss:.4f}")

            if avg_loss < best_loss - relative_improvement_threshold:
                best_loss = avg_loss
                patience = 0
            else:
                patience += 1
            if epochs_no_improvement > patience:
                logging.info("Early stopping")
                break


    def evaluate(self, file_name=""):
        self.student.eval()
        self.teacher.eval()

        correct, total_loss = 0, 0.0 

        all_labels = []
        all_preds = []
        all_teacher_logits = []
        all_student_logits = []

        criterion = nn.CrossEntropyLoss()

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 ** 2

        student_inference_times = []
        teacher_inference_times = []

        with torch.no_grad():
            for imgs, labels in self.testloader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                start_time_teacher = time.time()
                teacher_outs = self.teacher(imgs)
                teacher_inference_times.append(time.time() - start_time_teacher) 

                start_time_student = time.time()    
                student_outs = self.student(imgs)
                student_inference_times.append(time.time() - start_time_student)

                _, predicted = torch.max(student_outs, 1)
                total_loss += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(student_outs, labels)  
                total_loss += loss.item()

                all_teacher_logits.append(teacher_outs.cpu().numpy())
                all_student_logits.append(student_outs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

            accuracy = correct / total_loss  * 100
            avg_loss = total_loss / len(self.testloader)    
            precision = precision_score(all_labels, all_preds, average="weighted")
            recall = recall_score(all_labels, all_preds, average="weighted")
            f1 = f1_score(all_labels, all_preds, average="weighted")

            all_teacher_logits = np.concatenate(all_teacher_logits, axis=0)
            all_student_logits = np.concatenate(all_student_logits, axis=0)

            kl_div = torch.nn.functional.kl_div(
                torch.log_softmax(student_outs, dim=1), 
                torch.softmax(teacher_outs, dim=1), 
                reduction="batchmean"
            )

            teacher_mean = all_teacher_logits.mean(axis=0)
            student_mean = all_student_logits.mean(axis=0)

            logtis_correlation = np.corrcoef(teacher_mean, student_mean)[0, 1]

            avg_student_time = np.mean(student_inference_times) * 1000
            avg_teacher_time = np.mean(teacher_inference_times) * 1000

            mem_after = process.memory_info().rss / 1024 ** 2
            mem_usage = mem_after - mem_before

            results = pd.DataFrame({
                "Accuracy": [accuracy],
                "CrossEntropyLoss": [avg_loss],
                "Precision": [precision], 
                "Recall": [recall],
                "F1": [f1],
                "KL-Div": [kl_div.item()],
                "Logits Correlation": [logtis_correlation],   
                "Avg Student Inference Time (ms)": [avg_student_time],
                "Avg Teacher Inference Time (ms)": [avg_teacher_time],
                "Memory Usage (MB)": [mem_usage]
            }, index=[file_name])  
                
            results.to_csv(f"./logs/eval/{file_name}.csv", index=False)
            return results


if __name__ == "__main__":

    # Logging config
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("./logs/training.log", mode="w")
        ]
    )

    # Config parameters
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    teacher_paths = {
        "teacher": "./trained_models/trained_teacher_state_dict_1.pth",
        "teachercnn": "./trained_models/teachercnn_1.pth"
    }
    teachercnn_config_file = "./configs/teachercnn_params.json"
    with open(teachercnn_config_file, "r") as f:
        teachercnn_params = json.load(f) 

    student_config_file = "./configs/studentcnn_params.json"
    with open(student_config_file, "r") as f:
        studentcnn_params = json.load(f)

    # Training parameters 
    img_size = (28, 28)
    lr = 0.001
    temperature = 4.0   
    batch_size = 32
    epochs = 2
    alphas = [0]
    methods = ["kd"]

    # Train- and Testloader
    trainloader, testloader = get_dataloaders(batch_size=batch_size, resize=img_size) 

    # Training loop
    for teacher_name, teacher_path in teacher_paths.items():
        logging.info(f"Training student with teacher: {teacher_name}")  

        if teacher_name == "teachercnn":
            teacher = TeacherCNN(img_size=img_size,**teachercnn_params).to(device)
        else:
            teacher = Teacher().to(device)  

        teacher.load_state_dict(torch.load(teacher_path, map_location=device))
        teacher.eval()

        for method in methods:
            for alpha in alphas:
                logging.info(f"Training student with teacher: {teacher_name}, method: {method}, alpha: {alpha}")   

                trainer = TrainManager(
                    teacher=teacher,
                    student=StudentCNN(img_size=img_size, **studentcnn_params).to(device),
                    trainloader=trainloader,
                    testloader=testloader,
                    device=device,
                    method=method,
                    alpha=alpha,
                    temperature=temperature,
                    lr=lr
                )
                alpha_str = str(alpha).replace('.', '_')
                trainer.train(epochs=epochs)
                trainer.evaluate(file_name=f"{teacher_name}_{method}_{alpha_str}")
                torch.save(trainer.student.state_dict(), f"./trained_models/ds_model_{teacher_name}_{method}_{alpha_str}.pth")
    
    logging.info("Finished training")
