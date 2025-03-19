import optuna, torch, json, os, pandas as pd
from utils import get_dataloaders
from teacher import TeacherCNN


PARAM_GRID_FILE = "configs/params_teachercnn.json"
LOG_FILE = "logs/train_SCNN.json"
STUDENT_MODEL_SAVE_PATH = "./trained_models/best_teachercnn.pth"
EPOCHS = 50
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
best_overall_loss = float("inf")    
best_overall_model_params = None


with open(PARAM_GRID_FILE, "r") as f:
    PARAM_GRID = json.load(f)

def suggest_params(trial):
    return {
        key: trial.suggest_categorical(key, v) for key, v in PARAM_GRID.items()
    }

def objective(trial):
    global best_overall_loss, best_overall_model_params

    params = suggest_params(trial)
    trainloader, _ = get_dataloaders(params["batch_size"], resize=(28, 28))

    student = TeacherCNN(
        num_filters1=params["num_filters1"],
        num_filters2=params["num_filters2"],
        num_filters3=params["num_filters3"],
        kernel_size1=params["kernel_size1"],
        kernel_size2=params["kernel_size2"],
        kernel_size3=params["kernel_size3"],
        hidden_units=params["hidden_units"],
        padding1=params["padding1"],
        padding2=params["padding2"],
        padding3=params["padding3"],
        img_size=(28, 28)  
    ).to(DEVICE)

    student.train()
    optimizer = torch.optim.Adam(student.parameters(), lr=params["lr"])
    criterion = torch.nn.CrossEntropyLoss()

    best_loss = float("inf")
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for imgs, labels in trainloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            student_outs = student(imgs)
            loss = criterion(student_outs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(trainloader)
        trial.report(epoch_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_params = student.state_dict()

    if best_loss < best_overall_loss:
        best_overall_loss = best_loss
        best_overall_model_params = best_model_params

    return best_loss


if __name__ == "__main__":
    study = optuna.create_study(study_name="TeacherCNN Hyperparameter Tuning", direction="minimize")
    study.optimize(objective, n_trials=20)

    if best_overall_model_params:
        torch.save(best_overall_model_params, STUDENT_MODEL_SAVE_PATH)

    df = study.trials_dataframe()
    df.to_csv(LOG_FILE)

    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)






