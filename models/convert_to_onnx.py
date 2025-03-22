import torch
import onnx, json
from teacher import Teacher, TeacherCNN
from student import StudentCNN
from onnxruntime.quantization import quantize_dynamic, QuantType    

device = "mps" if torch.backends.mps.is_available() else "cpu"
dummy_input = torch.randn(1, 1, 28, 28).to(device)  


with open("./configs/trainer/trainer_config.json", "r") as f:
    config = json.load(f)

teacher = Teacher().to(device)
teachercnn = TeacherCNN(**config["teachercnn"]).to(device)
teacher_studentcnn = TeacherCNN(**config["teacher_studentcnn"]).to(device)

teacher.load_state_dict(torch.load(config["teacher_paths"]["teacher"], map_location=device))
teachercnn.load_state_dict(torch.load(config["teacher_paths"]["teachercnn"], map_location=device))
teacher_studentcnn.load_state_dict(torch.load(config["teacher_paths"]["teacher_studentcnn"], map_location=device))

torch.onnx.export(
    teacher, dummy_input, "./onnx_models/teacher.pth",
    export_params=True, 
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

torch.onnx.export(
    teachercnn, dummy_input, "./onnx_models/teachercnn.pth",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

torch.onnx.export(
    teacher_studentcnn, dummy_input, "./onnx_models/teacher_studentcnn.pth",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

quantized_teacher = quantize_dynamic("./onnx_models/teacher.pth", "./onnx_models/teacher_quantized.pth", weight_type=QuantType.QUInt8)
quantized_teachercnn = quantize_dynamic("./onnx_models/teachercnn.pth", "./onnx_models/teachercnn_quantized.pth", weight_type=QuantType.QUInt8)
quantized_teacher_studentcnn = quantize_dynamic("./onnx_models/teacher_studentcnn.pth", "./onnx_models/teacher_studentcnn_quantized.pth", weight_type=QuantType.QUInt8)

student_paths = ["ds_model_teachercnn_attention_0_5.pth", "ds_model_teacher_studentcnn_feature_0_5.pth", "ds_model_teacher_studentcnn_policy_0_5.pth", "ds_model_teacher_studentcnn_kd_0_5.pth"]

for student_path in student_paths:
    student = StudentCNN(**config["studentcnn"]).to(device)
    student.load_state_dict(torch.load(f"./trained_models/{student_path}", map_location=device))
    student.eval()


    torch.onnx.export(
        student, dummy_input, f"./onnx_models/{student_path.replace('.pth', '.onnx')}",
        export_params=True, 
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )

    quantized_student = quantize_dynamic(f"./onnx_models/{student_path.replace('.pth', '.onnx')}", f"./onnx_models/{student_path.replace('.pth', '_quantized.onnx')}", weight_type=QuantType.QUInt8)