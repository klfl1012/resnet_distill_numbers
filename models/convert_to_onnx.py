import torch
import onnx
from teacher import Teacher
from student import StudentCNN
from onnxruntime.quantization import quantize_dynamic, QuantType    

device = "mps" if torch.backends.mps.is_available() else "cpu"

teacher = Teacher().to(device)
teacher.load_state_dict(torch.load("./trained_models/trained_teacher_state_dict_1.pth", map_location=device))
teacher.eval()

student_params = {
        "num_filters1": 8,
        "num_filters2": 5,
        "kernel_size1": 1,
        "kernel_size2": 1,
        "padding1": 1,
        "padding2": 1,
        "padding3": 1,
        "hidden_units": 32,
        "img_size": (28, 28)
    }
student = StudentCNN(**student_params).to(device)
student.load_state_dict(torch.load("./trained_models/distilled_student_0.5_1.pth", map_location=device))
student.eval()

dummy_input = torch.randn(1, 1, 28, 28).to(device)  

torch.onnx.export(
    teacher, dummy_input, "./onnx_models/teacher.onnx",
    export_params=True, 
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

torch.onnx.export(
    student, dummy_input, "./onnx_models/student.onnx",
    export_params=True, 
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

quantized_student = quantize_dynamic("./onnx_models/student.onnx", "./onnx_models/student_quantized.onnx", weight_type=QuantType.QUInt8)
quantized_teacher = quantize_dynamic("./onnx_models/teacher.onnx", "./onnx_models/teacher_quantized.onnx", weight_type=QuantType.QUInt8)



