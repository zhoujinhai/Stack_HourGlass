import torch
import cv2
import copy
import sys
sys.path.append("../")
from config.config import get_config
from models.hour_glass import StackHourglass

def test_diff(ori_output, after_output):
    # 确保两个输出都是torch张量，并且都在CPU上
    ori_output = ori_output.cpu()
    after_output = after_output.cpu()
    
    # 计算最大绝对差异
    max_abs_diff = torch.max(torch.abs(ori_output - after_output))
    print("Max abs diff: ", max_abs_diff.item())
    
    # 检查argmax是否相同（仅当输出是分类任务的logits时适用）
    if ori_output.dim() > 0 and after_output.dim() == ori_output.dim():
        assert torch.argmax(ori_output) == torch.argmax(after_output), "Argmax differs between outputs"
    
    # 计算MSE差异
    mse_diff = torch.nn.MSELoss()(ori_output, after_output)
    print("MSE diff: ", mse_diff.item())

    
    
if __name__ == "__main__":
    # config
    config = get_config()

    # model
    device = config["device"]
    model = StackHourglass(config["n_stack"], config["in_dim"], config["n_kp"], config["n_hg_layer"])
    model.float().to(device)
    weight_path = "../exp/tooth_5/checkpoint170.pt"
    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    # input
    inputs = torch.randn(1, 3, 256, 256).to(device)
    ori_out = model(inputs)
    print("out.shape: ", ori_out.shape, ori_out)
    
    # fusing convolution with batch norm 
    # fuse_model(model)
    # after_out = model(inputs)
    # test_diff(ori_out[2], after_out[2])

    # convert
    onnx_path = weight_path.replace(".pt", ".onnx")
    torch.onnx.export(
        model,
        (inputs, ),
        onnx_path,
        verbose=False,
        input_names=['img'],
        output_names=["outs"],
        opset_version=12
    )
    print("converted successed!")
    
    import onnxruntime as ort 
    # 使用 ONNX Runtime 推理
    ort_session = ort.InferenceSession(onnx_path)
    outputs = ort_session.run(None, {'img': inputs.cpu().numpy()})

    # 输出 ONNX 结果
    print("ONNX output shape: ", outputs[0].shape)
    print("ONNX output: ", outputs[0])

    test_diff(ori_out, torch.from_numpy(outputs[0]))
    dnn_net = cv2.dnn.readNet(onnx_path)
    print("cv read onnx successed!")
