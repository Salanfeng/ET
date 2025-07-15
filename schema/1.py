import sys
import flatbuffers
from executorch.schema.executorch_flatbuffer import Program

TENSOR_TYPE_ID = 5

def scalar_type_to_str(scalar_type):
    return str(scalar_type)

def main(pte_path, output_path):
    with open(pte_path, "rb") as f:
        buf = f.read()
    program = Program.Program.GetRootAsProgram(buf, 0)

    with open(output_path, "w") as out:
        for ep_idx in range(program.ExecutionPlanLength()):
            ep = program.ExecutionPlan(ep_idx)
            out.write(f"ExecutionPlan[{ep_idx}] name: {ep.Name().decode() if ep.Name() else 'N/A'}\n")
            for v_idx in range(ep.ValuesLength()):
                val = ep.Values(v_idx)
                val_type = val.ValType()
                out.write(f"  Value[{v_idx}]: type={val_type}\n")
                # Tensor 类型判断
                if val_type == TENSOR_TYPE_ID:
                    tensor = val.Val()  # 直接用 Val() 获取 Tensor 对象
                    # 检查是否有 Tensor 方法
                    if hasattr(tensor, "ScalarType"):
                        name = None
                        if tensor.ExtraTensorInfo() and hasattr(tensor.ExtraTensorInfo(), "FullyQualifiedName") and tensor.ExtraTensorInfo().FullyQualifiedName():
                            name = tensor.ExtraTensorInfo().FullyQualifiedName().decode()
                        scalar_type = scalar_type_to_str(tensor.ScalarType())
                        sizes = [tensor.Sizes(i) for i in range(tensor.SizesLength())]
                        out.write(f"    Tensor: name={name}, scalar_type={scalar_type}, sizes={sizes}\n")
                        # 量化参数识别
                        if name and ("scale" in name or "zero_point" in name):
                            out.write(f"      [可能是量化参数]\n")
                        # 输出 ExtraTensorInfo
                        if tensor.ExtraTensorInfo():
                            out.write(f"      ExtraTensorInfo: {tensor.ExtraTensorInfo()}\n")
            out.write("\n")

if __name__ == "__main__":
    pte_path = "/home/pairshoe/syf/executorch/llama_qnn_74/hybrid_llama_qnn.pte"
    output_path = "pte_quant_info.txt"
    main(pte_path, output_path)