import onnxruntime as ort
from transformers import AutoTokenizer
from pathlib import Path

def load_model_and_tokenizer():
    model_path = (Path(__file__).parent.parent / "models" / "e5-large" / "model.onnx").resolve()
    tokenizer_dir = (Path(__file__).parent.parent / "models" / "e5-large").resolve()

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    
    # Create TRT cache dir if it doesn't exist
    cache_path = Path("/disk/trt_cache")
    cache_path.mkdir(parents=True, exist_ok=True)
    
    providers = [
        ("TensorrtExecutionProvider", {
            "device_id": 0,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(cache_path),
            "trt_fp16_enable": True,
        }),
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]
    
    session = ort.InferenceSession(
        str(model_path),
        providers=providers
    )
    return session, tokenizer
