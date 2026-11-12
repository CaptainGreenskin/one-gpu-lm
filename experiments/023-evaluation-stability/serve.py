"""Record actual generation settings while leaving the historical runtime intact."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.serving import local_backend
from one_gpu_lm.common.receipts import write_json

class RecordedRuntime(local_backend.Runtime):
    def __init__(self,model_path,profile,out,adapter=None):
        super().__init__(model_path,profile,out,adapter)
        import torch,transformers,peft
        write_json(out/'runtime-fingerprint.json',{'torch':torch.__version__,'transformers':transformers.__version__,
            'peft':peft.__version__,'cuda_build':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,
            'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
            'device':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0)),
            'attention':profile['attention'],'do_sample':False,'seed':101,'settings_changed_by_023':False})
local_backend.Runtime=RecordedRuntime
if __name__=='__main__':local_backend.main()
