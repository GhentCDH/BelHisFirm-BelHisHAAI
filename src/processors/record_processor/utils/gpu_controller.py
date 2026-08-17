import torch
import gc

class GPUController:

    @staticmethod
    def clear_gpu_memory() -> None:
        """ Clear GPU memory to prevent OOM errors.

            Returns: None
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Clear metal performance shader cache on Mac devices
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()