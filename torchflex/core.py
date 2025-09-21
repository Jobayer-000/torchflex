import torch
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
except Exception as e:
    print("torch_xla importing failed")
    print("Error:", e)


class DeviceManager:
    def __init__(self, backend="cpu"):
        self.backend = backend.lower()

        if self.backend == "cpu":
            self.device = torch.device("cpu")

        elif self.backend == "gpu":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")
            self.device = torch.device("cuda")

        elif self.backend == "tpu":
            self.device = xm.xla_device()

        else:
            raise ValueError(f"Unknown backend: {backend}")

    def to_device(self, obj):
        """Move tensor/model to the correct device."""
        return obj.to(self.device)

    def optimizer_step(self, optimizer, barrier=True):
        """Correct optimizer step for each backend."""
        if self.backend == "tpu":
            xm.optimizer_step(optimizer, barrier=barrier)
            xm.mark_step()
        else:
            optimizer.step()
    def is_master_ordinal(self):
        if self.backend == "tpu":
            return xm.is_master_ordinal()
        else:
            return True

    def loader(self, dataset, batch_size, shuffle=True, num_workers=4):
        """Correct dataloader for TPU/GPU/CPU."""
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        if self.backend == "tpu":
            loader = pl.MpDeviceLoader(loader, self.device)
        return loader
