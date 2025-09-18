import os

import pynvml


def print_all_gpu_pci_addresses() -> None:
    pynvml.nvmlInit()
    total = pynvml.nvmlDeviceGetCount()
    print(f"[PID {os.getpid()}] Total visible GPUs: {total}")
    for i in range(total):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
        print(
            f"CUDA index={i} -> PCI address={pci_info.domain:04x}:{pci_info.bus:02x}:{pci_info.device:02x}.{pci_info.pciDeviceId & 0x7}"
        )
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    print_all_gpu_pci_addresses()
