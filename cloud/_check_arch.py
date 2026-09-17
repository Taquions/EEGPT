"""Report whether this torch build has native kernels for the attached GPU.

A wheel that lacks a matching SASS entry can still run by JIT-compiling the
embedded PTX, but only if a `compute_*` entry is present and compatible. If
neither is there, every kernel launch fails at runtime -- which is a failure
worth discovering in a ten-minute test rather than three hours into a campaign.
"""
import torch

cap = torch.cuda.get_device_capability(0)
sm = "sm_%d%d" % cap
archs = torch.cuda.get_arch_list()

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("capability:", sm)
print("arch list:", archs)
print("native kernels for this device:", sm in archs)
print("ptx entries:", [a for a in archs if a.startswith("compute_")])
