"""
Experimental protocols for evaluating Vendi Topic Reduction.

This package contains modular implementations of different experimental protocols
for benchmarking topic reduction methods.
"""

from experiments.protocols.protocol_1 import run_protocol_1, print_protocol_1_summary
from experiments.protocols.protocol_2 import run_protocol_2, print_protocol_2_summary
from experiments.protocols.protocol_4 import run_protocol_4, print_protocol_4_summary

__all__ = [
    "run_protocol_1",
    "print_protocol_1_summary",
    "run_protocol_2",
    "print_protocol_2_summary",
    "run_protocol_4",
    "print_protocol_4_summary",
]
