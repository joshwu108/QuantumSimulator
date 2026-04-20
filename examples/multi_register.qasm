// Multiple quantum and classical registers
OPENQASM 2.0;
include "qelib1.inc";

qreg a[2];
qreg b[1];
creg ca[2];
creg cb[1];

// Prepare a Bell pair in register a
h a[0];
cx a[0], a[1];

// Apply X to register b
x b[0];

// Cross-register CNOT
cx a[1], b[0];

measure a[0] -> ca[0];
measure a[1] -> ca[1];
measure b[0] -> cb[0];
