// Demonstrates parametric rotation gates
OPENQASM 2.0;
include "qelib1.inc";

qreg q[1];
creg c[1];

// Rotate around X axis by pi/2 (creates superposition)
rx(pi/2) q[0];

// Then rotate around Z
rz(pi/4) q[0];

measure q[0] -> c[0];
