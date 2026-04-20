// Demonstrates the universal U3 gate
// U3(theta, phi, lambda) is the most general single-qubit rotation
OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg c[2];

// U3 equivalent to H gate: U3(pi/2, 0, pi)
u3(pi/2, 0, pi) q[0];

// U3 equivalent to X gate: U3(pi, 0, pi)
u3(pi, 0, pi) q[1];

measure q[0] -> c[0];
measure q[1] -> c[1];
