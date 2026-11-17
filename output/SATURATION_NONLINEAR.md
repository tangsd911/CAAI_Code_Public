# Input-saturation nonlinearity stress test

This test adds component-wise actuator saturation to the LTI plant; it is not a full nonlinear-plant validation.</br>
## humanoid
- reference bound = 5.973608
- factor=0.25, bound=1.493402: DRC sat=1.0000, LQR sat=0.9996.
- factor=0.50, bound=2.986804: DRC sat=1.0000, LQR sat=0.9991.
- factor=1.00, bound=5.973608: DRC sat=0.0109, LQR sat=0.0106.
- factor=2.00, bound=11.947216: DRC sat=0.0000, LQR sat=0.0000.
## microgrid
- reference bound = 0.420593
- factor=0.25, bound=0.105148: DRC sat=0.8583, LQR sat=0.5000.
- factor=0.50, bound=0.210297: DRC sat=0.3362, LQR sat=0.0256.
- factor=1.00, bound=0.420593: DRC sat=0.0096, LQR sat=0.0000.
- factor=2.00, bound=0.841187: DRC sat=0.0000, LQR sat=0.0000.
