"""Chapter 2: forward and backward values on the graph of one squared error."""

# One sample and the starting parameters.
x, y = 1.0, 2.0
w, b = 0.0, 0.0

# Forward pass: each node computes its value from its inputs.
u = w * x            # product node
y_hat = u + b        # prediction
r = y_hat - y        # residual
loss = r ** 2        # squared error

# Backward pass: each node multiplies the derivative arriving from above
# by its own local derivative (the chain rule).
dL_dr = 2 * r
dL_dyhat = dL_dr * 1.0
dL_db = dL_dyhat * 1.0
dL_du = dL_dyhat * 1.0
dL_dw = dL_du * x

if __name__ == "__main__":
    print(f"forward:  u = {u}, y_hat = {y_hat}, r = {r}, L = {loss}")
    print(f"backward: dL/dr = {dL_dr}, dL/db = {dL_db}, dL/dw = {dL_dw}")
