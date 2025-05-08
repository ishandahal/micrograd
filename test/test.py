import torch
import torch.nn.functional as F
from micrograd.micrograd import Value

# Simple test
def simple_test():
    tol = 1e-5
    y = torch.tensor([1.0])
    x1 = torch.tensor([2.0], requires_grad=True)
    x2 = torch.tensor([3.0], requires_grad=True)
    w1 = torch.tensor([-1.0], requires_grad=True)
    w2 = torch.tensor([1.5], requires_grad=True)
    res = w1 * x1 + w2 * x2
    act = F.relu(res)
    error = F.mse_loss(act, y)
    error.backward()
    
    x1_v = Value(x1.item())
    x2_v = Value(x2.item())
    w1_v = Value(w1.item())
    w2_v = Value(w2.item())
    res_v = w1_v * x1_v + w2_v * x2_v
    act_v = res_v.relu()
    error_v = (y.item() - act_v) ** 2
    error_v.backward()

    foo = abs(w1.grad - w1_v.grad)
    assert (w1.grad.item() - w1_v.grad) < tol, "weights are different"
    assert (w2.grad.item() - w2_v.grad) < tol, "weights are different"
    print("Simple test passed!")


def test_more_operations():
    a = torch.tensor([2.0]).double()
    b = torch.tensor([3.0], requires_grad=True)#.double()
    c = torch.tensor([-1.5]).double()
    d = torch.tensor([2.4], requires_grad=True)#.double()
    e = torch.tensor([1.0], requires_grad=True)#.double()
    f = a * b + 4
    f = f + 1
    g = 91 + c * d
    g = 5 + g.relu()
    h = (f * g).relu()
    # print(f"{h=}")
    h = h ** 2
    i = (h * e)
    # print(f"{h.item()=}, {e.item()=}")
    # print(f"{i=}")
    i.backward()
    # print(b.grad, d.grad, e.grad)
    b_d = b; d_d = d; e_d = e

    a = Value(a.item())
    b = Value(b.item())
    c = Value(c.item())
    d = Value(d.item())
    e = Value(e.item())
    f = a * b + 4
    f = f + 1
    g = 91 + c * d
    g = 5 + g.relu()
    h = (f * g).relu()
    # print(f"{h=}")
    h = h ** 2
    i = (h * e)
    i.backward()
    # print(f"{i=}")
    # print(b.grad, d.grad, e.grad)
    bv_d = b; dv_d = d; ev_d = e

    tol = 1e-5
    assert (bv_d.grad - b_d.grad) < tol, "weights are different"
    assert (dv_d.grad - d_d.grad) < tol, "weights are different"
    assert (ev_d.grad - e_d.grad) < tol, "weights are different"

def test_more_ops():

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a) 
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3 
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu() 
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-5
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    print(f"{amg.grad=}, {apt.grad.item()}")
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol


if __name__ == "__main__":
    simple_test()
    test_more_operations()
    test_more_ops()
