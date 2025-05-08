
class Value:
    """Class that implements basic autograd functionality similar to pytorch"""

    def __init__(self, data, child=(), label="", op=""):
        self.data = data
        self._prev = set(child)
        self.label = label
        self._op = op
        self.grad = 0 # Initially gradient is zero
        self._backward = lambda : None

    def __add__(self, other):
        if isinstance(other, (int, float)): other = Value(other)
        out = Value(self.data + other.data, child=(self, other), op="+")
        def backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = backward
        return out

    def __radd__(self, other):
        if not isinstance(other, (int, float)): print("Invalid add operation. Used non numeric type.")
        return self.__add__(Value(other))

    def __sub__(self, other):
        return self.__add__(other * -1)

    def __rsub__(self, other):
        other = Value(other)
        return other.__add__(-1 * self)

    def __mul__(self, other):
        if isinstance(other, (int, float)): other = Value(other)
        out = Value(self.data * other.data, child=(self, other), op="*")
        def backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = backward
        return out
        
    def __rmul__(self, other):
        return self.__mul__(Value(other))

    def __pow__(self, exp):
        if isinstance(exp, Value): exp = exp.data
        out = Value(self.data ** exp, child=(self,), op="exponentiate")
        def backward():
            # print("backward was called")
            # print(f"before update: {self.grad=}")
            self.grad += out.grad * (exp * self.data ** (exp - 1))
            # print(f"after update: {self.grad=}")
            # print(f"{self.grad=}, {out.data=}, {self.data=}, {exp=}")
        out._backward = backward
        return out

    def __neg__(self):
        # print("__neg__ was called")
        out = Value(-1 * self.data, child=(self,), op="negate")
        def backward():
            self.grad += out.grad * -1 
        out._backward = backward
        return out

    def __truediv__(self, other):
        if isinstance(other, (int, float)): other = Value(other)
        out = Value(self.data / other.data, child=(self, other), op="/")
        def backward():
            self.grad += out.grad * (1 / other.data)
            other.grad += out.grad * ( (- self.data) / (other.data ** 2) ) # (-x / y^2)
        out._backward = backward
        return out
        
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)): other = Value(other)
        self, other = other, self
        return self.__truediv__(other)
        
        
    def relu(self):
        res = max(0, self.data)
        out = Value(res, child=(self,))
        def backward():
            local_grad = 1 if self.data > 0 else 0
            self.grad += out.grad * local_grad
        out._backward = backward
        return out

    def backward(self):
        self.grad = 1 # Derivative wrt self is 1
        visited = set()
        topo = []
        def topological_sort(root: Value):
            if root not in visited:
                visited.add(root)
                for child in root._prev:
                    topological_sort(child)
                topo.append(root)
        topological_sort(self)
        _ = [node._backward() for node in reversed(topo)]
        
    def __repr__(self):
        return f"Val: {self.data}"
