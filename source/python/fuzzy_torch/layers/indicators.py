import torch

# Ненормированная гауссиана.
def gauss_pdf(input, out=None):
    out = torch.exp(-input**2 / 2)
    return out


class Hyperplane(torch.nn.Module):
    """
    Скалярное произведение и некоторая функция.
    """

    def __init__(self, in_features, function, weight=None, offset=None):
        super().__init__()
        self.in_features = in_features
        self.linear = torch.nn.Linear(in_features, 1)
        self.function = function

        with torch.no_grad():
            if not (weight is None):
                weight = torch.tensor(weight, dtype=torch.float32)
                weight = weight.reshape((in_features, 1))

                self.linear.weight = torch.nn.Parameter(weight)

            if not (offset is None):
                offset = torch.tensor(offset, dtype=torch.float32)
                offset = offset.reshape((1))

                self.linear.bias = torch.nn.Parameter(torch.norm(self.linear.weight) * (-offset))


    def forward(self, x):
        return self.function(self.linear(x))



class Sigmoid(Hyperplane):
    """
    Скалярное произведение и сигмоида.
    """

    def __init__(self, in_features, weight=None, offset=None):
        super().__init__(in_features, torch.sigmoid, weight, offset)



class Quadric(torch.nn.Module):
    """
    Линейное преобразование и некоторая функция.
    """

    def __init__(self, in_features, function, weight=None, offset=None):
        super().__init__()
        self.in_features = in_features
        self.linear = torch.nn.Linear(in_features, in_features)
        self.function = function

        with torch.no_grad():
            if not (weight is None):
                weight = torch.tensor(weight, dtype=torch.float32)
                weight = weight.reshape((in_features, in_features))

                self.linear.weight = torch.nn.Parameter(weight)

            if not (offset is None):
                offset = torch.tensor(offset, dtype=torch.float32)
                offset = offset.reshape((in_features))

                self.linear.bias = torch.nn.Parameter(torch.norm(self.linear.weight) * (-offset))


    def forward(self, x):
        return self.function(torch.linalg.norm(self.linear(x), dim=1, keepdim=True))


class Gaussian(Quadric):
    """
    Гауссиана.
    """

    def __init__(self, in_features, weight=None, offset=None):
        super().__init__(in_features, gauss_pdf, weight, offset)


