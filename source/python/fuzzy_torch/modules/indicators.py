import torch

# Ненормированная гауссиана.
def gauss_pdf(input, out=None):
    out = torch.exp(-input**2 / 2)
    return out


# Сигмоида с модулем.
def abs_sigmoid(input, out=None):
    doubled_input = 2 * input
    out = (doubled_input / (1 + torch.abs(doubled_input)) + 1) / 2
    return out


class Linear(torch.nn.Module):
    """
    Линейное преобразование и некоторая функция.
    """

    def __init__(self, in_features, out_features, function, weight=None, offset=None):
        super().__init__()
        self.in_features = in_features
        self.linear = torch.nn.Linear(in_features, out_features)
        self.function = function

        with torch.no_grad():
            if not (weight is None):
                weight = torch.tensor(weight, dtype=torch.float32)
                weight = weight.reshape((out_features, in_features))

                self.linear.weight = torch.nn.Parameter(weight)

            if not (offset is None):
                offset = torch.tensor(offset, dtype=torch.float32)
                offset = offset.reshape((out_features))

                self.linear.bias = torch.nn.Parameter(-self.linear.weight * offset)


    def forward(self, x):
        return self.function(self.linear(x))



class DotProductBased(Linear):
    """
    Скалярное произведение и некоторая функция.
    """

    def __init__(self, in_features, function, weight=None, offset=None):
        super().__init__(in_features, 1, function, weight, offset)


class Heaviside(DotProductBased):
    """
    Скалярное произведение и ступенька.
    """

    def __init__(self, in_features, weight=None, offset=None):
        super().__init__(
                in_features,
                lambda x : torch.heaviside(x, torch.tensor([0.5])),
                weight,
                offset)


class Sigmoid(DotProductBased):
    """
    Скалярное произведение и логистическая сигмоида.
    """

    def __init__(self, in_features, weight=None, offset=None):
        super().__init__(
                in_features,
                torch.sigmoid,#lambda x : torch.sigmoid(4 * x),
                weight,
                offset)


class AbsSigmoid(DotProductBased):
    """
    Скалярное произведение и сигмоида с модулем.
    """

    def __init__(self, in_features, weight=None, offset=None):
        super().__init__(
                in_features,
                abs_sigmoid,
                weight,
                offset)



class QuadricBased(Linear):
    """
    Линейное преобразование, норма и некоторая функция.
    """

    def __init__(self, in_features, function, weight=None, offset=None):
        super().__init__(
                in_features,
                in_features,
                lambda x : function(torch.norm(x, dim=1, keepdim=True)),
                weight,
                offset)


    def forward(self, x):
        return self.function(torch.linalg.norm(self.linear(x), dim=1, keepdim=True))


class Gaussian(QuadricBased):
    """
    Линейное преобразование, норма и гауссиана.
    """

    def __init__(self, in_features, weight=None, offset=None):
        super().__init__(in_features, gauss_pdf, weight, offset)
