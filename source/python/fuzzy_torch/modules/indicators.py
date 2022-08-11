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


# Треугольное распределение.
def triangle_pdf(input, out=None):
    out = torch.clamp(1 - torch.abs(input), min=0)
    return out



class Singletone(torch.nn.Module):
    """
    Индикаторная функция множества из одного элемента.
    """

    def __init__(self, offset=None):
        super().__init__()
        if offset is None:
            self.offset = torch.nn.Parameter(torch.zeros(1))
        else:
            self.offset = torch.nn.Parameter(offset)


    def forward(self, input):
        return torch.eq(input, self.offset)



# Отдельный набор одномерных масштабируемых функций принадлежности.
class ScalableIndicator(torch.nn.Module):
    """
    Базовый класс масштабируемых (вес + сдвиг) индикаторных функций.
    """

    def __init__(self, weight=None, offset=None):
        super().__init__()

        with torch.no_grad():
            if weight is None:
                self.weight = torch.nn.Parameter(torch.empty(1))
                torch.nn.init.uniform_(self.weight, -1, 1)
            else:
                self.weight = torch.nn.Parameter(torch.tensor(1) * weight)

            if offset is None:
                self.offset = torch.nn.Parameter(torch.empty(1))
                torch.nn.init.uniform_(self.offset, -1, 1)
            else:
                self.offset = torch.nn.Parameter(torch.tensor(1) * offset)


    def forward(self, input):
        raise NotImplementedError


class Sigmoid(ScalableIndicator):
    """
    Логистическая сигмоида.
    """

    def __init__(self, weight=None, offset=None):
        super().__init__(weight, offset)


    def forward(self, input):
        return torch.sigmoid(self.weight * (input - self.offset))


class LogSigmoid(ScalableIndicator):
    """
    Логистическая сигмоида в логарифмическом масштабе.
    """

    def __init__(self, weight=None, offset=None):
        super().__init__(weight, offset)


    def forward(self, input):
        return 1 / (torch.pow(input, -self.weight) * torch.exp(self.weight * self.offset) + 1)


class BiSigmoid(ScalableIndicator):
    """
    Логистическая сигмоида в логсигмоидном масштабе.
    """

    def __init__(self, weight=None, offset=None):
        super().__init__(weight, offset)


    def forward(self, input):
        return 1 / (torch.pow(1 / input - 1, self.weight) * torch.exp(self.weight * self.offset) + 1)


class Gaussian(ScalableIndicator):
    """
    Гауссиана.
    """

    def __init__(self, weight=None, offset=None):
        super().__init__(weight, offset)


    def forward(self, input):
        return gauss_pdf(self.weight * (input - self.offset))

