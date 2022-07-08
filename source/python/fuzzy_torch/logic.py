import torch

_EPSILON = 1.0e-7


class FuzzyLogic:
    """
    Базовый класс, реализующий нечеткую логику.
    """

    def __init__(self):
        pass


    @staticmethod
    def fuzzy_and(input, other, *, out=None):
        raise NotImplementedError


    @staticmethod
    def fuzzy_not(input, *, out=None):
        return 1 - input


    @staticmethod
    def fuzzy_or(input, other, *, out=None):
        out = fuzzy_not(fuzzy_and(fuzzy_not(input), fuzzy_not(other)))
        return out


    @staticmethod
    def fuzzy_impl(input, other, *, out=None):
        raise NotImplementedError



class Godel(FuzzyLogic):
    """
    Гёделева (min) логика.
    """

    @staticmethod
    def fuzzy_and(input, other, *, out=None):
        return torch.minimum(input, other, out=out)


    @staticmethod
    def fuzzy_or(input, other, *, out=None):
        return torch.maximum(input, other, out=out)


    @staticmethod
    def fuzzy_impl(input, other, *, out=None):
        out = torch.maximum(torch.le(input, other), other)
        return out



class Product(FuzzyLogic):
    """
    Вероятностная (произведение) логика.
    """

    @staticmethod
    def fuzzy_and(input, other, *, out=None):
        return torch.multiply(input, other, out=out)


    @staticmethod
    def fuzzy_or(input, other, *, out=None):
        out = input + other - input * other
        return out


    @staticmethod
    def fuzzy_impl(input, other, *, out=None, epsilon=_EPSILON):
        out = torch.minimum(torch.ones_like(input), (other + epsilon) / (input + epsilon))
        return out



class Lukasiewicz(FuzzyLogic):
    """
    Логика Лукасевича.
    """

    @staticmethod
    def fuzzy_and(input, other, *, out=None):
        out = torch.maximum(torch.zeros_like(input), input + other - 1)
        return out

    @staticmethod
    def fuzzy_or(input, other, *, out=None):
        out = torch.minimum(input + other, torch.ones_like(input))
        return out

    @staticmethod
    def fuzzy_impl(input, other, *, out=None):
        #condition = torch.le(input, other)
        #out = condition + (~condition) * (1 - input + other)
        out = torch.minimum(torch.ones_like(input), 1 - input + other)
        return out



class Nilpotent(FuzzyLogic):
    """
    Нильпотентная логика.
    """

    @staticmethod
    def fuzzy_and(input, other, *, out=None):
        out = torch.greater(input + other, 1) * torch.minimum(input, other)
        return out

    @staticmethod
    def fuzzy_or(input, other, *, out=None):
        sum_io = input + other
        out = torch.less(sum_io, torch.ones_like(sum_io)) * torch.maximum(input, other) + torch.ge(sum_io, 1)
        return out

    @staticmethod
    def fuzzy_impl(input, other, *, out=None):
        out = torch.maximum(torch.le(input, other), torch.maximum(1 - input, other))
        return out



class Hamacher(FuzzyLogic):
    """
    Логика Хамахера.
    """

    @staticmethod
    def fuzzy_and(input, other, *, out=None, epsilon=_EPSILON):
        prod_io = input * other
        out = prod_io / (input + other - prod_io + epsilon)
        return out

    @staticmethod
    def fuzzy_or(input, other, *, out=None, epsilon=_EPSILON):
        prod_io = input * other
        out = (input + other - 2 * prod_io + epsilon) / (1 - prod_io + epsilon)
        #out = (input + other) / (1 - input * other)
        return out

    @staticmethod
    def fuzzy_impl(input, other, *, out=None, epsilon=_EPSILON):
        prod_io = input * other
        condition = torch.le(input, other)
        out = condition + (~condition) * prod_io / ((input - other) * (1 + epsilon) + prod_io)
        return out
