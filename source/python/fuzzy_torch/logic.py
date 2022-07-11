import torch

_EPSILON = 1.0e-7


class FuzzyLogic:
    """
    Базовый класс, реализующий нечеткую логику.
    """

    def __init__(self):
        pass


    @classmethod
    def fuzzy_and(cls, input, other, *, out=None):
        raise NotImplementedError


    @classmethod
    def fuzzy_not(cls, input, *, out=None):
        return 1 - input


    @classmethod
    def fuzzy_or(cls, input, other, *, out=None):
        out = cls.fuzzy_not(cls.fuzzy_and(cls.fuzzy_not(input), cls.fuzzy_not(other)))
        return out


    @classmethod
    def fuzzy_impl(cls, input, other, *, out=None):
        raise NotImplementedError


    @classmethod
    def fuzzy_many_and(cls, *args, out=None):
        out = args[0]
        for i in range(1, len(args)):
            out = cls.fuzzy_and(out, args[i])
        return out


    @classmethod
    def fuzzy_many_or(cls, *args, out=None):
        out = args[0]
        for i in range(1, len(args)):
            out = cls.fuzzy_or(out, args[i])
        return out



class Godel(FuzzyLogic):
    """
    Гёделева (min) логика.
    """

    @classmethod
    def fuzzy_and(cls, input, other, *, out=None):
        return torch.minimum(input, other, out=out)


    @classmethod
    def fuzzy_or(cls, input, other, *, out=None):
        return torch.maximum(input, other, out=out)


    @classmethod
    def fuzzy_impl(cls, input, other, *, out=None):
        out = torch.maximum(torch.le(input, other), other)
        return out



class Product(FuzzyLogic):
    """
    Вероятностная (произведение) логика.
    """

    @classmethod
    def fuzzy_and(cls, input, other, *, out=None):
        return torch.multiply(input, other, out=out)


    @classmethod
    def fuzzy_or(cls, input, other, *, out=None):
        out = input + other - input * other
        return out


    @classmethod
    def fuzzy_impl(cls, input, other, *, out=None, epsilon=_EPSILON):
        out = torch.minimum(torch.ones_like(input), torch.clamp((other + epsilon) / (input + epsilon), 0, 1))
        return out



class Lukasiewicz(FuzzyLogic):
    """
    Логика Лукасевича.
    """

    @classmethod
    def fuzzy_and(cls, input, other, *, out=None):
        out = torch.maximum(torch.zeros_like(input), input + other - 1)
        return out

    @classmethod
    def fuzzy_or(cls, input, other, *, out=None):
        out = torch.minimum(input + other, torch.ones_like(input))
        return out

    @classmethod
    def fuzzy_impl(cls, input, other, *, out=None):
        #condition = torch.le(input, other)
        #out = condition + (~condition) * (1 - input + other)
        out = torch.minimum(torch.ones_like(input), 1 - input + other)
        return out



class Nilpotent(FuzzyLogic):
    """
    Нильпотентная логика.
    """

    @classmethod
    def fuzzy_and(cls, input, other, *, out=None):
        out = torch.greater(input + other, 1) * torch.minimum(input, other)
        return out

    @classmethod
    def fuzzy_or(cls, input, other, *, out=None):
        sum_io = input + other
        out = torch.less(sum_io, torch.ones_like(sum_io)) * torch.maximum(input, other) + torch.ge(sum_io, 1)
        return out

    @classmethod
    def fuzzy_impl(cls, input, other, *, out=None):
        out = torch.maximum(torch.le(input, other), torch.maximum(1 - input, other))
        return out



class Hamacher(FuzzyLogic):
    """
    Логика Хамахера.
    """

    @classmethod
    def fuzzy_and(cls, input, other, *, out=None, epsilon=_EPSILON):
        prod_io = input * other
        out = prod_io / (input + other - prod_io + epsilon)
        return out

    @classmethod
    def fuzzy_or(cls, input, other, *, out=None, epsilon=_EPSILON):
        prod_io = input * other
        out = torch.clamp((input + other - 2 * prod_io + epsilon) / (1 - prod_io + epsilon), 0, 1)
        # Почему-то иногда значение получается не из [0;1].
        # К регуляризации это не имеет никакого отношения.
        return out

    @classmethod
    def fuzzy_impl(cls, input, other, *, out=None, epsilon=_EPSILON):
        prod_io = input * other
        condition = torch.le(input, other)
        out = condition + (~condition) * prod_io / ((input - other) * (1 + epsilon) + prod_io)
        return out
