import torch

_EPSILON = 1.0e-7


class FuzzyLogic:
    """
    Базовый класс, реализующий нечеткую логику.
    """

    def __init__(self):
        pass


    @classmethod
    def _fuzzy_and(cls, input, other, *, out=None):
        """
        Бинарная операция "И".

        Параметры
        ---------
        input : torch.tensor
            Первый аргумент.
        other : torch.tensor
            Второй аргумент.
        out : torch.tensor, опционально
            Переменная для записи.
        """

        raise NotImplementedError


    @classmethod
    def fuzzy_not(cls, input, *, out=None):
        """
        Унарная операция "НЕ".

        Параметры
        ---------
        input : torch.tensor
            Аргумент.
        out : torch.tensor, опционально
            Переменная для записи.
        """

        # 1 - input
        return torch.add(torch.neg(input), 1, out=out)


    @classmethod
    def _fuzzy_or(cls, input, other, *, out=None):
        """
        Бинарная операция "ИЛИ".

        Параметры
        ---------
        input : torch.tensor
            Первый аргумент.
        other : torch.tensor
            Второй аргумент.
        out : torch.tensor, опционально
            Переменная для записи.
        """

        # NOT(AND(NOT(input), NOT(other)))
        return cls.fuzzy_not(cls._fuzzy_and(cls.fuzzy_not(input), cls.fuzzy_not(other)), out=out)


    @classmethod
    def fuzzy_impl(cls, input, other, *, out=None):
        """
        Бинарная операция "импликация".

        Параметры
        ---------
        input : torch.tensor
            Первый аргумент.
        other : torch.tensor
            Второй аргумент.
        out : torch.tensor, опционально
            Переменная для записи.
        """

        raise NotImplementedError


    @classmethod
    def fuzzy_equiv(cls, input, other, out=None):
        """
        Бинарная операция "эквивалентно".

        Параметры
        ---------
        input : torch.tensor
            Первый аргумент.
        other : torch.tensor
            Второй аргумент.
        out : torch.tensor, опционально
            Переменная для записи.
        """

        # AND(input -> other, other -> input)
        return cls.fuzzy_and(cls.fuzzy_impl(input, other), cls.fuzzy_impl(other, input), out=out)


    @classmethod
    def fuzzy_and(cls, *args, out=None):
        """
        Операция "И" (множество аргументов).

        Параметры
        ---------
        args : list(torch.tensor)
            Массив аргументов.
        out : torch.tensor, опционально
            Переменная для записи.
        """

        result = args[0]
        for index in range(1, len(args)):
            result = cls._fuzzy_and(result, args[index])

        if out is None:
            return result
        else:
            return out.copy_(result)


    @classmethod
    def fuzzy_or(cls, *args, out=None):
        """
        Операция "ИЛИ" (множество аргументов).

        Параметры
        ---------
        args : list(torch.tensor)
            Массив аргументов.
        out : torch.tensor, опционально
            Переменная для записи.
        """

        result = args[0]
        for index in range(1, len(args)):
            result = cls._fuzzy_or(result, args[index])

        if out is None:
            return result
        else:
            return out.copy_(result)



class Godel(FuzzyLogic):
    """
    Гёделева (min) логика.
    """

    @classmethod
    def _fuzzy_and(cls, input, other, *, out=None):
        return torch.minimum(input, other, out=out)


    @classmethod
    def _fuzzy_or(cls, input, other, *, out=None):
        return torch.maximum(input, other, out=out)


    @classmethod
    def fuzzy_impl(cls, input, other, *, out=None):
        return torch.maximum(torch.le(input, other), other, out=out)



class Product(FuzzyLogic):
    """
    Вероятностная (произведение) логика.
    """

    @classmethod
    def _fuzzy_and(cls, input, other, *, out=None):
        return torch.multiply(input, other, out=out)


    @classmethod
    def _fuzzy_or(cls, input, other, *, out=None):
        # input + other - input * other
        return torch.sub(input + other, input * other, out=out)


    @classmethod
    def fuzzy_impl(cls, input, other, *, out=None, epsilon=_EPSILON):
        return torch.minimum(torch.ones_like(input), torch.clamp((other + epsilon) / (input + epsilon), 0, 1), out=out)



class Lukasiewicz(FuzzyLogic):
    """
    Логика Лукасевича.
    """

    @classmethod
    def _fuzzy_and(cls, input, other, *, out=None):
        return torch.maximum(torch.zeros_like(input), input + other - 1, out=out)


    @classmethod
    def _fuzzy_or(cls, input, other, *, out=None):
        return torch.minimum(input + other, torch.ones_like(input), out=out)


    @classmethod
    def fuzzy_impl(cls, input, other, *, out=None):
        return torch.minimum(torch.ones_like(input), 1 - input + other, out=out)



class Nilpotent(FuzzyLogic):
    """
    Нильпотентная логика.
    """

    @classmethod
    def _fuzzy_and(cls, input, other, *, out=None):
        return torch.multiply(torch.greater(input + other, 1), torch.minimum(input, other), out=out)


    @classmethod
    def _fuzzy_or(cls, input, other, *, out=None):
        sum_io = input + other
        return torch.add(torch.less(sum_io, torch.ones_like(sum_io)) * torch.maximum(input, other), torch.ge(sum_io, 1), out=out)


    @classmethod
    def fuzzy_impl(cls, input, other, *, out=None):
        return torch.maximum(torch.le(input, other), torch.maximum(1 - input, other), out=out)



class Hamacher(FuzzyLogic):
    """
    Логика Хамахера.
    """

    @classmethod
    def _fuzzy_and(cls, input, other, *, out=None, epsilon=_EPSILON):
        prod_io = input * other
        return torch.div(prod_io, input + other - prod_io + epsilon, out=out)


    @classmethod
    def _fuzzy_or(cls, input, other, *, out=None, epsilon=_EPSILON):
        prod_io = input * other
        return torch.clamp((input + other - 2 * prod_io + epsilon) / (1 - prod_io + epsilon), 0, 1, out=out)
        # Почему-то иногда значение получается не из [0; 1].
        # К регуляризации это не имеет никакого отношения.
        # Требуется использовать clamp.


    @classmethod
    def fuzzy_impl(cls, input, other, *, out=None, epsilon=_EPSILON):
        prod_io = input * other
        condition = torch.le(input, other)
        return torch.add(condition, (~condition) * prod_io / ((input - other) * (1 + epsilon) + prod_io), out=out)
