import torch


class FuzzyTransition(torch.nn.Module):
    """
    Структура для описания нечеткого перехода.
    """

    def __init__(self, from_idx, to_idx, condition):
        super().__init__()
        self.from_idx = from_idx
        self.to_idx = to_idx
        self.condition = condition

    def forward(self, *args):
        return self.condition(*args)



class FuzzyFSA(torch.nn.Module):
    """
    Базовый класс нечеткого конечного автомата.
    """

    def __init__(self, logic, normalize=False):
        super().__init__()
        self.logic = logic
        self.normalize = normalize
        self.states = []
        self.transitions = []


    def forward(self, input, activations):
        raise NotImplementedError


    def normalize_activation(self, activation):
        """
        Нормализация активаций состояний (опцинально).
        """

        return activation / torch.sum(activation, dim=1, keepdim=True)



class TimeIndependentFFSA(FuzzyFSA):
    """
    Не зависящий от времени нечеткий конечный автомат.
    """

    def __init__(self, logic, normalize=False):
        super().__init__(logic=logic, normalize=normalize)


    def forward(self, input, activation):
        assert (activation.dim() == 2) and (activation.size()[1] == len(self.states))

        from_activation = torch.zeros_like(activation)
        to_activation   = torch.zeros_like(activation)

        for transition in self.transitions:
            # Не работает, так как используется in-place срезы (нарушает дерево производных).
            #condition = transition.condition(input).squeeze()
            #from_activations[:, transition.from_idx] = self.logic.fuzzy_or(from_activation[:, transition.from_idx], condition)
            #to_activations[:, transition.to_idx] = self.logic.fuzzy_or(to_activation[:, transition.to_idx], condition)

            # Условие перехода, размноженное на все состояния.
            # Конъюнкция активности состояния, из которого происходит переход, и условия самого перехода.
            condition = self.logic.fuzzy_and(activation[:, transition.from_idx, None], transition(input)).repeat(1, len(self.states))

            # Вспомогательные тензоры-множители.
            helper_from = torch.zeros_like(condition)
            helper_from[:, transition.from_idx] = 1

            helper_to = torch.zeros_like(condition)
            helper_to[:, transition.to_idx] = 1

            from_activation = self.logic.fuzzy_or(from_activation, helper_from * condition)
            to_activation = self.logic.fuzzy_or(to_activation, helper_to * condition)

        new_activation = self.logic.fuzzy_and(
            self.logic.fuzzy_or(activation, to_activation),
            self.logic.fuzzy_not(from_activation)
        )

        if self.normalize:
            new_activation = self.normalize_activation(new_activation)

        return new_activation



class ContinuousFuzzyTransition(FuzzyTransition):
    """
    Структура для описания плавного нечеткого перехода.
    """

    def __init__(self, from_idx, to_idx, condition, speed=1.0):
        super().__init__(from_idx=from_idx, to_idx=to_idx, condition=condition)
        self.speed = torch.nn.Parameter(torch.ones(1) * speed)

    def forward(self, *args):
        return self.condition(*args)



class TimeDependentFFSA(FuzzyFSA):
    """
    Зависящий от времени нечеткий конечный автомат.
    """

    def __init__(self, logic, normalize=False):
        super().__init__(logic=logic, normalize=normalize)


    def forward(self, input, activation, dt):
        assert (activation.dim() == 2) and (activation.size()[1] == len(self.states))

        from_activation = torch.zeros_like(activation)
        to_activation   = torch.zeros_like(activation)

        for transition in self.transitions:
            # Не работает, так как используется in-place срезы (нарушает дерево производных).
            #condition = transition.condition(input).squeeze()
            #from_activations[:, transition.from_idx] = self.logic.fuzzy_or(from_activation[:, transition.from_idx], condition)
            #to_activations[:, transition.to_idx] = self.logic.fuzzy_or(to_activation[:, transition.to_idx], condition)

            # Условие перехода, размноженное на все состояния.
            # Конъюнкция активности состояния, из которого происходит переход, и условия самого перехода.
            condition = self.logic.fuzzy_and(activation[:, transition.from_idx, None], transition(input)).repeat(1, len(self.states))
            speed = condition * transition.speed

            # Вспомогательные тензоры-множители.
            helper_from = torch.zeros_like(condition)
            helper_from[:, transition.from_idx] = 1

            helper_to = torch.zeros_like(condition)
            helper_to[:, transition.to_idx] = 1

            from_activation += helper_from * speed
            to_activation += helper_to * speed


        derivative = to_activation - from_activation

        new_activation = torch.clip(activation + derivative * dt, 0, 1)

        if self.normalize:
            new_activation = self.normalize_activation(new_activation)

        return new_activation
