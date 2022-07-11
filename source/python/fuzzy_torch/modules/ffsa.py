import torch


class FuzzyTransition:
    """
    Структура для описания нечеткого перехода.
    """

    def __init__(self, from_idx, to_idx, condition):
        self.from_idx = from_idx
        self.to_idx = to_idx
        self.condition = condition


class FuzzyFSA(torch.nn.Module):
    """
    Базовый класс нечеткого конечного автомата.
    """

    def __init__(self, logic):
        super().__init__()
        self.logic = logic
        self.states = []
        self.transitions = []


    def forward(self, input, activations):
        raise NotImplementedError



class TimeIndependentFFSA(FuzzyFSA):
    """
    Не зависящий от времени нечеткий конечный автомат.
    """

    def __init__(self, logic):
        super().__init__(logic=logic)


    def forward(self, input, activations):
        assert (activations.dim() == 2) and (activations.size()[1] == len(self.states))

        from_activations = torch.zeros_like(activations)
        to_activations   = torch.zeros_like(activations)

        for transition in self.transitions:
            # Не работает, так как используется in-place срезы (нарушает дерево производных).
            #condition = transition.condition(input).squeeze()
            #from_activations[:, transition.from_idx] = self.logic.fuzzy_or(from_activations[:, transition.from_idx], condition)
            #to_activations[:, transition.to_idx] = self.logic.fuzzy_or(to_activations[:, transition.to_idx], condition)

            # Условие перехода, размноженное на все состояния.
            condition = transition.condition(input).repeat(1, len(self.states))

            # Вспомогательные тензоры-множители.
            helper_from = torch.zeros_like(condition)
            helper_from[:, transition.from_idx] = 1

            helper_to = torch.zeros_like(condition)
            helper_to[:, transition.to_idx] = 1

            from_activations = self.logic.fuzzy_or(from_activations, helper_from * condition)
            to_activations = self.logic.fuzzy_or(to_activations, helper_to * condition)

        new_activations = self.logic.fuzzy_and(
            self.logic.fuzzy_or(activations, to_activations),
            self.logic.fuzzy_not(from_activations)
        )

        return new_activations
