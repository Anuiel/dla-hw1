import typing as tp

import editdistance


class Equatable[T](tp.Protocol):
    def eq(self, other: T) -> bool: ...

T = tp.TypeVar('T', bound=Equatable)


def calc_cer(target_text: tp.Collection[T], predicted_text: tp.Collection[T]) -> float:
    if len(target_text) == 0:
        return 1.0 if len(predicted_text) > 0 else 0.0

    edit_distance = editdistance.eval(target_text, predicted_text)
    cer = edit_distance / len(target_text)
    
    return cer


def calc_wer(target_text: str, predicted_text: str) -> float:
    return calc_cer(target_text.split(), predicted_text.split())
