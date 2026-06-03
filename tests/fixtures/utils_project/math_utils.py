def multiply(a: int, b: int) -> int:
    return a * b


def divide(a: int, b: int) -> int:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a // b


class AdvancedCalculator:
    def power(self, base: int, exp: int) -> int:
        return base ** exp

    def modulo(self, a: int, b: int) -> int:
        return a % b
