"""Calculadora de precision: AST con lista blanca de nodos y aritmetica Decimal.

No se usa eval(). Solo se evaluan nodos explicitamente permitidos, por lo que
nombres, llamadas a funciones, atributos e indexaciones quedan descartados.
"""

import ast
from decimal import Decimal, DivisionByZero, InvalidOperation, getcontext

getcontext().prec = 28

MAX_LONGITUD = 200
MAX_EXPONENTE = 100

_BINARIOS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Mod: lambda a, b: a % b,
}


class ErrorCalculo(ValueError):
    pass


def _evaluar(nodo: ast.AST) -> Decimal:
    if isinstance(nodo, ast.Expression):
        return _evaluar(nodo.body)

    if isinstance(nodo, ast.Constant):
        if isinstance(nodo.value, bool) or not isinstance(nodo.value, (int, float)):
            raise ErrorCalculo("solo se permiten numeros")
        return Decimal(str(nodo.value))

    if isinstance(nodo, ast.UnaryOp):
        if isinstance(nodo.op, ast.USub):
            return -_evaluar(nodo.operand)
        if isinstance(nodo.op, ast.UAdd):
            return _evaluar(nodo.operand)
        raise ErrorCalculo("operador unario no permitido")

    if isinstance(nodo, ast.BinOp):
        izquierda = _evaluar(nodo.left)
        derecha = _evaluar(nodo.right)

        if isinstance(nodo.op, ast.Pow):
            if abs(derecha) > MAX_EXPONENTE:
                raise ErrorCalculo(f"exponente mayor al maximo permitido ({MAX_EXPONENTE})")
            if derecha != derecha.to_integral_value():
                raise ErrorCalculo("el exponente debe ser entero")
            return izquierda ** int(derecha)

        operacion = _BINARIOS.get(type(nodo.op))
        if operacion is None:
            raise ErrorCalculo("operador no permitido")
        return operacion(izquierda, derecha)

    raise ErrorCalculo(f"expresion no permitida: {type(nodo).__name__}")


def calcular(expresion: str) -> Decimal:
    expresion = expresion.strip()
    if not expresion:
        raise ErrorCalculo("expresion vacia")
    if len(expresion) > MAX_LONGITUD:
        raise ErrorCalculo(f"expresion mayor a {MAX_LONGITUD} caracteres")

    try:
        arbol = ast.parse(expresion, mode="eval")
    except SyntaxError as exc:
        raise ErrorCalculo("sintaxis invalida") from exc

    try:
        return _evaluar(arbol)
    except DivisionByZero as exc:
        raise ErrorCalculo("division por cero") from exc
    except (InvalidOperation, OverflowError) as exc:
        raise ErrorCalculo("operacion invalida") from exc


def _formatear(valor: Decimal) -> str:
    normalizado = valor.normalize()
    if normalizado == normalizado.to_integral_value():
        return str(normalizado.quantize(Decimal(1)))
    return f"{normalizado:f}"


async def calculadora_ast(expresion: str) -> str:
    """Evalua una expresion matematica con precision decimal.

    Operadores permitidos: + - * / % ** y parentesis.

    Args:
        expresion: por ejemplo "300000 * 0.95 / 12".

    Returns:
        El resultado exacto, o un mensaje que empieza con ERROR si no se pudo evaluar.
    """
    try:
        return _formatear(calcular(expresion))
    except ErrorCalculo as exc:
        return f"ERROR: {exc}"
