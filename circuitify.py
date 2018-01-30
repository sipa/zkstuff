#!/usr/bin/python3

import re
import sys
import time

MODULUS = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

def modinv(n):
    assert(n != 0)
    return pow(n, MODULUS - 2, MODULUS)

class Linear:
    def __init__(self, real, const, *args):
        self.const = const % MODULUS
        self.real = real % MODULUS
        var = dict()
        for (name, val) in args:
            val = val % MODULUS
            if val == 0:
                continue
            if name not in var:
                var[name] = val
            else:
                cval = (var[name] + val) % MODULUS
                if cval != 0:
                    var[name] = cval
                else:
                    del var[name]
        self.var = var

    def __str__(self):
        terms = []
        for (name, val) in self.var.items():
            if val == 1:
                terms.append(name)
            elif (val + 1) % MODULUS == 0:
                terms.append("-%s" % name)
            elif (MODULUS - val < 10000):
                terms.append("-%i*%s" % (MODULUS - val, name))
            else:
                terms.append("%i*%s" % (val, name))
        if not terms or self.const != 0:
            if MODULUS - self.const < 10000:
                terms.append("-%i" % (MODULUS - self.const))
            else:
                terms.append("%i" % self.const)
        return " + ".join(terms)

    def is_const(self):
        return len(self.var) == 0

    def get_const(self):
        assert(self.is_const())
        assert(self.const == self.real)
        return self.const

    def get_real(self):
        return self.real

    def __add__(self, other):
        return Linear((self.real + other.real) % MODULUS, (self.const + other.const) % MODULUS, *(list(self.var.items()) + list(other.var.items())))

    def __sub__(self, other):
        return Linear((self.real + MODULUS - other.real) % MODULUS, (self.const + MODULUS - other.const) % MODULUS, *(list(self.var.items()) + [(name, MODULUS - val) for (name, val) in other.var.items()]))

    def __mul__(self, val):
        return Linear(self.real * val, self.const * val, *[(name, v * val) for (name, v) in self.var.items()])

    def __div__(self, val):
        if val == 0:
            raise Exception("Division by zero")
        inv = modinv(val)
        return Linear(self.real * inv, self.const * inv, *[(name, v * inv) for (name, v) in self.var.items()])

    def __eq__(self, other):
        return self.const == other.const and self.var == other.var

    def __lt__(self, other):
        if self.const < other.const: return True
        if self.const > other.const: return False
        return self.__str__() < other.__str__()

temp_count = 0
mul_count = 0
mul_data = []
cache = dict()
varset = dict()
eqs = []


def clean_expr(s):
    s = s.strip()
    if s == "" or s[0] != '(' or s[-1] != ')':
        return s
    depth = 1
    for i in range(1, len(s) - 1):
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
            if depth == 0:
                return s
    return clean_expr(s[1:-1])

VAR_RE = re.compile('[A-Za-z_][0-9a-zA-Z_]*')
SECRET_RE = re.compile('#(-?[0-9]+)')
NUM_RE = re.compile('[0-9]+')

def split_expr_binary(s, ops):
    i = 1
    depth = 0
    for i in range(len(s), 1, -1):
        if s[i - 1] == ')':
            depth += 1
        elif s[i - 1] == '(':
            depth -= 1
        elif depth == 0:
            for op in ops:
                if i - len(op) >= 0 and s[i-len(op):i] == op:
                    return (clean_expr(s[:i-len(op)]), op, clean_expr(s[i:]))
    return None

def new_mul(vall, valr):
    global mul_count
    global mul_data
    mul_count += 1
    mul_data.append((vall % MODULUS, valr % MODULUS, (vall * valr) % MODULUS))
    return (Linear(vall, 0, ("L%i" % mul_count, 1)), Linear(valr, 0, ("R%i" % mul_count, 1)), Linear(vall * valr, 0, ("O%i" % mul_count, 1)))

def new_temp(val):
    global temp_count
    temp_count += 1
    return Linear(val, 0, ("T%i" % temp_count, 1))

def new_const(val):
    return Linear(val, val)

def parse_expression(s):
    global cache
    global eqs
    global varset
    s = clean_expr(s)
    if s == "":
        raise Exception("Empty expression")
    sp = split_expr_binary(s, ["+", "-"])
    if sp:
        (left, op, right) = sp
        l = parse_expression(left)
        r = parse_expression(right)
        if op == '+':
            ret = l + r
        else:
            ret = l - r
        return ret
    sp = split_expr_binary(s, ["*", "/"])
    if sp:
        (left, op, right) = sp
        l = parse_expression(left)
        r = parse_expression(right)
        if l.is_const() and op == '*':
            return r * l.get_const()
        if r.is_const() and op == '*':
            return l * r.get_const()
        if r.is_const() and op == '/':
            return l / r.get_const()
        if op == '*' and r < r:
            (l, r) = (r, l)
        key = "%s {%s} %s" % (l, op, r)
        if key in cache:
            return cache[key]
        if op == '*':
            lv, rv, ret = new_mul(l.get_real(), r.get_real())
        else:
            ret, rv, lv = new_mul(l.get_real() * modinv(r.get_real()), r.get_real())
        assert(l.get_real() == lv.get_real())
        assert(r.get_real() == rv.get_real())
        eqs.append(l - lv)
        eqs.append(r - rv)
        cache[key] = ret
        return ret
    if s[0] == '-':
        return parse_expression(s[1:], MODULUS - 1)
    if VAR_RE.fullmatch(s):
        if s in varset:
            return varset[s]
        raise Exception("Variable '%s' not defined" % s)
    sp = SECRET_RE.fullmatch(s)
    if sp:
        return new_temp(int(sp.group(1)))
    if NUM_RE.fullmatch(s):
        return new_const(int(s))
    raise Exception("Cannot parse '%s'" % s)

def parse_statement(s):
    global varset
    global eqs
    sp = split_expr_binary(s, ["==", "="])
    if sp:
        (left, op, right) = sp
        if op == '=':
            if VAR_RE.fullmatch(left):
                expr = parse_expression(right)
                varset[left] = expr
            else:
                raise Exception("Assigning to non-variable '%s'" % left)
        if op == '==':
            l = parse_expression(left)
            r = parse_expression(right)
            assert(l.get_real() == r.get_real())
            eqs.append(l - r)
    else:
        raise Exception("Cannot execute '%s'" % s)

start = time.clock()
print("[%f] Parsing..." % 0)
lines = sys.stdin.readlines()
for line in lines:
    parse_statement(line)

print("[%f] Eliminating..." % (time.clock() - start))
for tnum in range(1, temp_count + 1):
    tnam = "T%i" % tnum
    c = 0
    cc = 0
    low = None
    leq = None
    for idx, eq in enumerate(eqs):
        if tnam in eq.var:
            cc += 1
            if low is None or c > len(eq.var):
                low = idx
                c = len(eq.var)
                leq = eq * modinv(eq.var[tnam])
    if cc > 1:
        neweqs = eqs
        for idx, eq in enumerate(eqs):
            if idx != low and tnam in eq.var:
                neweqs[idx] = eq - leq * eq.var[tnam]
    if cc > 0:
        eqs = neweqs[:low] + neweqs[low+1:]

print("[%f] Reducing..." % (time.clock() - start))
for vnam in ["L%i" % i for i in range(1, mul_count + 1)] + ["R%i" % i for i in range(1, mul_count + 1)] + ["O%i" % i for i in range(1, mul_count + 1)]:
    c = 0
    cc = 0
    low = None
    leq = None
    for idx, eq in enumerate(eqs):
        if vnam in eq.var:
            cc += 1
            if low is None or c > len(eq.var):
                low = idx
                c = len(eq.var)
                leq = eq * modinv(eq.var[vnam])
    if cc > 1:
        neweqs = eqs
        for idx, eq in enumerate(eqs):
            if idx != low and vnam in eq.var:
                neweqs[idx] = eq - leq * eq.var[vnam]
        eqs = neweqs

print("[%f] Maximizing 1s..." % (time.clock() - start))
neweqs = eqs
for idx, eq in enumerate(eqs):
    counts = dict()
    negations = dict()
    max_count = 0
    max_val = None
    for name, val in eq.var.items():
        negated = False
        if val & 1 == 0:
            val = MODULUS - val
            negated = True
        if val not in counts:
            counts[val] = 1
            negations[val] = 0
        else:
            counts[val] += 1
        negations[val] += negated
        if counts[val] > max_count:
            max_count = counts[val]
            max_val = val
    if negations[max_val] * 2 > counts[max_val]:
        neweqs[idx] = eq * (MODULUS - modinv(max_val))
    else:
        neweqs[idx] = eq * modinv(max_val)
eqs = neweqs

def encode_andytoshi_format():
    global eqs
    global mul_count
    ret = "%i,0,%i;" % (mul_count, len(eqs))
    for eq in eqs:
        for pos, (name, val) in enumerate(eq.var.items()):
            negative = False
            if val * 2 > MODULUS:
                negative = True
                val = MODULUS - val
            if negative:
                ret += "-"
            elif pos > 0:
                ret += "+"
            if val > 1:
                ret += "%i*" % val
            ret += name
        ret += "="
        val = eq.const
        if val * 2 > MODULUS:
            ret += "-"
            val = MODULUS - val
        ret += "%i;" % val
    return ret


print("[%f] Done" % (time.clock() - start))
print()
print(encode_andytoshi_format())

print()
print("Secret inputs:")
for i in range(1, mul_count + 1):
    print("L%i = %i" % (i, mul_data[i-1][0]))
    print("R%i = %i" % (i, mul_data[i-1][1]))
    print("O%i = %i" % (i, mul_data[i-1][0]))
