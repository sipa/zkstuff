#!/usr/bin/python3

import re
import sys
import time
import random

MODULUS = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
COST_SCALAR_MUL = 5
COST_SCALAR_NEG = 2
COST_SCALAR_COPY = 1

def equation_cost(var):
    if len(var) == 0:
        return 0
    total = 0
    negs = dict()
    counts = dict()
    high = 0
    for name, val in var.items():
        total += 1
        negate = 0
        if MODULUS - val < val:
            val = MODULUS - val
            negate = 1
        if val in counts:
            counts[val] += 1
            negs[val] += negate
        else:
            counts[val] = 1
            negs[val] = negate
        if counts[val] > high:
            high = val
    plus_one = max(negs[high], counts[high] - negs[high])
    neg_one = counts[high] - plus_one
    return COST_SCALAR_MUL * (total - plus_one - neg_one) + COST_SCALAR_NEG * neg_one + COST_SCALAR_COPY

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
        self.cost = equation_cost(var)

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
                if i - len(op) >= 1 and s[i-len(op):i] == op:
                    return (clean_expr(s[:i-len(op)]), op, clean_expr(s[i:]))
    return None

def new_mul(vall, valr):
    global mul_count
    global mul_data
    mul_data.append((vall % MODULUS, valr % MODULUS, (vall * valr) % MODULUS))
    ret = (Linear(vall, 0, ("L%i" % mul_count, 1)), Linear(valr, 0, ("R%i" % mul_count, 1)), Linear(vall * valr, 0, ("O%i" % mul_count, 1)))
    mul_count += 1
    return ret

def new_temp(val):
    global temp_count
    ret = Linear(val, 0, ("T%i" % temp_count, 1))
    temp_count += 1
    return ret

def new_const(val):
    return Linear(val, val)

def new_multiplication(l, r):
    global cache
    global eqs
    global varset
    if l.is_const():
        return r * l.get_const()
    if r.is_const():
        return l * r.get_const()
    if r < l:
        (l, r) = (r, l)
    key = "%s {*} %s" % (l, r)
    if key in cache:
        return cache[key]
    lv, rv, ret = new_mul(l.get_real(), r.get_real())
#    for var in varset:
#        if varset[var] == l:
#            varset[var] = lv
#        if varset[var] == r:
#            varset[var] = rv
    assert(l.get_real() == lv.get_real())
    assert(r.get_real() == rv.get_real())
    eqs.append(l - lv)
    eqs.append(r - rv)
    cache[key] = ret
    return ret

def new_division(l, r):
    global cache
    global eqs
    global varset
    if r.is_const():
        return l / r.get_const()
    key = "%s {/} %s" % (l, r)
    if key in cache:
        return cache[key]
    ret, rv, lv = new_mul(l.get_real() * modinv(r.get_real()), r.get_real())
#    for var in varset:
#        if varset[var] == l:
#            varset[var] = lv
#        if varset[var] == r:
#            varset[var] = rv
    assert(l.get_real() == lv.get_real())
    assert(r.get_real() == rv.get_real())
    eqs.append(l - lv)
    eqs.append(r - rv)
    cache[key] = ret
    return ret

def new_xor(l, r):
    return l + r - new_multiplication(l, r) * 2

def parse_expression(s):
    global cache
    global eqs
    global varset
    s = clean_expr(s)
    if s == "":
        raise Exception("Empty expression")
    sp = split_expr_binary(s, ["^"])
    if sp:
        (left, op, right) = sp
        l = parse_expression(left)
        r = parse_expression(right)
        ret = new_xor(l, r)
        assert(ret.get_real() == l.get_real() ^ r.get_real())
        return ret
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
        if op == '*':
            ret = new_multiplication(l, r)
            assert(ret.get_real() == (l.get_real() * r.get_real()) % MODULUS)
            return ret
        else:
            ret = new_division(l, r)
            assert(l.get_real() == (ret.get_real() * r.get_real()) % MODULUS)
            return new_division(l, r)
    if len(s) > 5 and s[:5] == 'bool(':
        ret = parse_expression(s[4:])
        m = new_multiplication(ret, ret - new_const(1))
        eqs.append(m)
        return ret
    if s[0] == '-':
        return parse_expression(s[1:]) * (MODULUS - 1)
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

def parse_expressions(s):
    sp = split_expr_binary(s, [","])
    if sp:
        (left, op, right) = sp
        return parse_expressions(left) + [parse_expression(right)]
    return [parse_expression(s)]

def parse_statement(s):
    global varset
    global eqs
    s = s.strip()
    if len(s) > 6 and s[0:6] == "debug ":
        expr = parse_expression(s[6:])
        print("DEBUG %s: %s [0x%x]\n" % (s[6:], expr, expr.get_real()))
        return
    sp = split_expr_binary(s, [":=", "=:", "==", "="])
    if sp:
        (left, op, right) = sp
        if op == ':=':
            bits = [x.strip() for x in left.split(",")]
            for bit in bits:
                if not VAR_RE.fullmatch(bit):
                    raise Exception("Bit '%s' is not a valid name in %s" % (bit, op))
            val = parse_expression(right)
            assert(len(bits) < 256) # don't support less-than constraint
            assert(val.get_real() >> len(bits) == 0)
            if val.is_const():
                bitvars = [new_const((val.get_real() >> i) & 1) for i in range(len(bits))]
            else:
                bitvars = [new_temp((val.get_real() >> i) & 1) for i in range(len(bits))]
                for i, bitvar in enumerate(bitvars):
                    eqs.append(new_multiplication(bitvar, bitvar - new_const(1)))
                    val = val - bitvar * (1 << i)
                eqs.append(val)
            for i in range(len(bits)):
                varset[bits[i]] = bitvars[i]
        elif op == '=:':
            bits = parse_expressions(right)
            if not VAR_RE.fullmatch(left):
                raise Exception("Int '%s' is not a valid name in %s" % (left, op))
            assert(len(bits) < 256)
            real = 0
            for i, bit in enumerate(bits):
                assert(bit.get_real() == 0 or bit.get_real() == 1)
                real = real + bit.get_real() * (1 << i)
            if all(bit.is_const() for bit in bits):
                val = new_const(real)
                varset[left] = val
            else:
                val = new_temp(real)
                varset[left] = val
                for i, bit in enumerate(bits):
                    val = val - bit * (1 << i)
                eqs.append(val)
        elif op == '=':
            if VAR_RE.fullmatch(left):
                expr = parse_expression(right)
                varset[left] = expr
            else:
                raise Exception("Assigning to non-variable '%s'" % left)
        elif op == '==':
            l = parse_expression(left)
            r = parse_expression(right)
            if l.get_real() != r.get_real():
                print("WARNING: in '%s', left is %i, right is %i" % (s, l.get_real(), r.get_real()))
            eqs.append(l - r)
    else:
        raise Exception("Cannot execute '%s'" % s)

def pivot_variable(eqs, vnam, eliminate=False):
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
        for idx, eq in enumerate(eqs):
            if idx != low and vnam in eq.var:
                eqs[idx] = eq - leq * eq.var[vnam]
    if eliminate and cc > 0:
        del eqs[low]


def encode_andytoshi_format():
    global eqs
    global mul_count
    ret = "%i,0,%i;" % (1<<(mul_count-1).bit_length(), len(eqs))
    for eq in eqs:
        for pos, (name, val) in enumerate(eq.var.items()):
            ret += " "
            negative = False
            if val * 2 > MODULUS:
                negative = True
                val = MODULUS - val
            if pos == 0 and negative:
                ret += "-"
            elif pos > 0 and negative:
                ret += "- "
            elif pos > 0:
                ret += "+ "
            if val > 1:
                ret += "%i*" % val
            ret += name
        ret += " = "
        val = (MODULUS - eq.const) % MODULUS
        if val * 2 > MODULUS:
            ret += "-"
            val = MODULUS - val
        ret += "%i;" % val
    return ret

def encode_scalar_const(v):
    if (v < 100):
        return ("SECP256K1_SCALAR_CONST(0, 0, 0, 0, 0, 0, 0, %i)" % v)
    vals = [(v >> (32 * (7 - i))) & 0xFFFFFFFF for i in range(8)]
    return ("SECP256K1_SCALAR_CONST(0x%08x, 0x%08x, 0x%08x, 0x%08x, 0x%08x, 0x%08x, 0x%08x, 0x%08x)" % (vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7]))

start = time.clock()
print("[%f] Parsing..." % 0)
for line in sys.stdin:
    parse_statement(line)

print("[%f] %i multiplications, %i temporaries, %i constraints, %i cost" % (time.clock() - start, mul_count, temp_count, len(eqs), sum(eq.cost for eq in eqs)))

print("[%f] Eliminating..." % (time.clock() - start))
tock = time.clock()
for tnum in range(0, temp_count):
    pivot_variable(eqs, "T%i" % tnum, True)
    now = time.clock()
    if (now - tock > 10):
        tock = now
        print("[%f] Eliminated %i/%i" % (now - start, tnum, temp_count))

eqs_cost = sum(eq.cost for eq in eqs)
print("[%f] %i multiplications, %i constraints, %i cost" % (time.clock() - start, mul_count, len(eqs), eqs_cost))
print("[%f] Reducing..." % (time.clock() - start))
tock = time.clock()
for i in range(mul_count):
    neweqs = eqs.copy()
    now = time.clock()
    if (now - tock > 20):
        tock = now
        print("[%f] Reduced to %i cost (step %i/%i)" % (now - start, eqs_cost, i, mul_count))
    for j in range(4):
        vnam = random.choice(["L%i","R%i","O%i"]) % random.randrange(mul_count)
        pivot_variable(neweqs, vnam)
        neweqs_cost = sum(eq.cost for eq in neweqs)
        if neweqs_cost < eqs_cost:
            eqs = neweqs
            eqs_cost = neweqs_cost
            break

print("[%f] %i multiplications, %i constraints, %i cost" % (time.clock() - start, mul_count, len(eqs), eqs_cost))

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



print("[%f] Done" % (time.clock() - start))
print()
print(encode_andytoshi_format())

print()
print("Secret inputs:")
print("L = {%s}" % (", ".join(encode_scalar_const(mul_data[i][0]) for i in range(mul_count))))
print("R = {%s}" % (", ".join(encode_scalar_const(mul_data[i][1]) for i in range(mul_count))))
print("O = {%s}" % (", ".join(encode_scalar_const(mul_data[i][2]) for i in range(mul_count))))
