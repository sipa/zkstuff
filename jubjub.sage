import hashlib
import random

p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
F = GF(p)
E = EllipticCurve(F, [0, 7])
n = E.order()
S = GF(n)
half = 1 / S(2)

bits = [random.randrange(2) for i in range(510)]

def hash_to_point(m):
    h = hashlib.sha256()
    h.update(m)
    for i in range(256):
        h2 = h
        h2.update(bytes([i]))
        m = h2.digest()
        x = 0
        for b in m:
            x = x * 256 + ord(b)
        x = F(x)
        try:
            p = E.lift_x(x)
        except:
            continue
        if p[1].is_square():
            return p
        else:
            return -p


G = [hash_to_point(bytes([0,0]))]
B = [hash_to_point(bytes([0,0]))]
for i in range(1, len(bits) + 1):
    G.append(hash_to_point(bytes([i % 256, i // 256])))
    H = G[-1] * int(half)
    assert(H + H == G[-1])
    B[0] += H
    B.append(H)

T1 = G[0]
for round in range(len(bits)/3):
    (b0, b1, b2) = bits[3 * round : 3 * round + 3]
    if b2 == 1:
        b0 = 1 - b0
        b1 = 1 - b1
    if b0 == 1: T1 += G[3 * round + 1]
    if b1 == 1: T1 += G[3 * round + 2]
    if b2 == 1: T1 += G[3 * round + 3]

T2 = B[0]
for round in range(len(bits)/3):
    (b0, b1, b2) = bits[3 * round : 3 * round + 3]
    if b2 == 1:
        b0 = 1 - b0
        b1 = 1 - b1
    T2 += (2*b0 - 1)*B[3 * round + 1]
    T2 += (2*b1 - 1)*B[3 * round + 2]
    T2 += (2*b2 - 1)*B[3 * round + 3]

T3 = B[0]
for round in range(len(bits)/3):
    (b0, b1, b2) = bits[3 * round : 3 * round + 3]
    P0 = B[1 + round*3]
    P1 = B[2 + round*3]
    P2 = B[3 + round*3]
    R = [P0+P1+P2, -P0+P1+P2, +P0-P1+P2, -P0-P1+P2]
    A = (2*b2 - 1) * R[b0 + 2*b1]
    T3 += A
 
assert(T1 == T2)
assert(T1 == T3)

print("x1 = %i" % B[0][0])
print("y1 = %i" % B[0][1])

# A*x*y + B*x*(1-y) + C*(1-x)*y + D*(1-x)*(1-y) =
# D + (B-D)x + (C-D)y + (A-B-C+D)xy

A1 = B[0]
for round in range(len(bits)/3):
    print("b0 = #%i" % bits[3 * round])
    print("b1 = #%i" % bits[3 * round + 1])
    print("b2 = #%i" % bits[3 * round + 2])
    print("b0 * (1 - b0) == 0")
    print("b1 * (1 - b1) == 0")
    print("b2 * (1 - b2) == 0")
    P0 = B[1 + 3*round]
    P1 = B[2 + 3*round]
    P2 = B[3 + 3*round]
    R = [P0+P1+P2, -P0+P1+P2, P0-P1+P2, -P0-P1+P2]
    print("x2 = %i + %i * b0 + %i * b1 + %i * (b0 * b1)" % (R[0][0], R[1][0] - R[0][0], R[2][0] - R[0][0], R[0][0] - R[1][0] - R[2][0] + R[3][0]))
    print("y2 = (2*b2 - 1) * (%i + %i * b0 + %i * b1 + %i * (b0 * b1))" % (R[0][1], R[1][1] - R[0][1], R[2][1] - R[0][1], R[0][1] - R[1][1] - R[2][1] + R[3][1]))
    print("lambda = (y2 - y1) / (x2 - x1)")
    print("x3 = lambda * lambda - x1 - x2")
    print("y3 = lambda * (x1 - x3) - y1")
    print("x1 = x3")
    print("y1 = y3")
print("x1 == %i" % T1[0])
