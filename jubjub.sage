import hashlib

p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
F = GF(p)
E = EllipticCurve(F, [0, 7])
n = E.order()
S = GF(n)

bits = [0,1,0,0,0,1,1,0,1,1,0,0,0,0,0,1,0,1,0,0,1,1,1,0,0,1,0,1,1,1,0,1,1,1]

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

G = []
B = [E.point(0,0)]
T = E.point(0,0)
for i in range(len(bits)):
    G.append(hash_to_point(bytes([i % 256, i // 256])))
    H = G[-1] * int(S(1) / S(2))
    B[0] += H
    B.append(H)
    if (bits[i]):
        T += G[-1]

print("x1 = %i" % B[0][0])
print("y1 = %i" % B[0][1])

# A*x*y + B*x*(1-y) + C*(1-x)*y + D*(1-x)*(1-y) =
# D + (B-D)x + (C-D)y + (A-B-C+D)xy

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
    R00 = -P0-P1-P2
    R01 = -P0+P1-P2
    R10 = +P0-P1-P2
    R11 = +P0+P1-P2
    print("x2 = %i + %i * b0 + %i * b1 + %i * (b0 * b1)" % (R11[0], R01[0] - R11[0], R10[0] - R11[0], R00[0] - R01[0] - R10[0] + R11[0]))
    print("y2 = (2*b2 - 1) * (%i + %i * b0 + %i * b1 + %i * (b0 * b1))" % (R11[1], R01[1] - R11[1], R10[1] - R11[1], R00[1] - R01[1] - R10[1] + R11[1]))
    print("lambda = (y2 - y1) / (x2 - x1)")
    print("x3 = lambda * lambda - x1 - x2")
    print("y3 = lambda * (x1 - x3) - y1")
    print("x1 = x3")
    print("y1 = y3")
print("x1 == %i" % T[0])
