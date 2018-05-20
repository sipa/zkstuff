import hashlib
import random

p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
F = GF(p)
E = EllipticCurve(F, [0, 7])
n = E.order()
S = GF(n)
half = 1 / S(2)

levels = 1
if len(sys.argv) > 1:
    levels = int(sys.argv[1])

def sha256(m):
    h = hashlib.sha256()
    h.update(m)
    return int(h.digest().encode('hex'), 16)

def hash_to_point(m):
    for i in range(256):
        x = sha256(bytes([i]) + m)
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
for i in range(510):
    G.append(hash_to_point(bytes([i % 256, i // 256])))
    H = G[-1] * int(half)
    assert(H + H == G[-1])
    B[0] += H
    B.append(H)

def native_hash(m):
    T = G[0]
    for i in range(2):
        for j in range(85):
            bitval = (m[i] >> (j * 3)) & 7
            (b0, b1, b2) = (bitval & 1, (bitval >> 1) & 1, (bitval >> 2) & 1)
            if b2 == 1:
                b0 = 1 - b0
                b1 = 1 - b1
            if b0 == 1: T += G[i * 255 + j * 3 + 1]
            if b1 == 1: T += G[i * 255 + j * 3 + 2]
            if b2 == 1: T += G[i * 255 + j * 3 + 3]
    return T[0]

base = sha256("base")
root = base
branch = []
directions = []

for i in range(levels):
    branchval = sha256("level %i" % i)
    dirval = sha256("direction %i" % i) & 1
    branch.append(branchval)
    directions.append(dirval)
    if dirval == 0:
        root = int(native_hash([root, branchval]))
    else:
        root = int(native_hash([branchval, root]))

def emit_step(level):
    print("dir = bool(#%i)" % directions[level])
    print("branch = #%i" % branch[level])
    print("diff = dir * (branch - root)")
    print("left = root + diff")
    print("right = branch - diff")
    print("%s,leftdrop := left" % (",".join(["b%i" % i for i in range(255)])))
    print("%s,rightdrop := right" % (",".join(["b%i" % i for i in range(255,510)])))
    print("x = %i" % B[0][0])
    print("y = %i" % B[0][1])
    for round in range(170):
        P0 = B[1 + 3*round]
        P1 = B[2 + 3*round]
        P2 = B[3 + 3*round]
        R = [P0+P1+P2, -P0+P1+P2, P0-P1+P2, -P0-P1+P2]
        print("inner = b%i * b%i" % (3 * round, 3 * round + 1))
        print("x%i = %i + %i * b%i + %i * b%i + %i * inner" % (round, R[0][0], R[1][0] - R[0][0], 3 * round, R[2][0] - R[0][0], 3 * round + 1, R[0][0] - R[1][0] - R[2][0] + R[3][0]))
        print("y%i = (2*b%i - 1) * (%i + %i * b%i + %i * b%i + %i * inner)" % (round, 3 * round + 2, R[0][1], R[1][1] - R[0][1], 3 * round, R[2][1] - R[0][1], 3 * round + 1, R[0][1] - R[1][1] - R[2][1] + R[3][1]))
        print("xd = x%i - x" % round)
        print("yd = y%i - y" % round)
        print("lambda = yd / xd")
        if round < 169:
            print("x = lambda*lambda - 2*x%i + xd" % round)
            print("y = lambda*(x%i - x) - y%i" % (round, round))
        else:
            print("root = lambda*lambda - 2*x%i + xd" % round)

print("root = %i" % base)
for i in range(levels):
    emit_step(i)
print("root == %i" % root)
