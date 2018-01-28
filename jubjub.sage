import hashlib

p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
F = GF(p)
E = EllipticCurve(F, [0, 7])
n = E.order()

print(n)

def hash_to_point(m):
    h = hashlib.sha256()
    h.update(m)
    for i in range(256):
        h2 = h
        h2.update(bytes([i]))
        m = h2.digest()
        for b in m:
            x = x * 256 + b
        x = F(x)
        if (x
        inter = pow(x,3,FIELD) + 7
        y = pow(inter, (FIELD+1)/4, FIELD)
        if pow(y, 2, FIELD) == inter:
            return (x,y)

print(hash_to_point(bytes([0,0])))
