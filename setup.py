from utils import *
import py_ecc.bn128 as b
from py_ecc.fields.field_elements import FQ as Field
from curve import ec_lincomb, G1Point, G2Point
from compiler.program import CommonPreprocessedInput
from verifier import VerificationKey
from dataclasses import dataclass
from poly import Polynomial, Basis
from functools import cache

# Recover the trusted setup from a file in the format used in
# https://github.com/iden3/snarkjs#7-prepare-phase-2
SETUP_FILE_G1_STARTPOS = 80
SETUP_FILE_POWERS_POS = 60

class f_inner(Field):
    field_modulus = b.curve_order

# Gets the first root of unity of a given group order
@cache
def get_root_of_unity(group_order):
    return f_inner(5) ** ((b.curve_order - 1) // group_order)

# Gets the full list of roots of unity of a given group order
@cache
def get_roots_of_unity(group_order):
    o = [f_inner(1), get_root_of_unity(group_order)]
    while len(o) < group_order:
        o.append(o[-1] * o[1])
    return o

# Fast Fourier transform, used to convert between polynomial coefficients
# and a list of evaluations at the roots of unity
# See https://vitalik.ca/general/2019/05/12/fft.html
def _fft(vals, modulus, roots_of_unity):
    if len(vals) == 1:
        return vals
    L = _fft(vals[::2], modulus, roots_of_unity[::2])
    R = _fft(vals[1::2], modulus, roots_of_unity[::2])
    o = [0 for i in vals]
    for i, (x, y) in enumerate(zip(L, R)):
        y_times_root = y * roots_of_unity[i]
        o[i] = (x + y_times_root) % modulus
        o[i + len(L)] = (x - y_times_root) % modulus
    return o

@dataclass
class Setup(object):
    #   ([1]₁, [x]₁, ..., [x^{d-1}]₁)
    # = ( G,    xG,  ...,  x^{d-1}G ), where G is a generator of G_2
    powers_of_x: list[G1Point]
    # [x]₂ = xH, where H is a generator of G_2
    X2: G2Point

    @classmethod
    def from_file(cls, filename):
        contents = open(filename, "rb").read()
        # Byte 60 gives you the base-2 log of how many powers there are
        powers = 2 ** contents[SETUP_FILE_POWERS_POS]
        # Extract G1 points, which start at byte 80
        values = [
            int.from_bytes(contents[i : i + 32], "little")
            for i in range(
                SETUP_FILE_G1_STARTPOS, SETUP_FILE_G1_STARTPOS + 32 * powers * 2, 32
            )
        ]
        assert max(values) < b.field_modulus
        # The points are encoded in a weird encoding, where all x and y points
        # are multiplied by a factor (for montgomery optimization?). We can
        # extract the factor because we know the first point is the generator.
        factor = b.FQ(values[0]) / b.G1[0]
        values = [b.FQ(x) / factor for x in values]
        powers_of_x = [(values[i * 2], values[i * 2 + 1]) for i in range(powers)]
        print("Extracted G1 side, X^1 point: {}".format(powers_of_x[1]))
        # Search for start of G2 points. We again know that the first point is
        # the generator.
        pos = SETUP_FILE_G1_STARTPOS + 32 * powers * 2
        target = (factor * b.G2[0].coeffs[0]).n
        while pos < len(contents):
            v = int.from_bytes(contents[pos : pos + 32], "little")
            if v == target:
                break
            pos += 1
        print("Detected start of G2 side at byte {}".format(pos))
        X2_encoding = contents[pos + 32 * 4 : pos + 32 * 8]
        X2_values = [
            b.FQ(int.from_bytes(X2_encoding[i : i + 32], "little")) / factor
            for i in range(0, 128, 32)
        ]
        X2 = (b.FQ2(X2_values[:2]), b.FQ2(X2_values[2:]))
        assert b.is_on_curve(X2, b.b2)
        print("Extracted G2 side, X^1 point: {}".format(X2))
        # assert b.pairing(b.G2, powers_of_x[1]) == b.pairing(X2, b.G1)
        # print("X^1 points checked consistent")
        return cls(powers_of_x, X2)

    # Encodes the KZG commitment that evaluates to the given values in the group
    def commit(self, values: Polynomial) -> G1Point:
        assert values.basis == Basis.LAGRANGE

        # Run inverse FFT to convert values from Lagrange basis to monomial basis
        roots = [x.n for x in get_roots_of_unity(len(values.values))]
        o, nvals = b.curve_order, [x.n for x in values.values]
        # Inverse FFT
        invlen = f_inner(1) / len(values.values)
        reversed_roots = [roots[0]] + roots[1:][::-1]
        monomial_coeffs = [f_inner(x) * invlen for x in _fft(nvals, o, reversed_roots)]

        # Compute linear combination of setup with monomial_coeffs
        assert len(monomial_coeffs) <= len(self.powers_of_x), "Cannot commit to polynomial of higher degree than trusted setup"
        commitment = G1Point((0,0))
        for coeff, power in zip(monomial_coeffs, self.powers_of_x):
            # TODO: update commitment using coeff*power [WIP]
            commitment = (commitment[0]+coeff * power, commitment[1])
        return commitment


    # Generate the verification key for this program with the given setup
    def verification_key(self, pk: CommonPreprocessedInput) -> VerificationKey:
        # Create the appropriate VerificationKey object
        return NotImplemented


if __name__ == "__main__":
    from compiler.program import Program
    print("===setup_test===")

    setup = Setup.from_file("test/powersOfTau28_hez_final_11.ptau")
    dummy_values = Polynomial(list(map(Scalar, [1, 2, 3, 4, 5, 6, 7, 8])), Basis.LAGRANGE)
    program = Program(["c <== a * b"], 8)
    commitment = setup.commit(dummy_values)
    assert commitment == G1Point((16120260411117808045030798560855586501988622612038310041007562782458075125622, 3125847109934958347271782137825877642397632921923926105820408033549219695465,))
    vk = setup.verification_key(program.common_preprocessed_input())
    assert (vk.w == 19540430494807482326159819597004422086093766032135589407132600596362845576832)
    print("Successfully created dummy commitment and verification key")