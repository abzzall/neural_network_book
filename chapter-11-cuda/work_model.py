"""CPU arithmetic model of launch coverage and idealised memory traffic."""


def coverage(elements, width):
    programs = (elements + width - 1) // width
    offsets = [p * width + lane for p in range(programs)
               for lane in range(width) if p * width + lane < elements]
    assert offsets == list(range(elements))
    return programs, programs * width - elements


if __name__ == '__main__':
    programs, masked = coverage(1003, 256)
    print(f'programs: {programs}, masked offsets: {masked}')
    print(f'vector-add FLOP per byte: {1 / 12:.6f}')
    # Leading-order 2*n**3 operations; each matrix transferred only once.
    n = 512
    print(f'ideal matmul FLOP per byte: {2 * n**3 / (3 * n**2 * 4):.6f}')
