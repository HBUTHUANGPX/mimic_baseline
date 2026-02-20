import time
import torch

from rsl_rl.modules import FSQQuantizer
from rsl_rl.modules.finite_scalar_quantization import FSQ as RefFSQ


def assert_close(a: torch.Tensor, b: torch.Tensor, tol: float = 1e-5) -> None:
    if not torch.allclose(a, b, atol=tol, rtol=0):
        diff = (a - b).abs().max().item()
        raise AssertionError(f"tensors not close, max diff={diff}")


def test_bound_and_quantize() -> None:
    fsq = FSQQuantizer(levels=[8, 5, 6, 7])
    z = torch.randn(128, fsq.dim) * 3.0
    z_b = fsq.bound(z)
    # z_b should be within [-1, 1]
    if (z_b > 1.0 + 1e-5).any() or (z_b < -1.0 - 1e-5).any():
        raise AssertionError("bound() out of expected range")

    z_q = fsq.quantize(z)
    if (z_q > 1.0 + 1e-5).any() or (z_q < -1.0 - 1e-5).any():
        raise AssertionError("quantize() out of [-1, 1]")


def test_codes_indices_roundtrip() -> None:
    fsq = FSQQuantizer(levels=[8, 5, 6, 7])
    z = torch.randn(64, fsq.dim)
    z_q, idx = fsq(z)
    z_q2 = fsq.indices_to_codes(idx)
    # Roundtrip should recover the same quantized codes
    assert_close(z_q, z_q2)


def test_forward_shapes() -> None:
    fsq = FSQQuantizer(levels=[4, 4, 4])
    z = torch.randn(10, 3)
    z_q, idx = fsq(z)
    if z_q.shape != z.shape:
        raise AssertionError(f"z_q shape {z_q.shape} != z shape {z.shape}")
    if idx.shape != (10,):
        raise AssertionError(f"idx shape {idx.shape} != (10,)")


def test_even_odd_levels_offsets() -> None:
    fsq = FSQQuantizer(levels=[4, 5])
    # even level uses offset 0.5, odd uses 0.0
    if not torch.allclose(fsq.offset, torch.tensor([0.5, 0.0], dtype=fsq.offset.dtype)):
        raise AssertionError("offset for even/odd levels incorrect")


def test_against_reference(preserve_symmetry: bool) -> None:
    levels = [8, 5, 6, 7]
    z = torch.randn(256, len(levels))
    fsq = FSQQuantizer(levels=levels, preserve_symmetry=preserve_symmetry)
    ref = RefFSQ(levels=levels, preserve_symmetry=preserve_symmetry, return_indices=False)

    z_q = fsq.quantize(z)
    ref_q = ref.quantize(z)

    if not torch.allclose(z_q, ref_q, atol=1e-4, rtol=0):
        diff = (z_q - ref_q).abs().max().item()
        raise AssertionError(f"quantize mismatch vs reference (preserve_symmetry={preserve_symmetry}), max diff={diff}")

    # indices round-trip should match between implementations
    _, idx = fsq(z)
    ref_idx = ref.codes_to_indices(ref_q)
    if not torch.equal(idx, ref_idx):
        raise AssertionError(f"indices mismatch vs reference (preserve_symmetry={preserve_symmetry})")


def main() -> None:
    test_bound_and_quantize()
    test_codes_indices_roundtrip()
    test_forward_shapes()
    test_even_odd_levels_offsets()
    test_against_reference(preserve_symmetry=False)
    test_against_reference(preserve_symmetry=True)
    # simple timing
    fsq = FSQQuantizer(levels=[8, 5, 6, 7])
    z = torch.randn(4096, fsq.dim)
    iters = 200
    # warmup
    for _ in range(20):
        _ = fsq(z)
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fsq(z)
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) * 1000.0 / iters
    print(f"FSQ forward avg: {avg_ms:.3f} ms over {iters} iters, shape={tuple(z.shape)}")
    print("FSQ minimal tests: OK")


if __name__ == "__main__":
    main()
