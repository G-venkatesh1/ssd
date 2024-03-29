tvm_config = dict(
    out="model.tar",
    opt_level=3,
    targets=["llvm"],
    compiler="cpu",
    host=None,
    port=None,
    timeout=60 * 60,  # 1h
    use_vm=False,
)


tvm_benchmark_config = dict(repeat=1, number=10, benchmark_only=False)
tvm_benchmark = True