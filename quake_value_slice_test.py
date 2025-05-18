# quake_value_slice_test.py
import cudaq


def test_quakevalue_slice():
    """
    Demonstrates:
     1) 3 total qubits, 3 float parameters
     2) Subset of qubits [1..3) => qubits #1,#2
     3) Perform a hadamard on that slice before calling a sub-kernel
     4) Sub-kernel uses the last 2 float params for rotation
     5) The first float param is used for a final rotation on qubit #0
    """

    # 1) Main kernel with 3 float parameters
    main_kernel, main_thetas = cudaq.make_kernel(list[float])

    # 2) Allocate 3 qubits
    qubits = main_kernel.qalloc(3)

    # 3) We'll do a final rotation on qubit #0 with the first parameter
    #    at the end, so we'll do that last. Let's proceed with slicing #1..2
    sub_qubits = qubits[1:3]  # qubits[1], qubits[2]
    sub_thetas = main_thetas[1:3]  # the 2nd and 3rd float

    # 4) Show how slicing can do more than sub-kernel calls:
    #    Apply hadamard across sub_qubits just by referencing the slice
    main_kernel.h(sub_qubits)

    # 5) Create a sub-kernel that takes 2 qubits + 2 float parameters
    sub_kernel, sk_qubits, sk_thetas = cudaq.make_kernel(cudaq.qvector, list[float])
    # sub_kernel => simple rotation
    sub_kernel.ry(sk_thetas[0], sk_qubits[0])
    sub_kernel.ry(sk_thetas[1], sk_qubits[1])

    # 6) Now call that sub-kernel in the main kernel on sub_qubits, sub_thetas
    main_kernel.apply_call(sub_kernel, sub_qubits, sub_thetas)

    # 7) Now do the final rotation with the *first param* main_thetas[0] on qubit #0
    main_kernel.rx(main_thetas[0], qubits[0])

    # 8) Measure
    main_kernel.mz(qubits)

    print("=== Kernel IR ===")
    print(main_kernel)

    # Provide some param values: param[0] is used on qubit #0,
    # param[1], param[2] used in sub-kernel for qubits #1,#2
    param_values = [0.15, 0.3, 0.45]
    result = cudaq.sample(main_kernel, param_values, shots_count=100)

    print("\nCircuit drawing:\n")
    print(cudaq.draw(main_kernel, param_values))
    print("\nSampled results:\n", result)


if __name__ == "__main__":
    test_quakevalue_slice()
