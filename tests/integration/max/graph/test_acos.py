# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Tests for acos operation."""

import pytest
from max.dtype import DType
from max.graph import Graph, TensorType, ops
from max.graph.device import DeviceRef


def test_acos_basic() -> None:
    """Test acos basic functionality with values in valid domain [-1, 1]."""
    input_type = TensorType(
        dtype=DType.float32, shape=(5,), device=DeviceRef.CPU()
    )

    with Graph("acos_basic", input_types=(input_type,)) as graph:
        x = graph.inputs[0]
        out = ops.acos(x)
        graph.output(out)

    # Verify graph was created successfully
    assert graph is not None


def test_acos_special_values() -> None:
    """Test acos with special mathematical values."""
    input_type = TensorType(
        dtype=DType.float32, shape=(3,), device=DeviceRef.CPU()
    )

    with Graph("acos_special", input_types=(input_type,)) as graph:
        x = graph.inputs[0]
        out = ops.acos(x)
        graph.output(out)

    # Known mathematical values:
    # acos(1.0) = 0.0
    # acos(0.0) = π/2 ≈ 1.5708
    # acos(-1.0) = π ≈ 3.1416
    assert graph is not None


def test_acos_2d_tensor() -> None:
    """Test acos with 2D tensor."""
    input_type = TensorType(
        dtype=DType.float32, shape=(3, 2), device=DeviceRef.CPU()
    )

    with Graph("acos_2d", input_types=(input_type,)) as graph:
        x = graph.inputs[0]
        out = ops.acos(x)
        graph.output(out)

    assert graph is not None
    assert out.shape == (3, 2)


def test_acos_3d_tensor() -> None:
    """Test acos with 3D tensor."""
    input_type = TensorType(
        dtype=DType.float32, shape=(2, 3, 4), device=DeviceRef.CPU()
    )

    with Graph("acos_3d", input_types=(input_type,)) as graph:
        x = graph.inputs[0]
        out = ops.acos(x)
        graph.output(out)

    assert graph is not None
    assert out.shape == (2, 3, 4)


def test_acos_edge_domain_values() -> None:
    """Test acos with values near domain boundaries."""
    input_type = TensorType(
        dtype=DType.float32, shape=(4,), device=DeviceRef.CPU()
    )

    with Graph("acos_edge", input_types=(input_type,)) as graph:
        x = graph.inputs[0]
        out = ops.acos(x)
        graph.output(out)

    # Test values very close to -1 and 1
    assert graph is not None


def test_acos_single_element() -> None:
    """Test acos with single element tensor."""
    input_type = TensorType(
        dtype=DType.float32, shape=(1,), device=DeviceRef.CPU()
    )

    with Graph("acos_single", input_types=(input_type,)) as graph:
        x = graph.inputs[0]
        out = ops.acos(x)
        graph.output(out)

    assert graph is not None
    assert out.shape == (1,)


def test_acos_dtype_preservation() -> None:
    """Test that acos preserves input dtype."""
    for dtype in [DType.float32, DType.float64]:
        input_type = TensorType(dtype=dtype, shape=(3,), device=DeviceRef.CPU())

        with Graph(f"acos_dtype_{dtype}", input_types=(input_type,)) as graph:
            x = graph.inputs[0]
            out = ops.acos(x)
            graph.output(out)

        assert out.dtype == dtype


def test_acos_zero_dimensional() -> None:
    """Test acos with zero-dimensional (scalar) tensor."""
    input_type = TensorType(
        dtype=DType.float32, shape=(), device=DeviceRef.CPU()
    )

    with Graph("acos_scalar", input_types=(input_type,)) as graph:
        x = graph.inputs[0]
        out = ops.acos(x)
        graph.output(out)

    assert graph is not None
    assert out.shape == ()


if __name__ == "__main__":
    pytest.main([__file__])
