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
"""Implements `IndexList` which is commonly used to represent N-D
indices.

You can import these APIs from the `utils` package. For example:

```mojo
from utils import IndexList
```
"""

from hashlib.hasher import Hasher
from sys import bitwidthof

from builtin.dtype import _int_type_of_width, _uint_type_of_width

from .static_tuple import StaticTuple

# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#


@always_inline
fn _reduce_and_fn(a: Bool, b: Bool) -> Bool:
    """Performs AND operation on two boolean inputs.

    Args:
        a: The first boolean input.
        b: The second boolean input.

    Returns:
        The result of AND operation on the inputs.
    """
    return a and b


# ===-----------------------------------------------------------------------===#
# Integer and Bool Tuple Utilities:
#   Utilities to operate on tuples of integers or tuples of bools.
# ===-----------------------------------------------------------------------===#


@always_inline
fn _int_tuple_binary_apply[
    binary_fn: fn[dtype: DType] (Scalar[dtype], Scalar[dtype]) -> Scalar[dtype],
](a: IndexList, b: __type_of(a)) -> __type_of(a):
    """Applies a given element binary function to each pair of corresponding
    elements in two tuples.

    Example Usage:
        var a: StaticTuple[Int, size]
        var b: StaticTuple[Int, size]
        var c = _int_tuple_binary_apply[size, Int.add](a, b)

    Args:
        a: Tuple containing lhs operands of the elementwise binary function.
        b: Tuple containing rhs operands of the elementwise binary function.

    Returns:
        Tuple containing the result.
    """

    var c = __type_of(a)()

    @parameter
    for i in range(a.size):
        var a_elem = a.__getitem__[i]()
        var b_elem = b.__getitem__[i]()
        c.__setitem__[i](binary_fn[a.element_type](a_elem, b_elem))

    return c


@always_inline
fn _int_tuple_compare[
    comp_fn: fn[dtype: DType] (Scalar[dtype], Scalar[dtype]) -> Bool,
](a: IndexList, b: __type_of(a)) -> StaticTuple[Bool, a.size]:
    """Applies a given element compare function to each pair of corresponding
    elements in two tuples and produces a tuple of Bools containing result.

    Example Usage:
        var a: StaticTuple[Int, size]
        var b: StaticTuple[Int, size]
        var c = _int_tuple_compare[size, Int.less_than](a, b)

    Args:
        a: Tuple containing lhs operands of the elementwise compare function.
        b: Tuple containing rhs operands of the elementwise compare function.

    Returns:
        Tuple containing the result.
    """

    var c = StaticTuple[Bool, a.size]()

    @parameter
    for i in range(a.size):
        var a_elem = a.__getitem__[i]()
        var b_elem = b.__getitem__[i]()
        c.__setitem__[i](comp_fn[a.element_type](a_elem, b_elem))

    return c


@always_inline
fn _bool_tuple_reduce[
    reduce_fn: fn (Bool, Bool) -> Bool,
](a: StaticTuple[Bool, _], init: Bool) -> Bool:
    """Reduces the tuple argument with the given reduce function and initial
    value.

    Example Usage:
        var a: StaticTuple[Bool, size]
        var c = _bool_tuple_reduce[size, _reduce_and_fn](a, True)

    Parameters:
        reduce_fn: Reduce function to accumulate tuple elements.

    Args:
        a: Tuple containing elements to reduce.
        init: Value to initialize the reduction with.

    Returns:
        The result of the reduction.
    """

    var c: Bool = init

    @parameter
    for i in range(a.size):
        c = reduce_fn(c, a.__getitem__[i]())

    return c


# ===-----------------------------------------------------------------------===#
# IndexList:
# ===-----------------------------------------------------------------------===#


fn _type_of_width[bitwidth: Int, unsigned: Bool]() -> DType:
    @parameter
    if unsigned:
        return _uint_type_of_width[bitwidth]()
    else:
        return _int_type_of_width[bitwidth]()


@register_passable("trivial")
struct IndexList[size: Int, *, element_type: DType = DType.int64](
    Comparable,
    Copyable,
    Defaultable,
    Hashable,
    Movable,
    Sized,
    Stringable,
    Writable,
):
    """A base struct that implements size agnostic index functions.

    Parameters:
        size: The size of the tuple.
        element_type: The underlying dtype of the integer element value.
    """

    alias _int_type = Scalar[element_type]
    """The underlying storage of the integer element value."""

    var data: StaticTuple[Self._int_type, size]
    """The underlying storage of the tuple value."""

    @always_inline
    fn __init__(out self):
        """Constructs a static int tuple of the given size."""
        self = 0

    @always_inline
    @implicit
    fn __init__(out self, data: StaticTuple[Self._int_type, size]):
        """Constructs a static int tuple of the given size.

        Args:
            data: The StaticTuple to construct the IndexList from.
        """
        constrained[
            element_type.is_integral(), "Element type must be of integral type."
        ]()
        self.data = data

    @doc_private
    @always_inline
    @implicit
    fn __init__(out self, value: __mlir_type.index):
        """Constructs a sized 1 static int tuple of given the element value.

        Args:
            value: The initial value.
        """
        constrained[size == 1]()
        constrained[
            element_type.is_integral(), "Element type must be of integral type."
        ]()
        self = Int(value)

    @always_inline
    @implicit
    fn __init__(out self, elems: (Int, Int)):
        """Constructs a static int tuple given a tuple of integers.

        Args:
            elems: The tuple to copy from.
        """
        constrained[
            element_type.is_integral(), "Element type must be of integral type."
        ]()
        var num_elements = len(elems)

        debug_assert(
            size == num_elements,
            "[IndexList] mismatch in the number of elements",
        )

        var tup = Self()

        @parameter
        for idx in range(2):
            tup[idx] = rebind[Int](elems[idx])

        self = tup

    @always_inline
    @implicit
    fn __init__(out self, elems: (Int, Int, Int)):
        """Constructs a static int tuple given a tuple of integers.

        Args:
            elems: The tuple to copy from.
        """
        constrained[
            element_type.is_integral(), "Element type must be of integral type."
        ]()
        var num_elements = len(elems)

        debug_assert(
            size == num_elements,
            "[IndexList] mismatch in the number of elements",
        )

        var tup = Self()

        @parameter
        for idx in range(3):
            tup[idx] = rebind[Int](elems[idx])

        self = tup

    @always_inline
    @implicit
    fn __init__(out self, elems: (Int, Int, Int, Int)):
        """Constructs a static int tuple given a tuple of integers.

        Args:
            elems: The tuple to copy from.
        """
        constrained[
            element_type.is_integral(), "Element type must be of integral type."
        ]()
        var num_elements = len(elems)

        debug_assert(
            size == num_elements,
            "[IndexList] mismatch in the number of elements",
        )

        var tup = Self()

        @parameter
        for idx in range(4):
            tup[idx] = rebind[Int](elems[idx])

        self = tup

    @always_inline
    fn __init__(out self, *elems: Int, __list_literal__: () = ()):
        """Constructs a static int tuple given a set of arguments.

        Args:
            elems: The elements to construct the tuple.
            __list_literal__: Specifies that this constructor can be used for
               list literals.
        """
        constrained[
            element_type.is_integral(), "Element type must be of integral type."
        ]()
        var num_elements = len(elems)

        debug_assert(
            size == num_elements,
            "[IndexList] mismatch in the number of elements",
        )

        var tup = Self()

        @parameter
        for idx in range(size):
            tup[idx] = elems[idx]

        self = tup

    @always_inline
    @implicit
    fn __init__(out self, elem: Int):
        """Constructs a static int tuple given a set of arguments.

        Args:
            elem: The elem to splat into the tuple.
        """
        constrained[
            element_type.is_integral(), "Element type must be of integral type."
        ]()
        self.data = __mlir_op.`pop.array.repeat`[
            _type = __mlir_type[
                `!pop.array<`, size.value, `, `, Self._int_type, `>`
            ]
        ](Self._int_type(elem))

    fn __init__(out self, *, other: Self):
        """Copy constructor.

        Args:
            other: The other tuple to copy from.
        """
        constrained[
            element_type.is_integral(), "Element type must be of integral type."
        ]()
        self.data = other.data

    @always_inline
    @implicit
    fn __init__(out self, values: VariadicList[Int]):
        """Creates a tuple constant using the specified values.

        Args:
            values: The list of values.
        """
        constrained[
            element_type.is_integral(), "Element type must be of integral type."
        ]()
        var num_elements = len(values)

        debug_assert(
            size == num_elements,
            "[IndexList] mismatch in the number of elements",
        )

        var tup = Self()

        @parameter
        for idx in range(size):
            tup[idx] = values[idx]

        self = tup

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Returns the size of the tuple.

        Returns:
            The tuple size.
        """
        return size

    @always_inline
    fn __getitem__[idx: Int](self) -> Int:
        """Gets an element from the tuple by index.

        Parameters:
            idx: The element index.

        Returns:
            The tuple element value.
        """
        return Int(self.data.__getitem__[idx]())

    @always_inline("nodebug")
    fn __getitem__[I: Indexer](self, idx: I) -> Int:
        """Gets an element from the tuple by index.

        Parameters:
            I: A type that can be used as an index.

        Args:
            idx: The element index.

        Returns:
            The tuple element value.
        """
        return Int(self.data[idx])

    @always_inline("nodebug")
    fn __setitem__[idx: Int](mut self, val: Int):
        """Sets an element in the tuple at the given static index.

        Parameters:
            idx: The element index.

        Args:
            val: The value to store.
        """
        self.data.__setitem__[idx](val)

    @always_inline("nodebug")
    fn __setitem__[idx: Int](mut self, val: Self._int_type):
        """Sets an element in the tuple at the given static index.

        Parameters:
            idx: The element index.

        Args:
            val: The value to store.
        """
        self.data.__setitem__[idx](val)

    @always_inline("nodebug")
    fn __setitem__(mut self, idx: Int, val: Int):
        """Sets an element in the tuple at the given index.

        Args:
            idx: The element index.
            val: The value to store.
        """
        self.data[idx] = val

    @always_inline("nodebug")
    fn as_tuple(self) -> StaticTuple[Int, size]:
        """Converts this IndexList to StaticTuple.

        Returns:
            The corresponding StaticTuple object.
        """
        var res = StaticTuple[Int, size]()

        @parameter
        for i in range(size):
            res[i] = Int(self.__getitem__[i]())
        return res

    @always_inline("nodebug")
    fn canonicalize(
        self,
        out result: IndexList[size, element_type = DType.int64],
    ):
        """Canonicalizes the IndexList.

        Returns:
            Canonicalizes the object.
        """
        return self.cast[DType.int64]()

    @always_inline
    fn flattened_length(self) -> Int:
        """Returns the flattened length of the tuple.

        Returns:
            The flattened length of the tuple.
        """
        var length: Int = 1

        @parameter
        for i in range(size):
            length *= self[i]

        return length

    @always_inline
    fn __add__(self, rhs: Self) -> Self:
        """Performs element-wise integer add.

        Args:
            rhs: Right hand side operand.

        Returns:
            The resulting index tuple.
        """

        @always_inline
        fn apply_fn[
            dtype: DType
        ](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
            return a + b

        return _int_tuple_binary_apply[apply_fn](self, rhs)

    @always_inline
    fn __sub__(self, rhs: Self) -> Self:
        """Performs element-wise integer subtract.

        Args:
            rhs: Right hand side operand.

        Returns:
            The resulting index tuple.
        """

        @always_inline
        fn apply_fn[
            dtype: DType
        ](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
            return a - b

        return _int_tuple_binary_apply[apply_fn](self, rhs)

    @always_inline
    fn __mul__(self, rhs: Self) -> Self:
        """Performs element-wise integer multiply.

        Args:
            rhs: Right hand side operand.

        Returns:
            The resulting index tuple.
        """

        @always_inline
        fn apply_fn[
            dtype: DType
        ](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
            return a * b

        return _int_tuple_binary_apply[apply_fn](self, rhs)

    @always_inline
    fn __floordiv__(self, rhs: Self) -> Self:
        """Performs element-wise integer floor division.

        Args:
            rhs: The elementwise divisor.

        Returns:
            The resulting index tuple.
        """

        @always_inline
        fn apply_fn[
            dtype: DType
        ](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
            return a // b

        return _int_tuple_binary_apply[apply_fn](self, rhs)

    @always_inline
    fn __rfloordiv__(self, rhs: Self) -> Self:
        """Floor divides rhs by this object.

        Args:
            rhs: The value to elementwise divide by self.

        Returns:
            The resulting index tuple.
        """
        return rhs // self

    @always_inline
    fn remu(self, rhs: Self) -> Self:
        """Performs element-wise integer unsigned modulo.

        Args:
            rhs: Right hand side operand.

        Returns:
            The resulting index tuple.
        """

        @always_inline
        fn apply_fn[
            dtype: DType
        ](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
            return a % b

        return _int_tuple_binary_apply[apply_fn](self, rhs)

    @always_inline
    fn __eq__(self, rhs: Self) -> Bool:
        """Compares this tuple to another tuple for equality.

        The tuples are equal if all corresponding elements are equal.

        Args:
            rhs: The other tuple.

        Returns:
            The comparison result.
        """

        @always_inline
        fn apply_fn[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Bool:
            return a == b

        return _bool_tuple_reduce[_reduce_and_fn](
            _int_tuple_compare[apply_fn](self.data, rhs.data), True
        )

    @always_inline
    fn __ne__(self, rhs: Self) -> Bool:
        """Compares this tuple to another tuple for non-equality.

        The tuples are non-equal if at least one element of LHS isn't equal to
        the corresponding element from RHS.

        Args:
            rhs: The other tuple.

        Returns:
            The comparison result.
        """
        return not (self == rhs)

    @always_inline
    fn __lt__(self, rhs: Self) -> Bool:
        """Compares this tuple to another tuple using LT comparison.

        A tuple is less-than another tuple if all corresponding elements of lhs
        is less than rhs.

        Note: This is **not** a lexical comparison.

        Args:
            rhs: Right hand side tuple.

        Returns:
            The comparison result.
        """

        @always_inline
        fn apply_fn[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Bool:
            return a < b

        return _bool_tuple_reduce[_reduce_and_fn](
            _int_tuple_compare[apply_fn](self.data, rhs.data), True
        )

    @always_inline
    fn __le__(self, rhs: Self) -> Bool:
        """Compares this tuple to another tuple using LE comparison.

        A tuple is less-or-equal than another tuple if all corresponding
        elements of lhs is less-or-equal than rhs.

        Note: This is **not** a lexical comparison.

        Args:
            rhs: Right hand side tuple.

        Returns:
            The comparison result.
        """

        @always_inline
        fn apply_fn[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Bool:
            return a <= b

        return _bool_tuple_reduce[_reduce_and_fn](
            _int_tuple_compare[apply_fn](self.data, rhs.data), True
        )

    @always_inline
    fn __gt__(self, rhs: Self) -> Bool:
        """Compares this tuple to another tuple using GT comparison.

        A tuple is greater-than than another tuple if all corresponding
        elements of lhs is greater-than than rhs.

        Note: This is **not** a lexical comparison.

        Args:
            rhs: Right hand side tuple.

        Returns:
            The comparison result.
        """

        @always_inline
        fn apply_fn[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Bool:
            return a > b

        return _bool_tuple_reduce[_reduce_and_fn](
            _int_tuple_compare[apply_fn](self.data, rhs.data), True
        )

    @always_inline
    fn __ge__(self, rhs: Self) -> Bool:
        """Compares this tuple to another tuple using GE comparison.

        A tuple is greater-or-equal than another tuple if all corresponding
        elements of lhs is greater-or-equal than rhs.

        Note: This is **not** a lexical comparison.

        Args:
            rhs: Right hand side tuple.

        Returns:
            The comparison result.
        """

        @always_inline
        fn apply_fn[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Bool:
            return a >= b

        return _bool_tuple_reduce[_reduce_and_fn](
            _int_tuple_compare[apply_fn](self.data, rhs.data), True
        )

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this IndexList value to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """

        writer.write("(")

        for i in range(size):
            if i != 0:
                writer.write(", ")

            var element = self[i]

            @parameter
            if bitwidthof[element_type]() == 32:
                writer.write(Int32(element))
            else:
                writer.write(Int64(element))

        # Single element tuples should be printed with a trailing comma.
        @parameter
        if size == 1:
            writer.write(",")

        writer.write(")")

    @no_inline
    fn __str__(self) -> String:
        """Get the tuple as a string.

        Returns:
            A string representation.
        """
        return String.write(self)

    @always_inline
    fn cast[
        dtype: DType
    ](self, out result: IndexList[size, element_type=dtype]):
        """Casts to the target DType.

        Parameters:
            dtype: The dtype to cast towards.

        Returns:
            The list casted to the target type.
        """
        constrained[dtype.is_integral(), "the target type must be integral"]()
        result = {}

        @parameter
        for i in range(size):
            result.data[i] = self.data.__getitem__[i]().cast[
                result.element_type
            ]()

    fn __hash__[H: Hasher](self, mut hasher: H):
        """Updates hasher with the underlying bytes.
        Parameters:
            H: The hasher type.
        Args:
            hasher: The hasher instance.
        """

        @parameter
        for i in range(size):
            hasher.update(self.data[i])


# ===-----------------------------------------------------------------------===#
# Factory functions for creating index.
# ===-----------------------------------------------------------------------===#
@always_inline
fn Index[
    T0: Intable, //, *, dtype: DType = DType.int64
](x: T0, out result: IndexList[1, element_type=dtype]):
    """Constructs a 1-D Index from the given value.

    Parameters:
        T0: The type of the 1st argument.
        dtype: The integer type of the underlying element.

    Args:
        x: The initial value.

    Returns:
        The constructed IndexList.
    """
    return __type_of(result)(Int(x))


@always_inline
fn Index[
    *, dtype: DType = DType.int64
](x: UInt, out result: IndexList[1, element_type=dtype]):
    """Constructs a 1-D Index from the given value.

    Parameters:
        dtype: The integer type of the underlying element.

    Args:
        x: The initial value.

    Returns:
        The constructed IndexList.
    """
    return __type_of(result)(Int(x))


@always_inline
fn Index[
    T0: Intable, T1: Intable, //, *, dtype: DType = DType.int64
](x: T0, y: T1, out result: IndexList[2, element_type=dtype]):
    """Constructs a 2-D Index from the given values.

    Parameters:
        T0: The type of the 1st argument.
        T1: The type of the 2nd argument.
        dtype: The integer type of the underlying element.

    Args:
        x: The 1st initial value.
        y: The 2nd initial value.

    Returns:
        The constructed IndexList.
    """
    return __type_of(result)(Int(x), Int(y))


@always_inline
fn Index[
    *, dtype: DType = DType.int64
](x: UInt, y: UInt, out result: IndexList[2, element_type=dtype]):
    """Constructs a 2-D Index from the given values.

    Parameters:
        dtype: The integer type of the underlying element.

    Args:
        x: The 1st initial value.
        y: The 2nd initial value.

    Returns:
        The constructed IndexList.
    """
    return __type_of(result)(Int(x), Int(y))


@always_inline
fn Index[
    T0: Intable,
    T1: Intable,
    T2: Intable, //,
    *,
    dtype: DType = DType.int64,
](x: T0, y: T1, z: T2, out result: IndexList[3, element_type=dtype]):
    """Constructs a 3-D Index from the given values.

    Parameters:
        T0: The type of the 1st argument.
        T1: The type of the 2nd argument.
        T2: The type of the 3rd argument.
        dtype: The integer type of the underlying element.

    Args:
        x: The 1st initial value.
        y: The 2nd initial value.
        z: The 3rd initial value.

    Returns:
        The constructed IndexList.
    """
    return __type_of(result)(Int(x), Int(y), Int(z))


@always_inline
fn Index[
    T0: Intable,
    T1: Intable,
    T2: Intable,
    T3: Intable, //,
    *,
    dtype: DType = DType.int64,
](x: T0, y: T1, z: T2, w: T3, out result: IndexList[4, element_type=dtype]):
    """Constructs a 4-D Index from the given values.

    Parameters:
        T0: The type of the 1st argument.
        T1: The type of the 2nd argument.
        T2: The type of the 3rd argument.
        T3: The type of the 4th argument.
        dtype: The integer type of the underlying element.

    Args:
        x: The 1st initial value.
        y: The 2nd initial value.
        z: The 3rd initial value.
        w: The 4th initial value.

    Returns:
        The constructed IndexList.
    """
    return __type_of(result)(Int(x), Int(y), Int(z), Int(w))


@always_inline
fn Index[
    T0: Intable,
    T1: Intable,
    T2: Intable,
    T3: Intable,
    T4: Intable, //,
    *,
    dtype: DType = DType.int64,
](
    x: T0,
    y: T1,
    z: T2,
    w: T3,
    v: T4,
    out result: IndexList[5, element_type=dtype],
):
    """Constructs a 5-D Index from the given values.

    Parameters:
        T0: The type of the 1st argument.
        T1: The type of the 2nd argument.
        T2: The type of the 3rd argument.
        T3: The type of the 4th argument.
        T4: The type of the 5th argument.
        dtype: The integer type of the underlying element.

    Args:
        x: The 1st initial value.
        y: The 2nd initial value.
        z: The 3rd initial value.
        w: The 4th initial value.
        v: The 5th initial value.

    Returns:
        The constructed IndexList.
    """
    return __type_of(result)(Int(x), Int(y), Int(z), Int(w), Int(v))


# ===-----------------------------------------------------------------------===#
# Utils
# ===-----------------------------------------------------------------------===#


@always_inline
fn product[size: Int](tuple: IndexList[size, **_], end_idx: Int = size) -> Int:
    """Computes a product of values in the tuple up to the given index.

    Parameters:
        size: The tuple size.

    Args:
        tuple: The tuple to get a product of.
        end_idx: The end index.

    Returns:
        The product of all tuple elements in the given range.
    """
    return product[size](tuple, 0, end_idx)


@always_inline
fn product[
    size: Int
](tuple: IndexList[size, **_], start_idx: Int, end_idx: Int) -> Int:
    """Computes a product of values in the tuple in the given index range.

    Parameters:
        size: The tuple size.

    Args:
        tuple: The tuple to get a product of.
        start_idx: The start index of the range.
        end_idx: The end index of the range.

    Returns:
        The product of all tuple elements in the given range.
    """
    var product: Int = 1
    for i in range(start_idx, end_idx):
        product *= tuple[i]
    return product
