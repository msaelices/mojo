# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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

from collections.dict import Dict, _DictKeyIter, _DictValueIter, _DictEntryIter


@value
struct Counter[V: KeyElement](Sized, CollectionElement, Boolable):
    """A container for counting hashable items.

    The value type must be specified statically, unlike a Python
    Counter, which can accept arbitrary value types.

    The value type must implement the `KeyElement` trait, as its values are
    stored in the dictionary as keys.

    Usage:

    ```mojo
    from collections import Counter
    var c = Counter[String](List("a", "a", "a", "b", "b", "c", "d", "c", "c"))
    print(c["a"]) # prints 3
    print(c["b"]) # prints 2
    ```
    Parameters:
        V: The value type to be counted. Currently must be KeyElement.
    """

    # Fields
    var _data: Dict[V, Int]

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(inout self):
        """Create a new, empty Counter object."""
        self._data = Dict[V, Int]()

    # TODO: Change List to Iterable when it is supported in Mojo
    fn __init__(inout self, items: List[V]):
        """Create a from an input iterable.

        Args:
            items: A list of items to count.
        """
        self._data = Dict[V, Int]()
        for item_ref in items:
            var item = item_ref[]
            self._data[item] = self._data.get(item, 0) + 1

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    def __getitem__(self, key: V) -> Int:
        """Get the count of a key.

        Args:
            key: The key to get the count of.

        Returns:
            The count of the key.
        """
        return self.get(key, 0)

    fn __setitem__(inout self, value: V, count: Int):
        """Set a value in the keyword Counter by key.

        Args:
            value: The value to associate with the specified count.
            count: The count to store in the Counter.
        """
        self._data[value] = count

    fn __iter__(ref [_]self: Self) -> _DictKeyIter[V, Int, __lifetime_of(self)]:
        """Iterate over the keyword dict's keys as immutable references.

        Returns:
            An iterator of immutable references to the Counter values.
        """
        return self._data.__iter__()

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    fn __len__(self) -> Int:
        """The number of elements currently stored in the Counter."""
        return len(self._data)

    fn __bool__(self) -> Bool:
        """Check if the Counter is empty or not.

        Returns:
            `False` if the Counter is empty, `True` if there is at least one element.
        """
        return bool(len(self))

    # ===-------------------------------------------------------------------===#
    # Comparison operators
    # ===-------------------------------------------------------------------===#

    fn __eq__(self, other: Self) -> Bool:
        """Check if all counts agree. Missing counts are treated as zero.

        Args:
            other: The other Counter to compare to.

        Returns:
            True if the two Counters are equal, False otherwise.
        """

        @parameter
        @always_inline
        fn is_eq(keys: _DictKeyIter[V, Int, _]) -> Bool:
            for e_ref in keys:
                var e = e_ref[]
                if self.get(e, 0) != other.get(e, 0):
                    return False
            return True

        return is_eq(self.keys()) and is_eq(other.keys())

    fn __ne__(self, other: Self) -> Bool:
        """Check if all counts disagree. Missing counts are treated as zero.

        Args:
            other: The other Counter to compare to.

        Returns:
            True if the two Counters are not equal, False otherwise.
        """
        return not self == other

    fn __le__(self, other: Self) -> Bool:
        """Check if all counts are less than or equal to the other Counter.

        Args:
            other: The other Counter to compare to.

        Returns:
            True if all counts are less than or equal to the other Counter, False otherwise.
        """

        @parameter
        @always_inline
        fn is_le(keys: _DictKeyIter[V, Int, _]) -> Bool:
            for e_ref in keys:
                var e = e_ref[]
                if self.get(e, 0) > other.get(e, 0):
                    return False
            return True

        return is_le(self.keys())

    fn __lt__(self, other: Self) -> Bool:
        """Check if all counts are less than in the other Counter.

        Args:
            other: The other Counter to compare to.

        Returns:
            True if all counts are less than in the other Counter, False otherwise.
        """
        return self <= other and self != other

    fn __gt__(self, other: Self) -> Bool:
        """Check if all counts are greater than in the other Counter.

        Args:
            other: The other Counter to compare to.

        Returns:
            True if all counts are greater than in the other Counter, False otherwise.
        """
        return other < self

    fn __ge__(self, other: Self) -> Bool:
        """Check if all counts are greater than or equal to the other Counter.

        Args:
            other: The other Counter to compare to.

        Returns:
            True if all counts are greater than or equal to the other Counter, False otherwise.
        """
        return other <= self

    # ===-------------------------------------------------------------------===#
    # Binary operators
    # ===-------------------------------------------------------------------===#

    fn __add__(self, other: Self) -> Self:
        """Add counts from two Counters.

        Args:
            other: The other Counter to add to this Counter.

        Returns:
            A new Counter with the counts from both Counters added together.
        """
        var result = Counter[V]()

        result.update(self)
        result.update(other)

        return +result^  # Remove zero and negative counts

    fn __iadd__(inout self, other: Self) raises:
        """Add counts from another Counter to this Counter.

        Args:
            other: The other Counter to add to this Counter.
        """
        self.update(other)
        self._keep_positive()

    fn _keep_positive(inout self) raises:
        """Remove zero and negative counts from the Counter."""
        for key_ref in self.keys():
            var key = key_ref[]
            if self[key] <= 0:
                _ = self.pop(key)

    # ===-------------------------------------------------------------------===#
    # Unary operators
    # ===-------------------------------------------------------------------===#

    fn __pos__(self) -> Self:
        """Return a shallow copy of the Counter.

        Returns:
            A shallow copy of the Counter.
        """
        var result = Counter[V]()
        for item_ref in self.items():
            var item = item_ref[]
            if item.value > 0:
                result[item.key] = item.value
        return result^

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn get(self, value: V) -> Optional[Int]:
        """Get a value from the counter.

        Args:
            value: The value to search for in the Counter.

        Returns:
            An optional value containing a copy of the value if it was present,
            otherwise an empty Optional.
        """
        return self._data.get(value)

    fn get(self, value: V, default: Int) -> Int:
        """Get a value from the Counter.

        Args:
            value: The value to search for in the counter.
            default: Default count to return.

        Returns:
            A copy of the value if it was present, otherwise default.
        """
        return self._data.get(value, default)

    fn pop(inout self, value: V) raises -> Int:
        """Remove a value from the Counter by value.

        Args:
            value: The value to remove from the Counter.

        Returns:
            The value associated with the key, if it was in the Counter.

        Raises:
            "KeyError" if the key was not present in the Counter.
        """
        return self._data.pop(value)

    fn pop(inout self, value: V, owned default: Int) raises -> Int:
        """Remove a value from the Counter by value.

        Args:
            value: The value to remove from the Counter.
            default: Optionally provide a default value to return if the value
                was not found instead of raising.

        Returns:
            The value associated with the key, if it was in the Counter.
            If it wasn't, return the provided default value instead.

        Raises:
            "KeyError" if the key was not present in the Counter and no
            default value was provided.
        """
        return self._data.pop(value, default)

    fn keys(ref [_]self: Self) -> _DictKeyIter[V, Int, __lifetime_of(self)]:
        """Iterate over the Counter's keys as immutable references.

        Returns:
            An iterator of immutable references to the Counter keys.
        """
        return self._data.keys()

    fn values(ref [_]self: Self) -> _DictValueIter[V, Int, __lifetime_of(self)]:
        """Iterate over the Counter's values as references.

        Returns:
            An iterator of references to the Counter values.
        """
        return self._data.values()

    fn items(ref [_]self: Self) -> _DictEntryIter[V, Int, __lifetime_of(self)]:
        """Iterate over the dict's entries as immutable references.

        Returns:
            An iterator of immutable references to the Counter entries.
        """
        return self._data.items()

    fn clear(inout self):
        """Remove all elements from the Counter."""
        self._data.clear()

    # Special methods for counter

    fn total(self) -> Int:
        """Return the total of all counts in the Counter.

        Returns:
            The total of all counts in the Counter.
        """
        var total = 0
        for count_ref in self.values():
            total += count_ref[]
        return total

    fn most_common(self, n: Int) -> List[CountTuple[V]]:
        """Return a list of the n most common elements and their counts from the most common to the least.

        Args:
            n: The number of most common elements to return.

        Returns:
            A list of the n most common elements and their counts.
        """
        var items: List[CountTuple[V]] = List[CountTuple[V]]()
        for item_ref in self._data.items():
            var item = item_ref[]
            var t = CountTuple[V](item.key, item.value)
            items.append(t)

        @parameter
        fn comparator(a: CountTuple[V], b: CountTuple[V]) -> Bool:
            return a < b

        sort[type = CountTuple[V], cmp_fn=comparator](items)
        return items[:n]

    fn elements(self) -> List[V]:
        """Return an iterator over elements repeating each as many times as its count.

        Returns:
            An iterator over the elements in the Counter.
        """
        var elements: List[V] = List[V]()
        for item_ref in self._data.items():
            var item = item_ref[]
            for _ in range(item.value):
                elements.append(item.key)
        return elements

    fn update(inout self, other: Self):
        """Update the Counter, like dict.update() but add counts instead of replacing them.

        Args:
            other: The Counter to update this Counter with.
        """
        for item_ref in other.items():
            var item = item_ref[]
            self._data[item.key] = self._data.get(item.key, 0) + item.value

    fn subtract(inout self, other: Self):
        """Subtract count. Both inputs and outputs may be zero or negative.

        Args:
            other: The Counter to subtract from this Counter.
        """
        for item_ref in other.items():
            var item = item_ref[]
            self[item.key] = self.get(item.key, 0) - item.value


struct CountTuple[V: KeyElement](
    CollectionElement,
):
    """A tuple representing a value and its count in a Counter.

    Parameters:
        V: The value in the Counter.
    """

    # Fields
    var _value: V
    """ The value in the Counter."""
    var _count: Int
    """ The count of the value in the Counter."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(inout self, value: V, count: Int):
        """Create a new CountTuple.

        Args:
            value: The value in the Counter.
            count: The count of the value in the Counter.
        """
        self._value = value
        self._count = count

    fn __copyinit__(inout self, other: Self):
        """Create a new CountTuple by copying another CountTuple.

        Args:
            other: The CountTuple to copy.
        """
        self._value = other._value
        self._count = other._count

    fn __moveinit__(inout self, owned other: Self):
        """Create a new CountTuple by moving another CountTuple.

        Args:
            other: The CountTuple to move.
        """
        self._value = other._value^
        self._count = other._count

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    fn __lt__(self, other: Self) -> Bool:
        """Compare two CountTuples by count, then by value.

        Args:
            other: The other CountTuple to compare to.

        Returns:
            True if this CountTuple is less than the other, False otherwise.
        """
        return self._count > other._count

    fn __eq__(self, other: Self) -> Bool:
        """Compare two CountTuples for equality.

        Args:
            other: The other CountTuple to compare to.

        Returns:
            True if the two CountTuples are equal, False otherwise.
        """
        return self._count == other._count

    @always_inline("nodebug")
    fn __getitem__(self, idx: Int) -> Variant[V, Int]:
        """Get an element in the tuple.

        Args:
            idx: The element to return.

        Returns:
            The value if idx is 0 and the count if idx is 1.
        """
        debug_assert(
            0 <= idx <= 1,
            "index must be within bounds",
        )
        if idx == 0:
            return self._value
        else:
            return self._count
