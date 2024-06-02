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
# RUN: %mojo %s

from collections.counter import Counter

from testing import assert_equal, assert_false, assert_raises, assert_true


def test_counter_construction():
    _ = Counter[Int]()
    _ = Counter[Int](List[Int]())
    _ = Counter[String](List[String]())


def test_counter_getitem():
    c = Counter[Int](List[Int](1, 2, 2, 3, 3, 3, 4))
    assert_equal(c[1], 1)
    assert_equal(c[2], 2)
    assert_equal(c[3], 3)
    assert_equal(c[4], 1)
    assert_equal(c[5], 0)


def test_iter():
    var c = Counter[String]()
    c["a"] = 1
    c["b"] = 2

    var keys = String("")
    for key in c:
        keys += key[]

    assert_equal(keys, "ab")


def test_iter_keys():
    var c = Counter[String]()
    c["a"] = 1
    c["b"] = 2

    var keys = String("")
    for key in c.keys():
        keys += key[]

    assert_equal(keys, "ab")


def test_iter_values():
    var c = Counter[String]()
    c["a"] = 1
    c["b"] = 2

    var sum = 0
    for value in c.values():
        sum += value[]

    assert_equal(sum, 3)


def test_iter_values_mut():
    var c = Counter[String]()
    c["a"] = 1
    c["b"] = 2

    for value in c.values():
        value[] += 1

    assert_equal(2, c["a"])
    assert_equal(3, c["b"])
    assert_equal(2, len(c))


def test_iter_items():
    var c = Counter[String]()
    c["a"] = 1
    c["b"] = 2

    var keys = String("")
    var sum = 0
    for entry in c.items():
        keys += entry[].key
        sum += entry[].value

    assert_equal(keys, "ab")
    assert_equal(sum, 3)


def test_total():
    var c = Counter[String]()
    c["a"] = 1
    c["b"] = 2

    assert_equal(c.total(), 3)


def test_most_common():
    var c = Counter[String]()
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3

    var most_common = c.most_common(2)
    assert_equal(len(most_common), 2)
    assert_equal(most_common[0][0][String], "c")
    assert_equal(most_common[0][1][Int], 3)
    assert_equal(most_common[1][0][String], "b")
    assert_equal(most_common[1][1][Int], 2)


def test_eq():
    var c1 = Counter[String]()
    c1["a"] = 1
    c1["b"] = 2
    c1["d"] = 0

    var c2 = Counter[String]()
    c2["a"] = 1
    c2["b"] = 2
    c2["c"] = 0

    assert_true(c1 == c2)

    c2["b"] = 3
    assert_false(c1 == c2)


def test_elements():
    var c = Counter[String]()
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3

    var elements = c.elements()

    assert_equal(len(elements), 6)
    assert_equal(elements[0], "a")
    assert_equal(elements[1], "b")
    assert_equal(elements[2], "b")
    assert_equal(elements[3], "c")
    assert_equal(elements[4], "c")
    assert_equal(elements[5], "c")


def test_update():
    var c1 = Counter[String]()
    c1["a"] = 1
    c1["b"] = 2

    var c2 = Counter[String]()
    c2["b"] = 3
    c2["c"] = 4

    c1.update(c2)

    assert_equal(c1["a"], 1)
    assert_equal(c1["b"], 5)
    assert_equal(c1["c"], 4)


def test_substract():
    var c1 = Counter[String]()
    c1["a"] = 4
    c1["b"] = 2
    c1["c"] = 0

    var c2 = Counter[String]()
    c2["a"] = 1
    c2["b"] = -2
    c2["c"] = 3

    c1.subtract(c2)

    assert_equal(c1["a"], 3)
    assert_equal(c1["b"], 4)
    assert_equal(c1["c"], -3)


def main():
    test_counter_construction()
    test_counter_getitem()
    test_elements()
    test_eq()
    test_iter()
    test_iter_keys()
    test_iter_items()
    test_iter_values()
    test_iter_values_mut()
    test_most_common()
    test_substract()
    test_total()
    test_update()