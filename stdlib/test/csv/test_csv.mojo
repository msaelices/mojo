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

from csv import reader
from pathlib import Path, _dir_of_current_file


def test_dialect():
    var csv_path = _dir_of_current_file() / "people.csv"
    with open(csv_path, "r") as csv_file:
        r = reader(csv_file, delimiter=",", quotechar='"')
        # TODO: Add more tests


def main():
    test_dialect()
