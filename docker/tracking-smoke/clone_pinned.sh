#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: clone_pinned.sh REPOSITORY_URL COMMIT DESTINATION" >&2
  exit 2
fi

repository="$1"
commit="$2"
destination="$3"

test -n "${commit}"
test ! -e "${destination}"

git init -q "${destination}"
git -C "${destination}" remote add origin "${repository}"
git -C "${destination}" fetch -q --depth 1 origin "${commit}"
git -C "${destination}" checkout -q --detach FETCH_HEAD

actual="$(git -C "${destination}" rev-parse HEAD)"
if [[ "${actual}" != "${commit}" ]]; then
  echo "source revision mismatch: expected ${commit}, got ${actual}" >&2
  exit 1
fi

git -C "${destination}" remote remove origin
find "${destination}/.git" -mindepth 1 -delete
rmdir "${destination}/.git"
