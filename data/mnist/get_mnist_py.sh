#!/usr/bin/env sh
# This scripts downloads the mnist data and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading..."

wget http://deeplearning.net/data/mnist/mnist.pkl.gz

echo "Done."
