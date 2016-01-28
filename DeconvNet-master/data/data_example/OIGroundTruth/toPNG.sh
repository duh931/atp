#!/bin/bash

for f in *.pgm; do
  convert ./"$f" ./"${f%.pgm}.png"
done

for f in *.ppm; do
  convert ./"$f" ./"${f%.ppm}.png"
