#!/bin/bash

for f in *.png; do
  convert -crop 500x500+0+0 ./"$f" ./"${f%.png}r.png"
done

