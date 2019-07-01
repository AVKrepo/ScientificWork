#!/bin/bash
echo "transforming image $1 to output.png"

../pjpeg2pan $1 image.pan
../prgb2hsl image.pan data1.pan
../pgetband 0 data1.pan data44.pan
../pgetband 1 data1.pan data45.pan
../pgetband 2 data1.pan data36.pan
../pdilatation 1 5 data36.pan data37.pan
../perosion 1 5 data37.pan data38.pan
../pmeanvalue data38.pan data41.pan
../psub data36.pan data38.pan data40.pan
../paddval data41.pan data40.pan data42.pan
../pimg2imc 4 data44.pan data45.pan data42.pan data43.pan
../phsl2rgb data43.pan out7.pan
../ppan2png out7.pan output.png

rm image.pan data1.pan data36.pan data37.pan data38.pan data40.pan data41.pan data42.pan data43.pan data44.pan data45.pan out7.pan
