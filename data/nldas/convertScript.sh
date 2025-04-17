#!/bin/sh

for fileName in *.grb; do
	echo "FileName $fileName"
	cdo -f nc4 copy $fileName $fileName.nc
done
echo "Done!"
