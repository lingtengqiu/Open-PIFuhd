#!/bin/sh
path=`basename $PWD`
cd ..
curvr_path=`pwd`
cd $path
array1=($1 $2 $3)
count=${#array1[@]}
for i in `seq 1 $count`
do
	save_path="${curvr_path}/save/${array1[$i-1]}/${path}"
	if [ ! -d "$save_path" ];then
		mkdir -p $save_path
	else
		echo " ${save_path} had exist!"
	fi
	if [ ! -d ${array1[$i-1]} ];then
		echo "build linke"
	else
		rm -rf ${array1[$i-1]}
	fi
	ln -s "${curvr_path}/save/${array1[$i-1]}/${path}" ${array1[$i-1]}
done

