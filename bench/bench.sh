#!/bin/bash

#
#    pcudaSAT: Simulating an efficient solution to SAT with active membranes on the GPU 
#    This simulator is published on:
#    J.M. Cecilia, J.M. García, G.D. Guerrero, M.A. Martínez-del-Amor, I. Pérez-Hurtado
#    M.J. Pérez-Jiménez. Simulating a P system based efficient solution to SAT by using
#    GPUs, Journal of Logic and Algebraic Programming, 79, 6 (2010), 317-325
#
#    pcudaSAT is a subproject of PMCGPU (Parallel simulators for Membrane 
#                                       Computing on the GPU)   
# 
#    Copyright (c) 2010 Miguel Á. Martínez-del-Amor (RGNC, University of Seville)
# 		        Ginés D. Guerrero (GACOP, University of Murcia)
#    
#    This file is part of pcudaSAT.
#  
#    pcudaSAT is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    pcudaSAT is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with pcudaSAT.  If not, see <http://www.gnu.org/licenses/>. 

# Commands and arguments


SIM="pcudaSAT"

if [ $SIM = "tsp" ];
then
	sel=-m
	sq=2
	pr=5
	echo Using TSP simulator, sel=$sel, seq=$sq, par=$pr
elif [ $SIM = "pcudaSAT" ]
then
	sel=-s
	sq=0
	pr=1
	echo Using pcudaSAT simulator, sel=$sel, seq=$sq, par=$pr
elif [ $SIM = "tsp_kernel" ];
then
        sel=-m
        sq=2
        pr=5
        echo Using TSP_KERNEL simulator, sel=$sel, seq=$sq, par=$pr
else
	echo Error selecting the simulator, select \"tsp\" or \"pcudaSAT\"
	exit
fi

WORK_DIR=$PWD
OUT_DIR=$PWD/output_$SIM
RES_DIR=$PWD/result_$SIM
OUT_PREFIX=pcuda_
IN_DIR=$PWD/input
BIN_DIR=$PWD/bin
BIN=$BIN_DIR/$SIM

mkdir -p $OUT_DIR
mkdir -p $RES_DIR
mkdir -p $BIN_DIR
cp $HOME/NVIDIA_GPU_Computing_SDK/C/bin/linux/release/pcudaSAT $BIN_DIR


cd $WORK_DIR

echo Running test for objects

for i in `seq 1 18`;
do
	echo running in_o$i.cnf

	for mode in $sq $pr;
	do
		echo running mode $mode

		$BIN $sel $mode -f $IN_DIR/in_o$i.cnf > $OUT_DIR/out_$mode\_o$i.1 2>>$OUT_DIR/err
		$BIN $sel $mode -f $IN_DIR/in_o$i.cnf > $OUT_DIR/out_$mode\_o$i.2 2>>$OUT_DIR/err
		$BIN $sel $mode -f $IN_DIR/in_o$i.cnf > $OUT_DIR/out_$mode\_o$i.3 2>>$OUT_DIR/err
	
		st=`grep -a Execution $OUT_DIR/out_$mode\_o$i.1 | tr " " "-" | cut -f3 -d-`
		nd=`grep -a Execution $OUT_DIR/out_$mode\_o$i.2 | tr " " "-" | cut -f3 -d-`
		rd=`grep -a Execution $OUT_DIR/out_$mode\_o$i.3 | tr " " "-" | cut -f3 -d-`

		res=`echo "scale=3;($st+$nd+$rd)/3"|bc`
		echo $res > $RES_DIR/out_$mode\_o$i
	done
done

echo Running test for membranes

for i in `seq 1 33`;
do
	echo running in_m$i.cnf

	for mode in $sq $pr;
	do
		echo running mode $mode

		$BIN $sel $mode -f $IN_DIR/in_m$i.cnf > $OUT_DIR/out_$mode\_m$i.1 2>>$OUT_DIR/err
		$BIN $sel $mode -f $IN_DIR/in_m$i.cnf > $OUT_DIR/out_$mode\_m$i.2 2>>$OUT_DIR/err
		$BIN $sel $mode -f $IN_DIR/in_m$i.cnf > $OUT_DIR/out_$mode\_m$i.3 2>>$OUT_DIR/err
	
		st=`grep -a Execution $OUT_DIR/out_$mode\_m$i.1 | tr " " "-" | cut -f3 -d-`
		nd=`grep -a Execution $OUT_DIR/out_$mode\_m$i.2 | tr " " "-" | cut -f3 -d-`
		rd=`grep -a Execution $OUT_DIR/out_$mode\_m$i.3 | tr " " "-" | cut -f3 -d-`

		res=`echo "scale=3;($st+$nd+$rd)/3"|bc`
		echo $res > $RES_DIR/out_$mode\_m$i
	done
done

echo Collecting data

cd $RES_DIR

for i in `seq 1 9`; do cat out_${sq}_o$i >> seq_obj-2-512_m-1024; done

for i in `seq 10 18`; do cat out_${sq}_o$i >> seq_obj-2-512_m-2048; done

for i in `seq 1 17`; do cat out_${sq}_m$i >> seq_obj-128_m-64-4M; done

for i in `seq 18 33`; do cat out_${sq}_m$i >> seq_obj-256_m-64-2M; done

for i in `seq 1 9`; do cat out_${pr}_o$i >> par_obj-2-512_m-1024; done

for i in `seq 10 18`; do cat out_${pr}_o$i >> par_obj-2-512_m-2048; done

for i in `seq 1 17`; do cat out_${pr}_m$i >> par_obj-128_m-64-4M; done

for i in `seq 18 33`; do cat out_${pr}_m$i >> par_obj-256_m-64-2M; done

echo Done.


