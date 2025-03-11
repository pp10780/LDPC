#script constants
LOG_FILENAME="matrix_results.txt" 

EBR_MIN=0.001
EBR_INCR=0.001
EBR_MAX=0.01

MAX_ITERATIONS=100

#create/open log file to append to
touch $LOG_FILENAME
echo ==================== >> $LOG_FILENAME

#get directory where the matrices are stored
echo write directory
read dirname
dir="$(ls $dirname)"

#print EBRs that will be used
for EBR in `seq $EBR_MIN $EBR_INCR $EBR_MAX`
	echo $EBR >> $LOG_FILENAME

#search all matrices in the directory
for matrix in $dir
do
	#don't do it for G
	if [ "${matrix: -1}" != "G" ]
	then
		prob_list=" "
		done=0
		#test all EBRs that will be used in the graph
		for EBR in `seq $EBR_MIN $EBR_INCR $EBR_MAX`
		do
			prob=0
			if [ $done == 0 ]
			then
				#iterate current EBR a number of times
				for i in `seq 1 1 $MAX_ITERATIONS`
				do
					#this is done in a very weird manner but it works and the "proper" way was not working
					G="$dirname/$matrix"
					G+=G
					./bin/ldpc $G $dirname/$matrix $EBR
					res=$?
					if [ $res == 0 ] 
					then 
						res=1
					else 
						res=0
					fi

					prob=$((prob+res))
				done
				#obtain probability of the matrix succefully decoding with the current EBR
				#these are intergers and should be floating point
				#prob=$((1-$prob/$MAX_ITERATIONS))
				if [ $prob == 0 ]
				then
					done=1
					echo done
				fi
			fi
			prob_list+=" $prob"
		done
		#log the results
		echo $matrix $prob_list>> $LOG_FILENAME
	fi
done
