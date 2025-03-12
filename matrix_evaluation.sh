#script constants
PROB_FILENAME="matrix_prob_results.txt" 
ITER_FILENAME="matrix_iter_results.txt" 

EBR_MIN=0.005
EBR_INCR=0.001
EBR_MAX=0.01

MAX_ITERATIONS=100

DELETE_BAD_MATRICES=0

#create/open log file to append to
touch $PROB_FILENAME
touch $ITER_FILENAME
echo ==================== >> $PROB_FILENAME
echo ==================== >> $ITER_FILENAME

#get directory where the matrices are stored
#echo write directory
#read dirname
dirname="proven_tests"
dir="$(ls $dirname)"


#print EBRs that will be used
prob_list=" "
for EBR in `seq $EBR_MIN $EBR_INCR $EBR_MAX`
do
	prob_list+=" $EBR"
done
echo $prob_list>> $PROB_FILENAME
echo $prob_list>> $ITER_FILENAME

#search all matrices in the directory
for matrix in $dir
do
	#don't do it for G
	if [ "${matrix: -1}" != "G" ]
	then
		prob_list=" "
		iter_list=" "
		done=0
		#test all EBRs that will be used in the graph
		for EBR in `seq $EBR_MIN $EBR_INCR $EBR_MAX`
		do
			prob=0
			iter=0
			if [ $done == 0 ]
			then
				#iterate current EBR a number of times
				for i in `seq 1 1 $MAX_ITERATIONS`
				do
					#this is done in a very weird manner but it works and the "proper" way was not working
					#G="$dirname/$matrix"
					#G+=G
					G="matrices/G1.csr"

					./bin/ldpc $G $dirname/$matrix $EBR -1 $i
					res=$?
					if [ $res != -1 ] 
					then
						prob=$(( prob+1 ))
						iter=$(( iter+res ))
					else
						res=0
					fi
				done
				if [ $prob == 0 ]
				then
					done=1
				fi

				#prob is currently the ammount of succeful iterations
				iter=$( echo "scale=2;$iter/$prob" | bc )
				#obtain probability of the matrix succefully decoding with the current EBR
				prob=$( echo "$prob/$MAX_ITERATIONS*100" | bc )
				
			fi
			iter_list+=" $iter"
			prob_list+=" $prob"
		done
		#delete matrix if it is bad
		#IFS= ' '
		if [ $DELETE_BAD_MATRICES == 1 ]; then
			read -ra prob_list_array <<< "$prob_list" 
			#bad matrices classification is if it can't correct first EBR
			if [ ${prob_list_array[0]} != 0 ]
			then
				rm "$dirname/$matrix"
				rm "$G"
			fi
		fi
		#log the results
		echo $matrix $prob_list>> $PROB_FILENAME
		echo $matrix $iter_list>> $ITER_FILENAME
	fi
done
