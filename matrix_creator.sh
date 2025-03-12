#script constants
LOG_FILENAME="matrix_creation_results.txt" 

SIZE_LIST=(10000 20000 50000 100000)
RATE_LIST=(0.2 0.1 0.05)
DENSITY_LIST=(3 5)

EFFICIENCY=20
MATRIX_ATEMPTS=50
MAX_ITERATIONS=100

G="BASE_G"
GOOD_DIR="proven_tests"
TEST_DIR="tests"

DELETE_BAD_MATRICES=1

#create/open log file to append to
touch $LOG_FILENAME
echo ==================== >> $LOG_FILENAME

for size in "${SIZE_LIST[@]}"
do
	for rate in "${RATE_LIST[@]}"
	do
		for density in "${DENSITY_LIST[@]}"
		do
			treshold_EBR=$( echo "scale=5;$rate / $EFFICIENCY" | bc -l )
			echo treshold: $treshold_EBR
			matrix_name="$size"_"$rate"_"$density"
			echo making $matrix_name
			for matrix in `seq 1 1 $MATRIX_ATEMPTS`
			do
				#create matrix
				tolerance=$( echo "$size * $rate" | bc )
				d=$( echo "$density / $rate" | bc )
				./bin/matrix "$TEST_DIR"/$matrix_name 1 $tolerance $size H $d $density

				#test if matrix is within useful level
				is_matrix_good=0
				for i in `seq 1 1 $MAX_ITERATIONS`
				do
					./bin/ldpc $G "$TEST_DIR"/$matrix_name $treshold_EBR -1 $i
					res=$?
					if [ $res != -1 ] 
					then
						is_matrix_good=1
						break
					fi
				done
				if [ $is_matrix_good == 1 ]
				then
					mv "$TEST_DIR"/"$matrix_name" "$GOOD_DIR"/"$matrix_name"
					num_errors=$( echo "scale=1;$size * $treshold_EBR" | bc )
					echo "$matrix_name" was a success for $treshold_EBR which is $num_errors errors >> $LOG_FILENAME
					break
				else
					rm "$TEST_DIR"/"$matrix_name"
				fi
			done
			if [ $is_matrix_good == 0 ]
			then
				echo "$matrix_name" was a failure for $treshold_EBR which is $num_errors errors >> $LOG_FILENAME
			fi
		done
	done
done
