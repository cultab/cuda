#!/bin/sh

RADIX=0
COUNTING=1
BITONIC=2

rm -f results.csv
echo '"method","size","threads","blocks","max_value", "time"' > results.csv

# radix sort needs at least 256 threads in total
method=$RADIX

for blocks_threads in "16 16" "8 32" "16 64" "8 128" "96 128" "192 128" "1 256" "4 256" "1 1024" "4 1024"; do
    blocks=$(echo "$blocks_threads" | cut -d' ' -f1)
    threads=$(echo "$blocks_threads" | cut -d' ' -f2)
    for size in 10 100 1000 10000 100000 1000000 10000000 100000000; do
        echo "./sort $method $size $threads $blocks"
        time=$(./sort "$method" "$size" "$threads" "$blocks" | tail -1 | cut -d ' ' -f4)
        echo "radix,$size,$threads,$blocks,-1,$time" >> results.csv
    done
done

# counting needs at least MAX_VALUE threads in total
method=$COUNTING

for blocks_threads in "16 16" "8 32" "16 64" "8 128" "96 128" "192 128" "1 256" "4 256" "1 1024" "4 1024"; do
    blocks=$(echo "$blocks_threads" | cut -d' ' -f1)
    threads=$(echo "$blocks_threads" | cut -d' ' -f2)
    for size in 10 100 1000 10000 100000 1000000 10000000 100000000; do
        for max_value in 2 16 64 256 1024; do
            total_threads=$(((blocks * threads)))
            # if we have enough threads for this max_value
            if [ $total_threads -ge $max_value ]; then
                echo "./sort $method $size $threads $blocks $max_value"
                time=$(./sort "$method" "$size" "$threads" "$blocks" "$max_value" | tail -1 | cut -d' ' -f4)
                echo "counting,$size,$threads,$blocks,$max_value,$time" >> results.csv
            fi
        done
    done
done

# bitonic sort needs an array with a size that's a power of 2
method=$BITONIC

for blocks_threads in "16 16" "8 32" "16 64" "8 128" "96 128" "192 128" "1 256" "4 256" "1 1024" "4 1024"; do
    blocks=$(echo "$blocks_threads" | cut -d' ' -f1)
    threads=$(echo "$blocks_threads" | cut -d' ' -f2)
    for size in 16 128 1024 16384 131072 1048576 16777216; do
        echo "./sort $method $size $threads $blocks"
        time=$(./sort "$method" "$size" "$threads" "$blocks" | tail -1 | cut -d ' ' -f4)
        echo "bitonic,$size,$threads,$blocks,-1,$time" >> results.csv
    done
done
