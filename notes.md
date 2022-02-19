nvidia-470.94_3
nvidia-libs-470.94_3
nvidia-gtklibs-470.94_3
nvidia-dkms-470.94_3

ask about:
    radix sort MSB and LSB stability
    Fast 4-way parallel radix sorting on GPUs:
        https://vgc.poly.edu/~csilva/papers/cgf.pdf#figure.2
        Linh Ha, Jens Krüger, Cláudio T. Silva†
        University of Utah


todo in order of importance:
    * [x] investigate KEYS_COUNT
    * [ ] better prefix sum
    * [x] try parallelizing move step: tried and failed, need implementation from Fast 4-way...
    * [ ] benchmarking code and results
    * [x] bitonic sort :^)
    * [ ] comments

shift instead of pow(2, *)
uint and unsigned int
int sizes 32vs64 in gpu

see: https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
