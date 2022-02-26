# 1 Εισαγωγή

Σε αυτή την εργασία θα δούμε τρεις αλγορίθμους ταξινόμησης και θα τους
υλοποιήσουμε σε CUDA. Οι αλγόριθμοι που επιλέχθηκαν είναι το radix sort,
counting sort και bitonic sort. Αρχικά θα δούμε την γενική ιδέα γύρω από
τον κάθε αλγόριθμο, όπως τυχόν προτερήματα και μειονεκτήματα. Ύστερα θα
δούμε την υλοποίηση του κάθε αλγόριθμου μαζί με κάποιες λεπτομέρειες της
υλοποίησης. Τέλος θα συγκρίνουμε τα αποτελέσματα επιδόσεων των
αλγορίθμων και θα προσπαθήσουμε να βγάλουμε συμπεράσματα.

# 2 Αλγόριθμοι

## 2.1 Counting sort

Το counting sort βασίζεται στην ιδέα ότι ξέροντας το πλήθος κάθε αριθμού
μια ακολουθίας μπορούμε να ανακατασκευάσουμε την ακολουθία. Το βασικό
πλεονέκτημά του είναι ότι η πολυπλοκότητα του είναι *O*(*m*) όπου *m*
είναι το μέγεθος του συνόλου των τιμών της ακολουθίας. Ταυτόχρονα αυτό
είναι και το βασικό του μειονέκτημα γιατί (ανάλογα με την υλοποίηση)
μεγαλύτερο σύνολο τιμών αυξάνει την μνήμη που θα πρέπει να
χρησιμοποιήσουμε. Δηλαδή το εύρος των τιμών που μπορεί να διαχειριστεί
περιορίζεται από την μνήμη που έχει στη διάθεσή του.

## 2.2 Radix sort

Το radix sort δεν είναι παρά μια επέκταση του counting sort ώστε να
αποφευχθεί το κύριο μειονέκτημά του –-το περιορισμένο εύρος τιμών της
ακολουθίας. Πιο συγκεκριμένα λειτουργεί σε φάσεις ταξινομώντας
παίρνοντας υπόψιν ένα υποσύνολο των ψηφίων των στοιχείων σε κάθε φάση.
Μπορεί να ξεκινάει από το πιο σημαντικό ψηφίο (MSB) ή το λιγότερο
σημαντικό ψηφίο (LSB).

## 2.3 Bitonic sort

Το bitonic sort βασίζεται στις ιδιότητες των διτονικών ακολουθιών. Είναι
ενδιαφέρον γιατί μπορεί να υλοποιηθεί σαν ένα sorting network όπως
γράφει ο Batcher (1968).

Ξεκινάει χτίζοντας αύξουσες και φθίνουσες υποακολουθίες εναλλάξ μήκους 2
στοιχείων. Συνεχίζει φτιάχνοντας διτονικές ακολουθίες από τις αύξουσες
και φθίνουσες εκμεταλλευόμενος το γεγονός ότι μια αύξουσα ακολουθία
ακολουθούμενη από μια φθίνουσα φτιάχνουν μια διτονική ακολουθία.
Συνεχίζει μέχρι να φτάσει σε μια διτονική ακολουθία. Σε αυτό το σημείο
συγχωνεύει τις υποακολουθίες εκμεταλλευόμενος των ιδιοτήτων που οι
διτονικές ακολουθίες έχουν σε σχέση με την ανισότητα των στοιχείων των
υποακολουθειών τους.

# 3 Υλοποίηση

Η υλοποίηση των αλγορίθμων έγινε σε CUDA (δοκιμάστηκε σε CUDA 10.2 11.5
και 11.6). Το πρόγραμμα δέχεται commandline arguments για τον αλγόριθμο
που θα τρέξει, το μέγεθος της ακολουθίας, τα blocks και threads.

Έτσι παράγει μια τυχαία ακολουθία με το μέγεθος που δώθηκε. Αφού
ταξινομήσει την ακολουθία επιστρέφει τον χρόνο που πήρε η ταξινόμηση
χωρίς να συμπεριλαμβάνει την αρχική και την τελική μεταφορά δεδομένων
από τον host στο device και το αντίστροφο.

Παρακάτω βλέπουμε το help menu του προγράμματος:

## 3.1 Prefix Sum

Το prefix sum χρησιμοποιείται από το radix sort και το counting sort. Η
υλοποίηση βασίζεται στον αλγόριθμο *All Partial Sums of an Array* που
περιγράφετε από τους Hillis και Steele (1986) στην σελίδα 1173. Έχει ένα
μειονέκτημα, χρειάζεται κατ᾽ελάχιστο size πλήθος συνολικών thread
(ανεξάρτητα του αριθμού των block).

``` cpp
__host__ unsigned int *prefix_sum(unsigned int *d_counts, size_t size, int blocks, int threads)
{
    unsigned int *d_in;
    unsigned int *d_out;
    unsigned int *d_temp;

    cudaErr(cudaMalloc((void **)&d_out, size * sizeof(unsigned int)));
    cudaErr(cudaMalloc((void **)&d_in, size * sizeof(unsigned int)));

    /*
     * Initialize in and out array to counts
     * but shifted once to the right,
     * the first element of each array is memset to 0.
     * (so that we can set it from host code)
     */
    cudaMemset(d_in, 0, 1);
    cudaMemset(d_out, 0, 1);
    cudaErr(cudaMemcpy(d_in + 1, d_counts, (size - 1) * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    cudaErr(cudaMemcpy(d_out + 1, d_counts, (size - 1) * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

    for (int j = 1; j <= floor(log2(size)); j += 1) {
        prefix_sum_kernel<<<blocks, threads>>>(d_in, d_out, j, size);
        cudaLastErr();

        // copy result back to input
        cudaErr(cudaMemcpy(d_in, d_out, size * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
        // swap in and out
        d_temp = d_in;
        d_in = d_out;
        d_out = d_temp;
    }

    // free out
    cudaErr(cudaFree(d_out));

    // NOTE: return input array (yes it's backwards)
    return d_in;
}
```

``` cpp
__global__ void prefix_sum_kernel(unsigned int *in, unsigned int *out, unsigned int j, size_t size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // don't go out of bounds
    if (tid < size) {
        if (tid >= __powf(2, j - 1)) {
            out[tid] += in[tid - (int)__powf(2, j - 1)];
        }
    }
}
```

## 3.2 Counting sort

Το πρώτο βήμα είναι η καταμέτρηση των τιμών της ακολουθίας, ουσιαστικά
δημιουργούμε ένα ιστόγραμμα των τιμών.

Όπως αναφέρθηκε στην ο αλγόριθμος γνωρίζει το πλήθος των τιμών του
συνόλου της ακολουθίας, έτσι φτιάχνουμε έναν πίνακα στην shared μνήμη
δυναμικά, δίνοντας το μέγεθος του πίνακα σαν το τρίτο argument του
kernel launch (`count<<<blocks,threads, sizeof(elem) * max_value>>`).

Σε αυτό τον πίνακα το κάθε block καταγράφει το πλήθος των τιμών που
συναντάει και στο τέλος ένα thread από κάθε block γράφει τα αποτελέσματα
στην global μνήμη.

``` cpp
__global__ void count(elem *array, size_t size, unsigned int *counts, elem max_value)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int block_tid = threadIdx.x;
    int block_stride = blockDim.x;

    extern __shared__ unsigned int local_counts[];

    // zero out the block local shared memory
    for (size_t i = block_tid; i < max_value; i += block_stride) {
        local_counts[i] = 0;
    }
    syncthreads();

    for (size_t i = tid; i < size; i += stride) {
        atomicAdd(&local_counts[array[i]], 1);
    }

    syncthreads();

    // copy per block results back to global memory
    for (size_t i = block_tid; i < max_value; i += block_stride) {
        atomicAdd(&(counts[i]), local_counts[i]);
    }
}
```

To επόμενο βήμα είναι ο υπολογισμός του prefix sum του ιστογράμματος
όπως είδαμε στην .

Τέλος χρησιμοποιόντας το prefix sum του ιστογράματος ξέρουμε την αρχική
θέση στην ταξινομημένη ακολουθία τις κάθε υποακολουθίας ίσων τιμών. Με
παραδοχή ότι ο αλγόριθμος δεν θα είναι σταθερός[1] μπορούμε παράλληλα να
μετακινήσουμε κάθε στοιχείο στη σωστή του θέση.

``` cpp
__global__ void counting_move(unsigned int *d_prefix_sums, elem *d_unsorted, elem *d_sorted, size_t size)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    elem cur_elem;
    unsigned int offset = 0;

    for (int i = tid; i < size; i += stride) {
        cur_elem = d_unsorted[i];
        offset = atomicAdd(&d_prefix_sums[cur_elem], 1);
        d_sorted[offset] = cur_elem;
    }
}
```

## 3.3 Radix sort

Όπως αναφέρθηκε στην ο radix sort μπορεί να υλοποιηθεί σαν επέκταση του
counting sort.

Αντί να ταξινομούμε τα στοιχεία της ακολουθίας ψηφίο-προς-ψηφίο στο
δεκαδικό σύστημα όπως θα έκανε μια αφελή υλοποίηση, κοιτάμε τα ψηφία των
στοιχείων στο δυαδικό σύστημα σε ομάδες των 8bit. Εφόσον ταξινομούμε
32bit αριθμούς χρειαζόμαστε 4 επαναλήψεις. Κάθε ομάδα των 8bit μπορεί να
αναπαραστήσει 2<sup>8</sup> = 256 ξεχωριστές τιμές, άρα το ιστόγραμμα
μας θα είναι πίνακας 256 τιμών

Δεν επιλέχθηκαν τυχαία ομάδες 8bit, είναι η μεγαλύτερη ομάδα η οποία
διαιρεί τέλεια 32bit αριθμούς και της οποίας το ιστόγραμμα χωράει στην
shared memory (περίπου 40.000 byte). Πιάνει στην μνήμη:
2<sup>8</sup> ⋅ sizeof(unsigned int) = 256 ⋅ 4 = 1024 byte
Η αμέσως μεγαλύτερη ομάδα που διαιρεί τέλεια 32bit αριθμούς είναι η
ομάδα μεγέθους 16bit η οποία πιάνει στη μνήμη πολύ περισσότερο χώρο από
όσο έχουμε διαθέσιμο:
2<sup>16</sup> ⋅ sizeof(unsigned int) = 65536 ⋅ 4 = 262144 byte

Εξετάζουμε μόνο τα δυαδικά ψηφία που αφορά την παρούσα επανάληψη
χρησιμοποιώντας ένα bitmask στο κάθε στοιχείο και μετατοπίζοντας το
αποτέλεσμα δεξιά μέχρι η ομάδα να βρίσκεται στα τελευταία 8 bit της
μνήμης.

Τελικά όπως είδαμε πριν στην το πρώτο βήμα είναι η δημιουργία του
ιστογράμματος των τιμών, αυτή τη φορά βέβαια χρησιμοποιώντας την
bitmasked τιμή των στοιχείων.

``` cpp
__global__ void count_masked(elem *array, size_t size, unsigned int *counts, unsigned int mask, size_t shift, size_t mask_size)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    /* block id and stride */
    int block_tid = threadIdx.x;
    int block_stride = blockDim.x;

    __shared__ unsigned int local_counts[KEYS_COUNT];

    // zero out the block local shared memory
    for (size_t i = block_tid; i < KEYS_COUNT; i += block_stride) {
        local_counts[i] = 0;
    }

    syncthreads();

    for (size_t i = tid; i < size; i += stride) {
        atomicAdd(&local_counts[(array[i] & mask) >> (mask_size * shift)], 1);
    }

    syncthreads();

    // copy per block results back to global memory
    for (size_t i = block_tid; i < KEYS_COUNT; i += block_stride) {
        atomicAdd(&(counts[i]), local_counts[i]);
    }
}
```

Ξανά, το επόμενο βήμα είναι ο υπολογισμός του prefix sum του
ιστογράμματος όπως είδαμε στην .

Τέλος απομένει η μετάθεση των τιμών στην σωστή θέση αυτής της
επανάληψης. Δυστυχώς, αυτή την φορά είναι πολύ σημαντικό ο αλγόριθμος να
είναι σταθερός από την μία επανάληψη στην επόμενη. Πρέπει να είναι
σταθερός γιατί στοιχεία που έχουν την ίδια bitmasked τιμή σε αυτή την
επανάληψη μπορεί να έχουν διαφορετική πραγματική τιμή και να χαθεί η
σειρά που δόθηκε στα στοιχεία σε μια προηγούμενη επανάληψη.

Τελικά δηλαδή δεν μπορούμε να μεταθέσουμε τα στοιχεία παράλληλα, άρα η
μετάθεση γίνεται σε host κώδικα, αυτό μας επιφέρει επιπλέον κόστος
μεταφοράς δεδομένων από το device στον host και πίσω.

``` cpp
__host__ void host_move(unsigned int *prefix_sums, elem *unsorted, elem *sorted, size_t size, unsigned int mask, unsigned long mask_size, unsigned long shift) {
    int offset = 0;

    for (size_t j = 0; j < size; ++j) {
        ulong masked_elem = (unsorted[j] & mask) >> (mask_size * shift);

        offset = prefix_sums[masked_elem];
        prefix_sums[masked_elem] += 1;
        sorted[offset] = unsorted[j];
    }
}
```

## 3.4 Bitonic sort

Υλοποιήθηκε παρόμοια με το prefix sum χρησιμοποιώντας ένα εξωτερικό
βρόχο και έναν εσωτερικό βρόχο που βρίσκεται στο kernel. Βασίζετε στην
συνάρτηση `impBitonicSort()` του Pitsianis (2008).

``` cpp
for (int k = 2; k <= (int)size; k *= 2) { // k is doubled every iteration
    for (int j = k/2; j > 0; j /= 2) { // j is halved at every iteration, with truncation of fractional parts
        bitonic_step<<<blocks, threads>>>(d_unsorted, size, k, j);
    }
}
```

``` cpp
__global__ void bitonic_step(elem* d_arr, size_t size, int k, int j)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x + gridDim.x;

    int tmp;
    int ij;
    int v;
    // i is the index of our first element
    for (int i = tid; i < size; i += stride) {
        // ij is the index of the second element, the one we compare the first with
        ij = i ^ j;
        // v keeps track of the order we want to sort the i'th chunk of size k,
        // if it's 0 it's ascending else it's descending
        v = i & k;

        // if the element we compare to is after our element ?
        if (ij > i) {
            // if the elements are not in the correct order that we want
            if ((v == 0 && d_arr[i] > d_arr[ij]) || (v != 0 && (d_arr[i] < d_arr[ij]))) {
                // swap the elements at index i and ij
                tmp = d_arr[i];
                d_arr[i] = d_arr[ij];
                d_arr[ij] = tmp;
            }
        }
    }
}
```

# 4 Αποτελέσματα

Οι δοκιμές έγιναν σε τοπική εγκατάσταση με 1050ti όπως και στον server
με ΤΙΤΑΝ RTX, δεν ήταν δυνατό να γίνουν δοκιμές στον server του
εργαστηρίου με την 750ti λόγο κάποιων ασυμβατοτήτων του κώδικα με το
compute capability 5.0 της 750ti.

Τα δεδομένα συλλέχθηκαν με την χρήση του script `bench.sh` και τα
διαγράμματα δημιουργήθηκαν με την χρήση του R script `graphs.R`. Τα
διαγράμματα δημιουργήθηκαν με σκοπό να επιδείξουν την επίδραση
διαφορετικών διατάξεων block/thread, την κάρτα γραφικών και τον
αλγόριθμο

Στο πρώτο διάγραμμα βλέπουμε τον χρόνο εκτέλεσης των αλγορίθμων με
διάφορες διατάξεις block/thread και στις δύο κάρτες. Μπορούμε να
διακρίνουμε ότι ο bitonic σε μικρές ακολουθίες έχει πολύ καλά
αποτελέσματα αλλά γρήγορα μένει πίσω όσο το μέγεθος τους μεγαλώνει. Ο
counting sort είναι καθαρά πιο γρήγορος από τον radix sort, αλλά βέβαια
λειτουργεί με σχετικά **πολύ** μικρό εύρος τιμών.

![Σύγκριση καρτών ανά διάφορες διατάξεις block/thread ανά
αλγόρυθμο](report.md_files/figure-gfm/unnamed-chunk-2-1.png)

Στο δέυτερο βλέπουμε την επίδραση που έχει η επιλογή διάταξης
block/thread στον χρόνο εκτλέλεσης. Η διαφορά είναι πιο εμφανής στο
counting sort και στο bitonic sort.

![Σύγκριση διάφορων διατάξεων block/thread ανά κάρτα και
αλγόριθμο](report.md_files/figure-gfm/unnamed-chunk-3-1.png)

Στο επόμενο βλέπουμε ακόμα πιο καθαρά αυτό το παρατηρήσαμε νωρίτερα, με
την διάταξη block/thread να έχει την μεγαλύτερη επιρροή στο bitonic sort
και μετά στο counting sort με το radix να μην φαίνεται να επηρεάζεται.

![Σύγκριση καρτών ανά αλγορίθμο με διαφόρες διτάξεις
block/thread](report.md_files/figure-gfm/unnamed-chunk-4-1.png)

Παρακάτω βλέπουμε αποτελέσματα μόνο για το radix sort.

![Επίδοση radix sort με διάφορες διατάξεις block/thread ανα
κάρτα](report.md_files/figure-gfm/unnamed-chunk-5-1.png)

Παρακάτω βλέπουμε αποτελέσματα για το counting sort με διαφορετικές
τιμές max value. Δεν μπορούμε να διακρίνουμε κανένα αποτέλεσμα. Αυτό δεν
μας ξαφνιάζει γιατί θα περιμέναμε η διαφορά να βρίσκεται στην μνήμη που
χρειαζόμαστε και όχι στον χρόνο εκτέλεσης.

![Επίδοση counting sort με διάφορες διατάξεις block/thread και max value
ανα κάρτα](report.md_files/figure-gfm/unnamed-chunk-6-1.png)

Τέλος βλέπουμε αποτελέσματα για το bitonic sort.

![Επίδοση bitonic sort με διάφορες διατάξεις block/thread ανα
κάρτα](report.md_files/figure-gfm/unnamed-chunk-7-1.png)

# 5 Συμπεράσματα

-   Το bitonic sort δεν είναι πολύ αποδοτικό για μεγάλες ακολουθίες αλλά
    είναι πολύ αποδοτικό για μικρές.

-   Η επίδοση του radix sort υπέφερε πιθανώς λόγο της ανάγκης σειριακής
    μετάβασης των στοιχείων, για αυτό η διάταξη block/thread δεν φάνηκε
    να έχει επίδραση.

-   Το counting sort είναι πολύ αποδοτικό.

-   Η 1050ti μερικές φορές είχε ασταθή επίδοση όπως είδαμε στα “καρφιά”
    που δημιουργήθηκαν σε πολλά διαγράμματα.

# 6 Πιθανός ενδιαφέροντα θέματα για μελλοντική εξερεύνηση

Μια υλοποίηση radix sort βασισμένη στην εργασία των Ha, Krüger, και
Silva (2009).

Διαφορές most significant digit radix sort με least significant radix
sort (όπως η παρούσα υλοποίηση).

Βελτιστοποίηση counting sort για ακολουθίες με εύρος τιμών πολύ
μικρότερο από το max value.

Βελτιστοποιημένη υλοποίηση του prefix sum.

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-networks" class="csl-entry">

Batcher, Kenneth. 1968. ‘Sorting networks and their applications’. Στο
*Proceed. AFIPS Spring Joint Comput. Conf.*, 32:307–14.
<https://doi.org/10.1145/1468075.1468121>.

</div>

<div id="ref-4way_radix" class="csl-entry">

Ha, Linh, Jens Krüger, και Claudio Silva. 2009. ‘Fast 4-way parallel
radix sorting on GPUs’. *Comput. Graph. Forum* 28 (Δεκέμβριος): 2368–78.
<https://doi.org/10.1111/j.1467-8659.2009.01542.x>.

</div>

<div id="ref-parallel_algos" class="csl-entry">

Hillis, W. Daniel, και Guy L. Steele. 1986. ‘Data Parallel Algorithms’.
*Commun. ACM* 29 (12): 1170–83. <https://doi.org/10.1145/7902.7903>.

</div>

<div id="ref-bitonic" class="csl-entry">

Pitsianis, Nikos. 2008. ‘bitonic.c code snippet’.
<https://courses.cs.duke.edu//fall08/cps196.1/Pthreads/bitonic.c>.

</div>

</div>

[1] stable sorting: όταν η σειρά δυο ίσων στοιχείων πριν και μετά την
ταξινόμηση μένει αδιάλακτη
