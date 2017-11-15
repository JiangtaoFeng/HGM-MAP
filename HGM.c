#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100
#define Gaussian_Table_Size 10000
#define MAX_GAUSSIAN 9
#define MAX_SENTENCE_LENGTH 1000

const int vocab_hash_size = 30000000;

typedef float real;

struct vocab_word {
    long long cn;
    char *word;
};

char train_file[MAX_STRING], context_word_file[MAX_STRING], word_file[MAX_STRING], weight_file[MAX_STRING];
struct vocab_word *vocab;
int window = 5, window2 = 10, num_threads = 8, dimension = 300;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, lines;
long long train_words = 0, word_count_actual = 0, iter = 3, file_size = 0;
long long correct_count, total_count;
real alpha = 0.025, beta = 0.001, starting_alpha, sample = 1e-5;
real *context_word_vec, *word_vec, *weight, *bias, *gaussian_table, *expTable;
clock_t start;

int negative = 5;
const int table_size = 1e8;
int *negTable;

int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++)
        if (!strcmp(str, argv[a])) {
            if (a == argc - 1) {
                printf("Argument missing for %s\n", str);
                exit(1);
            }
            return a;
        }
    return -1;
}

int getHash(char *word) {
    unsigned long long i, hash = 0;
    for (i = 0; i < strlen(word); ++i) {
        hash = hash * 257 + word[i];
    }
    hash = hash % vocab_hash_size;
    return hash;
}

int AddWordToVocab(char *word) {
    unsigned  int hash, length = strlen(word) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
    strcpy(vocab[vocab_size].word, word);
    vocab[vocab_size].cn = 1;
    vocab_size++;
    if (vocab_size + 2 >= vocab_max_size) {
        vocab_max_size += 1000;
        vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
    }
    hash = getHash(word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = vocab_size - 1;
    return vocab_size - 1;
}

void ReadWord(char *word, FILE *fin) {
    int i = 0;
    char ch;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == 13 || ch == 0) continue;
        if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
            if (i > 0)
                break;
            else
                continue;
        }
        word[i] = ch;
        i++;
        if (i >= MAX_STRING - 1)
            i--;
    }
    word[i] = 0;

}

int SearchWord(char * word) {
    unsigned int hash = getHash(word);
    while (1) {
        if (vocab_hash[hash] == -1) return -1;
        if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}

void initVocab() {

    char word[MAX_STRING];
    FILE *fin;
    long long i;
    for (i = 0; i < vocab_hash_size; ++i) vocab_hash[i] = -1;
    fin = fopen(train_file, "rb");
    if (fin == NULL) {
        printf("ERROR: Training Data File Not Found!");
        exit(1);
    }
    vocab_size = 0;
    while (!feof(fin)) {
        ReadWord(word, fin);
        if (feof(fin)) break;
        train_words++;
        if (train_words % 100000 == 0) {
            printf("Loading: %lldK%c", train_words / 1000, 13);
            fflush(stdout);
        }
        i = SearchWord(word);
        if (i == -1)
            AddWordToVocab(word);
        else
            vocab[i].cn++;
    }
    file_size = ftell(fin);
    fclose(fin);

}

void initNegTable() {

    int i, j;
    double power = 0.75, pow_sum = 0, prefix_pow;
    negTable = (int *)malloc(table_size * sizeof(int));
    for (i = 0; i < vocab_size; i++) pow_sum += pow(vocab[i].cn, power);
    i = 0;
    prefix_pow = pow(vocab[i].cn, power) / pow_sum;
    for (j = 0; j < table_size; j++) {
        negTable[j] = i;
        if (j / (double) table_size > prefix_pow) {
            i++;
            prefix_pow += pow(vocab[i].cn, power) / pow_sum;
        }
        if (i >= vocab_size) i = vocab_size - 1;
    }

}

void initNet() {

    long long i;
    unsigned long long next_random = clock();

    posix_memalign((void **)&context_word_vec, 128, (long long)vocab_size * dimension * sizeof(real));
    if (context_word_vec == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }
    for (i = 0; i < vocab_size * dimension; ++i) {
        next_random = next_random * (unsigned long long)25214903917 + 11;
        context_word_vec[i] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / dimension;
    }

    posix_memalign((void **)&word_vec, 128, (long long)vocab_size * dimension * sizeof(real));
    if (word_vec == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }
    for (i = 0; i < vocab_size * dimension; ++i) {
        next_random = next_random * (unsigned long long)25214903917 + 11;
        word_vec[i] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / dimension;
    }

    posix_memalign((void **)&weight, 128, (long long)dimension * window2 * sizeof(real));
    if (weight == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }
    for (i = 0; i < dimension * window2; i++) {
        next_random = next_random * (unsigned long long)25214903917 + 11;
        weight[i] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / window2;
    }

    posix_memalign((void **)&bias, 128, (long long)dimension * sizeof(real));
    if (bias == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }
    for (i = 0; i < dimension; i++) {
        next_random = next_random * (unsigned long long)25214903917 + 11;
        bias[i] = ((next_random & 0xFFFF) / (real)65536) - 0.5;
    }

}

int ReadWordIndex(FILE *fin)  {

    char word[MAX_STRING];
    ReadWord(word, fin);
    if (feof(fin)) return -1;
    return SearchWord(word);

}

void train_t(void *id) {

    int i, j, k, n, d;
    real label, direction, max_prob, sum_prob;
    long long word, target, sen[MAX_SENTENCE_LENGTH];
    unsigned long long word_count = 0, last_word_count = 0;
    long long sentence_length = 0, sentence_position = 0;
    long long local_correct_count = 0, local_total_count = 0;
    int local_iter = iter;
    clock_t now;
    unsigned long long next_random = (long long) (id + clock());
    real *context = (real *)calloc(dimension, sizeof(real));
    real *context_gradient = (real *)calloc(dimension, sizeof(real));
    long long *negative_samples = (long long *)calloc(negative, sizeof(long long));
    real *prob = (real *)calloc(negative + 1, sizeof(real));
    FILE *fi = fopen(train_file, "rb");
    fseek(fi ,file_size / (long long) num_threads * (long long) id, SEEK_SET);

    while (1) {
        if (word_count - last_word_count > 10000) {
            total_count += local_total_count;
            correct_count += local_correct_count;
            local_total_count = local_correct_count = 0;
            word_count_actual += word_count - last_word_count;
            last_word_count = word_count;
            now = clock();
            printf("Alpha: %f  Progress: %.2f%%  Correct/Total: %lld/%lld  Precision: %.4f%%  Words/thread/sec: %.2fk%c", alpha,
                   word_count_actual / (real)(iter * train_words + 1) * 100,
                   correct_count, total_count, correct_count / (real) (total_count + 1) * 100,
                   word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000), 13);
            fflush(stdout);
            alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
            if (alpha < starting_alpha * 0.0001)
                alpha = starting_alpha * 0.0001;
        }
        if (sentence_length == 0) {
            while (1) {
                word = ReadWordIndex(fi);
                if (feof(fi)) break;
                if (word == -1) continue;
                word_count++;
                if (word == 0) break;
                if (sample > 0) {
                    real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
                    next_random = next_random * (unsigned long long) 25214903917 + 11;
                    if (ran < (next_random & 0xFFFF) / (real)65536) continue;
                }
                sen[sentence_length++] = word;
                if (sentence_length >= MAX_SENTENCE_LENGTH) break;
            }
            sentence_position = 0;
        }
        if (feof(fi) || (word_count > train_words / num_threads)) {
            word_count_actual += word_count - last_word_count;
            local_iter--;
            if (local_iter == 0) break;
            word_count = 0;
            last_word_count = 0;
            sentence_length = 0;
            fseek(fi ,file_size / (long long) num_threads * (long long) id, SEEK_SET);
            continue;
        }
        word = sen[sentence_position];
        if (word == -1) continue;
        for (i = 0; i < dimension; i++) context[i] = 0;
        for (i = 0; i < dimension; i++) context_gradient[i] = 0;

        for (j = 0, d = 0; d < dimension; d++) {
            for (k = 0; k < window2; k++, j++) {
                int p = sentence_position - window + k;
                if (k >= window) p++;
                if (p < 0 || p >= MAX_SENTENCE_LENGTH) continue;
                context[d] += context_word_vec[sen[p] * dimension + d] * weight[j];
            }
            context[d] += bias[d];
        }

        for (n = 0; n < negative + 1; n++) {
            target = word;
            if (n < negative) {
                while (target == word) {
                    next_random = next_random * (unsigned long long) 25214903917 + 11;
                    target = negTable[(next_random >> 16) % table_size];
                }
                negative_samples[n] = target;
            }
            prob[n] = 0;
            for (d = 0; d < dimension; d++) {
                float diff = context[d] - word_vec[target * dimension + d];
                prob[n] -= diff * diff;
            }

            if (n == 0) max_prob = prob[n]; 
            else if (max_prob < prob[n]) max_prob = prob[n]; 

        }
       
        if (max_prob == prob[negative]) local_correct_count++;
        local_total_count++;
 
        sum_prob = 0;
        for (n = 0; n < negative + 1; n++) { 
            prob[n] -= max_prob; 
            if (prob[n] + MAX_GAUSSIAN < 0) prob[n] = 0;
            else prob[n] = gaussian_table[(long long)((-prob[n] / MAX_GAUSSIAN) * Gaussian_Table_Size)];
            sum_prob += prob[n];
        } 

        for (n = 0; n < negative + 1; n++) {
            if (n == negative) {
                target = word;
                label = 1;
            }
            else {
                target = negative_samples[n];
                label = 0;
            }
            prob[n] /= sum_prob;
            direction = (label - prob[n]) * alpha;
            for (d = 0, k = target * dimension; d < dimension; d++, k++) {
                context_gradient[d] += direction * (word_vec[k] - context[d]) - alpha * beta * context[d];
                word_vec[k] += direction * (context[d] - word_vec[k]) - alpha * beta * word_vec[k];
            } 
        }

        for (j = 0, d = 0; d < dimension; d++) {
            for (k = 0; k < window2; k++, j++) {
                int p = sentence_position - window + k;
                if (k >= window) p++;
                if (p < 0 || p >= MAX_SENTENCE_LENGTH) continue;
                float vec_gradient = context_gradient[d] * weight[j];
                weight[j] += context_gradient[d] * context_word_vec[sen[p] * dimension + d];
                context_word_vec[sen[p] * dimension + d] += vec_gradient;
            }
            bias[d] += context_gradient[d];
        }

        sentence_position++;
        if (sentence_position >= sentence_length) {
            sentence_length = 0;
            continue;
        }

    }

    fclose(fi);
    free(context);
    free(context_gradient);
    free(negative_samples);
    free(prob);
    pthread_exit(NULL);

}

void TrainModel() {

    int i, j;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));

    initVocab();
    initNegTable();
    initNet();

    start = clock();
    starting_alpha = alpha;
    correct_count = 0;
    total_count = 0;

    for (i = 0; i < num_threads; i++) pthread_create(&pt[i], NULL, train_t, (void *)i);
    for (i = 0; i < num_threads; i++) pthread_join(pt[i], NULL);

    FILE *fo = fopen(context_word_file, "wb");
    FILE *fe = fopen(word_file, "wb");
    for (i = 0; i < vocab_size; i++) {
        fprintf(fo, "%s ", vocab[i].word);
        fprintf(fe, "%s ", vocab[i].word);
        for (j = 0; j < dimension; j++) {
            fprintf(fo, "%lf ", context_word_vec[i * dimension + j]);
            fprintf(fe, "%lf ", word_vec[i * dimension + j]);
        }
        fprintf(fo, "\n");
        fprintf(fe, "\n");
    }
    fclose(fo);
    fclose(fe);

    FILE *fs = fopen(weight_file, "wb");
    for (i = 0; i < dimension * window2; i++) {
        fprintf(fs, "%f ", weight[i]);
    }
    for (i = 0; i < dimension; i++) {
        fprintf(fs, "%f ", bias[i]);
    }
    fclose(fs);
    free(pt);

}

int main(int argc, char **argv) {

    int i;
    if ((i = ArgPos((char *)"-d", argc, argv)) > 0) dimension = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-beta", argc, argv)) > 0) beta = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-context", argc, argv)) > 0) strcpy(context_word_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-word", argc, argv)) > 0) strcpy(word_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-weight", argc, argv)) > 0) strcpy(weight_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-window", argc, argv)) > 0) {
        window = atoi(argv[i + 1]);
        window2 = 2 * window;
    }
    if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
    vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
    gaussian_table = (real *)malloc((Gaussian_Table_Size + 1) * sizeof(real));
    for (i = 0; i < Gaussian_Table_Size; i++) {
        gaussian_table[i] = exp((i / (real) Gaussian_Table_Size) * MAX_GAUSSIAN);
    }

    TrainModel();
    printf("\r\n");
    return 0;

}
