// ssdprobe - expert-streaming read throughput: T threads, random chunk-aligned preads of one size from a big file.
// usage: ssdprobe FILE [chunk_kib=1792] [threads=6] [total_mib=4096] [direct=1]
// Read only.  direct=1 opens with O_DIRECT (page cache bypassed, as the engine's F_NOCACHE preads on macOS).
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static int fd;
static size_t chunk;
static uint64_t nchunks, per_thread;

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

static void* worker(void* arg) {
    uint64_t seed = (uint64_t) (uintptr_t) arg * 0x9e3779b97f4a7c15ull + 1;
    void* buf;
    if (posix_memalign(&buf, 16384, chunk)) return (void*) 1;
    for (uint64_t i = 0; i < per_thread; i++) {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        off_t off = (off_t) ((seed % nchunks) * chunk);
        if (pread(fd, buf, chunk, off) != (ssize_t) chunk) { perror("pread"); return (void*) 1; }
    }
    free(buf);
    return NULL;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s FILE [chunk_kib] [threads] [total_mib] [direct]\n", argv[0]); return 2; }
    chunk = (size_t) (argc > 2 ? atoi(argv[2]) : 1792) << 10;
    int threads = argc > 3 ? atoi(argv[3]) : 6;
    uint64_t total = (uint64_t) (argc > 4 ? atoi(argv[4]) : 4096) << 20;
    int direct = argc > 5 ? atoi(argv[5]) : 1;
    fd = open(argv[1], O_RDONLY | (direct ? O_DIRECT : 0));
    if (fd < 0) { perror("open"); return 1; }
    struct stat st;
    fstat(fd, &st);
    uint64_t size = (uint64_t) st.st_size;
    if (S_ISBLK(st.st_mode) && ioctl(fd, BLKGETSIZE64, &size) != 0) { perror("BLKGETSIZE64"); return 1; }
    nchunks = size / chunk;
    if (!nchunks) { fprintf(stderr, "file smaller than one chunk\n"); return 1; }
    per_thread = total / chunk / (uint64_t) threads;
    pthread_t th[64];
    double t0 = now_s();
    for (int i = 0; i < threads; i++) pthread_create(&th[i], NULL, worker, (void*) (uintptr_t) (i + 1));
    int bad = 0;
    for (int i = 0; i < threads; i++) { void* r; pthread_join(th[i], &r); bad |= r != NULL; }
    double dt = now_s() - t0;
    double bytes = (double) per_thread * threads * chunk;
    printf("chunk %4zu KiB x %2d threads, %s: %6.2f GB/s (%.0f reads/s, %.2f ms per read)\n", chunk >> 10, threads,
           direct ? "O_DIRECT" : "buffered", bytes / dt / 1e9, bytes / chunk / dt, dt * 1000 * threads / (bytes / chunk));
    return bad;
}
