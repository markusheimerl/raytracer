CC = clang
CFLAGS = -O3 -march=native -Wall -Wextra -I. -I..
LDFLAGS = -lm -lwebp -lwebpmux -lpthread -flto

ARCH ?= sm_87
CUDAFLAGS = --cuda-gpu-arch=$(ARCH) -x cuda -Wno-unknown-cuda-version
CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas

OBJS = raytracer.o scene.o \
       math/mat4.o math/ray.o math/vec3.o \
       geometry/aabb.o geometry/mesh.o \
       accel/bvh.o \
       render/camera.o render/light.o \
       utils/image.o utils/progress.o

raytracer.out: $(OBJS)
	$(CC) $(OBJS) $(CUDALIBS) $(LDFLAGS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c $< -o $@

run: raytracer.out
	@time ./raytracer.out

clean:
	rm -f *.out *.o */*.o *.webp