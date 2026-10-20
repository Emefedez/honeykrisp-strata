SHADERS := $(patsubst shaders/%.comp,build/%.spv,$(wildcard shaders/*.comp))

all: build/vkprobe $(SHADERS)

build/vkprobe: vkprobe.cpp | build
	$(CXX) -O2 -std=c++20 -Wall -o $@ $< -lvulkan

build/%.spv: shaders/%.comp | build
	glslc --target-env=vulkan1.3 -O -o $@ $<

build:
	mkdir -p build

clean:
	rm -rf build

.PHONY: all clean
