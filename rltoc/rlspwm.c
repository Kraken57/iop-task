
#include <stdio.h>
#include <math.h>
#include "weights.h"

#define RELU(x) ((x) > 0 ? (x) : 0)

void forward(float input, int output[1000]) {
    float layer1[512];
    float layer2[256];
    float layer3[128];
    float out[1000];

    // Layer 1
    for (int i = 0; i < 512; i++) {
        layer1[i] = w1[i] * input + b1[i];
        layer1[i] = RELU(layer1[i]);
    }

    // Layer 2
    for (int i = 0; i < 256; i++) {
        layer2[i] = 0;
        for (int j = 0; j < 512; j++) {
            layer2[i] += w2[i * 512 + j] * layer1[j];
        }
        layer2[i] += b2[i];
        layer2[i] = RELU(layer2[i]);
    }

    // Layer 3
    for (int i = 0; i < 128; i++) {
        layer3[i] = 0;
        for (int j = 0; j < 256; j++) {
            layer3[i] += w3[i * 256 + j] * layer2[j];
        }
        layer3[i] += b3[i];
        layer3[i] = RELU(layer3[i]);
    }

    // Output Layer
    for (int i = 0; i < 1000; i++) {
        out[i] = 0;
        for (int j = 0; j < 128; j++) {
            out[i] += w_out[i * 128 + j] * layer3[j];
        }
        out[i] += b_out[i];
        output[i] = out[i] > 0 ? 1 : 0;
    }
}

int main() {
    float modulation_index = 0.8f;
    int spwm[1000];

    forward(modulation_index, spwm);

    for (int i = 0; i < 1000; i++) {
        printf("%d,", spwm[i]);
    }

    return 0;
}
