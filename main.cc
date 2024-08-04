#include <iostream>
#include <fstream>
#include "stable_diffusion.h"

void save_image(const std::vector<uint8_t>& image_data, int width, int height, const std::string& filename) {

    std::ofstream rgb_file(filename, std::ios::out | std::ofstream::binary);
    std::copy(image_data.begin(), image_data.end(), std::ostreambuf_iterator<char>(rgb_file));
}

int main(int argc, char* argv[]) {
    std::string tflite_path = "model/stable_diffusion_v15_tflite/fp32_batch";
    std::string model_text_encoder = tflite_path + "/sd_text_encoder_fixed_batch.tflite";
    std::string model_first = tflite_path + "/sd_diffusion_model_v15_first_fixed_batch_fp32.tflite";
    std::string model_second = tflite_path + "/sd_diffusion_model_v15_second_fixed_batch_fp32.tflite";
    std::string model_decoder = tflite_path + "/sd_decoder_fixed_batch.tflite";

    StableDiffusion sd(model_text_encoder, model_first, model_second, model_decoder);

    std::string prompt = "Super cute fluffy cat warrior in armor, photorealistic, 4K, ultra detailed, vray rendering, unreal engine.";
    int num_steps = 20;
    int seed = 0;

    std::vector<uint8_t> image_data = sd.generate_image(prompt, num_steps, seed);

    std::string filename = "generated_image.raw";
    save_image(image_data, 512, 512, filename);

    std::cout << "Image saved to " << filename << std::endl;
    return 0;
}