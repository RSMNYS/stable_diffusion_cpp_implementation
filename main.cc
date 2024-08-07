#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "stable_diffusion.h"

// Function to save the image to a file
void save_image(const std::vector<uint8_t>& image_data, int width, int height, const std::string& filename) {
    std::ofstream rgb_file(filename, std::ios::out | std::ofstream::binary);
    std::copy(image_data.begin(), image_data.end(), std::ostreambuf_iterator<char>(rgb_file));
}

int main(int argc, char* argv[]) {
    // Define the paths for the model files
    std::string tflite_path = "model/stable_diffusion_v15_tflite/fp32_batch";
    std::string model_text_encoder = tflite_path + "/sd_text_encoder_fixed_batch.tflite";
    std::string model_first = tflite_path + "/sd_diffusion_model_v15_first_fixed_batch_fp32.tflite";
    std::string model_second = tflite_path + "/sd_diffusion_model_v15_second_fixed_batch_fp32.tflite";
    std::string model_decoder = tflite_path + "/sd_decoder_fixed_batch.tflite";

    // Create an instance of the StableDiffusion class
    StableDiffusion sd(model_text_encoder, model_first, model_second, model_decoder);

    // Variables to store user input
    std::string prompt;
    int num_steps;

    // Get user input for the prompt
    std::cout << "Enter the prompt for the image generation: ";
    std::getline(std::cin, prompt); // Use getline to capture the entire line including spaces

    // Get user input for the number of diffusion steps
    std::cout << "Enter the number of diffusion steps: ";
    std::cin >> num_steps;

    // Validate the number of steps
    if (num_steps <= 0) {
        std::cerr << "Invalid number of steps. Please enter a positive integer." << std::endl;
        return 1;
    }

    // Seed for the random number generator
    int seed = 0;

    // Generate the image using the provided prompt and number of steps
    std::vector<uint8_t> image_data = sd.generate_image(prompt, num_steps, seed);

    // Define the filename for the generated image
    std::string filename = "generated_image.raw";

    // Save the generated image to a file
    save_image(image_data, 512, 512, filename);

    // Inform the user that the image has been saved
    std::cout << "Image saved to " << filename << std::endl;
    
    return 0;
}