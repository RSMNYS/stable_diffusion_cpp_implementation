#ifndef STABLE_DIFFUSION_H
#define STABLE_DIFFUSION_H

#include <memory>
#include <string>
#include <vector>
#include "tensorflow/lite/interpreter.h"

class StableDiffusion {
public:
    StableDiffusion(const std::string& model_text_encoder,
                    const std::string& model_first,
                    const std::string& model_second,
                    const std::string& model_decoder);

    std::vector<float> encode_text(const std::vector<int>& encoded, const std::vector<int>& pos_ids);
    std::vector<float> decode(const std::vector<float>& latent);
    std::vector<float> diffusion_step(const std::vector<float>& latent,
                                      const std::vector<float>& t_emb,
                                      const std::vector<float>& context);
    std::vector<uint8_t> generate_image(const std::string& prompt, int num_steps, int seed);

private:
    std::unique_ptr<tflite::Interpreter> text_encoder;
    std::unique_ptr<tflite::Interpreter> first_model;
    std::unique_ptr<tflite::Interpreter> second_model;
    std::unique_ptr<tflite::Interpreter> decoder;

    void load_model(const std::string& model_path, std::unique_ptr<tflite::Interpreter>& interpreter);

    std::vector<float> run_inference(std::unique_ptr<tflite::Interpreter>& interpreter,
                                     const std::vector<int>& encoded,
                                     const std::vector<int>& pos_ids);

    std::vector<float> run_inference(std::unique_ptr<tflite::Interpreter>& interpreter,
                                     const std::vector<float>& input);

    std::vector<float> run_inference(std::unique_ptr<tflite::Interpreter>& interpreter,
                                     const std::vector<float>& latent,
                                     const std::vector<float>& t_emb,
                                     const std::vector<float>& context);

    std::vector<float> get_timestep_embedding(int timestep, int dim = 320, float max_period = 10000.0);
};

#endif // STABLE_DIFFUSION_H