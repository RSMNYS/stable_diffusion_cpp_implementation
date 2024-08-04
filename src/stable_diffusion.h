#ifndef STABLE_DIFFUSION_H
#define STABLE_DIFFUSION_H

#include <memory>
#include <string>
#include <vector>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"

class StableDiffusion
{
public:
    StableDiffusion(const std::string &model_text_encoder_path,
                    const std::string &model_first_path,
                    const std::string &model_second_path,
                    const std::string &model_decoder_path);

    std::vector<uint8_t> generate_image(const std::string &prompt, int num_steps, int seed);

private:
    std::string model_text_encoder_path;
    std::string model_first_path;
    std::string model_second_path;
    std::string model_decoder_path;

    std::unique_ptr<tflite::FlatBufferModel> text_encoder_model;
    std::unique_ptr<tflite::FlatBufferModel> first_model_model;
    std::unique_ptr<tflite::FlatBufferModel> second_model_model;
    std::unique_ptr<tflite::FlatBufferModel> decoder_model;

    std::unique_ptr<tflite::Interpreter> text_encoder;
    std::unique_ptr<tflite::Interpreter> first_model;
    std::unique_ptr<tflite::Interpreter> second_model;
    std::unique_ptr<tflite::Interpreter> decoder;

    void initialize_text_encoder();
    void initialize_diffusion_models();
    void initialize_decoder();

    std::vector<float> encode_prompt(const std::string &prompt);
    std::vector<float> encode_unconditional();
    std::vector<float> diffusion_process(const std::vector<float> &encoded_text,
                                         const std::vector<float> &unconditional_encoded_text,
                                         int num_steps, int seed);
    std::vector<uint8_t> decode_image(const std::vector<float> &latent);

    std::vector<float> diffusion_step(const std::vector<float> &latent,
                                      const std::vector<float> &t_emb,
                                      const std::vector<float> &context);

    int get_tensor_index_by_input_name(std::unique_ptr<tflite::Interpreter> &interpreter, const std::string &name);
    int get_tensor_index_by_output_name(std::unique_ptr<tflite::Interpreter> &interpreter, const std::string &name);

    void load_model(const std::string &model_path, std::unique_ptr<tflite::FlatBufferModel> &model, std::unique_ptr<tflite::Interpreter> &interpreter);

    std::vector<float> run_inference(std::unique_ptr<tflite::Interpreter> &interpreter,
                                     const std::vector<int> &encoded,
                                     const std::vector<int> &pos_ids);
    
    std::vector<float> run_inference(std::unique_ptr<tflite::Interpreter> &interpreter,
                                                  const std::vector<float> &input);

    std::vector<float> get_timestep_embedding(int timestep, int dim = 320, float max_period = 10000.0);
};

#endif // STABLE_DIFFUSION_H