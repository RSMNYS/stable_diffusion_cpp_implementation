#include <getopt.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <valarray>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "constants.h"
#include "bpe.h"

#include "stable_diffusion.h"
#include "scheduling_util.h"

using namespace std;

std::vector<float> get_normal(unsigned numbers, unsigned seed = 5, float mean = 0.0, float stddev = 1.0)
{
    std::default_random_engine generator(seed);
    std::normal_distribution<float> distribution(mean, stddev);

    std::vector<float> d;
    for (unsigned i = 0; i < numbers; i++)
        d.push_back(distribution(generator));

    return d;
}

StableDiffusion::StableDiffusion(const std::string &model_text_encoder,
                                 const std::string &model_first,
                                 const std::string &model_second,
                                 const std::string &model_decoder)
{
    std::cout << "Loading model: text_encoder" << std::endl;
    load_model(model_text_encoder, text_encoder);
    std::cout << "Loading model: first_model" << std::endl;
    load_model(model_first, first_model);
    std::cout << "Loading model: second_model" << std::endl;
    load_model(model_second, second_model);
    std::cout << "Loading model: decoder" << std::endl;
    load_model(model_decoder, decoder);
    std::cout << "Models loaded successfully" << std::endl;
}

// Encode text using the text encoder model
std::vector<float> StableDiffusion::encode_text(const std::vector<int>& encoded, const std::vector<int>& pos_ids)
{
    std::cout << "encode_text" << std::endl;
    return run_inference(text_encoder, encoded, pos_ids);
}

// Decode latent representation using the decoder model
std::vector<float> StableDiffusion::decode(const std::vector<float> &latent)
{
    return run_inference(decoder, latent);
}

std::vector<float> StableDiffusion::diffusion_step(const std::vector<float> &latent,
                                                   const std::vector<float> &t_emb,
                                                   const std::vector<float> &context)
{
    
    std::vector<float> first_output = run_inference(first_model, latent, t_emb, context);
    return run_inference(second_model, first_output);
}

std::vector<uint8_t> StableDiffusion::generate_image(const std::string &prompt, int num_steps, int seed)
{
    bpe bpe_encoder;
    std::cout << "Encoding prompt" << std::endl;
    auto encoded = bpe_encoder.encode(prompt);
    
    std::cout << "Encoded prompt: ";
    for (const auto& token: encoded) {
       std::cout << token << " ";
    }
    std::cout << std::endl;

    std::cout << "Generating position IDs." << std::endl;
    auto pos_ids = bpe_encoder.position_ids();
    std::cout << "Position IDs: ";
    for (const auto& id : pos_ids) {
        std::cout << id << " ";
    }
    std::cout << std::endl;

    // Add log before calling encode_text
    std::cout << "Calling encode_text with encoded and pos_ids." << std::endl;
    auto encoded_text = encode_text(encoded, pos_ids);
    std::cout << "Completed call to encode_text." << std::endl;

    // Add log before calling encode_text for unconditional tokens
    std::cout << "Calling encode_text with unconditioned tokens and pos_ids." << std::endl;
    auto unconditional_text = encode_text(bpe_encoder.unconditioned_tokens(), pos_ids);
    std::cout << "Completed call to encode_text for unconditioned tokens." << std::endl;

    float unconditional_guidance_scale = 7.5;
    auto noise = get_normal(64 * 64 * 4, seed);
    auto latent = noise;
    auto timesteps = get_timesteps(1, 1000, 1000 / num_steps);
    auto alphas_tuple = get_initial_alphas(timesteps);
    auto alphas = std::get<0>(alphas_tuple);
    auto alphas_prev = std::get<1>(alphas_tuple);

    for (int i = timesteps.size() - 1; i >= 0; i--)
    {
        std::cout << "Starting step " << (timesteps.size() - 1 - i) << " with timestep " << timesteps[i] << std::endl;

        auto latent_prev = latent;
        std::cout << "Generated latent_prev for step " << (timesteps.size() - 1 - i) << std::endl;

        auto t_emb = get_timestep_embedding(timesteps[i]);
        std::cout << "Generated t_emb for step " << (timesteps.size() - 1 - i) << std::endl;

        auto unconditional_latent = diffusion_step(latent, t_emb, unconditional_text);
        std::cout << "Generated unconditional_latent for step " << (timesteps.size() - 1 - i) << std::endl;

        latent = diffusion_step(latent, t_emb, encoded_text);
        std::cout << "Updated latent for step " << (timesteps.size() - 1 - i) << std::endl;

        std::valarray<float> l(latent.data(), latent.size());
        std::valarray<float> l_prev(latent_prev.data(), latent_prev.size());
        std::valarray<float> u(unconditional_latent.data(), unconditional_latent.size());
        l = u + unconditional_guidance_scale * (l - u);
        std::cout << "Updated valarray l for step " << (timesteps.size() - 1 - i) << std::endl;

        auto a_t = alphas[i];
        auto a_prev = alphas_prev[i];
        std::cout << "Alpha values for step " << (timesteps.size() - 1 - i) << " - a_t: " << a_t << ", a_prev: " << a_prev << std::endl;

        auto prev_x0 = (l_prev - sqrtf(1.0 - a_t) * l) / sqrtf(a_t);
        l = (l * sqrtf(1.0 - a_prev) + sqrtf(a_prev) * prev_x0);
        latent.assign(std::begin(l), std::end(l));
        std::cout << "Completed step " << (timesteps.size() - 1 - i) << std::endl;
    }

    auto decoded = decode(latent);
    std::valarray<float> d(decoded.data(), decoded.size());
    d = (d + 1) / 2 * 255;
    std::vector<uint8_t> decoded_uint8;
    for (auto e : d)
    {
        if (e > 255.0)
            e = 255;
        if (e < 0.0)
            e = 0;
        decoded_uint8.push_back((uint8_t)e);
    }
    return decoded_uint8;
}

void StableDiffusion::load_model(const std::string &model_path, std::unique_ptr<tflite::Interpreter> &interpreter)
{
    auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    interpreter->AllocateTensors();
}

// Run inference on text encoding model
std::vector<float> StableDiffusion::run_inference(std::unique_ptr<tflite::Interpreter>& interpreter,
                                                  const std::vector<int>& encoded,
                                                  const std::vector<int>& pos_ids)
{
    std::cout << "encode_text inference start" << std::endl;

    auto raw_interpreter = interpreter.get();

    std::cout << "Raw pointer created: " << static_cast<const void*>(raw_interpreter) << std::endl;

    const std::vector<int> inputs = raw_interpreter->inputs();
    std::copy(pos_ids.begin(), pos_ids.end(), raw_interpreter->typed_input_tensor<int>(0));
    std::copy(encoded.begin(), encoded.end(), raw_interpreter->typed_input_tensor<int>(1));

    if (raw_interpreter->Invoke() != kTfLiteOk) {
        std::cout << "Failed to invoke tflite!\n" << std::endl;
        exit(-1);
    }

    const std::vector<int> outputs = raw_interpreter->outputs();
    auto output = raw_interpreter->typed_tensor<float>(outputs[0]);
    return std::vector<float>(output, output + raw_interpreter->tensor(outputs[0])->bytes / 4);
}

// Run inference on decoder model
std::vector<float> StableDiffusion::run_inference(std::unique_ptr<tflite::Interpreter>& interpreter,
                                                  const std::vector<float> &input)
{
    auto inputs = interpreter->inputs();
    std::copy(input.begin(), input.end(), interpreter->typed_input_tensor<float>(0));

    interpreter->Invoke();

    auto outputs = interpreter->outputs();
    auto output = interpreter->typed_tensor<float>(outputs[0]);
    return std::vector<float>(output, output + interpreter->tensor(outputs[0])->bytes / 4);
}

// Run inference on diffusion model
std::vector<float> StableDiffusion::run_inference(std::unique_ptr<tflite::Interpreter>& interpreter,
                                                  const std::vector<float> &latent,
                                                  const std::vector<float> &t_emb,
                                                  const std::vector<float> &context)
{

    auto interpreter_ptr = interpreter.get();

    std::cout << "Raw pointer created: " << static_cast<const void*>(interpreter_ptr) << std::endl;

    auto inputs = interpreter_ptr->inputs();
    std::copy(latent.begin(), latent.end(), interpreter_ptr->typed_input_tensor<float>(2));
    std::copy(t_emb.begin(), t_emb.end(), interpreter_ptr->typed_input_tensor<float>(1));
    std::copy(context.begin(), context.end(), interpreter_ptr->typed_input_tensor<float>(0));

    interpreter_ptr->Invoke();

    auto outputs = interpreter_ptr->outputs();
    auto output = interpreter_ptr->typed_tensor<float>(outputs[0]);
    return std::vector<float>(output, output + interpreter_ptr->tensor(outputs[0])->bytes / 4);
}

std::vector<float> StableDiffusion::get_timestep_embedding(int timestep, int dim, float max_period)
{
    int half = dim / 2;
    std::vector<float> freqs(half);
    for (int i = 0; i < half; ++i)
    {
        freqs[i] = std::exp(-std::log(max_period) * i / half);
    }

    std::vector<float> args(half);
    for (int i = 0; i < half; ++i)
    {
        args[i] = timestep * freqs[i];
    }

    std::vector<float> embedding(2 * half);
    for (int i = 0; i < half; ++i)
    {
        embedding[i] = std::cos(args[i]);
        embedding[half + i] = std::sin(args[i]);
    }

    return embedding;
}
