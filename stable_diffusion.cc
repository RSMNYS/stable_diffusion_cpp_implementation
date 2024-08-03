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

StableDiffusion::StableDiffusion(const std::string &model_text_encoder_path,
                                 const std::string &model_first_path,
                                 const std::string &model_second_path,
                                 const std::string &model_decoder_path)
    : model_text_encoder_path(model_text_encoder_path),
      model_first_path(model_first_path),
      model_second_path(model_second_path),
      model_decoder_path(model_decoder_path) {}

void StableDiffusion::initialize_text_encoder()
{
    if (!text_encoder) {
        std::cout << "Loading text encoder model..." << std::endl;
        load_model(model_text_encoder_path, text_encoder_model, text_encoder);
    }
}

void StableDiffusion::initialize_diffusion_models()
{
    if (!first_model || !second_model) {
        std::cout << "Loading diffusion models..." << std::endl;
        load_model(model_first_path, first_model_model, first_model);
        load_model(model_second_path, second_model_model, second_model);
    }
}

void StableDiffusion::initialize_decoder()
{
    if (!decoder) {
        std::cout << "Loading decoder model..." << std::endl;
        load_model(model_decoder_path, decoder_model, decoder);
    }
}

std::vector<float> StableDiffusion::encode_prompt(const std::string &prompt)
{
    initialize_text_encoder();

    bpe bpe_encoder;
    auto encoded = bpe_encoder.encode(prompt);
    auto pos_ids = bpe_encoder.position_ids();

    return run_inference(text_encoder, encoded, pos_ids);
}

std::vector<float> StableDiffusion::encode_unconditional()
{
    initialize_text_encoder();

    bpe bpe_encoder;
    auto unconditioned_tokens = bpe_encoder.unconditioned_tokens();
    auto pos_ids = bpe_encoder.position_ids();

    return run_inference(text_encoder, unconditioned_tokens, pos_ids);
}

std::vector<float> StableDiffusion::diffusion_step(const std::vector<float> &latent,
                                                   const std::vector<float> &t_emb,
                                                   const std::vector<float> &context)
{
    
    std::vector<float> first_output = run_inference(first_model, latent, t_emb, context);
    return run_inference(second_model, first_output);
}

std::vector<float> StableDiffusion::diffusion_process(const std::vector<float> &encoded_text, const std::vector<float> &unconditional_encoded_text, int num_steps, int seed)
{
    initialize_diffusion_models();

    float unconditional_guidance_scale = 7.5;
    auto noise = get_normal(64 * 64 * 4, seed);
    auto latent = noise;

    auto timesteps = get_timesteps(1, 1000, 1000 / num_steps);
    auto alphas_tuple = get_initial_alphas(timesteps);
    auto alphas = std::get<0>(alphas_tuple);
    auto alphas_prev = std::get<1>(alphas_tuple);

    for (int i = timesteps.size() - 1; i >= 0; i--)
    {
        std::cout << "Step " << timesteps.size() - 1 - i << "\n";

        auto latent_prev = latent;
        auto t_emb = get_timestep_embedding(timesteps[i]);

        auto unconditional_latent = diffusion_step(latent, t_emb, unconditional_encoded_text);
        latent = diffusion_step(latent, t_emb, encoded_text);

        std::valarray<float> l(latent.data(), latent.size());
        std::valarray<float> l_prev(latent_prev.data(), latent_prev.size());
        std::valarray<float> u(unconditional_latent.data(), unconditional_latent.size());

        l = u + unconditional_guidance_scale * (l - u);

        auto a_t = alphas[i];
        auto a_prev = alphas_prev[i];

        auto prev_x0 = (l_prev - sqrtf(1.0 - a_t) * l) / sqrtf(a_t);
        l = (l * sqrtf(1.0 - a_prev) + sqrtf(a_prev) * prev_x0);
        latent.assign(std::begin(l), std::end(l));
    }

    return latent;
}

std::vector<uint8_t> StableDiffusion::decode_image(const std::vector<float> &latent)
{
    initialize_decoder();

    auto decoded = run_inference(decoder, latent);

    std::valarray<float> d(decoded.data(), decoded.size());
    d = (d + 1) / 2 * 255;
    std::vector<uint8_t> decoded_uint8;
    for (auto e : d)
    {
        if (e > 255.0)
            e = 255;
        if (e < 0.0)
            e = 0;
        decoded_uint8.push_back(static_cast<uint8_t>(e));
    }

    return decoded_uint8;
}

std::vector<uint8_t> StableDiffusion::generate_image(const std::string &prompt, int num_steps, int seed)
{
    std::cout << "Prompt encoding started" << std::endl;
    auto encoded_text = encode_prompt(prompt);
    auto unconditional_encoded_text = encode_unconditional();
    std::cout << "Diffusion process started" << std::endl;
    auto latent = diffusion_process(encoded_text, unconditional_encoded_text, num_steps, seed);
    std::cout << "Image decoding started" << std::endl;
    return decode_image(latent);
}

void StableDiffusion::load_model(const std::string &model_path, 
                                 std::unique_ptr<tflite::FlatBufferModel> &model, 
                                 std::unique_ptr<tflite::Interpreter> &interpreter)
{
    model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model) {
        std::cerr << "Failed to load model from: " << model_path << std::endl;
        exit(-1);
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    builder(&interpreter);

    if (!interpreter || interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to create interpreter for model: " << model_path << std::endl;
        exit(-1);
    }
}

std::vector<float> StableDiffusion::run_inference(std::unique_ptr<tflite::Interpreter> &interpreter,
                                                  const std::vector<int> &encoded,
                                                  const std::vector<int> &pos_ids)
{
    
    const std::vector<int> inputs = interpreter->inputs();
    std::copy(pos_ids.begin(), pos_ids.end(), interpreter->typed_input_tensor<int>(0));
    std::copy(encoded.begin(), encoded.end(), interpreter->typed_input_tensor<int>(1));

    if (interpreter->Invoke() != kTfLiteOk)
    {
        std::cerr << "Failed to invoke tflite!\n"
                  << std::endl;
        exit(-1);
    }

    const std::vector<int> outputs = interpreter->outputs();
    auto output = interpreter->typed_tensor<float>(outputs[0]);
    return std::vector<float>(output, output + interpreter->tensor(outputs[0])->bytes / 4);
}

std::vector<float> StableDiffusion::run_inference(std::unique_ptr<tflite::Interpreter> &interpreter,
                                                  const std::vector<float> &input)
{
    const std::vector<int> inputs = interpreter->inputs();
    std::copy(input.begin(), input.end(), interpreter->typed_input_tensor<float>(0));

    if (interpreter->Invoke() != kTfLiteOk)
    {
        std::cerr << "Failed to invoke tflite!\n"
                  << std::endl;
        exit(-1);
    }

    const std::vector<int> outputs = interpreter->outputs();
    auto output = interpreter->typed_tensor<float>(outputs[0]);
    return std::vector<float>(output, output + interpreter->tensor(outputs[0])->bytes / 4);
}

std::vector<float> StableDiffusion::run_inference(std::unique_ptr<tflite::Interpreter> &interpreter,
                                                  const std::vector<float> &latent,
                                                  const std::vector<float> &t_emb,
                                                  const std::vector<float> &context)
{
    
    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cerr << "Failed to allocate tensors!" << std::endl;
        exit(-1);
    }
    auto inputs = interpreter->inputs();
    std::copy(latent.begin(), latent.end(), interpreter->typed_input_tensor<float>(0));
    std::copy(t_emb.begin(), t_emb.end(), interpreter->typed_input_tensor<float>(1));
    std::copy(context.begin(), context.end(), interpreter->typed_input_tensor<float>(2));

    if (interpreter->Invoke() != kTfLiteOk)
    {
        std::cerr << "Failed to invoke tflite!\n"
                  << std::endl;
        exit(-1);
    }

    auto outputs = interpreter->outputs();
    auto output = interpreter->typed_tensor<float>(outputs[0]);
    return std::vector<float>(output, output + interpreter->tensor(outputs[0])->bytes / 4);
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
