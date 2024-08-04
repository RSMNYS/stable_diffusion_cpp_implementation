load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_copts")
load("@org_tensorflow//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")
load("//src:build_def.bzl", "android_linkopts")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "libsd",
    srcs = [
        "bpe.cc",
    ],
    hdrs = [
        "bpe.h",
    ],
    deps = [
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_binary(
    name = "image_generator",
    srcs = ["main.cc", "stable_diffusion.h", "stable_diffusion.cc", "scheduling_util.cc", "scheduling_util.h"],
    copts = tflite_copts(),
    linkopts = android_linkopts(),
    deps = [
        ":libsd",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)


