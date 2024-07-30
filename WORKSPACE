workspace(name = "stable_diffusion")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "org_tensorflow",
    sha256 = "c030cb1905bff1d2446615992aad8d8d85cbe90c4fb625cee458c63bf466bc8e",
    strip_prefix = "tensorflow-2.12.0",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/v2.12.0.tar.gz",
    ],
)

# Initialize tensorflow workspace.
load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

# Android.
android_sdk_repository(
    name = "androidsdk",
    api_level = 31,
)

android_ndk_repository(
    name = "androidndk",
    api_level = 21,
)
