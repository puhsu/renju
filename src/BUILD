load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

tf_cc_binary(
    name = "bernard",
    srcs = ["main.cpp", "mcts.hpp", "util.hpp", "rules.hpp", "policy.hpp", "rollout.hpp"],
    deps = [
         "//tensorflow/core:tensorflow",
    ]
)

