The main issue with your code is that you are using the Hugging Face `pipeline` interface to generate text, but you are not providing enough information to the model to generate coherent text. The `pipeline` interface is a simple, high-level interface that is designed to be easy to use, but it does not provide the same level of control as the lower-level `model` interface.

In particular, the `pipeline` interface does not allow you to specify the input context or the maximum length of the generated text. This means that the model is not able to generate text that is relevant to the input query or that is longer than the default maximum length.

To fix this issue, you should use the lower-level `model` interface to generate text. This interface allows you to specify the input context and the maximum length of the generated text, which should result in more coherent and relevant text.

Here is an example of how you can modify your code to use the lower-level `model` interface:

