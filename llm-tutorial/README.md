## Activations

|activation|description|formular|Derivative|
|---|---|---|---|
|relu|Rectified Linear Unit|$f(x) = \max(0, x)$|$f'(x) = max(0, 1)$|
|sigmoid|sigmoid|$\sigma(x) = \frac{1}{1 + e^{-x}}$|$\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$|
|tanh|Hyperbolic Tangent|$\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$|$\tanh(x) = 1 - \tanh^2(x)$|
|softmax|softmax|$\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} \quad \text{for } i = 1, \ldots, K$||
|leaky_relu|leaky_relu|$f(x) = \max(\alpha x, x)$|$f'(x) = \max(\alpha , 1)$|
|elu|elu|$f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \leq 0 \end{cases}$|$\text{ELU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \alpha e^x & \text{if } x \leq 0 \end{cases}$|
|swish|swish|$f(x) = x \cdot \sigma(\beta x), \sigma = sigmoid$|$f'(x) = \sigma(\beta x) + \beta x \cdot \sigma(\beta x) \left(1 - \sigma(\beta x)\right)$|
|gelu|gelu|$\text{GELU}(x) \approx 0.5x \left(1 + \tanh\left[\sqrt{\frac{2}{\pi}} \left(x + 0.044715x^3\right)\right]\right)$|$\text{GELU}'(x) = 0.5\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right]\right) + 0.5x \cdot \sqrt{\frac{2}{\pi}}(1 + 0.134145x^2) \cdot \left(1 - \tanh^2\left[\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right]\right)$|







## LLM Papers

### DeepSeek

[DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437)

[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/pdf/2501.12948)

[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/pdf/2402.03300)


[DeepSeek-VL: Towards Real-World Vision-Language Understanding](https://arxiv.org/pdf/2403.05525)

[DeepSeek LLM: Scaling Open-Source Language Models with Longtermism](https://arxiv.org/pdf/2401.02954)


[DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence](https://arxiv.org/pdf/2401.14196)

[DeepSeek-OCR: Contexts Optical Compression](https://arxiv.org/pdf/2510.18234)

### GPT 

[gpt-oss-120b & gpt-oss-20b Model Card](https://arxiv.org/pdf/2508.10925)

[GPT2:Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

[InstructGPT: Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155)

[whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/pdf/2212.04356)


### Llama

[LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/pdf/2302.13971)
[Llama 2: Open Foundation and Fine-Tuned Chat Models](https://scontent-hkg1-1.xx.fbcdn.net/v/t39.2365-6/10000000_662098952474184_2584067087619170692_n.pdf?_nc_cat=105&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=gM7vDM7WpqkQ7kNvwGJQrSb&_nc_oc=Adlxu_MwUy7ItLy_UPSHOeB_ezd4rKE0tWvutHkLOALlW0PC7MGJpbyfy_NP-m8hfRl5B-auodszFE0Xt-XWK4Hy&_nc_zt=14&_nc_ht=scontent-hkg1-1.xx&_nc_gid=KRFJcoh7Io2BFWiikAb9Qw&oh=00_AfgPrdRkvNaAsQoss_5MpmvVXbwAOp1v8ycGChelEuZGRQ&oe=6918DAFF)



