## Art & ML - Spring 2019

Shawn (Boxiang) Lyu, Chuning Yang, Edgar Xi, TJ (Tianjun) Ma, Zhiyu Bai

### Project 1

With a strong desire to further explore the Paradox panel topics, we want to know how the computer algorithm sees pictures differently and whether it has a similar bias as we humans do. We would also like to apply style transfer to our generated differences for an artistic rendering of the differences, and an out-of-box solution is used to upscale the transformed image.
This project is inspired by the adversarial attacks in machine learning. An adversarial attack consists of subtly modifying an original image in such a way that the changes are almost undetectable to the human eye. It is what caused the biased output in machine learning.

### Code and implementation
Machine interpretation of the difference between two pictures from different classes: we mainly used Foolbox and CleverHans, two libraries focused on adversarial attacks. Fast Signed Gradient Method (FSGM for short) [1] and Carlini and Wagner L2 attack [2] were chosen among the various attacks provided by these methods. The implementation is located in the `code/` directory.

The code for style transfer and DeepDream are largely taken from Keras examples and the tutorial given in class. Sample script for apply style transfer to a base image:
```bash
python neural_style_transfer.py base.png pollock_style-1.jpg [output prefix]
```


### Reference
[1] Goodfellow I J, Shlens J, Szegedy C. Explaining and harnessing adversarial examples[J]. arXiv preprint arXiv:1412.6572, 2014 https://arxiv.org/abs/1412.6572
[2] Carlini, N and Wagner, D. Towards Evaluating the Robustness of Neural Networks. arXiv preprint arXiv:1608.04644, 2016 https://arxiv.org/abs/1608.04644
