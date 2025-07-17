# AlexNet

## 핵심 아이디어

## 방법론

### DataSet
ImageNet is a dataset of over 15M labeled high resolution images with 22,000 categories.

This paper down-sampled the images to a fixed resolution of 256 x 256.

## 결론
On ILSVRC-2010, this paper achieves Top-1 test set error rates of 37.5% and Top-5 test set error rates of 17.0%.

On ILSVRC-2012, this paper achieves Top-5 error rate of 18.2%.

Averaging predictions of two CNN that pre-trained on entire Fall 2011 and five CNN gives an error rate of 15.3%.

On Fall 2009, this paper achieves Top1 error rates of 67.4% and Top-5 error rates of 40.9%.

Two images that have feature activation vectors with a small Euclidean separation are similar. And then Neural Network consider this two images to be similar.

This Euclidean distance between two 4096-dimenstional real-value vectors are inefficient, but short binary codes which compressed these vectors are efficient.

Paper's network demonstrates that depth is important for achieving high performance.
