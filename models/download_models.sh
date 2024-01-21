git lfs install

mkdir transcoder_python
wget https://dl.fbaipublicfiles.com/transcoder/pre_trained_models/translator_transcoder_size_from_DOBF.pth -O ./transcoder_python/model.pth
mkdir transcoder_cpp
wget https://dl.fbaipublicfiles.com/transcoder/pre_trained_models/TransCoder_model_1.pth -O ./transcoder_cpp/model.pth


git clone https://huggingface.co/uclanlp/plbart-java-cs
git clone https://huggingface.co/uclanlp/plbart-cs-java

git clone https://huggingface.co/uclanlp/plbart-single_task-java_en
git clone https://huggingface.co/uclanlp/plbart-single_task-en_java

git clone https://huggingface.co/Salesforce/codet5-base

git clone https://huggingface.co/Salesforce/codet5-base-codexglue-translate-java-cs
git clone https://huggingface.co/Salesforce/codet5-base-codexglue-translate-cs-java

git clone https://huggingface.co/Salesforce/codet5-base-codexglue-sum-java
git clone https://huggingface.co/Salesforce/codet5-base-codexglue-concode

git clone https://huggingface.co/facebook/incoder-1B
git clone https://huggingface.co/facebook/incoder-6B

git clone https://huggingface.co/bigcode/santacoder

git clone https://huggingface.co/bigcode/starcoderbase
