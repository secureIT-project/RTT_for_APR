python3 -m venv ./venv_rtt
source venv_rtt/bin/activate

pip install -r requirements.txt

git clone https://github.com/glample/fastBPE
cd fastBPE
python setup.py install
cd ..

cd jasper
javac -cp ".:lib/*" -d target src/main/java/clm/jasper/*.java src/main/java/clm/codet5/*.java src/main/java/clm/codegen/*.java src/main/java/clm/plbart/*.java src/main/java/clm/incoder/*.java src/main/java/clm/finetuning/*.java
cd ..


cd clm-apr/transcoder/tree-sitter
chmod +x tree-sitter-downloader.sh
./tree-sitter-downloader.sh
cd ../../..

cd clm-apr/quixbugs/evaluation/CodeBLEU/parser
chmod +x build.sh
./build.sh
cd ../../..
cp -r evaluation ./../defects4j/
cp -r evaluation ./../humaneval/
cd ../..


# We recommend downloading only the models you are going to use due to their size
# cd models
# chmod +x download_models.sh
# ./download_models.sh
# 
# For example, to run PLBART, run the following command:
# cd models
# git lfs install
# git clone https://huggingface.co/uclanlp/plbart-java-cs
# git clone https://huggingface.co/uclanlp/plbart-cs-java

# Follow Readme.md to install defects4j and make sure is added to the $PATH
defects4j env