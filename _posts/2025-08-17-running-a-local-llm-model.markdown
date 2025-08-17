---
layout: post
title:  "大模型动手系列：在本地运行大模型"
date:   2025-08-17 14:57:00 +0800
categories: AI LLM llama.cpp
---
纸上得来总觉浅，绝知此事要躬行，let's DO IT YOURSELF.

没有GPU，还要往下看吗？没问题，在本地运行一个大模型进行推理，现在是再简单不过了，这并不比运行一个java程序复杂多少。不需要GPU，用PC、普通服务器甚至手机都轻松跑大模型。

# 黑箱子：Ollama

在本地运行大模型，最简单的方式是用[Ollma](https://ollama.com)：


```bash
# 安装ollama docker镜像
> docker pull ollama/ollama

# 运行ollama server, 11434是对外提供api的端口
> docker run -itd -v ollama:/root/.ollama -p 11434:11434 --name ollama-server-docker ollama/ollama

# 观察ollama server运行
> docker ps
CONTAINER ID   IMAGE           COMMAND               CREATED      STATUS              PORTS                                           NAMES
355e1711d171   ollama/ollama   "/bin/ollama serve"   2 days ago   Up About a minute   0.0.0.0:11434->11434/tcp, :::11434->11434/tcp   ollama-server-docker
```

ollama-server是大模型部署的环境。接下来我们要用ollama命令行下载和运行模型。去哪里找模型呢？从Ollama官网的[Models](https://ollama.com/search)可以搜索各种开源的模型，我们选择deepseek-r1，可以看到在这个模型库下面的各个模型：

![deepseek-r1 models on https://ollama.com/library/deepseek-r1](/assets/2025-08-17/ollama-models-deepseek-r1.png "deepseeks model list")

让我们在本地部署一个小模型 deepseek-r1:1.5b

```bash
# 下载模型: ollama pull
> docker exec ollama-server-docker ollama pull deepseek-r1:1.5b
pulling manifest
pulling aabd4debf0c8: 100% ▕██████████████████▏ 1.1 GB
pulling c5ad996bda6e: 100% ▕██████████████████▏  556 B
pulling 6e4c38e1172f: 100% ▕██████████████████▏ 1.1 KB
pulling f4d24e9138dd: 100% ▕██████████████████▏  148 B
pulling a85fe2a2e58e: 100% ▕██████████████████▏  487 B
verifying sha256 digest
writing manifest
success

# 查看本地已下载的模型: ollama list
> docker exec ollama-server-docker ollama list
NAME                ID              SIZE      MODIFIED
deepseek-r1:1.5b    e0979632db5a    1.1 GB    4 hours ago

# 运行模型：ollama run
> docker exec ollama-server-docker ollama run deepseek-r1:1.5b

# 查看运行中的模型： ollama ps
> docker exec ollama-server-docker ollama ps
NAME                ID              SIZE      PROCESSOR    CONTEXT    UNTIL
deepseek-r1:1.5b    e0979632db5a    1.4 GB    100% CPU     4096       3 minutes from now
```


好了，deepseek-r1 1.5B跑起来了！现在让我们来使用这个模型，刚才我们在运行ollama server的时候指定api服务监听在11434端口，让我们用大模型来问个问题：is beijing a big city?

```bash
# 准备一个prompt
> cat query.json
{
    "model": "deepseek-r1:1.5b",
    "messages": [
        {
            "role": "user",
            "content": "is beijing a big city?"
        }
    ]
}

# 调用
curl -H "Content-Type: application-json" -d@./query.json http://localhost:11434/api/chat
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:32.611844467Z","message":{"role":"assistant","content":")\n\n"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:32.665931065Z","message":{"role":"assistant","content":"\u003c/think\u003e"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:32.727328824Z","message":{"role":"assistant","content":"\n\n"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:32.781767596Z","message":{"role":"assistant","content":"It"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:32.836942827Z","message":{"role":"assistant","content":" seems"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:32.891700896Z","message":{"role":"assistant","content":" like"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:32.946875426Z","message":{"role":"assistant","content":" you"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:33.005793781Z","message":{"role":"assistant","content":" might"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:33.060459595Z","message":{"role":"assistant","content":" be"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:33.115475679Z","message":{"role":"assistant","content":" asking"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:33.170179628Z","message":{"role":"assistant","content":" about"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:33.224486622Z","message":{"role":"assistant","content":" something"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:33.278913779Z","message":{"role":"assistant","content":","},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:33.333591467Z","message":{"role":"assistant","content":" but"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:33.38838538Z","message":{"role":"assistant","content":" your"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:33.443288333Z","message":{"role":"assistant","content":" question"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:33.498233589Z","message":{"role":"assistant","content":" isn"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:33.553611805Z","message":{"role":"assistant","content":"'t"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:33.608975795Z","message":{"role":"assistant","content":" fully"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:33.663724648Z","message":{"role":"assistant","content":" clear"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:33.718503233Z","message":{"role":"assistant","content":"."},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:33.773407342Z","message":{"role":"assistant","content":" Could"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:33.828404789Z","message":{"role":"assistant","content":" you"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:33.886930774Z","message":{"role":"assistant","content":" clarify"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:33.942498013Z","message":{"role":"assistant","content":" or"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:33.997716592Z","message":{"role":"assistant","content":" provide"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:34.052915777Z","message":{"role":"assistant","content":" more"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:34.108782817Z","message":{"role":"assistant","content":" details"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:34.163110822Z","message":{"role":"assistant","content":"?"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:34.217290039Z","message":{"role":"assistant","content":" Are"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:34.270806946Z","message":{"role":"assistant","content":" you"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:34.328451417Z","message":{"role":"assistant","content":" referring"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:34.382617907Z","message":{"role":"assistant","content":" to"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:34.437352875Z","message":{"role":"assistant","content":" a"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:34.491675969Z","message":{"role":"assistant","content":" specific"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:34.545505501Z","message":{"role":"assistant","content":" topic"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:34.603985341Z","message":{"role":"assistant","content":","},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:34.65801696Z","message":{"role":"assistant","content":" event"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:34.712364239Z","message":{"role":"assistant","content":","},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:34.767173784Z","message":{"role":"assistant","content":" or"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:34.821121942Z","message":{"role":"assistant","content":" situation"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:34.874991417Z","message":{"role":"assistant","content":" that"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:34.929022071Z","message":{"role":"assistant","content":" you"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:34.983836525Z","message":{"role":"assistant","content":"'d"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:35.038128057Z","message":{"role":"assistant","content":" like"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:35.092030257Z","message":{"role":"assistant","content":" help"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:35.146173124Z","message":{"role":"assistant","content":" with"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:35.200008509Z","message":{"role":"assistant","content":"?"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:35.254209271Z","message":{"role":"assistant","content":" For"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:35.308614347Z","message":{"role":"assistant","content":" example"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:35.363130214Z","message":{"role":"assistant","content":":\n\n"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:35.417722311Z","message":{"role":"assistant","content":"-"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:35.473350504Z","message":{"role":"assistant","content":" Are"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:35.528404738Z","message":{"role":"assistant","content":" you"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:35.583325183Z","message":{"role":"assistant","content":" asking"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:35.638589984Z","message":{"role":"assistant","content":" about"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:35.69364683Z","message":{"role":"assistant","content":" ["},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:35.748297632Z","message":{"role":"assistant","content":"specific"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:35.803251089Z","message":{"role":"assistant","content":" topic"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:35.857703282Z","message":{"role":"assistant","content":"]"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:35.912020401Z","message":{"role":"assistant","content":"?\n"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:35.968765933Z","message":{"role":"assistant","content":"-"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:36.026688758Z","message":{"role":"assistant","content":" Do"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:36.08286115Z","message":{"role":"assistant","content":" you"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:36.139617885Z","message":{"role":"assistant","content":" need"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:36.198146209Z","message":{"role":"assistant","content":" information"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:36.254119778Z","message":{"role":"assistant","content":" on"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:36.309970301Z","message":{"role":"assistant","content":" ["},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:36.365631147Z","message":{"role":"assistant","content":"event"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:36.421924057Z","message":{"role":"assistant","content":"/event"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:36.478106482Z","message":{"role":"assistant","content":"].\n"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:36.533800064Z","message":{"role":"assistant","content":"-"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:36.589263124Z","message":{"role":"assistant","content":" Or"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:36.644803272Z","message":{"role":"assistant","content":" is"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:36.700419614Z","message":{"role":"assistant","content":" there"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:36.756647013Z","message":{"role":"assistant","content":" something"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:36.816462262Z","message":{"role":"assistant","content":" particular"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:36.872915106Z","message":{"role":"assistant","content":" about"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:36.928870587Z","message":{"role":"assistant","content":" ["},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:36.988391308Z","message":{"role":"assistant","content":"subject"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:37.044554537Z","message":{"role":"assistant","content":"]"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:37.100257716Z","message":{"role":"assistant","content":" that"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:37.155711386Z","message":{"role":"assistant","content":" needs"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:37.211077537Z","message":{"role":"assistant","content":" explaining"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:37.2672704Z","message":{"role":"assistant","content":"?\n\n"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:37.32314017Z","message":{"role":"assistant","content":"Let"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:37.378762828Z","message":{"role":"assistant","content":" me"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:37.434389558Z","message":{"role":"assistant","content":" know"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:37.490460739Z","message":{"role":"assistant","content":" so"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:37.546824154Z","message":{"role":"assistant","content":" I"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:37.602264756Z","message":{"role":"assistant","content":" can"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:37.657471285Z","message":{"role":"assistant","content":" assist"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:37.713118329Z","message":{"role":"assistant","content":" you"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:37.768831549Z","message":{"role":"assistant","content":" better"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:37.82431745Z","message":{"role":"assistant","content":"!"},"done":false}
{"model":"deepseek-r1:1.5b","created_at":"2025-08-17T08:48:37.879806188Z","message":{"role":"assistant","content":""},"done_reason":"stop","done":true,"total_duration":7343401592,"load_duration":1970484943,"prompt_eval_count":3,"prompt_eval_duration":100862266,"eval_count":96,"eval_duration":5270895498}
```
对于北京是不是大城市这个问题，这个模型不是很自信。不过，用curl来访问这个模型实在是原始了，你可以用你喜欢的的ai客户端（比如chatbox.ai），或者sdk来访问它。

# 白盒子：llama.cpp从代码到运行

ollama是使用llama.cpp作为大模型的推理引擎的，刚才的方式确实让我们很快在本地运行一个大模型。进一步地，我们尝试从代码编译开始，搭建一套运行的推理服务，源码面前，了无秘密！

```bash
# 从github上克隆llama.cpp的代码
> git clone https://github.com/ggml-org/llama.cpp.git

# 编译
> mkdir llama.cpp-build 
> cd llama.cpp-build
> cmake ../llama.cpp
> make  # 等，等等等等。。。。遇到编译问题就解问题吧，多数是缺少依赖的包

# 编译完成
> ls ./bin
libggml-base.so                llama-cvector-generator  llama-gguf-split     llama-mtmd-cli         llama-server              test-backend-ops          test-json-partial            test-regex-partial
libggml-cpu.so                 llama-diffusion-cli      llama-gritlm         llama-parallel         llama-simple              test-barrier              test-json-schema-to-grammar  test-rope
libggml.so                     llama-embedding          llama-imatrix        llama-passkey          llama-simple-chat         test-c                    test-llama-grammar           test-sampling
libllama.so                    llama-eval-callback      llama-llava-cli      llama-perplexity       llama-speculative         test-chat                 test-log                     test-thread-safety
libmtmd.so                     llama-export-lora        llama-lookahead      llama-q8dot            llama-speculative-simple  test-chat-parser          test-model-load-cancel       test-tokenizer-0
llama-batched                  llama-finetune           llama-lookup         llama-quantize         llama-tokenize            test-chat-template        test-mtmd-c-api              test-tokenizer-1-bpe
llama-batched-bench            llama-gemma3-cli         llama-lookup-create  llama-qwen2vl-cli      llama-tts                 test-gbnf-validator       test-opt                     test-tokenizer-1-spm
llama-bench                    llama-gen-docs           llama-lookup-merge   llama-retrieval        llama-vdot                test-gguf                 test-quantize-fns
llama-cli                      llama-gguf               llama-lookup-stats   llama-run              test-arg-parser           test-grammar-integration  test-quantize-perf
llama-convert-llama2c-to-ggml  llama-gguf-hash          llama-minicpmv-cli   llama-save-load-state  test-autorelease          test-grammar-parser       test-quantize-stats
```

接下来我们要下载一个模型，就跟刚才用ollama pull下载模型一个概念。只不过刚才就只是敲了个命令，模型是什么，长什么样，载体是什么，有多大，我们是一点感觉都没有。这次我们都会接触到这些概念。访问huggingface的的[模型广场](http://huggingface.co/models)，并找到deepseek-ai/DeekSeek-R1-Distill-Qwen-7B这个模型：

![huggingface model deepseek-ai/DeekSeek-R1-Distill-Qwen-7B](/assets/2025-08-17/huggingface-deepseekr1-distill-qwen-7b.png "deepseek-ai/DeekSeek-R1-Distill-Qwen-7B")

我们下载模型：
```bash

# 安装huggingface-cli
> pip install -U "huggingface_hub[cli]"

# 下载模型
> hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
Fetching 11 files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:00<00:00, 17931.34it/s]
/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60

# 查看模型，模型主要是由四种文件构成的：
#   config.json: 模型的基本配置，超参数
#   model*.safetensors: 模型的参数，transformer中各种矩阵的权重，偏置；token embedding。
#   tokenizer_config.json/tokernizer.json：分词器设置
> ls -al --color=auto /root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60
total 20
drwxr-xr-x 3 root root 4096 Aug 14 18:52 .
drwxr-xr-x 3 root root 4096 Aug 14 10:26 ..
lrwxrwxrwx 1 root root   52 Aug 14 10:26 config.json                       -> ../../blobs/f9f95f99ff535f5cc8c3b97754a695e5d44690c3
drwxr-xr-x 2 root root 4096 Aug 14 10:26 figures
lrwxrwxrwx 1 root root   52 Aug 14 10:26 generation_config.json            -> ../../blobs/052ab54633116a634da950ab483233c4ace0aa82
lrwxrwxrwx 1 root root   52 Aug 14 10:26 .gitattributes                    -> ../../blobs/a6344aac8c09253b3b630fb776ae94478aa0275b
lrwxrwxrwx 1 root root   52 Aug 14 10:26 LICENSE                           -> ../../blobs/d42fae903e9fa07f3e8edb0db00a8d905ba49560
lrwxrwxrwx 1 root root   76 Aug 14 11:54 model-00001-of-000002.safetensors -> ../../blobs/8b27be8d43b81d0cd19b3113ef88caf1a61c2fc602de77e3cce97aedcec184b4
lrwxrwxrwx 1 root root   76 Aug 14 12:04 model-00002-of-000002.safetensors -> ../../blobs/1780a9502f56f592cb4d6262c8c76ca8ad36a86b0257e052c672d87f49707e80
lrwxrwxrwx 1 root root   52 Aug 14 10:26 model.safetensors.index.json      -> ../../blobs/d66b77d50ede97a02d92e1ba7f9e1ca2673d270f
lrwxrwxrwx 1 root root   52 Aug 14 10:26 README.md                         -> ../../blobs/1ae2b2ccda9cb58fb4179e30c1798b6e75980618
lrwxrwxrwx 1 root root   52 Aug 14 10:26 tokenizer_config.json             -> ../../blobs/9967ff32d94b21c94dc7e2b3bcbea295a46cde50
lrwxrwxrwx 1 root root   52 Aug 14 10:26 tokenizer.json                    -> ../../blobs/a34650995da6939a945c330eadb0687147ac3ef8
```

这个模型的保存格式是 huggingface 原生的，为了在 llama.cpp 中运行，我们需要把它转成需要的gguf格式（huggingface上有些模型也提供gguf格式）, gguf就是把模型所有的参数都打包在一起，并且支持量化进行压缩和优化：
```bash

# 进入llama.cpp代码目录
> cd llama.cpp

# 修改下面这个文件，在models这个数组中增加一个：{"name": "deepseek-r1-qwen", "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", },
> vi convert_hf_to_gguf_update.py
# 执行
> python ./convert_hf_to_gguf_update.py

# 把 huggingface 格式的模型转换为 gguf 格式
> python ./convert_hf_to_gguf.py --outfile ~/DeepSeek-R1-Distill-Qwen-7B.gguf /root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60
```

现在我们就可以运行这个模型了：
```bash

# 使用 llama-server 运行模型：
> cd llama.cpp-build
> ./bin/llama-server -m ~/DeepSeek-R1-Distill-Qwen-7B.gguf 
main: binding port with default address family
main: HTTP server is listening, hostname: 127.0.0.1, port: 8080, http threads: 7
main: loading model
srv    load_model: loading model '/root/DeepSeek-R1-Distill-Qwen-7B.gguf'
..... skip lots of output
..... skip lots of output
..... skip lots of output
main: server is listening on http://127.0.0.1:8080 - starting the main loop
srv  update_slots: all slots are idle
```

现在我们就可以跟刚才一样访问这个大模型了：
```bash
# 调用
> curl -H "Content-Type: application-json" -d@./query.json http://localhost:8080/api/chat
{"choices":[{"finish_reason":"stop","index":0,"message":{"role":"assistant","content":"<think>\n\n</think>\n\nYes, Beijing is considered a very big city. It is the capital city of China and one of the largest and most populous cities in the world."}}],"created":1755431514,"model":"deepseek-r1:1.5b","system_fingerprint":"b6176-5edf1592","object":"chat.completion","usage":{"completion_tokens":35,"prompt_tokens":10,"total_tokens":45},"id":"chatcmpl-mPOtqEZL1FseYIaIqv2xdvdL8lIzuIfn","timings":{"prompt_n":10,"prompt_ms":10271.016,"prompt_per_token_ms":1027.1016,"prompt_per_second":0.9736135159364955,"predicted_n":35,"predicted_ms":46028.278,"predicted_per_token_ms":1315.0936571428572,"predicted_per_second":0.7604021162816476}}
```

这个模型就显得自信，直接给出了问题的答案。这个模型跑得有点慢 :-(，大概只能跑到0.75 token/second。是因为这个版本是我修改过的Debug版本。

# 小总结

![components on a llm inference machines](/assets/2025-08-17/llm-inference-components.png)

## 认识推理引擎和训练引擎
我们成功地在一台服务器上跑起了两个deepseek-r1的模型，一个是1.5B，另外一个7B。llama.cpp在AI infra中，是一类被称之为***推理引擎***的组件。如果大家有听说用pytorch、tensorflow用编写大模型，那么llama.cpp和它们有什么区别呢：
* ***训练引擎***：在大模型训练中使用的编程框架，典型如pytorch。既要支持前向计算，也支持反向传播，支持反向梯度计算的自动推导。重量级。
* ***推理引擎***：专门用于推理的引擎，不支持反向传播，轻量级，着重考虑推理侧的各种优化。比如 PagedAttention，从GPU到树莓派等不同能力的计算资源的支持等。

除了llama.cpp外，常用的推理引擎还有 vllm，sglang。

## 原来大模型就是这些东西
我们还学会了从huggingface上下载模型，无非就是模型的超参数，各种矩阵的权重和偏置，token embedding。比如deepseek-r1 7B这个模型，下载下来只有8G的大小，包含70亿个参数，它已经表现出了惊人的智能。而人类的神经元只有860亿个，现在最大的大模型参数已经超过3140亿。如果scaling law继续生效的话，这不禁让人遐想联翩：大模型是不是通向AGI的真正道路？超越人类的智能是不是要到来了？

## 推理网关
大模型之上，通过api对外提供服务。如果用于生产服务，那么前端还得有一层推理网关，用于对整个推理集群间的请求进行负载均衡。推理网关大家在重点投入的地盘，它不就是一个负载均衡吗？这么说也不错，不过它跟传统的负载均衡有特殊之处在于：
* 传统的负载均衡处理短连接居多，大部分连接就持续几百毫秒，长连接的需求相对少。而大模型的api请求多数都是长连接。
* 传统的负载均衡调度算法比较简单，比如rr、wrr、lc、wlc等等，一般在负载均衡内部闭环就可以完成。而大模型的调度，由于每个请求的处理较重，且可能不同后端的处理能力不一样，需要由外部scheduler根据应用层的多种指标综合调度。

